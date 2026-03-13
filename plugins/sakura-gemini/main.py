import os
import sys
import time
import random
from collections import defaultdict

# Ensure the plugin directory is on sys.path so `context` package is importable
sys.path.insert(0, os.path.dirname(__file__))

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger, AstrBotConfig
from astrbot.api.message_components import Plain, Image
import astrbot.api.message_components as Comp

from context import ContextManager, ContextMessage, NoiseFilter, PromptGuard


@register("sakura_gemini", "heziqi", "群聊 AI 助手，支持文字+图片问答、双层上下文", "0.1.0")
class SakuraGeminiPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config

        self.cooldown_seconds = int(self.config.get("cooldown_seconds", 10))
        self.segment_length = int(self.config.get("segment_length", 300))
        self.min_msg_length = int(self.config.get("min_msg_length", 3))
        self.system_prompt = str(self.config.get(
            "system_prompt",
            "你是一个群聊 AI 助手。你可以看到群聊的历史消息作为背景信息，以及与当前提问者的对话历史。请根据上下文给出有帮助的回答。",
        ))

        # Proactive reply: fire after every N messages, with an extra probability gate
        self.proactive_every_n = int(self.config.get("proactive_every_n", 10))
        self.proactive_probability = float(self.config.get("proactive_probability", 0.3))
        self.proactive_prompt = str(self.config.get(
            "proactive_prompt",
            "你刚才看了一段群聊记录。作为群里的一员，用你的角色口吻随口插一句你想说的话，要自然，不要像在回答问题。只输出那一句话，不要解释。",
        ))

        self.ctx_mgr = ContextManager(
            group_ctx_max=int(self.config.get("group_ctx_max", 50)),
            user_ctx_max=int(self.config.get("user_ctx_max", 50)),
            ctx_expire_seconds=int(self.config.get("ctx_expire_seconds", 1800)),
        )

        self._db_path = self._get_db_path()
        self.ctx_mgr.load_from_db(self._db_path)

        # Per-user cooldown for @mention replies
        self._cooldowns: dict[tuple[str, str], float] = defaultdict(float)
        # Per-group: message counter and last proactive reply timestamp
        self._proactive_counters: dict[str, int] = defaultdict(int)
        self._proactive_last_ts: dict[str, float] = defaultdict(float)

        self._msg_count_since_save = 0
        self._save_interval = 20

        logger.info("Sakura Gemini 插件已加载，触发方式: @机器人")

    def _get_db_path(self) -> str:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(data_dir, exist_ok=True)
        return os.path.join(data_dir, "context_cache.db")

    def _extract_message_parts(self, event: AstrMessageEvent) -> tuple[str, list[str], bool, bool]:
        """Extract text, image URLs, has_image flag, and is_command flag from event."""
        text_parts = []
        image_urls = []

        for comp in event.message_obj.message:
            if isinstance(comp, Plain):
                text_parts.append(comp.text)
            elif isinstance(comp, Image):
                url = None
                if hasattr(comp, "url") and comp.url:
                    url = comp.url
                elif hasattr(comp, "file") and comp.file:
                    url = comp.file
                if url:
                    image_urls.append(url)

        text = "".join(text_parts).strip()
        is_command = text.startswith("/")
        has_image = len(image_urls) > 0
        return text, image_urls, has_image, is_command

    def _get_sender_id(self, event: AstrMessageEvent) -> str:
        if hasattr(event, "get_sender_id"):
            return str(event.get_sender_id())
        if hasattr(event.message_obj, "sender") and hasattr(event.message_obj.sender, "user_id"):
            return str(event.message_obj.sender.user_id)
        return "unknown"

    def _get_group_id(self, event: AstrMessageEvent) -> str:
        gid = event.message_obj.group_id if hasattr(event.message_obj, "group_id") else None
        return str(gid) if gid else event.unified_msg_origin

    def _split_reply(self, text: str, segment_length: int) -> list[str]:
        """Split reply into segments, preferring paragraph breaks."""
        if len(text) <= segment_length:
            return [text]

        segments = []
        while len(text) > segment_length:
            cut = text.rfind("\n\n", 0, segment_length)
            if cut == -1:
                cut = text.rfind("\n", 0, segment_length)
            if cut == -1:
                cut = segment_length
            segments.append(text[:cut].strip())
            text = text[cut:].strip()
        if text:
            segments.append(text)
        return segments

    async def _handle_query(self, event: AstrMessageEvent, question: str, image_urls: list[str], has_image: bool):
        """Core LLM query logic for @mention replies."""
        group_id = self._get_group_id(event)
        sender_id = self._get_sender_id(event)
        sender_name = event.get_sender_name()

        # Subcommand: clear memory
        if question == "清除记忆":
            self.ctx_mgr.clear_user_context(group_id, sender_id)
            yield event.plain_result("已清除你的对话记忆。")
            return

        # Prompt injection guard
        if PromptGuard.is_injection(question):
            yield event.chain_result([Comp.At(qq=sender_id), Comp.Plain(" 哼，这种小把戏想骗我？才不会上当呢！")])
            return

        # Empty input
        if not question and not has_image:
            yield event.chain_result([Comp.At(qq=sender_id), Comp.Plain(" 请输入你的问题")])
            return

        # Per-user cooldown
        now = time.time()
        cooldown_key = (group_id, sender_id)
        if now - self._cooldowns[cooldown_key] < self.cooldown_seconds:
            return

        provider = self.context.get_using_provider(event.unified_msg_origin)
        if not provider:
            yield event.chain_result([Comp.At(qq=sender_id), Comp.Plain(" 未配置 AI 服务，请联系管理员")])
            return

        messages = self.ctx_mgr.build_llm_messages(
            group_id=group_id,
            user_id=sender_id,
            current_content=question,
            system_prompt=self.system_prompt,
        )

        try:
            resp = await provider.text_chat(
                prompt=question or "请看图片",
                session_id=f"sakura_{group_id}_{sender_id}",
                contexts=messages[:-1],
                image_urls=image_urls or None,
                persist=False,
            )
            reply_text = resp.completion_text or "抱歉，我无法生成回复。"
        except Exception as e:
            logger.error(f"AI 调用失败: {e}")
            yield event.chain_result([Comp.At(qq=sender_id), Comp.Plain(" 请求失败，请稍后再试")])
            return

        self._cooldowns[cooldown_key] = now

        q_msg = ContextMessage(sender_id, sender_name, question or "[图片提问]", image_urls, now, False)
        a_msg = ContextMessage("bot", "Bot", reply_text, [], now, True)
        self.ctx_mgr.add_group_message(group_id, q_msg)
        self.ctx_mgr.add_group_message(group_id, a_msg)
        self.ctx_mgr.add_user_message(group_id, sender_id, q_msg)
        self.ctx_mgr.add_user_message(group_id, sender_id, a_msg)

        segments = self._split_reply(reply_text, self.segment_length)
        for i, segment in enumerate(segments):
            if i == 0:
                yield event.chain_result([Comp.At(qq=sender_id), Comp.Plain(f" {segment}")])
            else:
                yield event.plain_result(segment)

    async def _try_proactive_reply(self, event: AstrMessageEvent, group_id: str):
        """Send a proactive message if conditions are met."""
        if self.proactive_every_n <= 0 or self.proactive_probability <= 0:
            return

        self._proactive_counters[group_id] += 1
        if self._proactive_counters[group_id] < self.proactive_every_n:
            return

        # Hit the N-message threshold — reset counter and apply probability gate
        self._proactive_counters[group_id] = 0
        if random.random() > self.proactive_probability:
            return

        # Cooldown: don't fire too often even if N messages arrive quickly
        now = time.time()
        if now - self._proactive_last_ts[group_id] < self.cooldown_seconds:
            return

        provider = self.context.get_using_provider(event.unified_msg_origin)
        if not provider:
            return

        group_ctx = self.ctx_mgr.get_group_context(group_id)
        if not group_ctx:
            return

        lines = [
            f"{'[Bot]' if m.is_bot_reply else f'[{m.sender_name}]'}: {m.content}"
            for m in group_ctx[-self.proactive_every_n:]
        ]
        chat_summary = "\n".join(lines)

        try:
            resp = await provider.text_chat(
                prompt=self.proactive_prompt,
                session_id=f"sakura_proactive_{group_id}",
                contexts=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "system", "content": f"最近的群聊记录：\n{chat_summary}"},
                ],
                persist=False,
            )
            reply_text = (resp.completion_text or "").strip()
        except Exception as e:
            logger.error(f"主动发言调用失败: {e}")
            return

        if not reply_text:
            return

        self._proactive_last_ts[group_id] = now
        a_msg = ContextMessage("bot", "Bot", reply_text, [], now, True)
        self.ctx_mgr.add_group_message(group_id, a_msg)
        logger.info(f"主动发言 [{group_id}]: {reply_text[:30]}...")
        yield event.plain_result(reply_text)

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def on_group_message(self, event: AstrMessageEvent):
        """Passively record all group messages; occasionally reply proactively."""
        if event.is_at_or_wake_command:
            return

        text, image_urls, has_image, is_command = self._extract_message_parts(event)

        if NoiseFilter.should_filter(
            text=text, has_image=has_image, is_command=is_command, min_length=self.min_msg_length
        ):
            return

        group_id = self._get_group_id(event)
        sender_id = self._get_sender_id(event)
        sender_name = event.get_sender_name()

        self.ctx_mgr.add_group_message(
            group_id,
            ContextMessage(
                sender_id=sender_id,
                sender_name=sender_name,
                content=text,
                image_urls=image_urls,
                timestamp=time.time(),
                is_bot_reply=False,
            ),
        )

        self._msg_count_since_save += 1
        if self._msg_count_since_save >= self._save_interval:
            try:
                self.ctx_mgr.save_to_db(self._db_path)
            except Exception as e:
                logger.error(f"定期保存上下文失败: {e}")
            finally:
                self._msg_count_since_save = 0

        # Proactive reply check
        async for result in self._try_proactive_reply(event, group_id):
            yield result

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def on_at_mention(self, event: AstrMessageEvent):
        """Handle @bot mentions in group chat."""
        if not event.is_at_or_wake_command:
            return
        text, image_urls, has_image, is_command = self._extract_message_parts(event)
        if is_command:
            return
        async for result in self._handle_query(event, text, image_urls, has_image):
            yield result

    async def terminate(self):
        try:
            self.ctx_mgr.save_to_db(self._db_path)
            logger.info("Sakura Gemini 插件已卸载，上下文已保存")
        except Exception as e:
            logger.error(f"插件卸载时保存上下文失败: {e}")
