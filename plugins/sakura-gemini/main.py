import time
from collections import defaultdict

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger, AstrBotConfig
from astrbot.api.message_components import Plain, Image, Face
from astrbot.api.platform import MessageType
import astrbot.api.message_components as Comp

from context_manager import ContextManager, ContextMessage, NoiseFilter


@register("sakura_gemini", "heziqi", "群聊 AI 助手，支持文字+图片问答、双层上下文", "0.1.0")
class SakuraGeminiPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config

        self.trigger_word = str(self.config.get("trigger_word", "gemini"))
        self.cooldown_seconds = int(self.config.get("cooldown_seconds", 10))
        self.max_reply_length = int(self.config.get("max_reply_length", 500))
        self.min_msg_length = int(self.config.get("min_msg_length", 3))
        self.system_prompt = str(self.config.get("system_prompt", "你是一个群聊 AI 助手。你可以看到群聊的历史消息作为背景信息，以及与当前提问者的对话历史。请根据上下文给出有帮助的回答。"))

        self.ctx_mgr = ContextManager(
            group_ctx_max=int(self.config.get("group_ctx_max", 50)),
            user_ctx_max=int(self.config.get("user_ctx_max", 50)),
            ctx_expire_seconds=int(self.config.get("ctx_expire_seconds", 1800)),
        )

        self._db_path = self._get_db_path()
        self.ctx_mgr.load_from_db(self._db_path)

        self._cooldowns: dict[str, float] = defaultdict(float)
        self._msg_count_since_save = 0
        self._save_interval = 20

        logger.info(f"Sakura Gemini 插件已加载，触发词: {self.trigger_word}")

    def _get_db_path(self) -> str:
        import os
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
                if hasattr(comp, 'url') and comp.url:
                    url = comp.url
                elif hasattr(comp, 'file') and comp.file:
                    url = comp.file
                if url:
                    image_urls.append(url)

        text = "".join(text_parts).strip()
        is_command = text.startswith("/")
        has_image = len(image_urls) > 0
        return text, image_urls, has_image, is_command

    def _get_sender_id(self, event: AstrMessageEvent) -> str:
        if hasattr(event, 'get_sender_id'):
            return str(event.get_sender_id())
        if hasattr(event.message_obj, 'sender') and hasattr(event.message_obj.sender, 'user_id'):
            return str(event.message_obj.sender.user_id)
        return "unknown"

    def _get_group_id(self, event: AstrMessageEvent) -> str:
        gid = event.message_obj.group_id if hasattr(event.message_obj, 'group_id') else None
        return str(gid) if gid else event.unified_msg_origin

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def on_group_message(self, event: AstrMessageEvent):
        """Passively record all group messages to group context."""
        text, image_urls, has_image, is_command = self._extract_message_parts(event)

        # Skip gemini commands (handled by on_gemini_command)
        if text.lower().startswith(self.trigger_word):
            return

        if NoiseFilter.should_filter(text=text, has_image=has_image, is_command=is_command, min_length=self.min_msg_length):
            return

        group_id = self._get_group_id(event)
        sender_id = self._get_sender_id(event)
        sender_name = event.get_sender_name()

        msg = ContextMessage(
            sender_id=sender_id,
            sender_name=sender_name,
            content=text,
            image_urls=image_urls,
            timestamp=time.time(),
            is_bot_reply=False,
        )
        self.ctx_mgr.add_group_message(group_id, msg)

        self._msg_count_since_save += 1
        if self._msg_count_since_save >= self._save_interval:
            self.ctx_mgr.save_to_db(self._db_path)
            self._msg_count_since_save = 0

    @filter.command("gemini")
    async def on_gemini_command(self, event: AstrMessageEvent):
        """Handle gemini command: merge context, call LLM, reply."""
        group_id = self._get_group_id(event)
        sender_id = self._get_sender_id(event)
        sender_name = event.get_sender_name()

        text, image_urls, has_image, _ = self._extract_message_parts(event)

        # Strip trigger word
        question = text
        if question.lower().startswith(self.trigger_word):
            question = question[len(self.trigger_word):].strip()

        # Subcommand: clear memory
        if question == "清除记忆":
            self.ctx_mgr.clear_user_context(group_id, sender_id)
            yield event.plain_result("已清除你的对话记忆。")
            return

        # Empty input
        if not question and not has_image:
            yield event.chain_result([Comp.At(qq=sender_id), Comp.Plain(" 请输入你的问题")])
            return

        # Cooldown check
        now = time.time()
        if now - self._cooldowns[group_id] < self.cooldown_seconds:
            return
        self._cooldowns[group_id] = now

        # Get provider
        provider = self.context.get_using_provider(event.unified_msg_origin)
        if not provider:
            yield event.chain_result([Comp.At(qq=sender_id), Comp.Plain(" 未配置 AI 服务，请联系管理员")])
            return

        # Build context and call LLM
        messages = self.ctx_mgr.build_llm_messages(
            group_id=group_id,
            user_id=sender_id,
            current_content=question,
            current_image_urls=image_urls,
            system_prompt=self.system_prompt,
        )

        try:
            resp = await provider.text_chat(
                prompt=question or "请看图片",
                session_id=f"sakura_{group_id}_{sender_id}",
                contexts=messages[:-1],
                image_urls=image_urls if image_urls else None,
                persist=False,
            )
            reply_text = resp.completion_text or "抱歉，我无法生成回复。"
        except Exception as e:
            logger.error(f"Gemini 调用失败: {e}")
            yield event.chain_result([Comp.At(qq=sender_id), Comp.Plain(" 请求失败，请稍后再试")])
            return

        # Truncate if too long
        if len(reply_text) > self.max_reply_length:
            reply_text = reply_text[:self.max_reply_length] + "\n...(内容过长，已截取)"

        # Record to both context layers
        q_msg = ContextMessage(sender_id, sender_name, question or "[图片提问]", image_urls, now, False)
        a_msg = ContextMessage("bot", "Bot", reply_text, [], now, True)
        self.ctx_mgr.add_group_message(group_id, q_msg)
        self.ctx_mgr.add_group_message(group_id, a_msg)
        self.ctx_mgr.add_user_message(group_id, sender_id, q_msg)
        self.ctx_mgr.add_user_message(group_id, sender_id, a_msg)

        yield event.chain_result([Comp.At(qq=sender_id), Comp.Plain(f" {reply_text}")])

    async def terminate(self):
        self.ctx_mgr.save_to_db(self._db_path)
        logger.info("Sakura Gemini 插件已卸载，上下文已保存")
