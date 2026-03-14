import os
import sys
import time
import random
from collections import defaultdict
from datetime import datetime, timezone, timedelta

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
        self.render_probability = float(self.config.get("render_probability", 0.5))

        # When True, skip LLM calls and delegate @mention responses to angel_heart
        self.delegate_to_angel_heart = bool(self.config.get("delegate_to_angel_heart", False))

        # Admin QQ list for bot control commands (comma-separated)
        admin_raw = str(self.config.get("admin_qq_list", ""))
        self.admin_qq_list: set[str] = {x.strip() for x in admin_raw.split(",") if x.strip()}

        # Per-group pause state (resets on plugin reload)
        self._paused_groups: set[str] = set()

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

    def _build_agent_cfg(self):
        """Build MainAgentBuildConfig from AstrBot's global config.

        Reads provider_settings, timezone, cron tools toggle, etc. from the
        live AstrBot config so all framework features (persona, datetime,
        web search, tool calling) are applied when build_main_agent runs.
        """
        from astrbot.core.astr_main_agent import MainAgentBuildConfig

        cfg = self.context.get_config()
        settings = cfg.get("provider_settings", {})
        proactive_cfg = cfg.get("proactive_capability", {})

        return MainAgentBuildConfig(
            tool_call_timeout=settings.get("tool_call_timeout", 60),
            provider_settings=settings,
            add_cron_tools=proactive_cfg.get("add_cron_tools", True),
            timezone=cfg.get("timezone"),
            streaming_response=False,
        )

    async def _call_agent(
        self,
        event: AstrMessageEvent,
        prompt: str,
        user_contexts: list[dict],
        group_context_text: str,
        image_urls: list[str],
    ) -> str | None:
        """Unified LLM call via AstrBot's build_main_agent pipeline.

        Injects group context as an extra user content part so AstrBot's
        framework features (persona, datetime, web search, tools) all
        apply normally on top of our custom context.

        Returns completion text, or None on failure (caller handles error reply).
        """
        from astrbot.core.astr_main_agent import build_main_agent
        from astrbot.core.provider.entities import ProviderRequest
        from astrbot.core.provider.entites import TextPart
        from astrbot.core.astr_agent_run_util import run_agent

        req = ProviderRequest()
        req.prompt = prompt or "请看图片"
        req.image_urls = image_urls or []
        req.contexts = user_contexts  # user Q&A history (OpenAI format)

        if group_context_text:
            req.extra_user_content_parts.append(
                TextPart(text=f"<group_context>\n{group_context_text}\n</group_context>")
            )

        try:
            build_result = await build_main_agent(
                event=event,
                plugin_context=self.context,
                config=self._build_agent_cfg(),
                req=req,
                apply_reset=False,
            )
            if build_result is None:
                return None

            if build_result.reset_coro:
                await build_result.reset_coro

            async for _ in run_agent(
                build_result.agent_runner,
                max_step=10,
                show_tool_use=False,
            ):
                pass

            final_resp = build_result.agent_runner.get_final_llm_resp()
            return final_resp.completion_text if final_resp else None

        except Exception as e:
            logger.error(f"AI 调用失败: {e}")
            return None

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
            yield event.chain_result([Comp.At(qq=sender_id)])
            yield event.plain_result("哼，这种小把戏想骗我？才不会上当呢！")
            return

        # Empty input
        if not question and not has_image:
            yield event.chain_result([Comp.At(qq=sender_id)])
            yield event.plain_result("请输入你的问题")
            return

        # Per-user cooldown
        now = time.time()
        cooldown_key = (group_id, sender_id)
        if now - self._cooldowns[cooldown_key] < self.cooldown_seconds:
            return

        group_context_text = self.ctx_mgr.format_group_context(group_id)
        user_contexts = self.ctx_mgr.build_user_contexts(group_id, sender_id)

        reply_text = await self._call_agent(
            event=event,
            prompt=question,
            user_contexts=user_contexts,
            group_context_text=group_context_text,
            image_urls=image_urls,
        )

        if reply_text is None:
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
        use_render = random.random() < self.render_probability
        if use_render:
            # Send @mention separately so 传话筒 sees pure-text messages and renders them
            yield event.chain_result([Comp.At(qq=sender_id)])
            for segment in segments:
                yield event.plain_result(segment)
        else:
            # Merge @mention with text so 传话筒 skips rendering (has_non_text=True)
            full_text = "\n".join(segments)
            yield event.chain_result([Comp.At(qq=sender_id), Comp.Plain(f" {full_text}")])

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

        group_context_text = self.ctx_mgr.format_group_context(
            group_id, n=self.proactive_every_n
        )
        if not group_context_text:
            return

        reply_text = await self._call_agent(
            event=event,
            prompt=self.proactive_prompt,
            user_contexts=[],
            group_context_text=group_context_text,
            image_urls=[],
        )

        if not reply_text:
            return

        self._proactive_last_ts[group_id] = now
        a_msg = ContextMessage("bot", "Bot", reply_text, [], now, True)
        self.ctx_mgr.add_group_message(group_id, a_msg)
        logger.info(f"主动发言 [{group_id}]: {reply_text[:30]}...")
        yield event.plain_result(reply_text)

    _BOT_OFF_CMDS = {"暂停", "休息", "下线", "关闭", "/off"}
    _BOT_ON_CMDS = {"恢复", "上线", "开启", "/on"}

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def on_group_message(self, event: AstrMessageEvent):
        """Passively record all group messages; occasionally reply proactively."""
        if event.is_at_or_wake_command:
            return

        group_id = self._get_group_id(event)
        if group_id in self._paused_groups:
            return

        text, image_urls, has_image, is_command = self._extract_message_parts(event)

        if NoiseFilter.should_filter(
            text=text, has_image=has_image, is_command=is_command, min_length=self.min_msg_length
        ):
            return

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

        # Proactive reply check (disabled when delegating to angel_heart)
        if not self.delegate_to_angel_heart:
            async for result in self._try_proactive_reply(event, group_id):
                yield result

    _PLUGIN_KEYWORDS = (
        "点歌", "网易点歌", "QQ点歌", "酷狗点歌", "酷我点歌",
        "查歌词", "歌单收藏", "歌单取藏", "歌单列表", "歌单点歌",
        "群分析", "group_analysis", "分析设置", "设置格式",
    )

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def on_at_mention(self, event: AstrMessageEvent):
        """Handle @bot mentions in group chat."""
        if not event.is_at_or_wake_command:
            return
        text, image_urls, has_image, is_command = self._extract_message_parts(event)
        if is_command:
            return

        group_id = self._get_group_id(event)
        sender_id = self._get_sender_id(event)
        cmd = text.strip()

        # Bot control commands (admin only)
        if cmd in self._BOT_OFF_CMDS or cmd in self._BOT_ON_CMDS:
            if self.admin_qq_list and sender_id not in self.admin_qq_list:
                yield event.chain_result([Comp.At(qq=sender_id), Comp.Plain(" 你没有权限控制我哦")])
                return
            if cmd in self._BOT_OFF_CMDS:
                self._paused_groups.add(group_id)
                logger.info(f"Bot paused in group {group_id} by {sender_id}")
                yield event.plain_result("好，先去休息了，有事叫我。")
            else:
                self._paused_groups.discard(group_id)
                logger.info(f"Bot resumed in group {group_id} by {sender_id}")
                yield event.plain_result("回来了！")
            return

        # Silently ignore all @mentions when paused
        if group_id in self._paused_groups:
            return

        # Let other plugins handle their own keywords; add emoji only for those
        first_word = text.split()[0] if text.split() else ""
        if first_word in self._PLUGIN_KEYWORDS:
            try:
                msg_id = event.message_obj.message_id
                bot = getattr(event, "bot", None)
                if bot and msg_id:
                    await bot.call_action("set_msg_emoji_like", message_id=msg_id, emoji_id="212")
            except Exception:
                pass
            return
        # In delegate mode, let angel_heart handle the actual LLM response
        if self.delegate_to_angel_heart:
            return
        async for result in self._handle_query(event, text, image_urls, has_image):
            yield result

    async def terminate(self):
        try:
            self.ctx_mgr.save_to_db(self._db_path)
            logger.info("Sakura Gemini 插件已卸载，上下文已保存")
        except Exception as e:
            logger.error(f"插件卸载时保存上下文失败: {e}")
