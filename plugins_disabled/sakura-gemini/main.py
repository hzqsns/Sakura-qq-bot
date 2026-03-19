import os
import sys
import time
import random
from collections import defaultdict
from typing import Any
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
    _IMAGE_IDENTITY_HINTS = ("谁", "人物", "身份", "是不是", "识别", "哪位", "真人")

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config

        self.cooldown_seconds = int(self.config.get("cooldown_seconds", 10))
        self.segment_length = int(self.config.get("segment_length", 300))
        self.min_msg_length = int(self.config.get("min_msg_length", 3))

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

    def _get_event_chain(self, event: AstrMessageEvent) -> list[object]:
        try:
            chain = event.get_messages()
            if isinstance(chain, list):
                return chain
        except Exception:
            pass
        return getattr(event.message_obj, "message", []) or []

    def _append_image_candidates(self, image_urls: list[str], seen: set[str], comp: object):
        for key in ("url", "file", "path", "src", "base64", "data"):
            try:
                value = getattr(comp, key, None)
            except Exception:
                value = None
            if isinstance(value, str) and value and value not in seen:
                seen.add(value)
                image_urls.append(value)
        try:
            data = getattr(comp, "data", None)
        except Exception:
            data = None
        if isinstance(data, dict):
            for key in ("url", "file", "path", "src", "base64", "data"):
                value = data.get(key)
                if isinstance(value, str) and value and value not in seen:
                    seen.add(value)
                    image_urls.append(value)

    def _extract_text_and_images_from_chain(self, chain: list[object]) -> tuple[str, list[str]]:
        text_parts: list[str] = []
        image_urls: list[str] = []
        seen: set[str] = set()

        if not isinstance(chain, list):
            return "", []

        for comp in chain:
            try:
                if isinstance(comp, Plain):
                    text_parts.append(comp.text)
                elif isinstance(comp, Image):
                    self._append_image_candidates(image_urls, seen, comp)
                elif hasattr(Comp, "Node") and isinstance(comp, getattr(Comp, "Node")):
                    content = getattr(comp, "content", None)
                    if isinstance(content, list):
                        sub_text, sub_images = self._extract_text_and_images_from_chain(content)
                        if sub_text:
                            text_parts.append(sub_text)
                        for item in sub_images:
                            if item not in seen:
                                seen.add(item)
                                image_urls.append(item)
                elif hasattr(Comp, "Nodes") and isinstance(comp, getattr(Comp, "Nodes")):
                    nodes = getattr(comp, "nodes", None) or getattr(comp, "content", None)
                    if isinstance(nodes, list):
                        for node in nodes:
                            content = getattr(node, "content", None)
                            if isinstance(content, list):
                                sub_text, sub_images = self._extract_text_and_images_from_chain(content)
                                if sub_text:
                                    text_parts.append(sub_text)
                                for item in sub_images:
                                    if item not in seen:
                                        seen.add(item)
                                        image_urls.append(item)
                elif hasattr(Comp, "Forward") and isinstance(comp, getattr(Comp, "Forward")):
                    nodes = getattr(comp, "nodes", None) or getattr(comp, "content", None)
                    if isinstance(nodes, list):
                        for node in nodes:
                            content = getattr(node, "content", None)
                            if isinstance(content, list):
                                sub_text, sub_images = self._extract_text_and_images_from_chain(content)
                                if sub_text:
                                    text_parts.append(sub_text)
                                for item in sub_images:
                                    if item not in seen:
                                        seen.add(item)
                                        image_urls.append(item)
            except Exception as e:
                logger.warning(f"解析消息链片段失败: {e}")

        return "".join(text_parts).strip(), image_urls

    def _get_reply_message_id(self, reply_comp: object) -> str | None:
        for key in ("id", "message_id", "reply_id", "messageId", "message_seq"):
            value = getattr(reply_comp, key, None)
            if isinstance(value, (str, int)) and str(value):
                return str(value)
        data = getattr(reply_comp, "data", None)
        if isinstance(data, dict):
            for key in ("id", "message_id", "reply", "messageId", "message_seq"):
                value = data.get(key)
                if isinstance(value, (str, int)) and str(value):
                    return str(value)
        return None

    async def _call_get_msg(self, event: AstrMessageEvent, message_id: str) -> dict[str, Any] | None:
        if not (isinstance(message_id, str) and message_id.strip()):
            return None
        if (
            not hasattr(event, "bot")
            or not hasattr(event.bot, "api")
            or not hasattr(event.bot.api, "call_action")
        ):
            return None

        mid = message_id.strip()
        params_list: list[dict[str, Any]] = [
            {"message_id": mid},
            {"id": mid},
        ]
        if mid.isdigit():
            params_list.insert(1, {"message_id": int(mid)})
            params_list.append({"id": int(mid)})

        for params in params_list:
            try:
                ret = await event.bot.api.call_action("get_msg", **params)
                if isinstance(ret, dict):
                    data = ret.get("data")
                    return data if isinstance(data, dict) else ret
            except Exception:
                continue
        return None

    async def _resolve_image_refs_for_llm(
        self, event: AstrMessageEvent, image_refs: list[str]
    ) -> list[str]:
        refs = [ref for ref in image_refs if isinstance(ref, str) and ref.strip()]
        if not refs:
            return []
        try:
            from astrbot.core.utils.quoted_message.image_resolver import ImageResolver

            resolved = await ImageResolver(event).resolve_for_llm(refs)
            if resolved:
                logger.info(
                    "Sakura Gemini: resolved image refs for llm, input=%s output=%s",
                    len(refs),
                    len(resolved),
                )
                return resolved
        except Exception as e:
            logger.warning(f"Sakura Gemini: resolve image refs failed: {e}")
        return refs

    def _extract_from_onebot_message_payload(self, payload: Any) -> tuple[str, list[str]]:
        if isinstance(payload, dict):
            message = payload.get("message")
            if isinstance(message, list):
                text_parts: list[str] = []
                image_urls: list[str] = []
                seen: set[str] = set()
                for seg in message:
                    if not isinstance(seg, dict):
                        continue
                    seg_type = seg.get("type")
                    data = seg.get("data", {}) if isinstance(seg.get("data"), dict) else {}
                    if seg_type == "text":
                        text = data.get("text")
                        if isinstance(text, str):
                            text_parts.append(text)
                    elif seg_type == "image":
                        for key in ("url", "file", "path", "src", "base64"):
                            value = data.get(key)
                            if isinstance(value, str) and value and value not in seen:
                                seen.add(value)
                                image_urls.append(value)
                return "".join(text_parts).strip(), image_urls
        return "", []

    async def _extract_reply_message_parts(self, event: AstrMessageEvent) -> tuple[str, list[str]]:
        for comp in self._get_event_chain(event):
            try:
                if not isinstance(comp, Comp.Reply):
                    continue
            except Exception:
                continue

            for attr in ("message", "origin", "content"):
                payload = getattr(comp, attr, None)
                if isinstance(payload, list):
                    text, images = self._extract_text_and_images_from_chain(payload)
                    if text or images:
                        return text, images

            reply_id = self._get_reply_message_id(comp)
            if reply_id:
                payload = await self._call_get_msg(event, reply_id)
                text, images = self._extract_from_onebot_message_payload(payload)
                if text or images:
                    images = await self._resolve_image_refs_for_llm(event, images)
                    logger.info(
                        "Sakura Gemini: fetched quoted payload via get_msg, images=%s",
                        len(images),
                    )
                    return text, images
            break

        return "", []

    async def _extract_message_parts(
        self, event: AstrMessageEvent
    ) -> tuple[str, list[str], bool, bool]:
        """Extract text, image URLs, has_image flag, and is_command flag from event."""
        text, image_urls = self._extract_text_and_images_from_chain(self._get_event_chain(event))
        quoted_text, quoted_images = await self._extract_reply_message_parts(event)

        seen = set(image_urls)
        for image in quoted_images:
            if image not in seen:
                seen.add(image)
                image_urls.append(image)

        is_command = text.startswith("/")
        has_image = len(image_urls) > 0
        if has_image:
            image_urls = await self._resolve_image_refs_for_llm(event, image_urls)
            has_image = len(image_urls) > 0
        if quoted_images:
            logger.info(
                "Sakura Gemini: merged quoted images into current request, current=%s quoted=%s text=%r quoted_text=%r",
                len(image_urls) - len(quoted_images),
                len(quoted_images),
                text[:50],
                quoted_text[:50],
            )
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

    def _is_private_message(self, event: AstrMessageEvent) -> bool:
        try:
            is_private_chat = getattr(event, "is_private_chat", None)
            if callable(is_private_chat):
                return bool(is_private_chat())
        except Exception:
            pass
        try:
            mt = getattr(event, "get_message_type", None)
            mt = mt() if callable(mt) else None
            lowered = str(mt).lower()
            return "private" in lowered or "friend" in lowered
        except Exception:
            return False

    def _is_identity_image_query(self, question: str) -> bool:
        text = (question or "").strip()
        return any(word in text for word in self._IMAGE_IDENTITY_HINTS)

    def _build_effective_prompt(self, question: str, has_image: bool) -> str:
        text = (question or "").strip()
        if not has_image:
            return text

        if self._is_identity_image_query(text):
            return (
                "请基于图片内容回答，但不要识别、猜测或确认真实人物身份。"
                "请直接说明无法仅凭图片确认真实身份，然后继续描述画面里的人数、外观、服饰、动作和场景。"
                "不要使用角色扮演口吻，不要闲聊。"
                f"\n用户问题：{text or '这是谁'}"
            )

        return (
            "请基于图片内容直接回答，优先说明画面主体、人数、动作、服饰、场景和可见文字。"
            "回答要实用、准确、简洁，不要使用角色扮演口吻，不要跑题。"
            f"\n用户问题：{text or '请描述这张图'}"
        )

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

    def _build_system_prompt(self, has_image: bool) -> str:
        """Keep answers practical and avoid persona/tool interference."""
        if has_image:
            return (
                "你是一个实用型中文看图助手。"
                "请直接根据图片内容回答，优先描述人物数量、外观、动作、服饰、场景、物品、文字信息。"
                "不要角色扮演，不要使用二次元口癖，不要输出与问题无关的寒暄。"
                "如果用户想让你识别真人身份、判断是谁、是不是某个具体人物，你要明确说明无法确认身份，"
                "但可以继续描述外观、动作、场景和可见细节。"
                "如果图片里信息不够清楚，就明确说不确定的部分，同时给出你能确认的画面内容。"
                "回答使用自然中文，优先准确、简洁、实用。"
            )
        return (
            "你是一个实用型中文助手。"
            "请直接、准确回答用户问题，不要角色扮演，不要使用夸张口癖，不要输出与问题无关的内容。"
            "如果问题信息不足，就先基于现有信息给出最有帮助的回答。"
        )

    async def _call_agent(
        self,
        event: AstrMessageEvent,
        prompt: str,
        user_contexts: list[dict],
        group_context_text: str,
        image_urls: list[str],
    ) -> tuple[str | None, str | None]:
        """Call the active provider directly for predictable Q&A and image QA."""
        provider = self.context.get_using_provider(event.unified_msg_origin)
        if provider is None:
            return None, "当前没有可用的模型提供商，请检查 AstrBot 的 provider 配置。"

        effective_prompt = prompt or "请描述这张图"
        if group_context_text:
            effective_prompt = (
                f"{effective_prompt}\n\n"
                f"以下是最近群聊上下文，仅在有帮助时参考：\n{group_context_text}"
            )

        try:
            final_resp = await provider.text_chat(
                prompt=effective_prompt,
                image_urls=image_urls or [],
                contexts=user_contexts or None,
                system_prompt=self._build_system_prompt(bool(image_urls)),
            )
            if final_resp is None:
                return None, self._format_agent_error("", prompt, bool(image_urls))
            if final_resp.role == "err":
                logger.warning(f"LLM 返回错误响应: {final_resp.completion_text}")
                return None, self._format_agent_error(
                    final_resp.completion_text, prompt, bool(image_urls)
                )
            return final_resp.completion_text, None

        except Exception as e:
            logger.error(f"AI 调用失败: {e}")
            return None, self._format_agent_error(str(e), prompt, bool(image_urls))

    def _format_agent_error(self, error_text: str | None, prompt: str, has_image: bool) -> str:
        err = (error_text or "").lower()
        question = (prompt or "").strip()

        if "prohibited_content" in err or "content_filter" in err:
            if has_image and any(word in question for word in ("谁", "人物", "身份", "是不是", "识别")):
                return "这类图片我不能帮你直接识别人是谁，但可以帮你描述画面、穿着、动作和场景。你可以换个问法，比如“这张图里的人在做什么”。"
            if has_image:
                return "这张图我这次没法直接回答，你可以换个更具体、更中性的问法，我可以继续帮你描述画面内容。"
            return "这个问题我这次没法直接回答，你可以换个更具体的问法再试一次。"

        if has_image:
            return "这次图片请求处理失败了。你可以重新发送图片，或者直接发图再 @我试一次。"
        return "这次请求失败了，请稍后再试。"

    async def _handle_query(self, event: AstrMessageEvent, question: str, image_urls: list[str], has_image: bool):
        """Core LLM query logic for @mention replies."""
        group_id = self._get_group_id(event)
        sender_id = self._get_sender_id(event)
        sender_name = event.get_sender_name()
        is_private = self._is_private_message(event)

        # Subcommand: clear memory
        if question == "清除记忆":
            self.ctx_mgr.clear_user_context(group_id, sender_id)
            yield event.plain_result("已清除你的对话记忆。")
            return

        # Prompt injection guard
        if PromptGuard.is_injection(question):
            if not is_private:
                yield event.chain_result([Comp.At(qq=sender_id)])
            yield event.plain_result("哼，这种小把戏想骗我？才不会上当呢！")
            return

        # Empty input
        if not question and not has_image:
            if not is_private:
                yield event.chain_result([Comp.At(qq=sender_id)])
            yield event.plain_result("请输入你的问题")
            return

        # Per-user cooldown
        now = time.time()
        cooldown_key = (group_id, sender_id)
        if now - self._cooldowns[cooldown_key] < self.cooldown_seconds:
            return

        if has_image:
            group_context_text = ""
            user_contexts = []
        else:
            group_context_text = self.ctx_mgr.format_group_context(group_id)
            user_contexts = self.ctx_mgr.build_user_contexts(group_id, sender_id)

        effective_prompt = self._build_effective_prompt(question, has_image)

        reply_text, error_text = await self._call_agent(
            event=event,
            prompt=effective_prompt,
            user_contexts=user_contexts,
            group_context_text=group_context_text,
            image_urls=image_urls,
        )

        if reply_text is None:
            if is_private:
                yield event.plain_result(error_text or "请求失败，请稍后再试")
            else:
                yield event.chain_result([Comp.At(qq=sender_id), Comp.Plain(f" {error_text or '请求失败，请稍后再试'}")])
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
        if is_private:
            full_text = "\n".join(segments)
            yield event.plain_result(full_text)
            return

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

        reply_text, _error_text = await self._call_agent(
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

        text, image_urls, has_image, is_command = await self._extract_message_parts(event)

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
        text, image_urls, has_image, is_command = await self._extract_message_parts(event)
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
        event.should_call_llm(False)
        async for result in self._handle_query(event, text, image_urls, has_image):
            yield result

    @filter.event_message_type(filter.EventMessageType.PRIVATE_MESSAGE)
    async def on_private_message(self, event: AstrMessageEvent):
        """Handle direct private messages for clean debugging and image QA."""
        if not self._is_private_message(event):
            return

        logger.info("Sakura Gemini: handling private message via direct QA path")

        text, image_urls, has_image, is_command = await self._extract_message_parts(event)
        if is_command:
            return

        event.should_call_llm(False)
        async for result in self._handle_query(event, text, image_urls, has_image):
            yield result

    async def terminate(self):
        try:
            self.ctx_mgr.save_to_db(self._db_path)
            logger.info("Sakura Gemini 插件已卸载，上下文已保存")
        except Exception as e:
            logger.error(f"插件卸载时保存上下文失败: {e}")
