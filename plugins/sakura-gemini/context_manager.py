import json
import os
import sqlite3
import time
from collections import deque, defaultdict
from dataclasses import dataclass


@dataclass
class ContextMessage:
    sender_id: str
    sender_name: str
    content: str
    image_urls: list[str]
    timestamp: float
    is_bot_reply: bool

    def is_expired(self, expire_seconds: int) -> bool:
        return (time.time() - self.timestamp) > expire_seconds


class ContextManager:
    def __init__(
        self,
        group_ctx_max: int = 50,
        user_ctx_max: int = 50,
        ctx_expire_seconds: int = 1800,
    ):
        self.group_ctx_max = group_ctx_max
        self.user_ctx_max = user_ctx_max
        self.ctx_expire_seconds = ctx_expire_seconds
        # group_id -> deque of ContextMessage
        self._group_ctx: dict[str, deque[ContextMessage]] = defaultdict(
            lambda: deque(maxlen=self.group_ctx_max)
        )
        # group_id -> user_id -> deque of ContextMessage
        self._user_ctx: dict[str, dict[str, deque[ContextMessage]]] = defaultdict(dict)

    def add_group_message(self, group_id: str, msg: ContextMessage) -> None:
        self._group_ctx[group_id].append(msg)

    def add_user_message(self, group_id: str, user_id: str, msg: ContextMessage) -> None:
        if user_id not in self._user_ctx[group_id]:
            self._user_ctx[group_id][user_id] = deque(maxlen=self.user_ctx_max)
        self._user_ctx[group_id][user_id].append(msg)

    def get_group_context(self, group_id: str) -> list[ContextMessage]:
        if group_id not in self._group_ctx:
            return []
        return [
            msg for msg in self._group_ctx[group_id]
            if not msg.is_expired(self.ctx_expire_seconds)
        ]

    def get_user_context(self, group_id: str, user_id: str) -> list[ContextMessage]:
        if group_id not in self._user_ctx or user_id not in self._user_ctx[group_id]:
            return []
        return [
            msg for msg in self._user_ctx[group_id][user_id]
            if not msg.is_expired(self.ctx_expire_seconds)
        ]

    def clear_user_context(self, group_id: str, user_id: str) -> None:
        if group_id in self._user_ctx and user_id in self._user_ctx[group_id]:
            self._user_ctx[group_id][user_id].clear()

    def save_to_db(self, db_path: str) -> None:
        """Persist all context to SQLite."""
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS context_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ctx_type TEXT NOT NULL,
                    group_id TEXT NOT NULL,
                    user_id TEXT DEFAULT '',
                    sender_id TEXT NOT NULL,
                    sender_name TEXT NOT NULL,
                    content TEXT NOT NULL,
                    image_urls TEXT NOT NULL DEFAULT '[]',
                    timestamp REAL NOT NULL,
                    is_bot_reply INTEGER NOT NULL DEFAULT 0
                )
            """)
            conn.execute("DELETE FROM context_messages")

            for group_id, ctx_deque in self._group_ctx.items():
                for msg in ctx_deque:
                    conn.execute(
                        "INSERT INTO context_messages (ctx_type, group_id, sender_id, sender_name, content, image_urls, timestamp, is_bot_reply) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        ("group", group_id, msg.sender_id, msg.sender_name, msg.content, json.dumps(msg.image_urls), msg.timestamp, int(msg.is_bot_reply)),
                    )

            for group_id, users in self._user_ctx.items():
                for user_id, ctx_deque in users.items():
                    for msg in ctx_deque:
                        conn.execute(
                            "INSERT INTO context_messages (ctx_type, group_id, user_id, sender_id, sender_name, content, image_urls, timestamp, is_bot_reply) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            ("user", group_id, user_id, msg.sender_id, msg.sender_name, msg.content, json.dumps(msg.image_urls), msg.timestamp, int(msg.is_bot_reply)),
                        )

            conn.commit()
        finally:
            conn.close()

    def load_from_db(self, db_path: str) -> None:
        """Load context from SQLite, skipping expired messages."""
        if not os.path.exists(db_path):
            return
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.execute(
                "SELECT ctx_type, group_id, user_id, sender_id, sender_name, content, image_urls, timestamp, is_bot_reply FROM context_messages ORDER BY timestamp ASC"
            )
            for row in cursor:
                ctx_type, group_id, user_id, sender_id, sender_name, content, image_urls_json, timestamp, is_bot_reply = row
                msg = ContextMessage(
                    sender_id=sender_id,
                    sender_name=sender_name,
                    content=content,
                    image_urls=json.loads(image_urls_json),
                    timestamp=timestamp,
                    is_bot_reply=bool(is_bot_reply),
                )
                if msg.is_expired(self.ctx_expire_seconds):
                    continue
                if ctx_type == "group":
                    self.add_group_message(group_id, msg)
                elif ctx_type == "user":
                    self.add_user_message(group_id, user_id, msg)
        finally:
            conn.close()

    def build_llm_messages(
        self,
        group_id: str,
        user_id: str,
        current_content: str,
        current_image_urls: list[str],
        system_prompt: str,
    ) -> list[dict]:
        """Build the full message list for LLM call with dual-layer context."""
        messages = []

        # 1. System prompt
        messages.append({"role": "system", "content": system_prompt})

        # 2. Group context as background
        group_ctx = self.get_group_context(group_id)
        if group_ctx:
            group_lines = []
            for msg in group_ctx:
                prefix = "[Bot]" if msg.is_bot_reply else f"[{msg.sender_name}]"
                line = f"{prefix}: {msg.content}"
                if msg.image_urls:
                    line += " [图片]"
                group_lines.append(line)
            group_summary = "以下是最近的群聊记录：\n" + "\n".join(group_lines)
            messages.append({"role": "system", "content": group_summary})

        # 3. User conversation history
        user_ctx = self.get_user_context(group_id, user_id)
        for msg in user_ctx:
            role = "assistant" if msg.is_bot_reply else "user"
            messages.append({"role": role, "content": msg.content})

        # 4. Current question
        if current_image_urls:
            content = []
            if current_content:
                content.append({"type": "text", "text": current_content})
            for url in current_image_urls:
                content.append({"type": "image_url", "image_url": {"url": url}})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": current_content})

        return messages


class NoiseFilter:
    @staticmethod
    def should_filter(
        text: str,
        has_image: bool,
        is_command: bool,
        min_length: int = 3,
    ) -> bool:
        """Return True if the message should be filtered (not recorded)."""
        if is_command:
            return True
        if has_image:
            return False
        stripped = text.strip()
        if len(stripped) < min_length:
            return True
        return False
