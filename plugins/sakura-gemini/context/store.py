import json
import os
import sqlite3
from collections import deque, defaultdict

from .models import ContextMessage


class ContextManager:
    """
    Manages dual-layer conversation context (group-level + user-level).

    Storage: in-memory deques with maxlen cap (fast reads/writes).
    Persistence: SQLite for surviving restarts (save_to_db / load_from_db).
    Expiry: lazy — expired messages are filtered out at read time.
    """

    def __init__(
        self,
        group_ctx_max: int = 50,
        user_ctx_max: int = 50,
        ctx_expire_seconds: int = 1800,
    ):
        self.group_ctx_max = group_ctx_max
        self.user_ctx_max = user_ctx_max
        self.ctx_expire_seconds = ctx_expire_seconds

        # group_id -> deque[ContextMessage]
        self._group_ctx: dict[str, deque[ContextMessage]] = defaultdict(
            lambda: deque(maxlen=self.group_ctx_max)
        )
        # group_id -> user_id -> deque[ContextMessage]
        self._user_ctx: dict[str, dict[str, deque[ContextMessage]]] = defaultdict(dict)

    # ── Write ──────────────────────────────────────────────────────────────

    def add_group_message(self, group_id: str, msg: ContextMessage) -> None:
        self._group_ctx[group_id].append(msg)

    def add_user_message(self, group_id: str, user_id: str, msg: ContextMessage) -> None:
        if user_id not in self._user_ctx[group_id]:
            self._user_ctx[group_id][user_id] = deque(maxlen=self.user_ctx_max)
        self._user_ctx[group_id][user_id].append(msg)

    def clear_user_context(self, group_id: str, user_id: str) -> None:
        if group_id in self._user_ctx and user_id in self._user_ctx[group_id]:
            self._user_ctx[group_id][user_id].clear()

    # ── Read ───────────────────────────────────────────────────────────────

    def get_group_context(self, group_id: str) -> list[ContextMessage]:
        """Return non-expired group messages, oldest first."""
        return [
            msg for msg in self._group_ctx.get(group_id, [])
            if not msg.is_expired(self.ctx_expire_seconds)
        ]

    def get_user_context(self, group_id: str, user_id: str) -> list[ContextMessage]:
        """Return non-expired user messages, oldest first."""
        user_map = self._user_ctx.get(group_id, {})
        return [
            msg for msg in user_map.get(user_id, [])
            if not msg.is_expired(self.ctx_expire_seconds)
        ]

    # ── LLM message building ───────────────────────────────────────────────

    def build_llm_messages(
        self,
        group_id: str,
        user_id: str,
        current_content: str,
        system_prompt: str,
    ) -> list[dict]:
        """
        Build the full message list for an LLM call.

        Layout:
          1. system prompt
          2. group chat history as a system context block (background)
          3. user's prior Q&A pairs as user/assistant turns (continuity)
          4. current question as final user turn (text only — images are
             passed separately to provider.text_chat via image_urls kwarg)
        """
        messages: list[dict] = []

        messages.append({"role": "system", "content": system_prompt})

        group_ctx = self.get_group_context(group_id)
        if group_ctx:
            lines = []
            for msg in group_ctx:
                prefix = "[Bot]" if msg.is_bot_reply else f"[{msg.sender_name}]"
                line = f"{prefix}: {msg.content}"
                if msg.image_urls:
                    line += " [图片]"
                lines.append(line)
            messages.append({
                "role": "system",
                "content": "以下是最近的群聊记录：\n" + "\n".join(lines),
            })

        for msg in self.get_user_context(group_id, user_id):
            role = "assistant" if msg.is_bot_reply else "user"
            messages.append({"role": role, "content": msg.content})

        messages.append({"role": "user", "content": current_content or "请看图片"})

        return messages

    # ── Persistence ────────────────────────────────────────────────────────

    def save_to_db(self, db_path: str) -> None:
        """Flush all in-memory context to SQLite (full replace)."""
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS context_messages (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ctx_type    TEXT    NOT NULL,
                    group_id    TEXT    NOT NULL,
                    user_id     TEXT    NOT NULL DEFAULT '',
                    sender_id   TEXT    NOT NULL,
                    sender_name TEXT    NOT NULL,
                    content     TEXT    NOT NULL,
                    image_urls  TEXT    NOT NULL DEFAULT '[]',
                    timestamp   REAL    NOT NULL,
                    is_bot_reply INTEGER NOT NULL DEFAULT 0
                )
            """)
            conn.execute("DELETE FROM context_messages")

            for group_id, q in self._group_ctx.items():
                for msg in q:
                    conn.execute(
                        "INSERT INTO context_messages "
                        "(ctx_type, group_id, sender_id, sender_name, content, image_urls, timestamp, is_bot_reply) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        ("group", group_id, msg.sender_id, msg.sender_name,
                         msg.content, json.dumps(msg.image_urls), msg.timestamp, int(msg.is_bot_reply)),
                    )

            for group_id, users in self._user_ctx.items():
                for user_id, q in users.items():
                    for msg in q:
                        conn.execute(
                            "INSERT INTO context_messages "
                            "(ctx_type, group_id, user_id, sender_id, sender_name, content, image_urls, timestamp, is_bot_reply) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            ("user", group_id, user_id, msg.sender_id, msg.sender_name,
                             msg.content, json.dumps(msg.image_urls), msg.timestamp, int(msg.is_bot_reply)),
                        )

            conn.commit()
        finally:
            conn.close()

    def load_from_db(self, db_path: str) -> None:
        """Load context from SQLite, silently skip missing file, expired rows, or corrupted DB."""
        if not os.path.exists(db_path):
            return
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.execute(
                "SELECT ctx_type, group_id, user_id, sender_id, sender_name, "
                "content, image_urls, timestamp, is_bot_reply "
                "FROM context_messages ORDER BY timestamp ASC"
            )
            for row in cursor:
                ctx_type, group_id, user_id, sender_id, sender_name, \
                    content, image_urls_json, timestamp, is_bot_reply = row
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
        except sqlite3.DatabaseError:
            # Corrupted or schema-mismatched DB — start fresh rather than crash on startup
            pass
        finally:
            conn.close()
