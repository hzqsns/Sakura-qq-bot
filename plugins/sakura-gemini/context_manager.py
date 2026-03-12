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
