# plugins/sakura-gemini/tests/test_context_manager.py
import time
import pytest
from context_manager import ContextMessage, ContextManager


class TestContextMessage:
    def test_create_text_message(self):
        msg = ContextMessage(
            sender_id="12345",
            sender_name="Alice",
            content="hello world",
            image_urls=[],
            timestamp=time.time(),
            is_bot_reply=False,
        )
        assert msg.sender_id == "12345"
        assert msg.content == "hello world"
        assert msg.is_bot_reply is False

    def test_create_image_message(self):
        msg = ContextMessage(
            sender_id="12345",
            sender_name="Alice",
            content="look at this",
            image_urls=["https://example.com/img.jpg"],
            timestamp=time.time(),
            is_bot_reply=False,
        )
        assert len(msg.image_urls) == 1

    def test_is_expired(self):
        old_msg = ContextMessage(
            sender_id="12345",
            sender_name="Alice",
            content="old message",
            image_urls=[],
            timestamp=time.time() - 3600,  # 1 hour ago
            is_bot_reply=False,
        )
        assert old_msg.is_expired(expire_seconds=1800) is True

        new_msg = ContextMessage(
            sender_id="12345",
            sender_name="Alice",
            content="new message",
            image_urls=[],
            timestamp=time.time(),
            is_bot_reply=False,
        )
        assert new_msg.is_expired(expire_seconds=1800) is False


class TestContextManagerBasic:
    def test_init(self):
        mgr = ContextManager(
            group_ctx_max=50,
            user_ctx_max=50,
            ctx_expire_seconds=1800,
        )
        assert mgr is not None

    def test_add_group_message(self):
        mgr = ContextManager(group_ctx_max=50, user_ctx_max=50, ctx_expire_seconds=1800)
        msg = ContextMessage("u1", "Alice", "hello", [], time.time(), False)
        mgr.add_group_message("group1", msg)
        ctx = mgr.get_group_context("group1")
        assert len(ctx) == 1
        assert ctx[0].content == "hello"

    def test_add_user_message(self):
        mgr = ContextManager(group_ctx_max=50, user_ctx_max=50, ctx_expire_seconds=1800)
        msg = ContextMessage("u1", "Alice", "question", [], time.time(), False)
        mgr.add_user_message("group1", "u1", msg)
        ctx = mgr.get_user_context("group1", "u1")
        assert len(ctx) == 1

    def test_group_context_max_limit(self):
        mgr = ContextManager(group_ctx_max=3, user_ctx_max=50, ctx_expire_seconds=1800)
        for i in range(5):
            msg = ContextMessage("u1", "Alice", f"msg{i}", [], time.time(), False)
            mgr.add_group_message("group1", msg)
        ctx = mgr.get_group_context("group1")
        assert len(ctx) == 3
        assert ctx[0].content == "msg2"  # oldest kept

    def test_expired_messages_filtered(self):
        mgr = ContextManager(group_ctx_max=50, user_ctx_max=50, ctx_expire_seconds=60)
        old = ContextMessage("u1", "Alice", "old", [], time.time() - 120, False)
        new = ContextMessage("u1", "Alice", "new", [], time.time(), False)
        mgr.add_group_message("group1", old)
        mgr.add_group_message("group1", new)
        ctx = mgr.get_group_context("group1")
        assert len(ctx) == 1
        assert ctx[0].content == "new"

    def test_clear_user_context(self):
        mgr = ContextManager(group_ctx_max=50, user_ctx_max=50, ctx_expire_seconds=1800)
        msg = ContextMessage("u1", "Alice", "hello", [], time.time(), False)
        mgr.add_user_message("group1", "u1", msg)
        mgr.clear_user_context("group1", "u1")
        ctx = mgr.get_user_context("group1", "u1")
        assert len(ctx) == 0
