import os
import tempfile
import time

import pytest
from context import ContextMessage, ContextManager, NoiseFilter


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
        mgr = ContextManager(group_ctx_max=50, user_ctx_max=50, ctx_expire_seconds=1800)
        assert mgr.group_ctx_max == 50
        assert mgr.user_ctx_max == 50
        assert mgr.ctx_expire_seconds == 1800

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


class TestNoiseFilter:
    def test_short_text_filtered(self):
        assert NoiseFilter.should_filter(text="哈", has_image=False, is_command=False, min_length=3) is True

    def test_normal_text_passes(self):
        assert NoiseFilter.should_filter(text="今天天气不错", has_image=False, is_command=False, min_length=3) is False

    def test_image_message_passes_even_short(self):
        assert NoiseFilter.should_filter(text="", has_image=True, is_command=False, min_length=3) is False

    def test_command_filtered(self):
        assert NoiseFilter.should_filter(text="/help", has_image=False, is_command=True, min_length=3) is True

    def test_empty_text_no_image_filtered(self):
        assert NoiseFilter.should_filter(text="", has_image=False, is_command=False, min_length=3) is True

    def test_whitespace_only_filtered(self):
        assert NoiseFilter.should_filter(text="   ", has_image=False, is_command=False, min_length=3) is True

    def test_none_text_filtered(self):
        assert NoiseFilter.should_filter(text=None, has_image=False, is_command=False, min_length=3) is True


class TestSQLitePersistence:
    def setup_method(self, method):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "context_cache.db")

    def test_save_and_load(self):
        mgr = ContextManager(group_ctx_max=50, user_ctx_max=50, ctx_expire_seconds=1800)
        msg1 = ContextMessage("u1", "Alice", "hello", [], time.time(), False)
        msg2 = ContextMessage("u1", "Alice", "question", ["https://img.jpg"], time.time(), False)
        mgr.add_group_message("g1", msg1)
        mgr.add_group_message("g1", msg2)
        mgr.add_user_message("g1", "u1", msg2)

        mgr.save_to_db(self.db_path)

        mgr2 = ContextManager(group_ctx_max=50, user_ctx_max=50, ctx_expire_seconds=1800)
        mgr2.load_from_db(self.db_path)

        group_ctx = mgr2.get_group_context("g1")
        assert len(group_ctx) == 2
        assert group_ctx[1].image_urls == ["https://img.jpg"]

        user_ctx = mgr2.get_user_context("g1", "u1")
        assert len(user_ctx) == 1

    def test_load_skips_expired(self):
        mgr = ContextManager(group_ctx_max=50, user_ctx_max=50, ctx_expire_seconds=60)
        old = ContextMessage("u1", "Alice", "old", [], time.time() - 120, False)
        new = ContextMessage("u1", "Alice", "new", [], time.time(), False)
        mgr.add_group_message("g1", old)
        mgr.add_group_message("g1", new)
        mgr.save_to_db(self.db_path)

        mgr2 = ContextManager(group_ctx_max=50, user_ctx_max=50, ctx_expire_seconds=60)
        mgr2.load_from_db(self.db_path)
        ctx = mgr2.get_group_context("g1")
        assert len(ctx) == 1
        assert ctx[0].content == "new"

    def test_load_nonexistent_db(self):
        mgr = ContextManager(group_ctx_max=50, user_ctx_max=50, ctx_expire_seconds=1800)
        mgr.load_from_db("/nonexistent/path.db")
        assert mgr.get_group_context("g1") == []

    def test_load_corrupted_db(self):
        # Write garbage bytes to simulate a corrupted DB
        with open(self.db_path, "wb") as f:
            f.write(b"not a valid sqlite database")
        mgr = ContextManager(group_ctx_max=50, user_ctx_max=50, ctx_expire_seconds=1800)
        mgr.load_from_db(self.db_path)  # must not raise
        assert mgr.get_group_context("g1") == []


class TestContextMerging:
    def test_build_llm_context_with_both_layers(self):
        mgr = ContextManager(group_ctx_max=50, user_ctx_max=50, ctx_expire_seconds=1800)
        now = time.time()

        mgr.add_group_message("g1", ContextMessage("u2", "Bob", "anyone know about python?", [], now - 60, False))
        mgr.add_group_message("g1", ContextMessage("u3", "Charlie", "yeah it's great", [], now - 30, False))

        mgr.add_user_message("g1", "u1", ContextMessage("u1", "Alice", "what is rust?", [], now - 120, False))
        mgr.add_user_message("g1", "u1", ContextMessage("bot", "Bot", "Rust is a systems programming language.", [], now - 110, True))

        messages = mgr.build_llm_messages(
            group_id="g1",
            user_id="u1",
            current_content="explain generics",
            system_prompt="You are a helpful assistant.",
        )

        assert messages[0]["role"] == "system"
        assert "Bob" in messages[1]["content"]
        assert "Charlie" in messages[1]["content"]
        assert messages[2]["role"] == "user"
        assert "rust" in messages[2]["content"]
        assert messages[3]["role"] == "assistant"
        assert messages[-1]["role"] == "user"
        assert "generics" in messages[-1]["content"]

    def test_build_llm_context_empty(self):
        mgr = ContextManager(group_ctx_max=50, user_ctx_max=50, ctx_expire_seconds=1800)
        messages = mgr.build_llm_messages(
            group_id="g1",
            user_id="u1",
            current_content="hello",
            system_prompt="You are helpful.",
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_build_llm_context_empty_content_with_image(self):
        # When current_content is empty (image-only question), fallback text is used
        mgr = ContextManager(group_ctx_max=50, user_ctx_max=50, ctx_expire_seconds=1800)
        messages = mgr.build_llm_messages(
            group_id="g1",
            user_id="u1",
            current_content="",
            system_prompt="You are helpful.",
        )
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "请看图片"
