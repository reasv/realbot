import unittest
from unittest.mock import AsyncMock
import tempfile
import os

from src.matrix_chatbot import (
    MatrixBot,
    apply_generated_at_mentions,
    dedupe_swipe_jobs,
    matrix_content_mentions_user,
    swipe_action_for_emoji,
)


class MatrixMentionTests(unittest.TestCase):
    def test_explicit_mentions_metadata_detection(self):
        content = {
            "body": "hello",
            "m.mentions": {"user_ids": ["@bot:matrix.org"]},
        }
        self.assertTrue(
            matrix_content_mentions_user(content, "@bot:matrix.org", aliases=["BotNick"])
        )

    def test_textual_mentions_detection(self):
        content = {"body": "hey @BotNick, can you check this?"}
        self.assertTrue(
            matrix_content_mentions_user(content, "@bot:matrix.org", aliases=["BotNick"])
        )
        self.assertFalse(
            matrix_content_mentions_user(content, "@bot:matrix.org", aliases=["OtherName"])
        )

    def test_generated_mentions_unique_and_ambiguous(self):
        mapping = {"alice": "@alice:matrix.org"}

        def resolver(name: str) -> str | None:
            lowered = name.casefold()
            if lowered == "bob":
                return None
            return mapping.get(lowered)

        plain, formatted, mentions = apply_generated_at_mentions("hi @Alice", resolver)
        self.assertEqual(plain, "hi @Alice")
        self.assertIsNotNone(formatted)
        assert formatted is not None
        self.assertIn("matrix.to/#/@alice:matrix.org", formatted)
        self.assertEqual(mentions, ["@alice:matrix.org"])

        plain2, formatted2, mentions2 = apply_generated_at_mentions("hi @Bob", resolver)
        self.assertEqual(plain2, "hi @Bob")
        self.assertIsNone(formatted2)
        self.assertEqual(mentions2, [])


class MatrixSwipeTests(unittest.TestCase):
    def test_reaction_to_swipe_mapping(self):
        cfg = {"regen_emoji": "🔄", "prev_emoji": "◀️", "next_emoji": "▶️"}
        self.assertEqual(swipe_action_for_emoji("🔄", cfg), "regen")
        self.assertEqual(swipe_action_for_emoji("◀️", cfg), "prev")
        self.assertEqual(swipe_action_for_emoji("▶️", cfg), "next")
        self.assertIsNone(swipe_action_for_emoji("😀", cfg))

    def test_swipe_job_deduplication(self):
        jobs = [
            ("$a", "prev"),
            ("$a", "next"),
            ("$b", "regen"),
            ("$a", "regen"),
        ]
        self.assertEqual(dedupe_swipe_jobs(jobs), [("$b", "regen"), ("$a", "regen")])


class MatrixControlReactionStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_control_reaction_bookkeeping(self):
        bot = MatrixBot.__new__(MatrixBot)
        bot.client = object()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "control.json")
            bot._control_reactions_path = lambda room_id: path  # type: ignore
            bot._remove_control_reaction = AsyncMock(return_value=True)  # type: ignore

            bot._save_control_reactions("!room:example.org", [])
            bot._record_control_reaction("!room:example.org", "$message", "🔄", "$reaction")
            bot._record_control_reaction("!room:example.org", "$message", "🔄", "$reaction")

            loaded = bot._load_control_reactions("!room:example.org")
            self.assertEqual(len(loaded), 1)

            await bot._set_control_reaction("!room:example.org", "$message", "🔄", False)
            loaded_after = bot._load_control_reactions("!room:example.org")
            self.assertEqual(loaded_after, [])


class MatrixSendIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_send_paths_use_e2ee_safe_room_send_payloads(self):
        bot = MatrixBot.__new__(MatrixBot)
        response = type("Resp", (), {"event_id": "$event"})()
        fake_client = type("Client", (), {})()
        fake_client.room_send = AsyncMock(return_value=response)
        bot.client = fake_client

        await bot.send_message("!room:example.org", "hello", "<b>hello</b>", ["@u:hs"])
        await bot.send_edit("!room:example.org", "$target", "edited text")
        await bot.send_reaction("!room:example.org", "$target", "🔄")

        calls = fake_client.room_send.await_args_list
        self.assertEqual(len(calls), 3)

        send_call = calls[0].kwargs
        self.assertEqual(send_call["room_id"], "!room:example.org")
        self.assertEqual(send_call["message_type"], "m.room.message")
        self.assertTrue(send_call["ignore_unverified_devices"])
        self.assertEqual(send_call["content"]["body"], "hello")
        self.assertEqual(send_call["content"]["m.mentions"]["user_ids"], ["@u:hs"])

        edit_call = calls[1].kwargs
        self.assertEqual(edit_call["message_type"], "m.room.message")
        self.assertEqual(edit_call["content"]["m.relates_to"]["rel_type"], "m.replace")
        self.assertEqual(edit_call["content"]["m.relates_to"]["event_id"], "$target")

        reaction_call = calls[2].kwargs
        self.assertEqual(reaction_call["message_type"], "m.reaction")
        self.assertEqual(
            reaction_call["content"]["m.relates_to"],
            {"event_id": "$target", "rel_type": "m.annotation", "key": "🔄"},
        )
