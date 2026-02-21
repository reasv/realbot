import unittest
from unittest.mock import AsyncMock, patch
import tempfile
import os

from src.matrix_chatbot import (
    Membership,
    MessageType,
    MatrixBot,
    _respond_to_dms_enabled,
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

    def test_clean_content_replaces_matrix_mxid_mention(self):
        bot = MatrixBot.__new__(MatrixBot)
        bot._aliases = ["assistant"]

        cleaned = bot._clean_content("@assistant:bernkastel.pictures blah blah blah")
        self.assertEqual(cleaned, "{{char}} blah blah blah")

    def test_clean_content_keeps_plain_colon_punctuation(self):
        bot = MatrixBot.__new__(MatrixBot)
        bot._aliases = ["assistant"]

        cleaned = bot._clean_content("hey @assistant: can you check")
        self.assertEqual(cleaned, "hey {{char}}: can you check")


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


class MatrixDmConfigTests(unittest.TestCase):
    def test_respond_to_dms_toggle_reads_bot_section(self):
        self.assertTrue(_respond_to_dms_enabled({"bot": {"respond_to_dms": True}}))
        self.assertFalse(_respond_to_dms_enabled({"bot": {"respond_to_dms": False}}))
        self.assertFalse(_respond_to_dms_enabled({}))


class MatrixDmDetectionTests(unittest.TestCase):
    def test_is_direct_message_room_with_two_joined_members(self):
        bot = MatrixBot.__new__(MatrixBot)
        bot.bot_mxid = "@bot:example.org"
        bot.state_store = type("StateStore", (), {})()
        member = type("Member", (), {"membership": Membership.JOIN})
        bot.state_store.members = {
            "!dm:example.org": {
                "@bot:example.org": member(),
                "@alice:example.org": member(),
            }
        }

        self.assertTrue(bot._is_direct_message_room("!dm:example.org"))

    def test_is_direct_message_room_rejects_rooms_with_three_joined_members(self):
        bot = MatrixBot.__new__(MatrixBot)
        bot.bot_mxid = "@bot:example.org"
        bot.state_store = type("StateStore", (), {})()
        member = type("Member", (), {"membership": Membership.JOIN})
        bot.state_store.members = {
            "!group:example.org": {
                "@bot:example.org": member(),
                "@alice:example.org": member(),
                "@bob:example.org": member(),
            }
        }

        self.assertFalse(bot._is_direct_message_room("!group:example.org"))


class MatrixDmRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_on_message_routes_dm_when_enabled(self):
        room_id = "!dm:example.org"
        bot = MatrixBot.__new__(MatrixBot)
        bot.bot_mxid = "@bot:example.org"
        bot._aliases = ["bot"]
        bot._initial_sync_done = True
        bot.recentMessages = {}
        bot._constant_chat = AsyncMock()
        bot.state_store = type("StateStore", (), {})()
        member = type("Member", (), {"membership": Membership.JOIN})
        bot.state_store.members = {
            room_id: {
                "@bot:example.org": member(),
                "@alice:example.org": member(),
            }
        }

        content = type(
            "Content",
            (),
            {
                "msgtype": MessageType.TEXT,
                "body": "hello",
                "serialize": lambda self: {"body": "hello"},
            },
        )()
        evt = type(
            "Event",
            (),
            {"sender": "@alice:example.org", "room_id": room_id, "content": content},
        )()

        cfg = {
            "bot": {"respond_to_dms": True},
            "whitelist": {"always": [], "mentions": [], "rand": []},
        }
        with patch("src.matrix_chatbot.get_config", return_value=cfg):
            await bot._on_message(evt)

        bot._constant_chat.assert_awaited_once_with(room_id, evt)

    async def test_on_message_ignores_dm_when_disabled_and_unwhitelisted(self):
        room_id = "!dm:example.org"
        bot = MatrixBot.__new__(MatrixBot)
        bot.bot_mxid = "@bot:example.org"
        bot._aliases = ["bot"]
        bot._initial_sync_done = True
        bot.recentMessages = {}
        bot._constant_chat = AsyncMock()
        bot.state_store = type("StateStore", (), {})()
        member = type("Member", (), {"membership": Membership.JOIN})
        bot.state_store.members = {
            room_id: {
                "@bot:example.org": member(),
                "@alice:example.org": member(),
            }
        }

        content = type(
            "Content",
            (),
            {
                "msgtype": MessageType.TEXT,
                "body": "hello",
                "serialize": lambda self: {"body": "hello"},
            },
        )()
        evt = type(
            "Event",
            (),
            {"sender": "@alice:example.org", "room_id": room_id, "content": content},
        )()

        cfg = {
            "bot": {"respond_to_dms": False},
            "whitelist": {"always": [], "mentions": [], "rand": []},
        }
        with patch("src.matrix_chatbot.get_config", return_value=cfg):
            await bot._on_message(evt)

        bot._constant_chat.assert_not_awaited()


class MatrixInviteFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_room_member_invite_for_bot_uses_join_fallback(self):
        bot = MatrixBot.__new__(MatrixBot)
        bot.bot_mxid = "@bot:example.org"
        bot._join_invited_room = AsyncMock()  # type: ignore

        content = type("Content", (), {"membership": Membership.INVITE})()
        evt = type(
            "Event",
            (),
            {
                "content": content,
                "state_key": "@bot:example.org",
                "room_id": "!dm:example.org",
            },
        )()

        await bot._on_room_member(evt)
        bot._join_invited_room.assert_awaited_once_with(
            evt, source="room_member_invite"
        )

    async def test_room_member_non_invite_is_ignored(self):
        bot = MatrixBot.__new__(MatrixBot)
        bot.bot_mxid = "@bot:example.org"
        bot._join_invited_room = AsyncMock()  # type: ignore

        content = type("Content", (), {"membership": Membership.JOIN})()
        evt = type(
            "Event",
            (),
            {
                "content": content,
                "state_key": "@bot:example.org",
                "room_id": "!room:example.org",
            },
        )()

        await bot._on_room_member(evt)
        bot._join_invited_room.assert_not_awaited()
