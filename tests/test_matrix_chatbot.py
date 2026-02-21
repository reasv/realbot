import unittest
import asyncio
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
        plain_name_content = {"body": "hey BotNick, can you check this?"}
        self.assertFalse(
            matrix_content_mentions_user(
                plain_name_content, "@bot:matrix.org", aliases=["BotNick"]
            )
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


class MatrixReplyContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_process_message_places_reply_context_after_link_previews(self):
        bot = MatrixBot.__new__(MatrixBot)
        bot.bot_mxid = "@bot:example.org"
        bot._aliases = ["bot"]
        bot._get_display_name = AsyncMock(return_value="Alice")  # type: ignore
        bot._extract_images = AsyncMock(  # type: ignore
            return_value=(
                [],
                [{"url": "https://example.org", "title": "Title", "description": "Desc"}],
            )
        )
        bot._extract_reply_context = AsyncMock(  # type: ignore
            return_value=(
                "\n".join(
                    [
                        "[Reply Context]",
                        "The message above is replying to this previous message:",
                        "From: Bob",
                        "Message: hi",
                    ]
                ),
                [],
            )
        )

        content = type(
            "Content",
            (),
            {
                "msgtype": MessageType.TEXT,
                "body": "hello",
                "serialize": lambda self: {
                    "body": "hello",
                    "m.relates_to": {"m.in_reply_to": {"event_id": "$parent"}},
                },
            },
        )()
        evt = type(
            "Event",
            (),
            {
                "sender": "@alice:example.org",
                "room_id": "!room:example.org",
                "event_id": "$evt",
                "content": content,
            },
        )()

        with patch("src.matrix_chatbot.get_config", return_value={"matrix": {}}):
            msg = await bot._process_message(evt)

        text = msg["message"]
        self.assertIn("[Link Previews]", text)
        self.assertIn("[Reply Context]", text)
        self.assertLess(text.index("[Link Previews]"), text.index("[Reply Context]"))

    async def test_extract_reply_context_fetches_parent_and_attaches_image(self):
        bot = MatrixBot.__new__(MatrixBot)
        bot.bot_mxid = "@bot:example.org"
        bot._aliases = ["bot"]

        async def fake_get_member(_self, _room_id, user_id):
            if str(user_id) == "@bob:example.org":
                return type("Member", (), {"displayname": "Bob"})()
            return None

        bot.state_store = type("StateStore", (), {"get_member": fake_get_member})()

        replied_content = type(
            "ReplyContent",
            (),
            {"msgtype": MessageType.IMAGE, "body": "cat.png", "url": "mxc://hs/cat"},
        )()
        replied_evt = type(
            "ReplyEvent",
            (),
            {"sender": "@bob:example.org", "content": replied_content},
        )()
        bot.client = type("Client", (), {})()
        bot.client.get_event = AsyncMock(return_value=replied_evt)
        bot.client.download_media = AsyncMock(return_value=b"img-bytes")

        content = type(
            "Content",
            (),
            {
                "msgtype": MessageType.TEXT,
                "body": "ok",
                "serialize": lambda self: {
                    "body": "ok",
                    "m.relates_to": {"m.in_reply_to": {"event_id": "$parent"}},
                },
            },
        )()
        evt = type(
            "Event",
            (),
            {
                "sender": "@alice:example.org",
                "room_id": "!room:example.org",
                "event_id": "$evt",
                "content": content,
            },
        )()

        with patch(
            "src.matrix_chatbot.download_image_to_history",
            new=AsyncMock(return_value="history/images/reply/cat.png"),
        ):
            section, images = await bot._extract_reply_context(evt)

        bot.client.get_event.assert_awaited_once_with("!room:example.org", "$parent")
        bot.client.download_media.assert_awaited_once_with("mxc://hs/cat")
        self.assertIsNotNone(section)
        assert section is not None
        self.assertIn("[Reply Context]", section)
        self.assertIn("From: Bob", section)
        self.assertIn("Message: cat.png", section)
        self.assertIn("Replied Message Image Filename(s): cat.png", section)
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0]["source"], "matrix_reply")
        self.assertEqual(images[0]["filename"], "cat.png")

    async def test_process_message_preserves_quoted_reply_text(self):
        bot = MatrixBot.__new__(MatrixBot)
        bot.bot_mxid = "@bot:example.org"
        bot._aliases = ["bot"]
        bot._get_display_name = AsyncMock(return_value="Alice")  # type: ignore
        bot._extract_images = AsyncMock(return_value=([], []))  # type: ignore
        bot._extract_reply_context = AsyncMock(return_value=(None, []))  # type: ignore

        content = type(
            "Content",
            (),
            {
                "msgtype": MessageType.TEXT,
                "body": "> <@bob:example.org> old message\n\nnew reply only",
                "serialize": lambda self: {
                    "body": "> <@bob:example.org> old message\n\nnew reply only",
                    "m.relates_to": {"m.in_reply_to": {"event_id": "$parent"}},
                },
            },
        )()
        evt = type(
            "Event",
            (),
            {
                "sender": "@alice:example.org",
                "room_id": "!room:example.org",
                "event_id": "$evt",
                "content": content,
            },
        )()

        with patch("src.matrix_chatbot.get_config", return_value={"matrix": {}}):
            msg = await bot._process_message(evt)

        self.assertEqual(
            msg["message"], "> <@bob:example.org> old message\n\nnew reply only"
        )


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


class MatrixInferenceSchedulingTests(unittest.IsolatedAsyncioTestCase):
    async def test_process_messages_schedules_other_rooms_while_one_room_inflight(self):
        bot = MatrixBot.__new__(MatrixBot)
        bot.pendingMessages = {"!a:example.org": [{"user": "alice", "message": "one"}]}
        bot.pendingSwipes = {}
        bot._inflight_rooms = set()
        bot._inflight_tasks = {}

        gate_a = asyncio.Event()
        gate_b = asyncio.Event()
        started: list[str] = []

        async def fake_process_room(
            room_id: str, messages: list[dict], swipe_jobs: list[tuple[str, str]]
        ) -> None:
            started.append(room_id)
            if room_id == "!a:example.org":
                await gate_a.wait()
            elif room_id == "!b:example.org":
                await gate_b.wait()

        bot._process_room = fake_process_room  # type: ignore

        await bot._process_messages()
        await asyncio.sleep(0)

        self.assertIn("!a:example.org", started)
        self.assertIn("!a:example.org", bot._inflight_rooms)

        bot.pendingMessages["!b:example.org"] = [{"user": "bob", "message": "two"}]
        await bot._process_messages()
        await asyncio.sleep(0)

        self.assertIn("!b:example.org", started)
        self.assertIn("!b:example.org", bot._inflight_rooms)

        gate_a.set()
        gate_b.set()
        if bot._inflight_tasks:
            await asyncio.gather(*bot._inflight_tasks.values(), return_exceptions=True)

    async def test_process_messages_keeps_per_room_execution_sequential(self):
        bot = MatrixBot.__new__(MatrixBot)
        bot.pendingMessages = {"!a:example.org": [{"user": "alice", "message": "first"}]}
        bot.pendingSwipes = {}
        bot._inflight_rooms = set()
        bot._inflight_tasks = {}

        gate = asyncio.Event()
        calls: list[int] = []

        async def fake_process_room(
            room_id: str, messages: list[dict], swipe_jobs: list[tuple[str, str]]
        ) -> None:
            calls.append(len(messages))
            if len(calls) == 1:
                await gate.wait()

        bot._process_room = fake_process_room  # type: ignore

        await bot._process_messages()
        await asyncio.sleep(0)
        self.assertEqual(calls, [1])

        bot.pendingMessages["!a:example.org"] = [{"user": "alice", "message": "second"}]
        await bot._process_messages()
        await asyncio.sleep(0)
        self.assertEqual(calls, [1])

        gate.set()
        if bot._inflight_tasks:
            await asyncio.gather(*bot._inflight_tasks.values(), return_exceptions=True)

        await bot._process_messages()
        await asyncio.sleep(0)
        self.assertEqual(calls, [1, 1])


class MatrixTypingIndicatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_process_room_refreshes_typing_until_finished(self):
        bot = MatrixBot.__new__(MatrixBot)
        bot._typing_timeout_ms = 30000
        bot._typing_refresh_seconds = 0.01

        typing_states: list[bool] = []
        gate = asyncio.Event()

        async def fake_set_typing(room_id: str, typing: bool) -> None:
            typing_states.append(typing)

        async def fake_handle_inference(room_id: str, messages: list[dict]) -> None:
            await gate.wait()

        bot._set_typing = fake_set_typing  # type: ignore
        bot._handle_swipes = AsyncMock()  # type: ignore
        bot._handle_inference = AsyncMock(side_effect=fake_handle_inference)  # type: ignore

        room_task = asyncio.create_task(
            bot._process_room("!room:example.org", [{"user": "u", "message": "m"}], [])
        )
        await asyncio.sleep(0.03)
        gate.set()
        await room_task

        self.assertGreaterEqual(sum(1 for state in typing_states if state), 2)
        self.assertTrue(typing_states)
        self.assertFalse(typing_states[-1])
