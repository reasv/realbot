import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src import openai_inference


class ChatInferencePendingIdTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_inference_stores_pending_message_id_for_string_channel(self):
        cfg = {
            "openai": {
                "ctx_message_limit": 10,
                "ctx_image_limit": 2,
                "stopping_strings": ["\n"],
                "stopping_strings_limit": -1,
            }
        }
        with tempfile.TemporaryDirectory() as td:
            history_path = os.path.join(td, "room.json")
            run_inference_mock = AsyncMock(return_value={"message": "assistant: hello there"})
            with patch.dict(os.environ, {"BOT_NAME": "assistant"}, clear=False), patch(
                "src.openai_inference._history_file_for_channel", return_value=history_path
            ), patch("src.openai_inference.get_config", return_value=cfg), patch(
                "src.openai_inference.run_inference",
                new=run_inference_mock,
            ):
                reply, pending_id = await openai_inference.chat_inference(
                    "!room:example.org",
                    [{"user": "alice", "message": "hi"}],
                )

                self.assertEqual(reply, "hello there")
                self.assertIsInstance(pending_id, str)
                self.assertTrue(pending_id.startswith("pending:"))

                history = openai_inference.load_channel_history("!room:example.org")
                last = history["messages"][-1]
                self.assertEqual(last["user"], "{{char}}")
                self.assertEqual(last["messageId"], pending_id)
                self.assertEqual(
                    run_inference_mock.await_args.kwargs.get("channel_id"),
                    "!room:example.org",
                )
                self.assertEqual(run_inference_mock.await_args.kwargs.get("is_dm"), False)

    async def test_chat_inference_forwards_is_dm_flag_to_run_inference(self):
        cfg = {
            "openai": {
                "ctx_message_limit": 10,
                "ctx_image_limit": 2,
                "stopping_strings": ["\n"],
                "stopping_strings_limit": -1,
            }
        }
        with tempfile.TemporaryDirectory() as td:
            history_path = os.path.join(td, "room.json")
            run_inference_mock = AsyncMock(return_value={"message": "assistant: hello there"})
            with patch.dict(os.environ, {"BOT_NAME": "assistant"}, clear=False), patch(
                "src.openai_inference._history_file_for_channel", return_value=history_path
            ), patch("src.openai_inference.get_config", return_value=cfg), patch(
                "src.openai_inference.run_inference",
                new=run_inference_mock,
            ):
                await openai_inference.chat_inference(
                    "!dm:example.org",
                    [{"user": "alice", "message": "hi"}],
                    is_dm=True,
                )

            self.assertEqual(
                run_inference_mock.await_args.kwargs.get("channel_id"),
                "!dm:example.org",
            )
            self.assertEqual(run_inference_mock.await_args.kwargs.get("is_dm"), True)


class ChatInferenceEmptyReplyTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_inference_does_not_store_empty_assistant_reply(self):
        cfg = {
            "openai": {
                "ctx_message_limit": 10,
                "ctx_image_limit": 2,
                "stopping_strings": ["\n"],
                "stopping_strings_limit": -1,
            }
        }
        with tempfile.TemporaryDirectory() as td:
            history_path = os.path.join(td, "room.json")
            seeded = {
                "messages": [
                    {"user": "alice", "message": "hello"},
                    {"user": "{{char}}", "message": "", "messageId": "pending:old-empty"},
                ]
            }
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(seeded, f)

            with patch.dict(os.environ, {"BOT_NAME": "assistant"}, clear=False), patch(
                "src.openai_inference._history_file_for_channel", return_value=history_path
            ), patch("src.openai_inference.get_config", return_value=cfg), patch(
                "src.openai_inference.run_inference",
                new=AsyncMock(return_value={"message": ""}),
            ):
                result = await openai_inference.chat_inference(
                    "!room:example.org",
                    [{"user": "bob", "message": "yo"}],
                )

            self.assertIsNone(result)
            history = openai_inference.load_channel_history("!room:example.org")
            self.assertEqual(
                [m for m in history["messages"] if m.get("user") == "{{char}}"],
                [],
            )
            self.assertEqual(history["messages"][-1]["user"], "bob")
            self.assertEqual(history["messages"][-1]["message"], "yo")


class RunInferenceOverrideFileTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_inference_creates_missing_override_file(self):
        cfg = {
            "openai": {
                "max_tokens": 32,
                "model": "fake-model",
                "api_url": "http://localhost:5000/v1",
            }
        }
        fake_create = AsyncMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))]
            )
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
        )

        with tempfile.TemporaryDirectory() as td:
            override_path = os.path.join(td, "missing-overrides.json")
            with patch.dict(
                os.environ,
                {
                    "BOT_NAME": "assistant",
                    "LLM_API_KEY": "test-key",
                    "SAMPLING_OVERRIDE_FILE": override_path,
                },
                clear=False,
            ), patch("src.openai_inference.get_config", return_value=cfg), patch(
                "src.openai_inference.openai.AsyncOpenAI", return_value=fake_client
            ):
                result = await openai_inference.run_inference(
                    [{"role": "user", "content": "hello"}]
                )

            self.assertEqual(result, {"message": "hello"})
            self.assertTrue(os.path.exists(override_path))
            with open(override_path, "r", encoding="utf-8") as f:
                self.assertEqual(f.read().strip(), "{}")

        self.assertEqual(fake_create.await_count, 1)
        self.assertEqual(fake_create.await_args.kwargs["extra_body"], {})

    async def test_run_inference_logs_full_response_when_log_file_is_set(self):
        class FakeCompletion:
            def __init__(self, content: str):
                self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]

            def model_dump(self, mode: str = "json"):
                return {
                    "id": "cmpl_test",
                    "choices": [{"message": {"content": "hello"}}],
                    "usage": {"total_tokens": 12},
                }

        cfg = {
            "openai": {
                "max_tokens": 32,
                "model": "fake-model",
                "api_url": "http://localhost:5000/v1",
            }
        }
        fake_create = AsyncMock(return_value=FakeCompletion("hello"))
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
        )

        with tempfile.TemporaryDirectory() as td:
            log_path = os.path.join(td, "openai-response.jsonl")
            with patch.dict(
                os.environ,
                {
                    "BOT_NAME": "assistant",
                    "LLM_API_KEY": "test-key",
                    "OPENAI_RESPONSE_LOG_FILE": log_path,
                },
                clear=False,
            ), patch("src.openai_inference.get_config", return_value=cfg), patch(
                "src.openai_inference.openai.AsyncOpenAI", return_value=fake_client
            ):
                result = await openai_inference.run_inference(
                    [{"role": "user", "content": "hello"}]
                )

            self.assertEqual(result, {"message": "hello"})
            with open(log_path, "r", encoding="utf-8") as f:
                rows = [line.strip() for line in f.readlines() if line.strip()]
            self.assertEqual(len(rows), 1)
            payload = json.loads(rows[0])
            self.assertEqual(payload["model"], "fake-model")
            self.assertEqual(payload["api_url"], "http://localhost:5000/v1")
            self.assertEqual(payload["request"]["model"], "fake-model")
            self.assertEqual(payload["request"]["max_tokens"], 32)
            self.assertEqual(payload["request"]["extra_body"], {})
            self.assertEqual(payload["system_prompt_template_filename"], None)
            self.assertEqual(payload["response"]["id"], "cmpl_test")


class RunInferenceSystemPromptTemplateTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_inference_uses_configured_system_prompt_template_file(self):
        fake_create = AsyncMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))]
            )
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
        )

        with tempfile.TemporaryDirectory() as td:
            template_path = os.path.join(td, "example_system_prompt.txt")
            with open(template_path, "w", encoding="utf-8") as f:
                f.write("You are {{assistant_username}}. Reply with one sentence.")

            cfg = {
                "openai": {
                    "max_tokens": 32,
                    "model": "fake-model",
                    "api_url": "http://localhost:5000/v1",
                    "system_prompt_template_dir": td,
                    "system_prompt_template_name": "example_system_prompt.txt",
                }
            }

            with patch.dict(
                os.environ,
                {
                    "BOT_NAME": "assistant",
                    "LLM_API_KEY": "test-key",
                },
                clear=False,
            ), patch("src.openai_inference.get_config", return_value=cfg), patch(
                "src.openai_inference.openai.AsyncOpenAI", return_value=fake_client
            ):
                result = await openai_inference.run_inference(
                    [{"role": "user", "content": "hello"}]
                )

        self.assertEqual(result, {"message": "hello"})
        self.assertEqual(fake_create.await_count, 1)
        messages = fake_create.await_args.kwargs["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "You are assistant. Reply with one sentence.")

    async def test_run_inference_file_template_keeps_single_braces_literal(self):
        fake_create = AsyncMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))]
            )
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
        )

        with tempfile.TemporaryDirectory() as td:
            template_path = os.path.join(td, "example_system_prompt.txt")
            with open(template_path, "w", encoding="utf-8") as f:
                f.write('JSON example: {"k":"v"}; user={{assistant_username}}')

            cfg = {
                "openai": {
                    "max_tokens": 32,
                    "model": "fake-model",
                    "api_url": "http://localhost:5000/v1",
                    "system_prompt_template_dir": td,
                    "system_prompt_template_name": "example_system_prompt.txt",
                }
            }

            with patch.dict(
                os.environ,
                {
                    "BOT_NAME": "assistant",
                    "LLM_API_KEY": "test-key",
                },
                clear=False,
            ), patch("src.openai_inference.get_config", return_value=cfg), patch(
                "src.openai_inference.openai.AsyncOpenAI", return_value=fake_client
            ):
                result = await openai_inference.run_inference(
                    [{"role": "user", "content": "hello"}]
                )

        self.assertEqual(result, {"message": "hello"})
        self.assertEqual(fake_create.await_count, 1)
        messages = fake_create.await_args.kwargs["messages"]
        self.assertEqual(messages[0]["content"], 'JSON example: {"k":"v"}; user=assistant')

    async def test_run_inference_logs_template_filename_when_file_template_is_used(self):
        class FakeCompletion:
            def __init__(self, content: str):
                self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]

            def model_dump(self, mode: str = "json"):
                return {
                    "id": "cmpl_test_template",
                    "choices": [{"message": {"content": "hello"}}],
                }

        fake_create = AsyncMock(return_value=FakeCompletion("hello"))
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
        )

        with tempfile.TemporaryDirectory() as td:
            template_name = "example_system_prompt.txt"
            template_path = os.path.join(td, template_name)
            with open(template_path, "w", encoding="utf-8") as f:
                f.write("You are {{assistant_username}}.")

            log_path = os.path.join(td, "openai-response.jsonl")
            cfg = {
                "openai": {
                    "max_tokens": 32,
                    "model": "fake-model",
                    "api_url": "http://localhost:5000/v1",
                    "system_prompt_template_dir": td,
                    "system_prompt_template_name": template_name,
                }
            }

            with patch.dict(
                os.environ,
                {
                    "BOT_NAME": "assistant",
                    "LLM_API_KEY": "test-key",
                    "OPENAI_RESPONSE_LOG_FILE": log_path,
                },
                clear=False,
            ), patch("src.openai_inference.get_config", return_value=cfg), patch(
                "src.openai_inference.openai.AsyncOpenAI", return_value=fake_client
            ):
                result = await openai_inference.run_inference(
                    [{"role": "user", "content": "hello"}]
                )

            self.assertEqual(result, {"message": "hello"})
            with open(log_path, "r", encoding="utf-8") as f:
                rows = [line.strip() for line in f.readlines() if line.strip()]
            self.assertEqual(len(rows), 1)
            payload = json.loads(rows[0])
            self.assertEqual(payload["system_prompt_template_filename"], template_name)
            self.assertEqual(payload["request"]["messages"][0]["role"], "system")

    async def test_run_inference_uses_channel_override_template_name(self):
        fake_create = AsyncMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))]
            )
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
        )

        with tempfile.TemporaryDirectory() as td:
            default_template_path = os.path.join(td, "example_system_prompt.txt")
            with open(default_template_path, "w", encoding="utf-8") as f:
                f.write("DEFAULT: {{assistant_username}}")

            channel_template_path = os.path.join(td, "matrix_room_prompt.txt")
            with open(channel_template_path, "w", encoding="utf-8") as f:
                f.write("OVERRIDE: {{assistant_username}}")

            cfg = {
                "openai": {
                    "max_tokens": 32,
                    "model": "fake-model",
                    "api_url": "http://localhost:5000/v1",
                    "system_prompt_template_dir": td,
                    "system_prompt_template_name": "example_system_prompt.txt",
                    "system_prompt_template_channel_overrides": {
                        "!room:example.org": "matrix_room_prompt.txt",
                    },
                }
            }

            with patch.dict(
                os.environ,
                {
                    "BOT_NAME": "assistant",
                    "LLM_API_KEY": "test-key",
                },
                clear=False,
            ), patch("src.openai_inference.get_config", return_value=cfg), patch(
                "src.openai_inference.openai.AsyncOpenAI", return_value=fake_client
            ):
                result = await openai_inference.run_inference(
                    [{"role": "user", "content": "hello"}],
                    channel_id="!room:example.org",
                )

        self.assertEqual(result, {"message": "hello"})
        self.assertEqual(fake_create.await_count, 1)
        messages = fake_create.await_args.kwargs["messages"]
        self.assertEqual(messages[0]["content"], "OVERRIDE: assistant")

    async def test_run_inference_uses_dm_template_when_is_dm_without_channel_override(self):
        fake_create = AsyncMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))]
            )
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
        )

        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "example_system_prompt.txt"), "w", encoding="utf-8") as f:
                f.write("GLOBAL: {{assistant_username}}")
            with open(os.path.join(td, "dm_prompt.txt"), "w", encoding="utf-8") as f:
                f.write("DM: {{assistant_username}}")

            cfg = {
                "openai": {
                    "max_tokens": 32,
                    "model": "fake-model",
                    "api_url": "http://localhost:5000/v1",
                    "system_prompt_template_dir": td,
                    "system_prompt_template_name": "example_system_prompt.txt",
                    "system_prompt_template_dm_name": "dm_prompt.txt",
                }
            }

            with patch.dict(
                os.environ,
                {
                    "BOT_NAME": "assistant",
                    "LLM_API_KEY": "test-key",
                },
                clear=False,
            ), patch("src.openai_inference.get_config", return_value=cfg), patch(
                "src.openai_inference.openai.AsyncOpenAI", return_value=fake_client
            ):
                result = await openai_inference.run_inference(
                    [{"role": "user", "content": "hello"}],
                    channel_id="!dm-room:example.org",
                    is_dm=True,
                )

        self.assertEqual(result, {"message": "hello"})
        messages = fake_create.await_args.kwargs["messages"]
        self.assertEqual(messages[0]["content"], "DM: assistant")

    async def test_run_inference_channel_override_beats_dm_override(self):
        fake_create = AsyncMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))]
            )
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
        )

        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "example_system_prompt.txt"), "w", encoding="utf-8") as f:
                f.write("GLOBAL: {{assistant_username}}")
            with open(os.path.join(td, "dm_prompt.txt"), "w", encoding="utf-8") as f:
                f.write("DM: {{assistant_username}}")
            with open(os.path.join(td, "channel_prompt.txt"), "w", encoding="utf-8") as f:
                f.write("CHANNEL: {{assistant_username}}")

            cfg = {
                "openai": {
                    "max_tokens": 32,
                    "model": "fake-model",
                    "api_url": "http://localhost:5000/v1",
                    "system_prompt_template_dir": td,
                    "system_prompt_template_name": "example_system_prompt.txt",
                    "system_prompt_template_dm_name": "dm_prompt.txt",
                    "system_prompt_template_channel_overrides": {
                        "!dm-room:example.org": "channel_prompt.txt",
                    },
                }
            }

            with patch.dict(
                os.environ,
                {
                    "BOT_NAME": "assistant",
                    "LLM_API_KEY": "test-key",
                },
                clear=False,
            ), patch("src.openai_inference.get_config", return_value=cfg), patch(
                "src.openai_inference.openai.AsyncOpenAI", return_value=fake_client
            ):
                result = await openai_inference.run_inference(
                    [{"role": "user", "content": "hello"}],
                    channel_id="!dm-room:example.org",
                    is_dm=True,
                )

        self.assertEqual(result, {"message": "hello"})
        messages = fake_create.await_args.kwargs["messages"]
        self.assertEqual(messages[0]["content"], "CHANNEL: assistant")


class UsernamePrefixStrippingTests(unittest.TestCase):
    def test_strip_repeated_prefix_at_start(self):
        text = "assistant: assistant: hello there"
        stripped = openai_inference._strip_assistant_username_prefixes(text, "assistant")
        self.assertEqual(stripped, "hello there")

    def test_strip_prefix_on_each_non_empty_line_when_all_lines_match(self):
        text = "assistant: line one\n\nassistant: line two\nassistant: line three"
        stripped = openai_inference._strip_assistant_username_prefixes(text, "assistant")
        self.assertEqual(stripped, "line one\n\nline two\nline three")

    def test_keep_non_uniform_multiline_prefixes(self):
        text = "assistant: line one\nbob: line two\nassistant: line three"
        stripped = openai_inference._strip_assistant_username_prefixes(text, "assistant")
        self.assertEqual(stripped, "line one\nbob: line two\nassistant: line three")

    def test_strip_prefix_with_leading_whitespace(self):
        text = " kuroneko:  lmao bro went from murderous rage"
        stripped = openai_inference._strip_assistant_username_prefixes(text, "kuroneko")
        self.assertEqual(stripped, "lmao bro went from murderous rage")
