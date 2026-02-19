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
            with patch.dict(os.environ, {"BOT_NAME": "assistant"}, clear=False), patch(
                "src.openai_inference._history_file_for_channel", return_value=history_path
            ), patch("src.openai_inference.get_config", return_value=cfg), patch(
                "src.openai_inference.run_inference",
                new=AsyncMock(return_value={"message": "assistant: hello there"}),
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
