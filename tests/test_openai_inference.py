import os
import tempfile
import unittest
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

