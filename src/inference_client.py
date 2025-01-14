import json
from typing import List
import openai
import asyncio
import os
from dotenv import load_dotenv
from typing import List, Dict

from .utils import format_chat_history, normalize_chat_history
class InferenceClient:
    def __init__(self):
        return

    def get_inference_timeout(self):
        load_dotenv()
        return int(os.getenv("INFERENCE_TIMEOUT", 30))

    def get_bot_name(self):
        load_dotenv()
        username = os.getenv("BOT_NAME")
        assert username is not None, "Error. Please set the BOT_NAME environment variable."
        return username

    def get_openai_client(self):
        load_dotenv()
        api_url = os.getenv("OPENAI_API_URL")
        api_key = os.getenv("OPENAI_API_KEY")
        assert api_url is not None, "Error. Please set the OPENAI_API_URL environment variable."
        assert api_key is not None, "Error. Please set the OPENAI_API_KEY environment variable."
        client = openai.OpenAI(
            base_url=api_url,
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=self.get_inference_timeout()
        )
        return client
    
    def load_history(self, channelID: int | str, botname: str):
        history_file = f"history/{botname}/{channelID}.json"
        try:
            with open(history_file, "r") as f:
                history: dict[str, list] = json.load(f)
        except:
            return []
        return history['messages']
    
    def save_history(self, channelID: int | str, botname: str, history: List[dict[str, str]]):
        history_file = f"history/{botname}/{channelID}.json"
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        with open(history_file, "w") as f:
            json.dump({"messages": history}, f, indent=2)

    def append_history_reply(self, channelID: int | str, botname: str, reply: str):
        history = self.load_history(channelID, botname)
        history.append({
            'user': "{{char}}",
            'message': reply
        })
        self.save_history(channelID, botname, history)

    def format_chat(self, history: List[dict[str, str]], max_ctx: int = 128):
        username = self.get_bot_name()
        formatted_messages = []
        for msg in history[-max_ctx:]:
            message = msg.copy()
            msg_user: str = message['user']
            msg_content: str = message['message']
            msg_content = msg_content.replace("{{char}}", username)

            if msg_user == "{{char}}":
                formatted_messages.append({
                    "role": "assistant",
                    "content": msg_content
                })
            else:
                formatted_messages.append({
                    "role": "user",
                    "content": f"{msg_user}: {msg_content}"
                })
        return format_chat_history(formatted_messages, username)

    async def run_inference(self, messages: List[dict[str, str]]):
        print(messages)
        client = self.get_openai_client()
        fallback_timeout = self.get_inference_timeout() + 2
        try:
            completion = await client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,  # type: ignore
            )
            return completion.choices[0].message.content
        except asyncio.TimeoutError:
            print(f"Inference forcefully timed out after {fallback_timeout} seconds")
            return None
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            return None
    
    def clean_generated_message(self, message: str):
        # Remove "username: " from the start of the message
        username = self.get_bot_name()
        if message.startswith(f"{username}: "):
            message = message[len(f"{username}: "):]
        return message