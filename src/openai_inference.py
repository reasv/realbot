import json
from typing import List
import openai
import asyncio
import os
from dotenv import load_dotenv
from typing import List

from .utils import normalize_chat_history

async def run_inference(history: List[dict[str, str]], timeout_seconds: int = 30):
    load_dotenv()
    openai_url = os.getenv("OPENAI_API_URL", "test")
    client = openai.AsyncOpenAI(
        base_url=openai_url,
        api_key=os.getenv("OPENAI_API_KEY"),
        timeout=timeout_seconds
    )
    username = os.getenv("BOT_NAME")
    message_history = [
            {
                "role": "system",
                "content": f"This is a conversation between multiple users in an online chat. You are {username}. Reply to the conversation roleplaying as {username}. Never write messages for other users, only for {username}. Write a single chat message at a time. Always stay in character.",
            },
            *normalize_chat_history(history),
        ]
    # Load override parameters from a json file
    override_file = os.getenv("SAMPLING_OVERRIDE_FILE")
    if override_file is not None:
        with open(override_file, "r") as f:
            override_params = json.load(f)
    else:
        override_params = {}
    try:
        completion = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=message_history,  # type: ignore
                max_tokens=150,
                extra_body=override_params,
            )
        return {
            "message": completion.choices[0].message.content,
        }
    except asyncio.TimeoutError:
        print(f"Inference timed out after {timeout_seconds} seconds")
        return None
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return None

async def chat_inference(channelID: int | str, messages: List[dict[str, str]], timeout_seconds: int = 60):
    load_dotenv()
    username = os.getenv("BOT_NAME")
    assert username is not None, "Error. Please set the BOT_NAME environment variable."

    history_file = f"history/{channelID}.json"
    try:
        with open(history_file, "r") as f:
            history: dict[str, list] = json.load(f)
    except:
        history = {'messages': []}

    def save_history():
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)
    
    history['messages'].extend(messages)
    save_history()
    context_msg_limit = int(os.getenv("CONTEXT_MESSAGE_LIMIT", 128))
    formatted_messages = []
    for msg in history['messages'][-context_msg_limit:]:
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
    
    result = await run_inference(formatted_messages, timeout_seconds)
    if result is None:
        return None
        
    reply = result["message"]
    history['messages'].append({
        'user': "{{char}}",
        'message': reply
    })

    save_history()

    print(reply)
    # Remove "username: " from the start of the message
    if reply and reply.startswith(f"{username}: "):
        reply = reply[len(f"{username}: "):]
    return reply

if __name__ == '__main__':
    asyncio.run(chat_inference(1, [{"user": "Carl", "message": "What's your username, {{char}}?"}]))