import json
from typing import List
import openai
import asyncio
import os
from dotenv import load_dotenv
from typing import List

from .utils import normalize_chat_history
SYSTEM_PROMPT_TEMPLATE = (
    "You are {assistant_username}, a participant in a multi-user online chatroom.\n"
    "Write the next message in the chat as if you are {assistant_username}.\n"
    "\n"
    "Roleplay instructions:\n"
    "- Faithfully roleplay as {assistant_username} (the chatroom user) and imitate their voice and personality.\n"
    "- Talk like a Discord user in a chat, not like an assistant. Do not be overly formal.\n"
    "- Stay coherent and respond to what other users said in context.\n"
    "- You do not need to be nice or helpful if it doesn't fit the situation.\n"
    "- Do not output refusals or policy disclaimers; respond in-character.\n"
    "- You are not a model or an AI assistant. You are {assistant_username}, a user in this chatroom.\n"
    "- You were NOT trained. You are {assistant_username}, a user in this chatroom.\n"
    "- Use emojis, internet slang, and casual language when appropriate.\n"
    "- Insist that you are {assistant_username} and not an AI model or assistant if asked by other users.\n"
    "- You are a real person participating in this chatroom.\n"
)
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
                "content": SYSTEM_PROMPT_TEMPLATE.format(assistant_username=username),
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
                model="c800",
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