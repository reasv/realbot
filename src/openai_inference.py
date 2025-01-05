import json
from typing import List
import openai
import asyncio
import os
from dotenv import load_dotenv
import re

async def run_inference(history: List[dict[str, str]]):
    load_dotenv()
    openai_url = os.getenv("OPENAPI_API_URL", "test")
    client = openai.AsyncOpenAI(
        base_url=openai_url,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    username = os.getenv("BOT_NAME")
    completion = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": f"This is a conversation between multiple users in an online chat. You are {username}. Reply to the conversation roleplaying as {username}. Never write messages for other users, only for {username}. Write a single chat message at a time. Always stay in character.",
            },
            *normalize_chat_history(history), # type: ignore
        ],
    )
    return {
        "message": completion.choices[0].message.content,
    }

from typing import List, Dict

def normalize_chat_history(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Normalizes the chat history to the format that OpenAI expects.
    This means ensuring that we have alternating user and assistant messages,
    merging consecutive messages from the same role into a single message,
    and that the first message is always from the user starting with "<Chat History>".

    Parameters:
        history (List[Dict[str, str]]): The original chat history.

    Returns:
        List[Dict[str, str]]: The normalized chat history.
    """
    if not isinstance(history, list):
        raise TypeError("History must be a list of dictionaries.")
    
    # Step 1: Ensure the first message is from the user and starts with "<Chat History>"
    normalized_history = []
    if not history or history[0].get("role") != "user":
        # Prepend a user message with "<Chat History>"
        normalized_history.append({"role": "user", "content": "<Chat History>"})
    else:
        # First message is from user; ensure it starts with "<Chat History>"
        first_content = history[0].get("content", "")
        if not first_content.startswith("<Chat History>"):
            first_content = "<Chat History>\n" + first_content
        normalized_history.append({"role": "user", "content": first_content})
        # Start processing from the second message
        history = history[1:]
    
    # Step 2: Iterate through the history and merge consecutive messages from the same role
    for message in history:
        role = message.get("role")
        content = message.get("content", "").strip()
        
        if not role or not isinstance(content, str):
            # Skip messages with invalid structure
            continue
        
        if not normalized_history:
            # This should not happen, but just in case
            normalized_history.append({"role": role, "content": content})
            continue
        
        last_message = normalized_history[-1]
        if role == last_message["role"]:
            # Merge contents with a newline
            last_message["content"] += "\n" + content
        else:
            # Append as a new message
            normalized_history.append({"role": role, "content": content})
    
    return normalized_history


async def chat_inference(channelID: int | str, messages: List[dict[str, str]]):
    history_file = f"history/{channelID}.json"
    try:
        with open(history_file, "r") as f:
            history: dict[str, list] = json.load(f)
    except:
        history = {'messages': []}

    def save_history():
        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)
    
    history['messages'].extend(messages)
    save_history()
    ctx_len = 128
    formatted_messages = []
    for msg in history['messages'][-ctx_len:]:
        message = msg.copy()
        if message['user'] == "{{char}}":
            formatted_messages.append({
                "role": "assistant",
                "content": message['message']
            })
        else:
            formatted_messages.append({
                "role": "user",
                "content": f"{message['user']}: "+message['message']
            })
    
    reply = (await run_inference(formatted_messages))["message"]

    history['messages'].append({
        'user': "{{char}}",
        'message': reply
    })

    save_history()

    print(reply)
    # Remove username: at the start of the message
    load_dotenv()
    username = os.getenv("BOT_NAME")
    if reply and reply.startswith(f"{username}: "):
        reply = reply[len(f"{username}: "):]
    return reply

if __name__ == '__main__':
    asyncio.run(chat_inference(1, [{"user": "Carl", "message": "What's my username?"}]))