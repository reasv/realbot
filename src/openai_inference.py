import json
from typing import List
import openai
import asyncio
import os
from dotenv import load_dotenv

async def run_inference(history: List[dict[str, str]]):
    load_dotenv()
    client = openai.AsyncOpenAI(
        base_url=os.getenv("OPENAPI_API_URL", "https://api.openai.com"),
        api_key=os.getenv("OPENAI_API_KEY")
    )
    completion = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "This is a conversation between multiple users in an online chat. You are rei. Reply to the conversation as if you are rei."
            },
            {
                "role": "user",
                "content": "<Chat History>"
            },
            *history # type: ignore
        ],
    )
    return {
        "message": completion.choices[0].message.content,
    }

async def chat_inference(channelID: str, messages: List[dict[str, str]]):
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
    return reply

if __name__ == '__main__':
    asyncio.run(chat_inference("exampleChannel", [{"user": "Carl", "message": "What's my username?"}]))