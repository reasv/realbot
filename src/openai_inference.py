import copy
import json
from typing import Any, Dict, List
import openai
import asyncio
import os
from dotenv import load_dotenv

from .chat_image_utils import load_image_as_data_url
from .utils import get_config, normalize_chat_history
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
async def run_inference(history: List[dict[str, Any]], timeout_seconds: int = 30):
    load_dotenv()
    openai_url = os.getenv("OPENAI_API_URL", "test")
    client = openai.AsyncOpenAI(
        base_url=openai_url,
        api_key=os.getenv("LLM_API_KEY"),
        timeout=timeout_seconds
    )
    username = os.getenv("BOT_NAME")
    config = get_config()
    openai_model = config.get("openai", {}).get("model", "default")
    print(f"Using model: {openai_model}")
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
                model=openai_model,
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

async def chat_inference(channelID: int | str, messages: List[dict[str, Any]], timeout_seconds: int = 60):
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
    context_image_limit = int(os.getenv("CONTEXT_IMAGE_LIMIT", 8))
    recent_messages = [copy.deepcopy(m) for m in history['messages'][-context_msg_limit:]]
    recent_messages = enforce_image_limit(recent_messages, context_image_limit)

    formatted_messages: List[Dict[str, Any]] = []
    for msg in recent_messages:
        formatted = build_openai_message(msg, username)
        if formatted:
            formatted_messages.append(formatted)
    
    result = await run_inference(formatted_messages, timeout_seconds)
    if result is None:
        return None
        
    reply = result["message"]
    # Strip username: prefix if present
    if reply:
        while reply.startswith(f"{username}: "):
            reply = reply[len(f"{username}: "):]
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


def enforce_image_limit(messages: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    """
    Trim image references across the provided messages so that at most `limit`
    remain, removing the oldest ones first.
    """
    if limit <= 0:
        for msg in messages:
            msg.pop("images", None)
        return messages

    total_images = sum(len(msg.get("images", [])) for msg in messages)
    excess = total_images - limit
    if excess <= 0:
        return messages

    for msg in messages:
        images = msg.get("images")
        if not images:
            continue
        remove_count = min(len(images), excess)
        if remove_count:
            del images[:remove_count]
            excess -= remove_count
            if not images:
                msg.pop("images", None)
        if excess <= 0:
            break
    return messages


def build_openai_message(message: Dict[str, Any], username: str) -> Dict[str, Any] | None:
    """
    Converts a stored history record into the structure expected by OpenAI's chat API.
    """
    msg_user = message.get("user", "")
    msg_content = str(message.get("message", "") or "")
    msg_content = msg_content.replace("{{char}}", username)
    role = "assistant" if msg_user == "{{char}}" else "user"
    if role == "assistant":
        # strip username prefix if present
        while msg_content.startswith(f"{username}: "):
            msg_content = msg_content[len(f"{username}: "):]
        text_prefix = f"{username}: {msg_content}"
    else:
        text_prefix = f"{msg_user}: {msg_content}"

    segments: List[Dict[str, Any]] = []
    if text_prefix:
        segments.append({"type": "text", "text": text_prefix})

    for image in message.get("images", []):
        segment = build_image_segment(image)
        if segment:
            segments.append(segment)

    if not segments:
        return None
    return {"role": role, "content": segments}


def build_image_segment(image: Dict[str, Any]) -> Dict[str, Any] | None:
    """
    Converts an image metadata entry into an OpenAI content segment.
    """
    image_url = image.get("url")
    local_path = image.get("local_path")
    if local_path:
        resolved = local_path
        if not os.path.isabs(resolved):
            resolved = os.path.abspath(resolved)
        try:
            image_url = load_image_as_data_url(resolved)
        except Exception as e:
            print(f"Failed to load cached image {resolved}: {e}")
            image_url = None

    if not image_url:
        return None

    return {
        "type": "image_url",
        "image_url": {
            "url": image_url,
            "detail": "auto",
        },
    }
