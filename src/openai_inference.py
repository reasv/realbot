import copy
import json
from typing import Any, Dict, List, Iterable
import uuid
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


def _as_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            if isinstance(item, str):
                out.append(item)
        return out
    return []


def _dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _consume_stopping_run(text: str, start: int, stopping_strings: List[str]) -> int:
    """
    Consumes a maximal run of one or more stopping strings that are immediately adjacent.
    """
    end = start
    while end < len(text):
        matched = None
        for s in stopping_strings:
            if text.startswith(s, end):
                matched = s
                break
        if matched is None:
            break
        end += len(matched)
    return end


def _strip_trailing_whitespace_stops(text: str, whitespace_stops: List[str]) -> str:
    if not text or not whitespace_stops:
        return text

    while True:
        removed = False
        for s in whitespace_stops:
            if text.endswith(s):
                text = text[: -len(s)]
                removed = True
                break
        if not removed:
            return text


def truncate_message_by_stopping_strings(
    message: str | None,
    stopping_strings: Any,
    limit: Any,
) -> str:
    """
    If `limit != -1`, truncates `message` after the (limit + 1)th run of one or more
    consecutive stopping strings, counting runs globally across all stopping strings.

    Trailing stopping strings that are made up only of whitespace characters are stripped
    from the end after truncation.
    """
    if not message:
        return ""

    try:
        limit_int = int(limit)
    except Exception:
        limit_int = -1

    if limit_int == -1:
        return message

    stops = _as_string_list(stopping_strings)
    stops = [s for s in stops if isinstance(s, str) and s != ""]
    if not stops:
        return message

    # Prefer longer matches first to avoid consuming partial stopping strings.
    stops = _dedupe_preserve_order(sorted(stops, key=len, reverse=True))
    whitespace_stops = [s for s in stops if s.isspace()]

    run_count = 0
    in_run = False
    i = 0
    while i < len(message):
        matched = None
        for s in stops:
            if message.startswith(s, i):
                matched = s
                break

        if matched is None:
            in_run = False
            i += 1
            continue

        if not in_run:
            run_count += 1
            in_run = True
            if run_count == limit_int + 1:
                end = _consume_stopping_run(message, i, stops)
                truncated = message[:end]
                return _strip_trailing_whitespace_stops(truncated, whitespace_stops)

        i += len(matched)

    return message


def _history_file_for_channel(channelID: int | str) -> str:
    return f"history/{channelID}.json"


def load_channel_history(channelID: int | str) -> dict[str, Any]:
    history_file = _history_file_for_channel(channelID)
    try:
        with open(history_file, "r") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict) and isinstance(loaded.get("messages"), list):
            return loaded
    except Exception:
        pass
    return {"messages": []}


def save_channel_history(channelID: int | str, history: dict[str, Any]) -> None:
    history_file = _history_file_for_channel(channelID)
    os.makedirs(os.path.dirname(history_file), exist_ok=True)
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)


def _new_pending_message_id() -> str:
    return f"pending:{uuid.uuid4()}"


async def run_inference(history: List[dict[str, Any]], timeout_seconds: int = 30):
    load_dotenv()
    openai_url = os.getenv("OPENAI_API_URL", "http://localhost:5000/v1")
    config = get_config()
    max_tokens = config.get("openai", {}).get("max_tokens", 150)
    openai_url = config.get("openai", {}).get("api_url", openai_url)
    client = openai.AsyncOpenAI(
        base_url=openai_url,
        api_key=os.getenv("LLM_API_KEY"),
        timeout=timeout_seconds
    )
    username = os.getenv("BOT_NAME")
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
                max_tokens=max_tokens,
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


async def _generate_reply_from_stored_messages(
    stored_messages: List[dict[str, Any]],
    username: str,
    timeout_seconds: int,
) -> str | None:
    config = get_config()
    try:
        context_msg_limit = int(config.get("openai", {}).get("ctx_message_limit", 4))
    except Exception:
        context_msg_limit = 4

    try:
        context_image_limit = int(config.get("openai", {}).get("ctx_image_limit", 2))
    except Exception:
        context_image_limit = 2

    if context_msg_limit <= 0:
        recent_messages: List[dict[str, Any]] = []
    else:
        recent_messages = [copy.deepcopy(m) for m in stored_messages[-context_msg_limit:]]

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
    if reply:
        while reply.startswith(f"{username}: "):
            reply = reply[len(f"{username}: "):]

    openai_cfg = config.get("openai", {})
    reply = truncate_message_by_stopping_strings(
        reply,
        openai_cfg.get("stopping_strings", ["\n"]),
        openai_cfg.get("stopping_strings_limit", -1),
    )
    return reply


async def chat_inference(channelID: int | str, messages: List[dict[str, Any]], timeout_seconds: int = 60):
    load_dotenv()
    username = os.getenv("BOT_NAME")
    assert username is not None, "Error. Please set the BOT_NAME environment variable."

    history = load_channel_history(channelID)
    
    history['messages'].extend(messages)
    save_channel_history(channelID, history)

    reply = await _generate_reply_from_stored_messages(history["messages"], username, timeout_seconds)
    if reply is None:
        return None

    pending_message_id = _new_pending_message_id()

    assistant_message: dict[str, Any] = {
        "user": "{{char}}",
        "message": reply,
        "messageId": pending_message_id,
    }
    history["messages"].append(assistant_message)
    save_channel_history(channelID, history)

    print(reply)
    # Remove "username: " from the start of the message
    if reply and reply.startswith(f"{username}: "):
        reply = reply[len(f"{username}: "):]
    return (reply, pending_message_id)

if __name__ == '__main__':
    asyncio.run(chat_inference(1, [{"user": "Carl", "message": "What's your username, {{char}}?"}]))


def finalize_assistant_message_id(channelID: int | str, pendingMessageId: str, messageId: str) -> bool:
    """
    Update the assistant message with `messageId == pendingMessageId` to have the
    provided platform message ID. To minimize work, it first checks the latest
    assistant message, then falls back to a reverse scan.
    """
    history = load_channel_history(channelID)
    messages = history.get("messages")
    if not isinstance(messages, list):
        return False

    pending_id_str = str(pendingMessageId)
    real_id_str = str(messageId)

    # Fast path: latest assistant message matches pending id.
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("user") != "{{char}}":
            continue
        if msg.get("messageId") == pending_id_str:
            msg["messageId"] = real_id_str
            save_channel_history(channelID, history)
            return True
        break

    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("user") != "{{char}}":
            continue
        if msg.get("messageId") == pending_id_str:
            msg["messageId"] = real_id_str
            save_channel_history(channelID, history)
            return True

    return False


def finalize_last_assistant_message_id(channelID: int | str, messageId: str) -> bool:
    """
    Legacy helper retained for compatibility. Prefer finalize_assistant_message_id.
    """
    history = load_channel_history(channelID)
    messages = history.get("messages")
    if not isinstance(messages, list):
        return False

    message_id_str = str(messageId)
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("user") != "{{char}}":
            continue
        existing = msg.get("messageId")
        if isinstance(existing, str) and existing.startswith("pending:"):
            msg["messageId"] = message_id_str
            save_channel_history(channelID, history)
            return True
        return False

    return False


def _find_message_index_by_id(history: dict[str, Any], messageId: str) -> int | None:
    messages = history.get("messages")
    if not isinstance(messages, list):
        return None

    message_id_str = str(messageId)
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("messageId") == message_id_str:
            return i
    return None


def _ensure_swipe_fields(msg: dict[str, Any]) -> None:
    current = str(msg.get("message", "") or "")
    swipes = msg.get("swipes")
    if not isinstance(swipes, list) or not all(isinstance(s, str) for s in swipes):
        swipes = [current]
        msg["swipes"] = swipes
        msg["swipeIndex"] = 0
        return

    if not swipes:
        msg["swipes"] = [current]
        msg["swipeIndex"] = 0
        return

    idx = msg.get("swipeIndex")
    if not isinstance(idx, int):
        idx = -1

    if 0 <= idx < len(swipes) and swipes[idx] == current:
        return

    try:
        found = swipes.index(current)
    except ValueError:
        swipes.append(current)
        msg["swipeIndex"] = len(swipes) - 1
        return

    msg["swipeIndex"] = found


async def swipe_regenerate(channelID: int | str, messageId: str, timeout_seconds: int = 60) -> str | None:
    """
    Regenerate a prior assistant message (nondestructive).
    - Initializes swipes/swipeIndex if missing.
    - Appends a new swipe, updates message + swipeIndex, saves to disk.
    """
    load_dotenv()
    username = os.getenv("BOT_NAME")
    assert username is not None, "Error. Please set the BOT_NAME environment variable."

    history = load_channel_history(channelID)
    idx = _find_message_index_by_id(history, messageId)
    if idx is None:
        print(f"[swipes] messageId not found in channel {channelID}: {messageId}")
        return None

    messages = history.get("messages")
    assert isinstance(messages, list)
    target = messages[idx]
    if not isinstance(target, dict) or target.get("user") != "{{char}}":
        print(f"[swipes] messageId is not an assistant message in channel {channelID}: {messageId}")
        return None

    _ensure_swipe_fields(target)

    context_messages: List[dict[str, Any]] = []
    for m in messages[:idx]:
        if isinstance(m, dict):
            context_messages.append(m)

    reply = await _generate_reply_from_stored_messages(context_messages, username, timeout_seconds)
    if reply is None:
        return None

    target.setdefault("swipes", [])
    if not isinstance(target.get("swipes"), list):
        target["swipes"] = []
    target["swipes"].append(reply)
    target["swipeIndex"] = len(target["swipes"]) - 1
    target["message"] = reply

    save_channel_history(channelID, history)
    return reply


def swipe_prev(channelID: int | str, messageId: str) -> str | None:
    history = load_channel_history(channelID)
    idx = _find_message_index_by_id(history, messageId)
    if idx is None:
        print(f"[swipes] messageId not found in channel {channelID}: {messageId}")
        return None

    messages = history.get("messages")
    if not isinstance(messages, list) or not (0 <= idx < len(messages)):
        return None

    msg = messages[idx]
    if not isinstance(msg, dict) or msg.get("user") != "{{char}}":
        return None

    _ensure_swipe_fields(msg)
    swipes = msg.get("swipes")
    swipe_index = msg.get("swipeIndex")
    if not isinstance(swipes, list) or not isinstance(swipe_index, int):
        return None

    if swipe_index <= 0:
        return None

    new_index = swipe_index - 1
    msg["swipeIndex"] = new_index
    msg["message"] = swipes[new_index]
    save_channel_history(channelID, history)
    return msg["message"]


def swipe_next(channelID: int | str, messageId: str) -> str | None:
    history = load_channel_history(channelID)
    idx = _find_message_index_by_id(history, messageId)
    if idx is None:
        print(f"[swipes] messageId not found in channel {channelID}: {messageId}")
        return None

    messages = history.get("messages")
    if not isinstance(messages, list) or not (0 <= idx < len(messages)):
        return None

    msg = messages[idx]
    if not isinstance(msg, dict) or msg.get("user") != "{{char}}":
        return None

    _ensure_swipe_fields(msg)
    swipes = msg.get("swipes")
    swipe_index = msg.get("swipeIndex")
    if not isinstance(swipes, list) or not isinstance(swipe_index, int):
        return None

    if swipe_index >= len(swipes) - 1:
        return None

    new_index = swipe_index + 1
    msg["swipeIndex"] = new_index
    msg["message"] = swipes[new_index]
    save_channel_history(channelID, history)
    return msg["message"]


def get_swipe_nav_state(channelID: int | str, messageId: str) -> tuple[bool, bool] | None:
    """
    Returns (has_prev, has_next) for the assistant message identified by messageId.
    If the message has no swipes data, returns (False, False).
    """
    history = load_channel_history(channelID)
    idx = _find_message_index_by_id(history, messageId)
    if idx is None:
        return None

    messages = history.get("messages")
    if not isinstance(messages, list) or not (0 <= idx < len(messages)):
        return None

    msg = messages[idx]
    if not isinstance(msg, dict) or msg.get("user") != "{{char}}":
        return None

    swipes = msg.get("swipes")
    swipe_index = msg.get("swipeIndex")
    if not isinstance(swipes, list) or not all(isinstance(s, str) for s in swipes):
        return (False, False)
    if not isinstance(swipe_index, int):
        return (False, False)

    if swipe_index < 0 or swipe_index >= len(swipes):
        return (False, False)

    has_prev = swipe_index > 0
    has_next = swipe_index < len(swipes) - 1
    return (has_prev, has_next)


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
