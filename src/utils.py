import tomllib
from typing import Dict, List, Any
from deepmerge import always_merger

def get_config():
    with open('default.config.toml', 'rb') as f:
        config = tomllib.load(f)
    try:
        with open('user.config.toml', 'rb') as f:
            overrides = tomllib.load(f)
    except:
        overrides = {}

    config = always_merger.merge(config, overrides)
    return config

def dequote(s):
    """
    If a string has single or double quotes around it, remove them.
    Make sure the pair of quotes match.
    If a matching pair of quotes is not found,
    or there are less than 2 characters, return the string unchanged.
    """
    if (len(s) >= 2 and s[0] == s[-1]) and s.startswith(("'", '"')):
        return s[1:-1]
    return s

def _ensure_content_list(content: Any) -> List[Dict[str, Any]]:
    """
    Converts a string or list of OpenAI message segments into a normalized list.
    """
    normalized: List[Dict[str, Any]] = []
    if isinstance(content, str):
        text = content.strip()
        if text:
            normalized.append({"type": "text", "text": text})
        return normalized

    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype == "text":
                text_val = part.get("text")
                if isinstance(text_val, str):
                    text_val = text_val.strip()
                    if text_val:
                        normalized.append({"type": "text", "text": text_val})
            elif ptype == "image_url":
                image_url = part.get("image_url")
                if isinstance(image_url, dict) and image_url.get("url"):
                    segment = {"type": "image_url", "image_url": {"url": image_url["url"]}}
                    if "detail" in image_url:
                        segment["image_url"]["detail"] = image_url["detail"]
                    normalized.append(segment)
    return normalized


def normalize_chat_history(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
    normalized_history: List[Dict[str, Any]] = []
    if not history or history[0].get("role") != "user":
        normalized_history.append({"role": "user", "content": [{"type": "text", "text": "<Chat History>"}]})
    else:
        first_content = _ensure_content_list(history[0].get("content"))
        if first_content:
            first_segment = first_content[0]
            if first_segment.get("type") == "text":
                if not first_segment.get("text", "").startswith("<Chat History>"):
                    first_segment["text"] = "<Chat History>\n" + first_segment.get("text", "")
            else:
                first_content.insert(0, {"type": "text", "text": "<Chat History>"})
        else:
            first_content = [{"type": "text", "text": "<Chat History>"}]
        normalized_history.append({"role": "user", "content": first_content})
        history = history[1:]
    
    # Step 2: Iterate through the history and merge consecutive messages from the same role
    for message in history:
        role = message.get("role")
        content = _ensure_content_list(message.get("content"))

        if not role or not content:
            continue

        if not normalized_history:
            # This should not happen, but just in case
            normalized_history.append({"role": role, "content": content})
            continue

        last_message = normalized_history[-1]
        if role == last_message["role"]:
            # Merge contents by extending the segment list
            last_message["content"].extend(content)
        else:
            # Append as a new message
            normalized_history.append({"role": role, "content": content})
    
    return normalized_history

def format_chat_history(history: List[Dict[str, str]], username) -> List[Dict[str, str]]:
    return  [
            {
                "role": "system",
                "content": f"This is a conversation between multiple users in an online chat. You are {username}. Reply to the conversation roleplaying as {username}. Never write messages for other users, only for {username}. Write a single chat message at a time. Always stay in character.",
            },
            *normalize_chat_history(history),
        ]
