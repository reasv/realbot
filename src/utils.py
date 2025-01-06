import tomllib
from typing import Dict, List
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

def format_chat_history(history: List[Dict[str, str]], username) -> List[Dict[str, str]]:
    return  [
            {
                "role": "system",
                "content": f"This is a conversation between multiple users in an online chat. You are {username}. Reply to the conversation roleplaying as {username}. Never write messages for other users, only for {username}. Write a single chat message at a time. Always stay in character.",
            },
            *normalize_chat_history(history),
        ]
