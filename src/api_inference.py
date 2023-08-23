import json
from typing import List

import requests

from src.utils import get_config, get_character

def run_inference(user_input, history, last_username: str):
    config = get_config()

    character = get_character()
    name = character['name']
    description = character['description'].replace("{{char}}", name).replace("{{user}}", last_username)

    context = config['prompt']['prompt_format'].format(name=name, description=description)
    user_name = ""
    bot_name = config["prompt"]["bot_name_format"].format(name=name)
    request = {
        'user_input': user_input,
        'max_new_tokens': 250,
        'auto_max_new_tokens': False,
        'history': history,
        'mode': 'instruct',  # Valid options: 'chat', 'chat-instruct', 'instruct'
        'character': 'Example',
        'instruction_template': 'Airoboros-v1.2',  # Will get autodetected if unset
        'your_name': user_name,
        'name1': user_name, # Optional
        'name2': name, # Optional
        'context': f"{context}\n\n", # Optional
        # 'greeting': 'greeting', # Optional
        'name1_instruct': user_name, # Optional
        'name2_instruct': bot_name, # Optional
        'context_instruct': f"{context}\n\n", # Optional
        # 'turn_template': 'turn_template', # Optional
        'regenerate': False,
        '_continue': False,

        # Generation params. If 'preset' is set to different than 'None', the values
        # in presets/preset-name.yaml are used instead of the individual numbers.
        'preset': 'None',
        'do_sample': True,
        'temperature': 0.95,
        'top_p': 0.1,
        'typical_p': 1,
        'epsilon_cutoff': 0,  # In units of 1e-4
        'eta_cutoff': 0,  # In units of 1e-4
        'tfs': 1,
        'top_a': 0,
        'repetition_penalty': 1.18,
        'repetition_penalty_range': 0,
        'top_k': 40,
        'min_length': 0,
        'no_repeat_ngram_size': 0,
        'num_beams': 1,
        'penalty_alpha': 0,
        'length_penalty': 1,
        'early_stopping': False,
        'mirostat_mode': 0,
        'mirostat_tau': 5,
        'mirostat_eta': 0.1,
        'guidance_scale': 1,
        'negative_prompt': '',
        'stop_at_newline': True,
        'seed': -1,
        'add_bos_token': True,
        'truncation_length': 2048,
        'ban_eos_token': False,
        'skip_special_tokens': True,
        'stopping_strings': ['/n']
    }

    with open('generation_params_override.json', 'r') as f:
        overrides = json.load(f)
        request.update(overrides)

    HOST = config['system']['api_host']
    URI = (config['system'].get('uri', None) or 'http://{}/api/v1/chat').format(HOST)

    response = requests.post(URI, json=request)

    if response.status_code == 200:
        result = response.json()['results'][0]['history']
        print(json.dumps(result, indent=4))
        return {
            "message": result['visible'][-1][1],
            "history": result
        }

def chat_inference(channelID: str, messages: List[dict[str, str]]):
    history_file = f"history/{channelID}.json"
    try:
        with open(history_file, "r") as f:
            history: dict[str, list] = json.load(f)
    except:
        history = {'internal': [], 'visible': []}
    
    name: str = get_character()['name']

    config = get_config()
    prompt = ""
    last_username = ""
    for message in messages:
        message['user'] = message['user'].replace("{{char}}", name)
        message['message'] = message['message'].replace("{{char}}", name)
        print(message)
        prompt += config["prompt"]["user_message_format"].format(user=message['user'], message=message['message'])
        last_username = message['user']

    result = run_inference(prompt, history, last_username)
    history = result["history"]
    
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)

    return result["message"]

if __name__ == '__main__':
    chat_inference("exampleChannel", "Carl", "What's my name?")