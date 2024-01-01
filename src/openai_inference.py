import json
from typing import List
import openai
import asyncio
from utils import get_config, get_character

async def run_inference(history, last_username: str):
    config = get_config()
    character = get_character()

    name = character['name']
    description = character['description'].replace("{{char}}", name).replace("{{user}}", last_username)

    context = config['prompt']['prompt_format'].format_map({'character': name, 'user': last_username, 'description': description})

    bot_name = config["prompt"]["bot_name_format"].format_map({'character': name, 'user': last_username})
    prompt = context+history+bot_name

    HOST = config['system']['api_host']
    URI = (config['system'].get('uri', None) or 'http://{}/v1/').format(HOST)
    print(URI)
    client = openai.AsyncOpenAI(
        base_url=URI,
        api_key="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    )

    completion = await client.completions.create(
        model="LoneStriker_zephyr-7b-alpha-8.0bpw-h6-exl2",
        prompt=prompt,
        max_tokens=250,
        stop=["\n\n", "</m>", "</s>"],
    )

    reply: str = completion.choices[0].text

    return {
        "message": reply,
    }

    # request = {
    #     'user_input': user_input,
    #     'max_new_tokens': 250,
    #     'auto_max_new_tokens': False,
    #     'history': history,
    #     'mode': 'instruct',  # Valid options: 'chat', 'chat-instruct', 'instruct'
    #     'character': 'Example',
    #     'instruction_template': 'Airoboros-v1.2',  # Will get autodetected if unset
    #     'your_name': user_name,
    #     'name1': user_name, # Optional
    #     'name2': name, # Optional
    #     'context': f"{context}\n\n", # Optional
    #     # 'greeting': 'greeting', # Optional
    #     'name1_instruct': user_name, # Optional
    #     'name2_instruct': bot_name, # Optional
    #     'context_instruct': f"{context}\n\n", # Optional
    #     # 'turn_template': 'turn_template', # Optional
    #     'regenerate': False,
    #     '_continue': False,

    #     # Generation params
    #     'do_sample': True,
    #     'temperature': 0.95,
    #     'top_p': 0.1,
    #     'typical_p': 1,
    #     'epsilon_cutoff': 0,  # In units of 1e-4
    #     'eta_cutoff': 0,  # In units of 1e-4
    #     'tfs': 1,
    #     'top_a': 0,
    #     'repetition_penalty': 1.18,
    #     'repetition_penalty_range': 0,
    #     'top_k': 40,
    #     'min_length': 0,
    #     'no_repeat_ngram_size': 0,
    #     'num_beams': 1,
    #     'penalty_alpha': 0,
    #     'length_penalty': 1,
    #     'early_stopping': False,
    #     'mirostat_mode': 0,
    #     'mirostat_tau': 5,
    #     'mirostat_eta': 0.1,
    #     'guidance_scale': 1,
    #     'negative_prompt': '',
    #     'stop_at_newline': False,
    #     'seed': -1,
    #     'add_bos_token': True,
    #     'truncation_length': 2048,
    #     'ban_eos_token': True,
    #     'skip_special_tokens': True,
    #     'stopping_strings': []
    # }

    # with open('generation_params_override.json', 'r') as f:
    #     overrides = json.load(f)
    #     request.update(overrides)

    

    
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

    character_name: str = get_character()['name']
    config = get_config()

    prompt = ""
    last_username = ""
    for message in history['messages'][-30:]:
        message_format = config["prompt"]["user_message_format"]
        if message['user'] == "{{char}}":
            message_format = config["prompt"]["bot_message_format"]

        message['user'] = message['user'].replace("{{char}}", character_name)
        message['message'] = message['message'].replace("{{char}}", character_name)
        print(message)
        prompt += message_format.format_map({'user': message['user'], 'message': message['message']})

        last_username = message['user']

    reply = (await run_inference(prompt, last_username))["message"]
    
    history['messages'].append({
        'user': "{{char}}",
        'message': reply
    })

    save_history()

    print(reply)
    return reply

if __name__ == '__main__':
    asyncio.run(chat_inference("exampleChannel", [{"user": "Carl", "message": "Who are you?"}]))