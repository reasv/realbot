import tomllib

def get_config():
    with open('config.toml', 'rb') as f:
        config = tomllib.load(f)
    return config

def get_character():
    name = get_config()['prompt']['character_card']
    with open(f'characters/{name}.toml', 'rb') as f:
        char = tomllib.load(f)
    return char