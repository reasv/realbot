import tomllib
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

def get_character():
    name = get_config()['prompt']['character_card']
    with open(f'characters/{name}.toml', 'rb') as f:
        char = tomllib.load(f)
    return char