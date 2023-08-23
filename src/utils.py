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