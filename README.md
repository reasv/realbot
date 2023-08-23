# realbot
A discord bot that connects with a user accounts using discord-self and simulates a real user through LLMs

This bot uses `discord-self` to control "real" user accounts as opposed to regular "bot" accounts, which violates the discord TOS and could get the user account banned.
As such, the bot appears indistinguishable from a real user to others, aside from its potentially insane ramblings.

## Getting started
Requires Python 3.11.

This bot will connect to an instance of [text-generation-webui](https://github.com/oobabooga/text-generation-webui) for inference so ensure you have one running.

Rename `example.env` to `.env` and add your discord token inside it.

Install dependencies
```
$ pip install -r requirements.txt
```

Run the bot
```
$ python main.py
```
The bot will not respond until you have set up the whitelist, see below.

## Configuration
In order to change the bot's settings, rename `user.config.toml.example` to `user.config.toml` and add your settings overrides over the defaults there. You can see the default configuration in `default.config.toml`,
Don't edit the latter, make all your local configuration changes within `user.config.toml`

Settings can be changed at runtime, since the bot re-loads most settings files on every message it receives, and it always opens the configuration files and character sheets in read-only mode, then closes them after reading them.
Feel free to edit your character profiles and whitelists while the bot is running, in order to instantly see the effects.

Chat history files can also be deleted or edited at runtime, but if the bot was in the process of writing a new message for the corresponding channel while you edit/delete the file, the history file will be overwritten/re-written on disk after, so your changes might be lost.

### Whitelist
You need to whitelist channel IDs in which you want the bot to interact, by adding them to the appropriate whitelist in the configuration file.

The bot will never respond to messages outside of whitelisted channels.

Putting a channel ID under `always` means the bot will respond to every single message from this channel.

`mentions` means it will only respond when @mentioned.

`rand` puts the bot in RandomChat mode for said channel(s), which means the bot will respond semi-randomly, more closely emulating human behaviour.

The parameters for RandomChat can be configured through the appropriate configuration section.

### Generation parameters
Parameters used for inference such as temperature, rep penalty, mirostat, etc. can be set in `generation_params_override.json`.
These will override the default configuration which you can see in `src/api_inference.py`

Potentially this can give you you full control over how `text-generation-webui`'s API is used.