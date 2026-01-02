# realbot

A discord bot that connects with a user accounts using `discord-self` and simulates a real user through LLMs

This bot uses `discord-self` to control "real" user accounts as opposed to regular "bot" accounts, which violates the discord TOS and could get the user account banned.
As such, the bot appears indistinguishable from a real user to others, aside from its potentially insane ramblings.

## Getting started

Requires Python 3.11.

This bot will connect to an instance of [text-generation-webui](https://github.com/oobabooga/text-generation-webui) for inference so ensure you have one running.

Copy `.env.example` and rename it to `.env` then add your discord token and the URL of your OpenAI-compatible chat completion API.

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

In order to change the bot's settings, make a copy of `user.config.toml.example` and rename it to `user.config.toml`, then add your settings overrides there. You can see the default configuration in `default.config.toml`,
Don't edit the latter, make all your local configuration changes within `user.config.toml`

Settings can be changed at runtime, since the bot re-loads most settings files on every message it receives, and it always opens the configuration files and character sheets in read-only mode, then closes them after reading them.
Feel free to edit your character profiles and whitelists while the bot is running, in order to instantly see the effects.

Chat history files can also be deleted or edited at runtime, but if the bot was in the process of writing a new message for the corresponding channel while you edit/delete the file, the history file will be overwritten/re-written on disk after, so your changes might be lost.

### Whitelist

You need to whitelist channel IDs in which you want the bot to interact, by adding them to the appropriate whitelist in the configuration file.

The bot will never respond to messages outside of whitelisted channels.

Putting a channel ID under `always` means the bot will respond to every single message from this channel.

`mentions` means it will only respond when @mentioned, and to targeted replies.

`rand` puts the bot in RandomChat mode for said channel(s), which means the bot will respond semi-randomly, more closely emulating human behaviour.

The parameters for RandomChat can be configured through the appropriate configuration section.

### Image context

Image attachments from Discord and links shared on IRC are automatically captured and included in the prompt that is sent to the LLM. A cached copy of every downloaded image is written to `history/images/` so chat history can be replayed later. If you want to limit how many recent images are preserved in the prompt, set `CONTEXT_IMAGE_LIMIT` in your `.env` file (defaults to `8`). The oldest images are dropped first once that cap is exceeded.
