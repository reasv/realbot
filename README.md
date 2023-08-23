# realbot
A discord bot that connects with a user accounts using discord-self and simulates a real user through LLMs

This bot uses `discord-self` to control "real" user accounts as opposed to regular "bot" accounts, which violates the discord TOS and could get the user account banned.
As such, the bot appears indistinguishable from a real user to others, aside from its potentially insane ramblings.

## Getting started
Rename `example.env` to `.env` and add your discord token inside it.

Install dependencies
```
$ pip install -r requirements.txt
```

Run the bot
```
$ python main.py
```
