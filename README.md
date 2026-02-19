# realbot

LLM-powered chat bot with multiple transports:
- Discord (user account via `discord.py-self`)
- IRC
- Matrix (user account via `matrix-nio`, including E2EE rooms)

## Getting started

Requires Python 3.11.

Copy `.env.example` to `.env` and configure credentials.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Matrix E2EE prerequisite

For encrypted Matrix rooms, `matrix-nio[e2e]` requires libolm on the system.
Install libolm before installing requirements if your platform does not already provide it.

## Run

Run Discord only:

```bash
python run_discord.py
```

Run IRC only:

```bash
python run_irc.py
```

Run Matrix only:

```bash
python run_matrix.py
```

Run the legacy dual-process Discord+IRC launcher:

```bash
python main.py
```

## Environment variables

General:
- `BOT_NAME`
- `LLM_API_KEY`
- `OPENAI_API_URL`
- `SAMPLING_OVERRIDE_FILE`

Discord:
- `DISCORD_TOKEN`

IRC:
- `IRC_SERVER`
- `IRC_PORT`
- `IRC_NICKNAME`
- `IRC_CHANNELS`

Matrix:
- `MATRIX_HOMESERVER`
- `MATRIX_USER_ID`
- `MATRIX_ACCESS_TOKEN`
- `MATRIX_DEVICE_ID`
- `MATRIX_STORE_PATH` (default: `history/matrix_store`)
- `MATRIX_SYNC_TIMEOUT_MS` (default: `30000`)

Matrix auth is token-based only. Room joins/invite acceptance are not handled by bot logic; join rooms externally with the bot account.

## Configuration

Create `user.config.toml` from `user.config.toml.example`.
Defaults live in `default.config.toml`.

### Whitelist

The bot only responds in whitelisted destinations.
Whitelist entries support mixed ID formats:
- Discord numeric channel IDs
- IRC channel names (e.g. `#ai`)
- Matrix room IDs (e.g. `!roomid:matrix.org`)

Buckets:
- `always`: respond to every message
- `mentions`: respond only when mentioned
- `rand`: random-chat behavior

### Random chat

Configured under `[randomChat]`.
Controls engagement chance, session duration, cooldown, and whether mentions can bypass failed roll.

### Swipes

Configured under `[swipes]`.
Supports regenerate/prev/next controls, whitelists, and optional auto-react controls.
`user_whitelist`, `channel_whitelist`, and `auto_react_channel_whitelist` accept both integers and strings.

### OpenAI output truncation

Optional truncation controls under `[openai]`:
- `stopping_strings`
- `stopping_strings_limit`

### Image context

- Discord attachments and embeds are captured.
- IRC image URLs are captured.
- Matrix image events and image URLs are captured, including encrypted-room media when decryption succeeds.

Cached images are written under `history/images/` so history replay can include image context.
