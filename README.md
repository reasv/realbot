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
- `OPENAI_RESPONSE_LOG_FILE` (optional; appends request/response JSONL entries, including `system_prompt_template_filename`)

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
- `MATRIX_IMPORT_KEYS_PATH` (optional, path to exported Megolm keys file)
- `MATRIX_IMPORT_KEYS_PASSWORD` (optional, passphrase for imported keys)

Matrix auth is password based. The bot auto-accepts room invites.

If logs show `Received undecrypted event ...`, the device is missing room keys.
You can either:

- Verify this device and wait for new messages, or
- Export E2EE room keys from another client (e.g. Element) and set
  `MATRIX_IMPORT_KEYS_PATH` + `MATRIX_IMPORT_KEYS_PASSWORD` so the bot can import them on startup.

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

### Global bot options

Configured under `[bot]`.

- `respond_to_dms`: when `true`, Matrix DMs (1:1 rooms) are handled even if not whitelisted.

### Random chat

Configured under `[randomChat]`.
Controls engagement chance, session duration, cooldown, and whether mentions can bypass failed roll.

### Swipes

Configured under `[swipes]`.
Supports regenerate/prev/next controls, whitelists, and optional auto-react controls.
`user_whitelist`, `channel_whitelist`, and `auto_react_channel_whitelist` accept both integers and strings.

### OpenAI output truncation

Optional prompt/truncation controls under `[openai]`:

- `system_prompt_template_dir` (optional directory containing prompt template files; default `prompts`)
- `system_prompt_template_name` (optional template filename, e.g. `example_system_prompt.txt`; supports `{{assistant_username}}`)
- `system_prompt_template_dm_name` (optional DM-wide template filename)
- `system_prompt_template_channel_overrides` (optional channel ID -> template filename map)
- `stopping_strings`
- `stopping_strings_limit`

Priority order when selecting prompt templates:
`channel override > DM override > global template`

Example:

```toml
[openai]
system_prompt_template_dir = "prompts"
system_prompt_template_name = "example_system_prompt.txt"
system_prompt_template_dm_name = "dm_prompt.txt"

[openai.system_prompt_template_channel_overrides]
"!roomid:matrix.org" = "matrix_room_prompt.txt"
"#ai" = "irc_channel_prompt.txt"
"1143322712169787493" = "discord_channel_prompt.txt"
```

### Image context

- Discord attachments and embeds are captured.
- IRC image URLs are captured.
- Matrix image events and image URLs are captured, including encrypted-room media when decryption succeeds.

Cached images are written under `history/images/` so history replay can include image context.
