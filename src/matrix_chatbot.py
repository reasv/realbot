"""Matrix chatbot using Simple-Matrix-Bot-Lib with E2EE support."""

import asyncio
import datetime
import html
import json
import os
import random
import re
from collections import deque
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Compatibility: some homeservers (Conduit, Conduwuit, etc.) omit the
# ``one_time_key_counts`` field from key-upload responses.  matrix-nio's
# ``Schemas.keys_upload`` marks it *required*, so every key-upload response
# fails validation and is returned as ``KeysUploadError``.  nio then
# retries on every sync cycle, spamming the log and potentially interfering
# with E2EE setup.  Patching the schema before any client is created
# fixes this.
# ---------------------------------------------------------------------------
def _patch_nio_key_upload_schema() -> None:
    try:
        from nio.schemas import Schemas  # type: ignore[import-untyped]

        for schema_name in ("keys_upload", "sync"):
            schema = getattr(Schemas, schema_name, None)
            if not isinstance(schema, dict):
                continue
            req = schema.get("required")
            if not isinstance(req, list):
                continue
            for key in ("one_time_key_counts", "device_one_time_keys_count"):
                try:
                    req.remove(key)
                except ValueError:
                    pass
    except Exception:
        pass


_patch_nio_key_upload_schema()

import simplematrixbotlib as botlib

from .chat_image_utils import build_remote_image_record
from .openai_inference import (
    chat_inference,
    finalize_assistant_message_id,
    finalize_last_assistant_message_id,
    get_swipe_nav_state,
    swipe_next,
    swipe_prev,
    swipe_regenerate,
)
from .utils import dequote, get_config, is_whitelisted_id, normalize_id_set


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class RandomChat:
    endTime: datetime.datetime
    nextChatTime: datetime.datetime
    lastEventId: str


# ---------------------------------------------------------------------------
# Pure helper functions (exported for tests)
# ---------------------------------------------------------------------------


def swipe_action_for_emoji(emoji: str, cfg: dict) -> str | None:
    """Map a reaction emoji to a swipe action name, or ``None``."""
    if emoji == cfg.get("regen_emoji", "\U0001f504"):
        return "regen"
    if emoji == cfg.get("prev_emoji", "\u25c0\ufe0f"):
        return "prev"
    if emoji == cfg.get("next_emoji", "\u25b6\ufe0f"):
        return "next"
    return None


def matrix_content_mentions_user(
    content: dict,
    user_id: str,
    aliases: list[str] | None = None,
) -> bool:
    """Return True if *content* mentions *user_id* via ``m.mentions`` or textual ``@alias``."""
    # 1. Structured mentions metadata
    mentions = content.get("m.mentions")
    if isinstance(mentions, dict):
        uids = mentions.get("user_ids")
        if isinstance(uids, list) and user_id in uids:
            return True

    # 2. Textual @alias in body
    body = content.get("body", "")
    if body and aliases:
        for alias in aliases:
            if not alias:
                continue
            pattern = re.compile(
                r"(?:^|(?<=\s)|(?<=@))"
                + re.escape(alias)
                + r"(?=\s|$|[,\.!?\;:\)\]\}])",
                re.IGNORECASE,
            )
            if pattern.search(body):
                return True
    return False


def apply_generated_at_mentions(
    text: str,
    resolver,
) -> tuple[str, str | None, list[str]]:
    """Convert ``@Name`` patterns in LLM output to Matrix mention pills.

    *resolver* is ``resolver(name) -> user_id | None``.

    Returns ``(plain_body, formatted_body | None, mentioned_user_ids)``.
    """
    if "@" not in text:
        return text, None, []

    boundary = set(" \t\n\r\"'`,!?;:()[]{}<>")
    mentioned: list[str] = []
    parts: list[str] = []
    has_pill = False
    i = 0

    while i < len(text):
        ch = text[i]

        if ch != "@":
            parts.append(ch)
            i += 1
            continue

        # Must follow a boundary (or start of string)
        if i > 0 and text[i - 1] not in boundary:
            parts.append(ch)
            i += 1
            continue

        # Must have a non-space character after @
        if i + 1 >= len(text) or text[i + 1].isspace():
            parts.append(ch)
            i += 1
            continue

        tail = text[i + 1 :]
        line_match = re.match(r"^[^\r\n]+", tail)
        raw = (line_match.group(0) if line_match else tail)[:64].strip()
        words = raw.split()
        if not words:
            parts.append(ch)
            i += 1
            continue

        counts = [1]
        if len(words) >= 2:
            counts.append(2)

        replaced = False
        for k in counts:
            base = " ".join(words[:k])
            candidates = [base]
            stripped = base.rstrip("\"'`.,!?;:()[]{}<>")
            if stripped and stripped != base:
                candidates.append(stripped)

            for cand in candidates:
                end = i + 1 + len(cand)
                if end < len(text) and text[end] not in boundary:
                    continue
                uid = resolver(cand)
                if not uid:
                    continue
                pill = f'<a href="https://matrix.to/#/{uid}">@{cand}</a>'
                parts.append(pill)
                mentioned.append(uid)
                has_pill = True
                i = end
                replaced = True
                break

            if replaced:
                break

        if not replaced:
            parts.append(ch)
            i += 1

    formatted = "".join(parts) if has_pill else None
    return text, formatted, mentioned


def dedupe_swipe_jobs(jobs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Keep only the last action per event_id, preserving order of last occurrence."""
    last: dict[str, tuple[int, str]] = {}
    for idx, (eid, action) in enumerate(jobs):
        last[eid] = (idx, action)
    return [
        (eid, action)
        for eid, (_, action) in sorted(last.items(), key=lambda kv: kv[1][0])
    ]


# ---------------------------------------------------------------------------
# Bot class
# ---------------------------------------------------------------------------


class MatrixBot:
    """Matrix chatbot with E2EE, swipe controls, and random-chat support."""

    def __init__(self):
        # Environment
        self.homeserver: str = os.getenv("MATRIX_HOMESERVER", "")
        self.user_id: str = os.getenv("MATRIX_USER_ID", "")
        self.bot_name: str = os.getenv("BOT_NAME", "Bot")
        store_path: str = os.getenv("MATRIX_STORE_PATH", "./history/matrix_store/")
        os.makedirs(store_path, exist_ok=True)

        # Aliases for textual mention detection
        self._aliases: list[str] = [self.bot_name]
        localpart = (
            self.user_id.split(":")[0].lstrip("@") if ":" in self.user_id else ""
        )
        if localpart and localpart.casefold() != self.bot_name.casefold():
            self._aliases.append(localpart)

        # --- simplematrixbotlib setup ---
        creds_kw: dict[str, Any] = {
            "homeserver": self.homeserver,
            "username": self.user_id,
            "session_stored_file": "history/matrix_session.txt",
        }
        access_token = os.getenv("MATRIX_ACCESS_TOKEN", "")
        password = os.getenv("MATRIX_PASSWORD", "")
        if access_token:
            creds_kw["access_token"] = access_token
        elif password:
            creds_kw["password"] = password

        self.creds = botlib.Creds(**creds_kw)

        self.bot_config = botlib.Config()
        self.bot_config.encryption_enabled = True
        self.bot_config.ignore_unverified_devices = True
        self.bot_config.store_path = store_path
        self.bot_config.join_on_invite = False

        self.sbot = botlib.Bot(creds=self.creds, config=self.bot_config)
        self.client: Any = None  # nio.AsyncClient -- set on startup

        # Runtime state
        self.pendingMessages: dict[str, list[dict[str, Any]]] = {}
        self.pendingSwipes: dict[str, list[tuple[str, str]]] = {}
        self.randomChats: dict[str, RandomChat] = {}
        self.recentMessages: dict[str, deque] = {}
        self._bg_started: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _register_handlers(self):
        @self.sbot.listener.on_startup
        async def _startup(room_id):
            if not self._bg_started:
                self._bg_started = True
                self.client = self.sbot.async_client
                print(f"[Matrix] Logged in as {self.client.user_id}")
                await self._bootstrap_encryption()
                asyncio.create_task(self._inference_loop())

        @self.sbot.listener.on_message_event
        async def _message(room, message):
            await self._on_message(room, message)

        @self.sbot.listener.on_reaction_event
        async def _reaction(room, event, reaction):
            await self._on_reaction(room, event, reaction)

        # Handle undecryptable events -- request missing session keys
        try:
            import nio as _nio

            @self.sbot.listener.on_custom_event(_nio.MegolmEvent)
            async def _undecryptable(room, event):
                await self._on_megolm_event(room, event)
        except Exception:
            pass

    def run(self):
        self._register_handlers()
        self.sbot.run()

    # ------------------------------------------------------------------
    # E2EE bootstrap
    # ------------------------------------------------------------------

    async def _bootstrap_encryption(self) -> None:
        """Upload device keys, query room members, and establish Olm sessions.

        Must be called once after the async_client is ready.
        """
        if not self.client:
            return

        # 1. Upload our device keys + one-time keys so other devices can
        #    discover us and include us in future Megolm sessions.
        try:
            resp = await self.client.keys_upload()
            rtype = type(resp).__name__
            if "Error" in rtype:
                print(
                    f"[Matrix] Key upload returned {rtype}: {getattr(resp, 'message', resp)}"
                )
            else:
                print(f"[Matrix] Device keys uploaded successfully")
        except Exception as exc:
            print(f"[Matrix] Key upload failed: {exc}")

        # 2. Query device keys for every user in our rooms so we know
        #    their devices and can establish Olm sessions when sending.
        users_to_query: set[str] = set()
        for room in self.client.rooms.values():
            for uid in room.users:
                if uid != self.user_id:
                    users_to_query.add(uid)

        if users_to_query:
            try:
                resp = await self.client.keys_query()
                rtype = type(resp).__name__
                if "Error" in rtype:
                    print(
                        f"[Matrix] Key query returned {rtype}: {getattr(resp, 'message', resp)}"
                    )
                else:
                    print(
                        f"[Matrix] Queried device keys for {len(users_to_query)} user(s)"
                    )
            except Exception as exc:
                print(f"[Matrix] Key query failed: {exc}")

        # 3. Claim one-time keys for devices we don't have Olm sessions
        #    with yet (enables sending encrypted messages immediately).
        if hasattr(self.client, "olm") and self.client.olm:
            missing: dict[str, list[str]] = {}
            for uid in users_to_query:
                devices = self.client.device_store.active_user_devices(uid)
                for dev in devices:
                    if not self.client.olm.session_store.get(dev.curve25519):
                        missing.setdefault(uid, []).append(dev.device_id)

            if missing:
                try:
                    resp = await self.client.keys_claim(missing)
                    rtype = type(resp).__name__
                    n = sum(len(ds) for ds in missing.values())
                    if "Error" in rtype:
                        print(
                            f"[Matrix] Key claim returned {rtype}: {getattr(resp, 'message', resp)}"
                        )
                    else:
                        print(f"[Matrix] Claimed one-time keys for {n} device(s)")
                except Exception as exc:
                    print(f"[Matrix] Key claim failed: {exc}")

        print("[Matrix] E2EE bootstrap complete")

    async def _on_megolm_event(self, room, event) -> None:
        """Handle an undecryptable Megolm event by requesting the session key."""
        sid = getattr(event, "session_id", "?")
        sender = getattr(event, "sender", "?")
        print(
            f"[Matrix] Undecryptable event in {room.room_id} from {sender} "
            f"(session {sid}) -- requesting key"
        )
        if not self.client:
            return
        try:
            await self.client.request_room_key(event)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _on_message(self, room, message):
        sender = message.sender
        room_id = room.room_id
        content = message.source.get("content", {})
        body = content.get("body", "").strip()
        msgtype = content.get("msgtype", "")

        # Buffer ALL messages (including own) for random-chat context
        if room_id not in self.recentMessages:
            self.recentMessages[room_id] = deque(maxlen=200)
        processed = self._process_message(room, message)
        self.recentMessages[room_id].append(processed)

        # Don't respond to our own messages
        if sender == self.user_id:
            return

        # Only act on text messages
        if msgtype != "m.text":
            return
        if not body:
            return

        # Whitelist routing
        is_mentioned = matrix_content_mentions_user(
            content, self.user_id, self._aliases
        )

        if is_mentioned and _is_whitelisted(room_id, "mentions"):
            return await self._constant_chat(room_id, processed)

        if _is_whitelisted(room_id, "always"):
            return await self._constant_chat(room_id, processed)

        if _is_whitelisted(room_id, "rand"):
            return await self._random_chat(room_id, processed, is_mentioned)

    async def _on_reaction(self, room, event, reaction):
        sender = event.source.get("sender", "")
        if sender == self.user_id:
            return

        cfg = get_config().get("swipes", {}) or {}
        if not cfg.get("enabled", False):
            return

        if not self._swipe_allowed(cfg, sender, room.room_id):
            return

        action = swipe_action_for_emoji(reaction, cfg)
        if not action:
            return

        target = (
            event.source.get("content", {}).get("m.relates_to", {}).get("event_id", "")
        )
        if not target:
            return

        print(f"[{room.room_id}] Swipe {action} on {target}")
        self.pendingSwipes.setdefault(room.room_id, []).append((target, action))

    # ------------------------------------------------------------------
    # Message processing
    # ------------------------------------------------------------------

    def _process_message(self, room, message) -> dict[str, Any]:
        sender = message.sender
        content = message.source.get("content", {})
        body = content.get("body", "").strip()
        event_id = message.source.get("event_id", "")
        msgtype = content.get("msgtype", "")

        username = self._get_display_name(room, sender)
        cleaned = self._clean_content(body)

        out: dict[str, Any] = {
            "user": username,
            "message": cleaned,
            "messageId": event_id,
        }

        images = self._extract_images(content, msgtype)
        if images:
            out["images"] = images
        return out

    def _get_display_name(self, room, user_id: str) -> str:
        if user_id == self.user_id:
            return "{{char}}"
        try:
            user = room.users.get(user_id)
            if user and getattr(user, "display_name", None):
                return user.display_name
        except Exception:
            pass
        # Fall back to localpart
        if ":" in user_id:
            return user_id.split(":")[0].lstrip("@")
        return user_id

    def _clean_content(self, content: str) -> str:
        """Replace textual mentions of the bot with ``{{char}}``."""
        for alias in self._aliases:
            if alias:
                content = re.sub(
                    r"@?" + re.escape(alias),
                    "{{char}}",
                    content,
                    flags=re.IGNORECASE,
                )
        return content.strip()

    @staticmethod
    def _extract_images(content: dict, msgtype: str) -> list[dict[str, Any]]:
        images: list[dict[str, Any]] = []

        # m.image message type
        if msgtype == "m.image":
            url = content.get("url", "")
            if url:
                images.append(build_remote_image_record(url, source="matrix_image"))
            enc = content.get("file", {})
            if isinstance(enc, dict) and enc.get("url"):
                images.append(
                    build_remote_image_record(
                        enc["url"], source="matrix_encrypted_image"
                    )
                )

        # Image URLs in text body
        body = content.get("body", "")
        if body:
            exts = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")
            for m in re.finditer(r"(https?://\S+)", body, re.I):
                url = m.group(1).rstrip(").,>")
                if any(url.lower().split("?")[0].endswith(e) for e in exts):
                    images.append(build_remote_image_record(url, source="matrix_link"))

        return images

    # ------------------------------------------------------------------
    # Chat behaviours
    # ------------------------------------------------------------------

    async def _constant_chat(self, room_id: str, msg: dict[str, Any]):
        self._add_to_queue(room_id, msg)

    async def _random_chat(self, room_id: str, msg: dict[str, Any], is_mentioned: bool):
        cfg = get_config()["randomChat"]
        now = datetime.datetime.now()
        current = self.randomChats.get(room_id)

        if current:
            if now < current.endTime:
                current.lastEventId = msg.get("messageId", "")
                self._add_to_queue(room_id, msg)
                return
            if now < current.nextChatTime:
                print(
                    f"[{room_id}] Not starting new conversation until "
                    f"{current.nextChatTime}"
                )
                return

        # Roll for engagement
        if random.randint(0, cfg["engagement_chance"]) != 0:
            if not is_mentioned or not cfg.get("respond_to_mentions", True):
                print(f"[{room_id}] Random chat roll failed")
                return

        # Fetch recent messages from buffer
        recent = list(self.recentMessages.get(room_id, []))
        msgs = recent[-cfg.get("message_history_limit", 10) :]

        # Check for overlap with previous session
        if current and msgs:
            for m in msgs:
                if m.get("messageId") == current.lastEventId:
                    print(f"[{room_id}] Insufficient new messages")
                    return

        if not msgs:
            return

        # Guard against race condition
        if current != self.randomChats.get(room_id):
            return

        end = now + datetime.timedelta(
            seconds=random.randint(
                cfg["min_chat_duration_seconds"], cfg["max_chat_duration_seconds"]
            )
        )
        self.randomChats[room_id] = RandomChat(
            endTime=end,
            nextChatTime=end
            + datetime.timedelta(
                minutes=random.randint(
                    cfg["min_downtime_minutes"], cfg["max_downtime_minutes"]
                )
            ),
            lastEventId=msg.get("messageId", ""),
        )

        for m in msgs:
            self._add_to_queue(room_id, m)

        print(
            f"[{room_id}] Starting random chat until {end}. "
            f"Next earliest: {self.randomChats[room_id].nextChatTime}"
        )

    def _add_to_queue(self, room_id: str, msg: dict[str, Any]):
        extra = f" [{len(msg['images'])} image(s)]" if msg.get("images") else ""
        print(f"[{room_id}] {msg['user']}: {msg['message']}{extra}")
        self.pendingMessages.setdefault(room_id, []).append(msg)

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------

    async def _inference_loop(self):
        print("[Matrix] Background inference loop started")
        while True:
            try:
                await self._process_messages()
            except Exception as exc:
                print(f"[Matrix] inference loop error: {exc}")
            await asyncio.sleep(0.5)

    async def _process_messages(self):
        room_ids = list(
            set(self.pendingMessages.keys()) | set(self.pendingSwipes.keys())
        )
        tasks = []
        for rid in room_ids:
            pending = self.pendingMessages.get(rid, [])
            swipes = self.pendingSwipes.get(rid, [])
            if not pending and not swipes:
                continue
            self.pendingMessages[rid] = []
            self.pendingSwipes[rid] = []
            tasks.append(self._process_room(rid, pending, swipes))
        if tasks:
            await asyncio.gather(*tasks)

    async def _process_room(
        self,
        room_id: str,
        messages: list[dict[str, Any]],
        swipe_jobs: list[tuple[str, str]],
    ):
        # Show typing indicator
        try:
            await self._set_typing(room_id, True)
        except Exception:
            pass

        try:
            await self._handle_swipes(room_id, swipe_jobs)
            result = await self._handle_inference(room_id, messages)
        finally:
            try:
                await self._set_typing(room_id, False)
            except Exception:
                pass

        await self._handle_response(room_id, result)

    async def _handle_swipes(self, room_id: str, swipe_jobs: list[tuple[str, str]]):
        if not swipe_jobs:
            return

        scfg = get_config().get("swipes", {}) or {}
        prev_e = str(scfg.get("prev_emoji", "\u25c0\ufe0f"))
        next_e = str(scfg.get("next_emoji", "\u25b6\ufe0f"))

        for eid, action in dedupe_swipe_jobs(swipe_jobs):
            new_text: str | None = None
            try:
                if action == "regen":
                    new_text = await swipe_regenerate(room_id, eid)
                elif action == "prev":
                    new_text = swipe_prev(room_id, eid)
                elif action == "next":
                    new_text = swipe_next(room_id, eid)
            except Exception as e:
                print(f"[{room_id}] Swipe {action} failed for {eid}: {e}")
                continue

            if new_text:
                cleaned = _clean_response(new_text)
                try:
                    await self.send_edit(room_id, eid, cleaned)
                except Exception as e:
                    print(f"[{room_id}] Swipe edit failed for {eid}: {e}")

            # Update navigation reactions
            try:
                nav = get_swipe_nav_state(room_id, eid)
                if nav is not None:
                    hp, hn = nav
                    await self._set_control_reaction(room_id, eid, prev_e, hp)
                    await self._set_control_reaction(room_id, eid, next_e, hn)
            except Exception as e:
                print(f"[{room_id}] Swipe nav update failed: {e}")

    async def _handle_inference(self, room_id: str, messages: list[dict[str, Any]]):
        if not messages:
            return None
        try:
            return await chat_inference(room_id, messages)
        except Exception as e:
            print(f"[{room_id}] chat_inference error: {e}")
            return None

    async def _handle_response(self, room_id: str, result):
        response: str | None = None
        pending_id: str | None = None

        if isinstance(result, tuple) and len(result) == 2:
            response, pending_id = result
        elif isinstance(result, str):
            response = result

        if not response:
            return

        cleaned = _clean_response(response)
        room_obj = self.client.rooms.get(room_id) if self.client else None
        plain, fmt, mention_ids = self._resolve_mentions(cleaned, room_obj)

        sent_eid = await self.send_message(room_id, plain, fmt, mention_ids)
        if not sent_eid:
            return

        # Store the real platform message ID
        try:
            if pending_id:
                finalize_assistant_message_id(room_id, pending_id, sent_eid)
            else:
                finalize_last_assistant_message_id(room_id, sent_eid)
        except Exception as e:
            print(f"[{room_id}] Failed to store messageId: {e}")

        # Swipe control reactions
        scfg = get_config().get("swipes", {}) or {}
        try:
            await self._cleanup_recorded_control_reactions(room_id)
        except Exception as e:
            print(f"[{room_id}] Reaction cleanup failed: {e}")

        if scfg.get("auto_react_controls", False):
            ar_wl = normalize_id_set(scfg.get("auto_react_channel_whitelist", []))
            if (not ar_wl) or (room_id in ar_wl):
                regen = str(scfg.get("regen_emoji", "\U0001f504"))
                try:
                    r_eid = await self.send_reaction(room_id, sent_eid, regen)
                    if r_eid:
                        self._record_control_reaction(room_id, sent_eid, regen, r_eid)
                except Exception as e:
                    print(f"[{room_id}] Auto-react failed: {e}")

    # ------------------------------------------------------------------
    # @-mention resolution in generated responses
    # ------------------------------------------------------------------

    def _resolve_mentions(self, text: str, room) -> tuple[str, str | None, list[str]]:
        if not room or "@" not in text:
            return text, None, []

        users = getattr(room, "users", {}) or {}

        def resolver(name: str) -> str | None:
            key = name.casefold()
            if key in {"everyone", "here"}:
                return None
            for uid, u in users.items():
                dn = getattr(u, "display_name", None)
                if dn and dn.casefold() == key:
                    return uid
                lp = uid.split(":")[0].lstrip("@") if ":" in uid else ""
                if lp and lp.casefold() == key:
                    return uid
            return None

        return apply_generated_at_mentions(text, resolver)

    # ------------------------------------------------------------------
    # Matrix send primitives
    # ------------------------------------------------------------------

    async def send_message(
        self,
        room_id: str,
        body: str,
        formatted_body: str | None = None,
        mention_user_ids: list[str] | None = None,
    ) -> str | None:
        content: dict[str, Any] = {"body": body, "msgtype": "m.text"}
        if formatted_body:
            content["format"] = "org.matrix.custom.html"
            content["formatted_body"] = formatted_body
        content["m.mentions"] = {"user_ids": mention_user_ids or []}

        try:
            resp = await self.client.room_send(
                room_id=room_id,
                message_type="m.room.message",
                content=content,
                ignore_unverified_devices=True,
            )
            return getattr(resp, "event_id", None)
        except Exception as e:
            print(f"[{room_id}] send_message failed: {e}")
            return None

    async def send_edit(
        self,
        room_id: str,
        target_event_id: str,
        new_text: str,
    ) -> str | None:
        content: dict[str, Any] = {
            "m.relates_to": {
                "rel_type": "m.replace",
                "event_id": target_event_id,
            },
            "m.new_content": {"body": new_text, "msgtype": "m.text"},
            "body": new_text,
            "msgtype": "m.text",
        }
        try:
            resp = await self.client.room_send(
                room_id=room_id,
                message_type="m.room.message",
                content=content,
                ignore_unverified_devices=True,
            )
            return getattr(resp, "event_id", None)
        except Exception as e:
            print(f"[{room_id}] send_edit failed: {e}")
            return None

    async def send_reaction(
        self,
        room_id: str,
        target_event_id: str,
        emoji: str,
    ) -> str | None:
        content = {
            "m.relates_to": {
                "event_id": target_event_id,
                "rel_type": "m.annotation",
                "key": emoji,
            }
        }
        try:
            resp = await self.client.room_send(
                room_id=room_id,
                message_type="m.reaction",
                content=content,
                ignore_unverified_devices=True,
            )
            return getattr(resp, "event_id", None)
        except Exception as e:
            print(f"[{room_id}] send_reaction failed: {e}")
            return None

    async def _set_typing(self, room_id: str, typing: bool):
        if self.client:
            await self.client.room_typing(room_id, typing, timeout=30000)

    # ------------------------------------------------------------------
    # Swipe permission check
    # ------------------------------------------------------------------

    @staticmethod
    def _swipe_allowed(cfg: dict, user_id: str, room_id: str) -> bool:
        uwl = normalize_id_set(cfg.get("user_whitelist", []))
        cwl = normalize_id_set(cfg.get("channel_whitelist", []))
        if not uwl and not cwl:
            return True
        if not uwl:
            return room_id in cwl
        if not cwl:
            return user_id in uwl
        return user_id in uwl or room_id in cwl

    # ------------------------------------------------------------------
    # Control-reaction bookkeeping
    # ------------------------------------------------------------------

    def _control_reactions_path(self, room_id: str) -> str:
        return os.path.join("history", "control-reactions", f"{room_id}.json")

    def _load_control_reactions(self, room_id: str) -> list[dict[str, str]]:
        path = self._control_reactions_path(room_id)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict) and isinstance(data.get("reactions"), list):
                out: list[dict[str, str]] = []
                for item in data["reactions"]:
                    if not isinstance(item, dict):
                        continue
                    mid = item.get("messageId")
                    emoji = item.get("emoji")
                    if not isinstance(mid, str) or not mid:
                        continue
                    if not isinstance(emoji, str) or not emoji:
                        continue
                    entry: dict[str, str] = {"messageId": mid, "emoji": emoji}
                    reid = item.get("reactionEventId")
                    if isinstance(reid, str) and reid:
                        entry["reactionEventId"] = reid
                    out.append(entry)
                return out
        except Exception:
            pass
        return []

    def _save_control_reactions(
        self, room_id: str, reactions: list[dict[str, str]]
    ) -> None:
        path = self._control_reactions_path(room_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"reactions": reactions}, fh, indent=2)

    def _record_control_reaction(
        self,
        room_id: str,
        message_id: str,
        emoji: str,
        reaction_event_id: str,
    ) -> None:
        existing = self._load_control_reactions(room_id)
        key = (message_id, emoji)
        if key not in {(r.get("messageId"), r.get("emoji")) for r in existing}:
            existing.append(
                {
                    "messageId": message_id,
                    "emoji": emoji,
                    "reactionEventId": reaction_event_id,
                }
            )
        self._save_control_reactions(room_id, existing)

    def _unrecord_control_reaction(
        self, room_id: str, message_id: str, emoji: str
    ) -> None:
        existing = self._load_control_reactions(room_id)
        filtered = [
            r
            for r in existing
            if not (r["messageId"] == message_id and r["emoji"] == emoji)
        ]
        if len(filtered) != len(existing):
            self._save_control_reactions(room_id, filtered)

    async def _remove_control_reaction(
        self, room_id: str, reaction_event_id: str
    ) -> bool:
        """Redact a reaction event to remove it."""
        if not self.client:
            return False
        try:
            await self.client.room_redact(room_id, reaction_event_id)
            return True
        except Exception as e:
            print(f"[{room_id}] Failed to redact reaction {reaction_event_id}: {e}")
            return False

    async def _set_control_reaction(
        self,
        room_id: str,
        message_id: str,
        emoji: str,
        should_have: bool,
    ) -> None:
        if should_have:
            reid = await self.send_reaction(room_id, message_id, emoji)
            if reid:
                self._record_control_reaction(room_id, message_id, emoji, reid)
            return

        # Remove: find recorded reaction event ID and redact it
        records = self._load_control_reactions(room_id)
        for rec in records:
            if rec.get("messageId") == message_id and rec.get("emoji") == emoji:
                reid = rec.get("reactionEventId", "")
                if reid:
                    await self._remove_control_reaction(room_id, reid)
                break
        self._unrecord_control_reaction(room_id, message_id, emoji)

    async def _cleanup_recorded_control_reactions(self, room_id: str) -> None:
        """Remove all recorded control reactions (called before adding new ones)."""
        records = self._load_control_reactions(room_id)
        if not records:
            return
        remaining: list[dict[str, str]] = []
        for rec in records:
            reid = rec.get("reactionEventId", "")
            if reid:
                ok = await self._remove_control_reaction(room_id, reid)
                if not ok:
                    remaining.append(rec)
            else:
                remaining.append(rec)
        self._save_control_reactions(room_id, remaining)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _is_whitelisted(room_id: str, wtype: str = "always") -> bool:
    cfg = get_config()
    return is_whitelisted_id(room_id, cfg.get("whitelist", {}).get(wtype, []))


def _clean_response(resp: str) -> str:
    return dequote(html.unescape(resp).strip())


def run_bot():
    """Entry-point called by *main.py*."""
    bot = MatrixBot()
    bot.run()
