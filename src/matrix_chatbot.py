"""Matrix chatbot using mautrix-python with E2EE support.

Provides the same feature set as the Discord bot:
  - Whitelist-based routing (always / mentions / rand)
  - Random-chat engagement with configurable timing
  - Swipe (regenerate / prev / next) via Matrix reactions
  - @-mention resolution in generated text (Matrix pill links)
  - Image extraction from m.image events and URLs in text
  - Background inference loop with 0.5 s polling
  - Typing indicators while generating
  - Control-reaction bookkeeping on disk
  - Full E2EE support via OlmMachine (auto-encrypt / auto-decrypt)
"""

from __future__ import annotations

import asyncio
import datetime
import html
import json
import logging
import os
import random
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

from mautrix.client import Client
from mautrix.client.encryption_manager import DecryptionDispatcher
from mautrix.client.state_store.memory import MemoryStateStore
from mautrix.client.syncer import InternalEventType
from mautrix.crypto import OlmMachine
from mautrix.crypto.attachments import decrypt_attachment
from mautrix.crypto.store.asyncpg import PgCryptoStore
from mautrix.types import (
    EventID,
    EventType,
    Membership,
    MessageType,
    RoomID,
    UserID,
)
from mautrix.util.async_db import Database

from .chat_image_utils import build_remote_image_record, download_image_to_history
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

log = logging.getLogger(__name__)

# ── Module-level helpers ─────────────────────────────────────────────


def swipe_action_for_emoji(emoji: str, cfg: dict) -> str | None:
    """Map an emoji string to a swipe action using the supplied config."""
    if not emoji or not cfg:
        return None
    if emoji == cfg.get("regen_emoji"):
        return "regen"
    if emoji == cfg.get("prev_emoji"):
        return "prev"
    if emoji == cfg.get("next_emoji"):
        return "next"
    return None


def matrix_content_mentions_user(
    content: dict, user_id: str, aliases: list[str]
) -> bool:
    """Return *True* if *content* mentions the bot via ``m.mentions`` or body text."""
    # 1. Structured m.mentions
    mentions = content.get("m.mentions", {})
    if isinstance(mentions, dict):
        user_ids = mentions.get("user_ids", [])
        if isinstance(user_ids, list) and user_id in user_ids:
            return True
    # 2. Textual @alias in body
    body = content.get("body", "")
    if isinstance(body, str):
        for alias in aliases:
            if re.search(rf"(?<!\w)@?{re.escape(alias)}(?!\w)", body, re.IGNORECASE):
                return True
    return False


_MENTION_BOUNDARY = set(" \t\n\"'`,!?;:()[]{}<>")


def _is_mention_boundary(ch: str) -> bool:
    return ch in _MENTION_BOUNDARY


def apply_generated_at_mentions(
    text: str, resolver: Callable[[str], str | None]
) -> tuple[str, str | None, list[str]]:
    """Scan *text* for ``@Name`` patterns, resolving via *resolver*.

    Returns ``(plain_body, formatted_body | None, mentioned_user_ids)``.
    ``formatted_body`` is only set when at least one mention was resolved.
    """
    if "@" not in text:
        return (text, None, [])

    mentioned_ids: list[str] = []
    has_mentions = False
    plain_parts: list[str] = []
    html_parts: list[str] = []

    i = 0
    while i < len(text):
        ch = text[i]
        if ch != "@":
            plain_parts.append(ch)
            html_parts.append(html.escape(ch))
            i += 1
            continue

        # Must be at a word boundary
        if i > 0 and not _is_mention_boundary(text[i - 1]):
            plain_parts.append(ch)
            html_parts.append(ch)
            i += 1
            continue

        # Must have a non-space char following @
        if i + 1 >= len(text) or text[i + 1].isspace():
            plain_parts.append(ch)
            html_parts.append(ch)
            i += 1
            continue

        tail = text[i + 1 :]
        line_end = tail.find("\n")
        if line_end == -1:
            line_end = len(tail)
        segment = tail[: min(line_end, 64)]
        words = segment.split()

        if not words:
            plain_parts.append(ch)
            html_parts.append(ch)
            i += 1
            continue

        replaced = False
        for k in (1, 2):
            if k > len(words):
                break
            candidate_raw = " ".join(words[:k])
            # Re-slice from tail so spacing is exact
            candidate_base = tail[: len(candidate_raw)]

            variants = [candidate_base]
            stripped = candidate_base.rstrip("\"'`.,!?;:()[]{}<>")
            if stripped and stripped != candidate_base:
                variants.append(stripped)

            for candidate in variants:
                after_idx = i + 1 + len(candidate)
                if after_idx < len(text) and not _is_mention_boundary(text[after_idx]):
                    continue

                user_id = resolver(candidate)
                if not user_id:
                    continue

                pill = (
                    f'<a href="https://matrix.to/#/{html.escape(user_id)}">'
                    f"{html.escape(candidate)}</a>"
                )
                plain_parts.append(f"@{candidate}")
                html_parts.append(pill)
                if user_id not in mentioned_ids:
                    mentioned_ids.append(user_id)
                has_mentions = True
                i = after_idx
                replaced = True
                break
            if replaced:
                break

        if not replaced:
            plain_parts.append(ch)
            html_parts.append(ch)
            i += 1

    plain = "".join(plain_parts)
    if has_mentions:
        return (plain, "".join(html_parts), mentioned_ids)
    return (plain, None, mentioned_ids)


def dedupe_swipe_jobs(
    jobs: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Keep only the *last* action per message-id, preserving order."""
    last_by_msg: dict[str, tuple[int, str]] = {}
    for idx, (msg_id, action) in enumerate(jobs):
        last_by_msg[msg_id] = (idx, action)
    return [
        (msg_id, action)
        for msg_id, (_, action) in sorted(last_by_msg.items(), key=lambda kv: kv[1][0])
    ]


def _is_whitelisted(room_id: str, wtype: str) -> bool:
    cfg = get_config()
    whitelist_values = cfg.get("whitelist", {}).get(wtype, [])
    return is_whitelisted_id(room_id, whitelist_values)


def _clean_response(resp: str) -> str:
    return dequote(html.unescape(resp).strip())


# ── State store with crypto-side find_shared_rooms ───────────────────


class BotStateStore(MemoryStateStore):
    """``MemoryStateStore`` extended with ``find_shared_rooms`` for the
    crypto ``StateStore`` interface used by ``OlmMachine``."""

    async def find_shared_rooms(self, user_id: UserID) -> list[RoomID]:
        shared: list[RoomID] = []
        for room_id, members in self.members.items():
            member = members.get(user_id)
            if not member or member.membership != Membership.JOIN:
                continue
            if await self.is_encrypted(room_id):
                shared.append(room_id)
        return shared


# ── Custom decryption dispatcher ─────────────────────────────────────


class RetryingDecryptionDispatcher(DecryptionDispatcher):
    """DecryptionDispatcher that waits for missing Megolm sessions before
    giving up.  The stock dispatcher logs a warning and drops the event;
    this one calls ``wait_for_session()`` (up to 5 s) so key-shares that
    arrive in the same or next sync batch can still satisfy the decrypt."""

    async def handle(self, evt: Any) -> None:
        try:
            decrypted = await self.client.crypto.decrypt_megolm_event(evt)
        except Exception as first_err:
            session_id = getattr(getattr(evt, "content", None), "session_id", None)
            if session_id:
                log.debug(
                    "Waiting up to 5 s for session %s (event %s)…",
                    session_id,
                    evt.event_id,
                )
                arrived = await self.client.crypto.wait_for_session(
                    evt.room_id, session_id, timeout=5
                )
                if arrived:
                    try:
                        decrypted = await self.client.crypto.decrypt_megolm_event(evt)
                    except Exception as retry_err:
                        self.client.crypto_log.warning(
                            "Failed to decrypt %s after session wait: %s",
                            evt.event_id,
                            retry_err,
                        )
                        return
                else:
                    self.client.crypto_log.warning(
                        "Failed to decrypt %s: %s (session %s never arrived)",
                        evt.event_id,
                        first_err,
                        session_id,
                    )
                    return
            else:
                self.client.crypto_log.warning(
                    "Failed to decrypt %s: %s", evt.event_id, first_err
                )
                return
        self.client.dispatch_event(decrypted, evt.source)


# ── Data classes ─────────────────────────────────────────────────────


@dataclass
class RandomChat:
    endTime: datetime.datetime
    nextChatTime: datetime.datetime
    lastEventId: str


# ── Main bot class ───────────────────────────────────────────────────


class MatrixBot:
    """Full-featured Matrix chatbot with E2EE, swipes, random chat, and
    mention resolution -- feature-parity with the Discord implementation."""

    # Populated during __init__ / run()
    client: Client
    state_store: BotStateStore
    crypto_db: Database
    crypto_store: PgCryptoStore
    crypto_machine: OlmMachine

    def __init__(self) -> None:
        self.homeserver: str = os.environ["MATRIX_HOMESERVER"]
        self.bot_mxid: str = os.environ["MATRIX_USER_ID"]
        self.bot_name: str = os.environ.get("BOT_NAME", "")
        if not self.bot_name:
            self.bot_name = self.bot_mxid.split(":")[0].lstrip("@")
        self.access_token: str = os.environ.get("MATRIX_ACCESS_TOKEN", "")
        self.password: str = os.environ.get("MATRIX_PASSWORD", "")
        self.device_id: str = os.environ.get("MATRIX_DEVICE_ID", "")
        self.store_path: str = os.environ.get(
            "MATRIX_STORE_PATH", "./history/matrix_store/"
        )

        # Queues
        self.pendingMessages: dict[str, list[dict]] = {}
        self.pendingSwipes: dict[str, list[tuple[str, str]]] = {}
        self.randomChats: dict[str, RandomChat] = {}
        self.recentMessages: dict[str, deque] = {}

        # Aliases used for mention detection
        localpart = self.bot_mxid.split(":")[0].lstrip("@")
        self._aliases: list[str] = [self.bot_name]
        if localpart.casefold() != self.bot_name.casefold():
            self._aliases.append(localpart)

    # ── lifecycle ────────────────────────────────────────────────────

    async def run(self) -> None:
        """Async entry-point: login, set up E2EE, sync, run inference loop."""
        os.makedirs(self.store_path, exist_ok=True)

        # Restore persisted session (access_token / device_id) if available
        self._load_session()

        # Stores
        self.state_store = BotStateStore()

        # Persistent crypto store (SQLite)
        db_path = os.path.join(self.store_path, "crypto.db")
        self.crypto_db = Database.create(
            url=f"sqlite:{db_path}",
            upgrade_table=PgCryptoStore.upgrade_table,
        )
        await self.crypto_db.start()
        self.crypto_store = PgCryptoStore(
            account_id=self.bot_mxid,
            pickle_key="realbot_pickle_key",
            db=self.crypto_db,
        )

        # When the crypto store has no existing Olm account (fresh DB), we
        # MUST create a new device so the homeserver broadcasts
        # device_lists.changed and other clients discover our identity keys.
        # Re-using a stale device_id (e.g. from MATRIX_DEVICE_ID env var)
        # skips that notification, so senders never learn about us.
        existing_account = await self.crypto_store.get_account()
        if existing_account is None and self.device_id:
            log.info(
                "Fresh crypto store: discarding stale device_id %s "
                "to force new device creation on login",
                self.device_id,
            )
            self.device_id = ""
            self.access_token = ""  # token is tied to old device

        # Client
        self.client = Client(
            mxid=self.bot_mxid,
            device_id=self.device_id,
            base_url=self.homeserver,
            token=self.access_token,
            state_store=self.state_store,
            sync_store=self.crypto_store,
        )

        # Login if we have no access token
        if not self.access_token:
            if not self.password:
                raise RuntimeError(
                    "Either MATRIX_ACCESS_TOKEN or MATRIX_PASSWORD must be set"
                )
            resp = await self.client.login(
                password=self.password,
                device_name="realbot",
            )
            self.access_token = resp.access_token
            self.device_id = resp.device_id
            self._save_session()
            log.info("Logged in as %s (device %s)", self.bot_mxid, self.device_id)

        # E2EE
        await self._setup_e2ee()

        # Event handlers
        self.client.add_event_handler(EventType.ROOM_MESSAGE, self._on_message)
        self.client.add_event_handler(EventType.REACTION, self._on_reaction)
        self.client.add_event_handler(InternalEventType.INVITE, self._on_invite)

        # NOTE: we do NOT set ignore_initial_sync because it drops ALL
        # events including to-device key shares needed for E2EE.  Instead we
        # gate room-event handlers behind _initial_sync_done.
        self._initial_sync_done = False

        # Wait for first successful sync before entering inference loop
        sync_ready = asyncio.Event()

        async def _on_first_sync(_data: Any) -> None:
            sync_ready.set()

        self.client.add_event_handler(InternalEventType.SYNC_SUCCESSFUL, _on_first_sync)

        self.client.start(filter_data=None)
        log.info("Waiting for initial sync...")
        await sync_ready.wait()
        self._initial_sync_done = True
        log.info("Sync established -- entering inference loop")

        # Remove the one-shot handler
        try:
            self.client.remove_event_handler(
                InternalEventType.SYNC_SUCCESSFUL, _on_first_sync
            )
        except (KeyError, ValueError):
            pass

        # Inference loop
        try:
            while True:
                await self._process_messages()
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass
        finally:
            self.client.stop()
            if hasattr(self, "crypto_db"):
                await self.crypto_db.stop()

    # ── session persistence ──────────────────────────────────────────

    _SESSION_FILE = "session.json"

    def _session_path(self) -> str:
        return os.path.join(self.store_path, self._SESSION_FILE)

    def _load_session(self) -> None:
        path = self._session_path()
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not self.access_token and data.get("access_token"):
                self.access_token = data["access_token"]
            if not self.device_id and data.get("device_id"):
                self.device_id = data["device_id"]
            log.info("Restored session from %s", path)
        except FileNotFoundError:
            pass
        except Exception as exc:
            log.warning("Failed to load session from %s: %s", path, exc)

    def _save_session(self) -> None:
        path = self._session_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "access_token": self.access_token,
                    "device_id": self.device_id,
                    "user_id": self.bot_mxid,
                },
                fh,
            )

    # ── E2EE setup ───────────────────────────────────────────────────

    async def _setup_e2ee(self) -> None:
        self.crypto_machine = OlmMachine(
            client=self.client,
            crypto_store=self.crypto_store,
            state_store=self.state_store,
        )
        # Setting client.crypto auto-registers the stock DecryptionDispatcher
        self.client.crypto = self.crypto_machine

        # Replace with our retry-capable dispatcher that waits for session
        # keys instead of silently dropping undecryptable events.
        self.client.remove_dispatcher(DecryptionDispatcher)
        self.client.add_dispatcher(RetryingDecryptionDispatcher)

        await self.crypto_machine.load()
        # Upload device identity keys and one-time keys to the homeserver
        # so other clients can discover us and share Megolm session keys.
        await self.crypto_machine.share_keys()
        log.info(
            "E2EE initialised (device %s, curve25519 %s)",
            self.device_id,
            self.crypto_machine.account.identity_keys.get("curve25519", "?"),
        )

        # Verify our keys are actually on the server
        try:
            key_resp = await self.client.query_keys({self.bot_mxid: []})
            server_devices = key_resp.get("device_keys", {}).get(self.bot_mxid, {})
            log.info(
                "Server reports %d device(s) for %s: %s",
                len(server_devices),
                self.bot_mxid,
                list(server_devices.keys()),
            )
            our_dev = server_devices.get(self.device_id, {})
            server_curve = our_dev.get("keys", {}).get(
                f"curve25519:{self.device_id}", "NOT FOUND"
            )
            local_curve = self.crypto_machine.account.identity_keys.get(
                "curve25519", "?"
            )
            if server_curve == local_curve:
                log.info("Key verification OK: server curve25519 matches local")
            else:
                log.error(
                    "KEY MISMATCH: server has curve25519=%s but local is %s",
                    server_curve,
                    local_curve,
                )
        except Exception as exc:
            log.warning("Failed to query device keys from server: %s", exc)

        # Cross-sign our device using the recovery key so other clients
        # see it as verified (not "unverified session").
        recovery_key = os.environ.get("MATRIX_RECOVERY_KEY", "").strip()
        if recovery_key:
            try:
                await self.crypto_machine.verify_with_recovery_key(recovery_key)
                log.info("Device cross-signed via recovery key -- verified!")
            except Exception as exc:
                log.warning("Cross-signing via recovery key failed: %s", exc)
        else:
            log.info(
                "No MATRIX_RECOVERY_KEY set -- device will appear unverified. "
                "Set the env var to your recovery key to enable cross-signing."
            )

    # ── event handlers ───────────────────────────────────────────────

    async def _on_invite(self, evt: Any) -> None:
        """Auto-join rooms the bot is invited to."""
        room_id = getattr(evt, "room_id", None)
        if not room_id:
            return
        try:
            await self.client.join_room(room_id)
            log.info("Joined room %s after invite", room_id)
        except Exception as exc:
            log.warning("Failed to join %s: %s", room_id, exc)

    async def _on_message(self, evt: Any) -> None:
        """Handle incoming ``m.room.message`` events (including decrypted)."""
        # Ignore events from initial sync (to-device events are still processed
        # by OlmMachine handlers for E2EE key exchange).
        if not self._initial_sync_done:
            return
        # Ignore own messages
        if str(evt.sender) == self.bot_mxid:
            return

        content = evt.content
        msgtype = getattr(content, "msgtype", None)
        body = getattr(content, "body", None)

        room_id = str(evt.room_id)

        # Buffer every event for random-chat history replay
        if room_id not in self.recentMessages:
            self.recentMessages[room_id] = deque(maxlen=200)
        self.recentMessages[room_id].append(evt)

        # Only respond to text-bearing messages and images
        if msgtype == MessageType.IMAGE:
            pass  # image-only messages are valid -- handled below
        elif msgtype not in (MessageType.TEXT, MessageType.EMOTE):
            return
        elif not body or not body.strip():
            return

        # Serialise content for mention detection
        raw_content: dict = {}
        if hasattr(content, "serialize"):
            try:
                raw_content = content.serialize()
            except Exception:
                raw_content = {}

        is_mentioned = matrix_content_mentions_user(
            raw_content, self.bot_mxid, self._aliases
        )

        # Whitelist routing
        if _is_whitelisted(room_id, "always"):
            return await self._constant_chat(room_id, evt)

        if is_mentioned and _is_whitelisted(room_id, "mentions"):
            return await self._constant_chat(room_id, evt)

        if _is_whitelisted(room_id, "rand"):
            return await self._random_chat(room_id, evt, is_mentioned)

        # If mentioned in a mentions channel that didn't match above,
        # or not whitelisted at all -- ignore.

    async def _on_reaction(self, evt: Any) -> None:
        """Handle ``m.reaction`` events for swipe controls."""
        if not self._initial_sync_done:
            return
        if str(evt.sender) == self.bot_mxid:
            return

        config = get_config()
        swipes_cfg = config.get("swipes", {}) or {}
        if not swipes_cfg.get("enabled", False):
            return

        room_id = str(evt.room_id)
        if not self._swipe_allowed(swipes_cfg, str(evt.sender), room_id):
            return

        relates_to = getattr(evt.content, "relates_to", None)
        if not relates_to:
            return

        emoji = getattr(relates_to, "key", None)
        target_event_id = str(getattr(relates_to, "event_id", ""))
        if not emoji or not target_event_id:
            return

        action = swipe_action_for_emoji(emoji, swipes_cfg)
        if not action:
            return

        log.info("[%s] Swipe %s on %s", room_id, action, target_event_id)
        queue = self.pendingSwipes.setdefault(room_id, [])
        queue.append((target_event_id, action))

    # ── chat behaviours ──────────────────────────────────────────────

    async def _constant_chat(self, room_id: str, evt: Any) -> None:
        msg = await self._process_message(evt)
        self._add_to_queue(room_id, msg)

    async def _random_chat(
        self, room_id: str, evt: Any, is_mentioned: bool = False
    ) -> None:
        config = get_config().get("randomChat", {})
        current = self.randomChats.get(room_id)

        # Active session -- keep chatting
        if current and datetime.datetime.now() < current.endTime:
            current.lastEventId = str(evt.event_id)
            msg = await self._process_message(evt)
            self._add_to_queue(room_id, msg)
            return

        # Cooldown period
        if current and datetime.datetime.now() < current.nextChatTime:
            if is_mentioned and config.get("respond_to_mentions", True):
                msg = await self._process_message(evt)
                self._add_to_queue(room_id, msg)
            return

        # Roll for engagement
        chance = config.get("engagement_chance", 10)
        if random.randint(0, chance) != 0:
            if is_mentioned and config.get("respond_to_mentions", True):
                msg = await self._process_message(evt)
                self._add_to_queue(room_id, msg)
            return

        # Start new random chat session
        msgs: list[dict] = []
        recent = list(self.recentMessages.get(room_id, []))
        history_limit = config.get("message_history_limit", 10)

        for past_evt in recent[-history_limit:]:
            if current and str(past_evt.event_id) == current.lastEventId:
                # Not enough new messages since last session
                return
            msgs.append(await self._process_message(past_evt))

        if not msgs:
            msgs.append(await self._process_message(evt))

        # Guard against concurrent session creation
        if current is not self.randomChats.get(room_id):
            return

        end_time = datetime.datetime.now() + datetime.timedelta(
            seconds=random.randint(
                config.get("min_chat_duration_seconds", 40),
                config.get("max_chat_duration_seconds", 500),
            )
        )
        self.randomChats[room_id] = RandomChat(
            endTime=end_time,
            nextChatTime=end_time
            + datetime.timedelta(
                minutes=random.randint(
                    config.get("min_downtime_minutes", 5),
                    config.get("max_downtime_minutes", 20),
                )
            ),
            lastEventId=str(evt.event_id),
        )

        for msg in msgs:
            self._add_to_queue(room_id, msg)

        log.info(
            "[%s] Random chat started until %s",
            room_id,
            end_time.isoformat(timespec="seconds"),
        )

    # ── message processing ───────────────────────────────────────────

    async def _process_message(self, evt: Any) -> dict:
        room_id = str(evt.room_id)
        username = await self._get_display_name(room_id, str(evt.sender))

        msgtype = getattr(evt.content, "msgtype", None)
        body = getattr(evt.content, "body", None) or ""

        cleaned = self._clean_content(body)

        processed: dict[str, Any] = {
            "user": username,
            "message": cleaned,
            "messageId": str(evt.event_id),
        }

        images = await self._extract_images(evt)
        if images:
            processed["images"] = images
            # LLMs like Claude error on image-only messages with no text,
            # so ensure there is always *some* text content.
            if not processed["message"]:
                processed["message"] = "image.png"

        return processed

    async def _get_display_name(self, room_id: str, user_id: str) -> str:
        if user_id == self.bot_mxid:
            return "{{char}}"

        member = await self.state_store.get_member(RoomID(room_id), UserID(user_id))
        if member and member.displayname:
            return member.displayname

        # Fallback: extract localpart from MXID
        return user_id.split(":")[0].lstrip("@")

    def _clean_content(self, text: str) -> str:
        """Replace @bot-alias mentions with ``{{char}}``."""
        for alias in self._aliases:
            text = re.sub(
                rf"(?<!\w)@?{re.escape(alias)}(?!\w)",
                "{{char}}",
                text,
                flags=re.IGNORECASE,
            )
        return text

    async def _extract_images(self, evt: Any) -> list[dict]:
        images: list[dict] = []
        content = evt.content
        room_id = str(evt.room_id)

        config = get_config()
        matrix_cfg = config.get("matrix", {})
        max_preview_links = int(matrix_cfg.get("max_preview_links", 3))
        max_images = int(matrix_cfg.get("max_images_per_message", 5))

        # ── 1. m.image attachment ────────────────────────────────────────
        # In E2EE rooms, content.url is None; the media is at content.file
        # (an EncryptedFile with url, key, iv, hashes) and must be decrypted.
        msgtype = getattr(content, "msgtype", None)
        if msgtype == MessageType.IMAGE:
            url = getattr(content, "url", None)
            enc_file = getattr(content, "file", None)
            filename = getattr(content, "body", "image.png") or "image.png"
            media_url = url or (enc_file.url if enc_file else None)
            if media_url:
                try:
                    data = await self.client.download_media(media_url)
                    # Decrypt if this is an encrypted attachment
                    if enc_file and not url:
                        data = decrypt_attachment(
                            data,
                            enc_file.key.key,
                            enc_file.hashes["sha256"],
                            enc_file.iv,
                        )
                    local = await download_image_to_history(
                        room_id, str(media_url), filename, data=data
                    )
                    images.append(
                        {
                            "source": "matrix_image",
                            "url": str(media_url),
                            "filename": filename,
                            "local_path": local,
                        }
                    )
                except Exception as exc:
                    log.warning(
                        "[%s] Failed to download m.image %s: %s",
                        room_id,
                        media_url,
                        exc,
                    )

        if len(images) >= max_images:
            return images[:max_images]

        # ── 2. URLs in text body ─────────────────────────────────────────
        body = getattr(content, "body", None) or ""
        image_url_re = re.compile(
            r"https?://\S+\.(?:png|jpg|jpeg|gif|webp)(?:\?\S*)?",
            re.IGNORECASE,
        )
        general_url_re = re.compile(r"https?://[^\s<>\"')\]]+", re.IGNORECASE)

        # Collect all URLs, split into direct-image vs preview candidates
        all_urls: list[str] = []
        seen: set[str] = set()
        for match in general_url_re.finditer(body):
            url_str = match.group(0).rstrip(".,;:!?")
            if url_str not in seen:
                seen.add(url_str)
                all_urls.append(url_str)

        direct_image_urls: list[str] = []
        preview_candidates: list[str] = []
        for url_str in all_urls:
            if image_url_re.fullmatch(url_str):
                direct_image_urls.append(url_str)
            else:
                preview_candidates.append(url_str)

        # ── 2a. Direct image URLs -- download locally ────────────────────
        for img_url in direct_image_urls:
            if len(images) >= max_images:
                break
            try:
                local = await download_image_to_history(room_id, img_url)
                images.append(
                    {
                        "source": "matrix_url",
                        "url": img_url,
                        "local_path": local,
                    }
                )
            except Exception as exc:
                log.warning(
                    "[%s] Failed to download URL image %s: %s",
                    room_id,
                    img_url,
                    exc,
                )
                # Fall back to remote-only record so inference can try the raw URL
                images.append(build_remote_image_record(img_url, source="matrix_url"))

        if len(images) >= max_images:
            return images[:max_images]

        # ── 2b. Non-image URLs -- fetch link previews via homeserver ─────
        # The homeserver fetches og:image (like Discord's embed previews)
        # and caches it as an mxc:// URL we can download.
        # Limited to max_preview_links per message to control bandwidth.
        previews_fetched = 0
        for page_url in preview_candidates:
            if len(images) >= max_images:
                break
            if previews_fetched >= max_preview_links:
                break
            previews_fetched += 1
            try:
                preview = await self.client.get_url_preview(page_url)
                og_image = getattr(preview, "image", None)
                if og_image and getattr(og_image, "url", None):
                    mxc_url = og_image.url
                    data = await self.client.download_media(mxc_url)
                    filename = "preview.jpg"
                    local = await download_image_to_history(
                        room_id, str(mxc_url), filename, data=data
                    )
                    images.append(
                        {
                            "source": "matrix_preview",
                            "url": page_url,
                            "preview_title": getattr(preview, "title", "") or "",
                            "local_path": local,
                        }
                    )
            except Exception as exc:
                log.warning(
                    "[%s] Link preview failed for %s: %s",
                    room_id,
                    page_url,
                    exc,
                )

        return images[:max_images]

    # ── queue management ─────────────────────────────────────────────

    def _add_to_queue(self, room_id: str, msg: dict) -> None:
        extra = ""
        if msg.get("images"):
            extra = f" [{len(msg['images'])} image(s)]"
        log.info("[%s] %s: %s%s", room_id, msg["user"], msg["message"], extra)
        self.pendingMessages.setdefault(room_id, []).append(msg)

    # ── inference loop ───────────────────────────────────────────────

    async def _process_messages(self) -> None:
        room_ids = list(
            set(self.pendingMessages.keys()) | set(self.pendingSwipes.keys())
        )
        tasks: list = []

        for room_id in room_ids:
            pending = self.pendingMessages.get(room_id, [])
            swipe_jobs = self.pendingSwipes.get(room_id, [])
            if not pending and not swipe_jobs:
                continue
            self.pendingMessages[room_id] = []
            self.pendingSwipes[room_id] = []
            tasks.append(self._process_room(room_id, pending, swipe_jobs))

        if tasks:
            await asyncio.gather(*tasks)

    async def _process_room(
        self,
        room_id: str,
        messages: list[dict],
        swipe_jobs: list[tuple[str, str]],
    ) -> None:
        # Typing indicator
        try:
            await self.client.set_typing(RoomID(room_id), timeout=30000)
        except Exception:
            pass

        try:
            if swipe_jobs:
                await self._handle_swipes(room_id, swipe_jobs)
            if messages:
                await self._handle_inference(room_id, messages)
        finally:
            try:
                await self.client.set_typing(RoomID(room_id), timeout=0)
            except Exception:
                pass

    # ── swipe handling ───────────────────────────────────────────────

    async def _handle_swipes(
        self, room_id: str, swipe_jobs: list[tuple[str, str]]
    ) -> None:
        swipes_cfg = get_config().get("swipes", {}) or {}
        prev_emoji = str(swipes_cfg.get("prev_emoji", "\u25c0\ufe0f"))
        next_emoji = str(swipes_cfg.get("next_emoji", "\u25b6\ufe0f"))

        jobs = dedupe_swipe_jobs(swipe_jobs)

        for msg_id, action in jobs:
            new_text: str | None = None
            try:
                if action == "regen":
                    new_text = await swipe_regenerate(room_id, msg_id)
                elif action == "prev":
                    new_text = swipe_prev(room_id, msg_id)
                elif action == "next":
                    new_text = swipe_next(room_id, msg_id)
            except Exception as exc:
                log.warning(
                    "[%s] Swipe %s failed for %s: %s", room_id, action, msg_id, exc
                )
                continue

            if new_text:
                cleaned = _clean_response(new_text)
                plain, formatted, _mention_ids = await self._resolve_mentions(
                    cleaned, room_id
                )
                try:
                    await self.send_edit(room_id, msg_id, plain, formatted)
                except Exception as exc:
                    log.warning("[%s] Swipe edit failed: %s", room_id, exc)

            # Update navigation arrows
            try:
                nav = get_swipe_nav_state(room_id, msg_id)
                if nav:
                    has_prev, has_next = nav
                    await self._set_control_reaction(
                        room_id, msg_id, prev_emoji, has_prev
                    )
                    await self._set_control_reaction(
                        room_id, msg_id, next_emoji, has_next
                    )
            except Exception as exc:
                log.warning("[%s] Nav reaction update failed: %s", room_id, exc)

    # ── inference handling ───────────────────────────────────────────

    async def _handle_inference(self, room_id: str, messages: list[dict]) -> None:
        try:
            result = await chat_inference(room_id, messages)
        except Exception as exc:
            log.error("[%s] Inference failed: %s", room_id, exc)
            return

        response: str | None = None
        pending_id: str | None = None

        if isinstance(result, tuple) and len(result) == 2:
            response, pending_id = result
        elif isinstance(result, str):
            response = result

        if not response:
            return

        cleaned = _clean_response(response)
        plain, formatted, mention_ids = await self._resolve_mentions(cleaned, room_id)

        event_id = await self.send_message(room_id, plain, formatted, mention_ids)
        if not event_id:
            return

        # Finalise the assistant message ID in history
        try:
            if pending_id:
                finalize_assistant_message_id(room_id, pending_id, str(event_id))
            else:
                finalize_last_assistant_message_id(room_id, str(event_id))
        except Exception as exc:
            log.warning("[%s] Failed to finalise message id: %s", room_id, exc)

        # Control reactions: cleanup old, add auto-react
        swipes_cfg = get_config().get("swipes", {}) or {}
        try:
            await self._cleanup_recorded_control_reactions(room_id)
        except Exception as exc:
            log.warning("[%s] Control reaction cleanup failed: %s", room_id, exc)

        if swipes_cfg.get("enabled", False) and swipes_cfg.get(
            "auto_react_controls", False
        ):
            auto_wl = normalize_id_set(swipes_cfg.get("auto_react_channel_whitelist"))
            if not auto_wl or room_id in auto_wl:
                regen_emoji = str(swipes_cfg.get("regen_emoji", "\U0001f504"))
                try:
                    react_eid = await self.send_reaction(
                        room_id, str(event_id), regen_emoji
                    )
                    self._record_control_reaction(
                        room_id,
                        str(event_id),
                        regen_emoji,
                        str(react_eid),
                    )
                except Exception as exc:
                    log.warning("[%s] Auto-react failed: %s", room_id, exc)

    # ── send primitives ──────────────────────────────────────────────

    async def send_message(
        self,
        room_id: str,
        body: str,
        formatted_body: str | None = None,
        mention_ids: list[str] | None = None,
    ) -> EventID | None:
        """Send a text message, optionally with HTML formatting and mentions."""
        content: dict[str, Any] = {
            "msgtype": "m.text",
            "body": body,
        }
        if formatted_body:
            content["format"] = "org.matrix.custom.html"
            content["formatted_body"] = formatted_body
        content["m.mentions"] = {"user_ids": mention_ids} if mention_ids else {}
        try:
            return await self.client.send_message_event(
                RoomID(room_id), EventType.ROOM_MESSAGE, content
            )
        except Exception as exc:
            log.error("[%s] send_message failed: %s", room_id, exc)
            return None

    async def send_edit(
        self,
        room_id: str,
        target_event_id: str,
        body: str,
        formatted_body: str | None = None,
    ) -> EventID | None:
        """Edit an existing message via ``m.replace``."""
        new_content: dict[str, Any] = {
            "body": body,
            "msgtype": "m.text",
        }
        if formatted_body:
            new_content["format"] = "org.matrix.custom.html"
            new_content["formatted_body"] = formatted_body

        content: dict[str, Any] = {
            "body": body,
            "msgtype": "m.text",
            "m.new_content": new_content,
            "m.relates_to": {
                "rel_type": "m.replace",
                "event_id": target_event_id,
            },
        }
        try:
            return await self.client.send_message_event(
                RoomID(room_id), EventType.ROOM_MESSAGE, content
            )
        except Exception as exc:
            log.error("[%s] send_edit failed: %s", room_id, exc)
            return None

    async def send_reaction(
        self, room_id: str, target_event_id: str, key: str
    ) -> EventID:
        """Send a reaction (``m.annotation``)."""
        return await self.client.react(RoomID(room_id), EventID(target_event_id), key)

    # ── typing indicator ─────────────────────────────────────────────

    async def _set_typing(self, room_id: str, typing: bool) -> None:
        try:
            timeout = 30000 if typing else 0
            await self.client.set_typing(RoomID(room_id), timeout=timeout)
        except Exception:
            pass

    # ── @-mention resolution ─────────────────────────────────────────

    async def _resolve_mentions(
        self, text: str, room_id: str
    ) -> tuple[str, str | None, list[str]]:
        """Resolve ``@Name`` patterns to Matrix pill links."""
        await self._ensure_room_members(room_id)
        members = await self.state_store.get_member_profiles(
            RoomID(room_id), memberships=(Membership.JOIN,)
        )

        def resolver(name: str) -> str | None:
            lowered = name.strip().casefold()
            if not lowered or lowered in {"everyone", "here"}:
                return None
            for uid, member in members.items():
                if member.displayname and member.displayname.casefold() == lowered:
                    return str(uid)
                lp = str(uid).split(":")[0].lstrip("@")
                if lp.casefold() == lowered:
                    return str(uid)
            return None

        return apply_generated_at_mentions(text, resolver)

    async def _ensure_room_members(self, room_id: str) -> None:
        """Fetch and cache the full member list for *room_id* if missing."""
        rid = RoomID(room_id)
        if await self.state_store.has_full_member_list(rid):
            return
        try:
            members = await self.client.get_joined_members(rid)
            await self.state_store.set_members(
                rid, members, only_membership=Membership.JOIN
            )
        except Exception as exc:
            log.warning("Failed to fetch members for %s: %s", room_id, exc)

    # ── swipe permission ─────────────────────────────────────────────

    @staticmethod
    def _swipe_allowed(swipes_cfg: dict, user_id: str, room_id: str) -> bool:
        user_wl = normalize_id_set(swipes_cfg.get("user_whitelist"))
        chan_wl = normalize_id_set(swipes_cfg.get("channel_whitelist"))
        if not user_wl and not chan_wl:
            return True
        if not user_wl:
            return room_id in chan_wl
        if not chan_wl:
            return user_id in user_wl
        return user_id in user_wl or room_id in chan_wl

    # ── control reaction bookkeeping ─────────────────────────────────

    def _control_reactions_path(self, room_id: str) -> str:
        return os.path.join("history", "control-reactions", f"{room_id}.json")

    def _load_control_reactions(self, room_id: str) -> list[dict]:
        path = self._control_reactions_path(room_id)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                return []
            raw = data.get("reactions", [])
            if not isinstance(raw, list):
                return []
            out: list[dict] = []
            for item in raw:
                if not isinstance(item, dict):
                    continue
                msg_id = item.get("messageId", "")
                emoji = item.get("emoji", "")
                react_id = item.get("reactionEventId", "")
                if msg_id and emoji:
                    out.append(
                        {
                            "messageId": msg_id,
                            "emoji": emoji,
                            "reactionEventId": react_id,
                        }
                    )
            return out
        except Exception:
            return []

    def _save_control_reactions(self, room_id: str, reactions: list[dict]) -> None:
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
        for rec in existing:
            if rec["messageId"] == message_id and rec["emoji"] == emoji:
                rec["reactionEventId"] = reaction_event_id
                self._save_control_reactions(room_id, existing)
                return
        existing.append(
            {
                "messageId": message_id,
                "emoji": emoji,
                "reactionEventId": reaction_event_id,
            }
        )
        self._save_control_reactions(room_id, existing)

    async def _remove_control_reaction(
        self, room_id: str, reaction_event_id: str
    ) -> bool:
        try:
            await self.client.redact(RoomID(room_id), EventID(reaction_event_id))
            return True
        except Exception:
            return False

    async def _set_control_reaction(
        self,
        room_id: str,
        message_id: str,
        emoji: str,
        should_have: bool,
    ) -> None:
        if should_have:
            try:
                react_eid = await self.send_reaction(room_id, message_id, emoji)
                self._record_control_reaction(
                    room_id, message_id, emoji, str(react_eid)
                )
            except Exception:
                pass
            return

        # Remove
        existing = self._load_control_reactions(room_id)
        to_remove = [
            r for r in existing if r["messageId"] == message_id and r["emoji"] == emoji
        ]
        for rec in to_remove:
            rid = rec.get("reactionEventId", "")
            if rid:
                await self._remove_control_reaction(room_id, rid)
        remaining = [
            r
            for r in existing
            if not (r["messageId"] == message_id and r["emoji"] == emoji)
        ]
        self._save_control_reactions(room_id, remaining)

    async def _cleanup_recorded_control_reactions(self, room_id: str) -> None:
        records = self._load_control_reactions(room_id)
        if not records:
            return
        remaining: list[dict] = []
        for rec in records:
            rid = rec.get("reactionEventId", "")
            if rid:
                ok = await self._remove_control_reaction(room_id, rid)
                if not ok:
                    remaining.append(rec)
            else:
                remaining.append(rec)
        self._save_control_reactions(room_id, remaining)


# ── Entry point ──────────────────────────────────────────────────────


def run_bot() -> None:
    """Start the Matrix bot (blocking)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    bot = MatrixBot()
    asyncio.run(bot.run())
