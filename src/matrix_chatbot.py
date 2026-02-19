import asyncio
import contextlib
import datetime
import html
import json
import os
import random
import re
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Sequence

from dotenv import load_dotenv

try:
    from nio import (
        Api,
        AsyncClient,
        AsyncClientConfig,
        KeysQueryResponse,
        LocalProtocolError,
        MatrixRoom,
        MegolmEvent,
        ReactionEvent,
        RoomMessage,
        UnknownEncryptedEvent,
    )

    NIO_AVAILABLE = True
except Exception:
    Api = Any  # type: ignore
    AsyncClient = None  # type: ignore
    AsyncClientConfig = None  # type: ignore
    KeysQueryResponse = Any  # type: ignore
    LocalProtocolError = Exception  # type: ignore
    MatrixRoom = Any  # type: ignore
    MegolmEvent = Any  # type: ignore
    ReactionEvent = Any  # type: ignore
    RoomMessage = Any  # type: ignore
    UnknownEncryptedEvent = Any  # type: ignore
    NIO_AVAILABLE = False

from .chat_image_utils import (
    build_remote_image_record,
    download_image_to_history,
    ensure_channel_image_dir,
    sanitize_filename,
)
from .openai_inference import (
    chat_inference,
    finalize_assistant_message_id,
    finalize_last_assistant_message_id,
    get_swipe_nav_state,
    swipe_next,
    swipe_prev,
    swipe_regenerate,
)
from .utils import dequote, get_config, is_whitelisted_id, normalize_id, normalize_id_set

IMAGE_URL_RE = re.compile(r"(https?://\S+)", re.IGNORECASE)
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")


@dataclass
class RandomChat:
    endTime: datetime.datetime
    nextChatTime: datetime.datetime
    lastMessageId: str


def _is_mention_boundary_char(ch: str) -> bool:
    return ch.isspace() or ch in "\"'`,!?;:()[]{}<>"


def _extract_localpart(user_id: str) -> str:
    if not isinstance(user_id, str):
        return ""
    if not user_id.startswith("@") or ":" not in user_id:
        return ""
    return user_id[1:].split(":", 1)[0]


def _unique_preserve(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def matrix_content_mentions_user(
    content: dict[str, Any], user_id: str, aliases: Iterable[str]
) -> bool:
    """
    Checks explicit Matrix mention metadata first, then falls back to textual checks.
    """
    mentions_obj = content.get("m.mentions")
    if isinstance(mentions_obj, dict):
        user_ids = mentions_obj.get("user_ids")
        if isinstance(user_ids, list):
            for uid in user_ids:
                if isinstance(uid, str) and uid == user_id:
                    return True

    search_space = "\n".join(
        str(v)
        for v in (content.get("body"), content.get("formatted_body"))
        if isinstance(v, str)
    )
    if not search_space:
        return False

    if user_id in search_space:
        return True

    for alias in aliases:
        alias = alias.strip()
        if not alias:
            continue
        alias_pattern = re.compile(rf"(?<!\w)@{re.escape(alias)}(?!\w)", re.IGNORECASE)
        if alias_pattern.search(search_space):
            return True
    return False


def swipe_action_for_emoji(emoji: str, swipes_cfg: dict[str, Any]) -> str | None:
    regen_emoji = str(swipes_cfg.get("regen_emoji", "🔄"))
    prev_emoji = str(swipes_cfg.get("prev_emoji", "◀️"))
    next_emoji = str(swipes_cfg.get("next_emoji", "▶️"))
    if emoji == regen_emoji:
        return "regen"
    if emoji == prev_emoji:
        return "prev"
    if emoji == next_emoji:
        return "next"
    return None


def dedupe_swipe_jobs(swipe_jobs: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
    """
    Keeps only the latest action for each target message while preserving
    chronological order between distinct targets.
    """
    last_by_message: dict[str, tuple[int, str]] = {}
    for i, (message_id, action) in enumerate(swipe_jobs):
        last_by_message[message_id] = (i, action)

    return [
        (message_id, action)
        for message_id, (i, action) in sorted(last_by_message.items(), key=lambda kv: kv[1][0])
    ]


def build_reaction_content(target_event_id: str, emoji: str) -> dict[str, Any]:
    return {
        "m.relates_to": {
            "event_id": target_event_id,
            "rel_type": "m.annotation",
            "key": emoji,
        }
    }


def build_text_content(
    body: str,
    formatted_body: str | None = None,
    mention_user_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    content: dict[str, Any] = {"msgtype": "m.text", "body": body}
    if formatted_body:
        content["format"] = "org.matrix.custom.html"
        content["formatted_body"] = formatted_body
    if mention_user_ids:
        content["m.mentions"] = {"user_ids": _unique_preserve(str(uid) for uid in mention_user_ids)}
    return content


def build_edit_content(
    body: str,
    target_event_id: str,
    formatted_body: str | None = None,
    mention_user_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    content = build_text_content(
        body=f"* {body}",
        formatted_body=(f"* {formatted_body}" if formatted_body else None),
        mention_user_ids=mention_user_ids,
    )
    content["m.new_content"] = build_text_content(
        body=body,
        formatted_body=formatted_body,
        mention_user_ids=mention_user_ids,
    )
    content["m.relates_to"] = {"rel_type": "m.replace", "event_id": target_event_id}
    return content


def apply_generated_at_mentions(
    content: str, resolver: Callable[[str], str | None]
) -> tuple[str, str | None, list[str]]:
    """
    Converts generated @mentions into Matrix mention pills when resolvable.
    Returns (plain_body, formatted_body_or_none, mentioned_user_ids).
    """
    if "@" not in content:
        return content, None, []

    out_plain: list[str] = []
    out_formatted: list[str] = []
    mention_user_ids: list[str] = []
    mention_seen: set[str] = set()
    replaced_any = False

    i = 0
    while i < len(content):
        ch = content[i]
        if ch != "@":
            out_plain.append(ch)
            out_formatted.append(html.escape(ch))
            i += 1
            continue

        if i > 0 and not _is_mention_boundary_char(content[i - 1]):
            out_plain.append(ch)
            out_formatted.append(html.escape(ch))
            i += 1
            continue

        if i + 1 >= len(content) or content[i + 1].isspace():
            out_plain.append(ch)
            out_formatted.append(html.escape(ch))
            i += 1
            continue

        tail = content[i + 1 :]
        stop_match = re.match(r"^[^\r\n]+", tail)
        raw_segment = (stop_match.group(0) if stop_match else tail)[:64].strip()
        if not raw_segment:
            out_plain.append(ch)
            out_formatted.append(html.escape(ch))
            i += 1
            continue

        words = raw_segment.split()
        if not words:
            out_plain.append(ch)
            out_formatted.append(html.escape(ch))
            i += 1
            continue

        word_counts_to_try = [1]
        if len(words) >= 2:
            word_counts_to_try.append(2)

        replaced = False
        for k in word_counts_to_try:
            candidate_base = " ".join(words[:k])
            if not candidate_base:
                continue

            if not tail[: len(candidate_base)].casefold() == candidate_base.casefold():
                continue

            candidates_to_try = [candidate_base]
            candidate_stripped = candidate_base.rstrip("\"'`.,!?;:()[]{}<>")
            if candidate_stripped and candidate_stripped.casefold() != candidate_base.casefold():
                candidates_to_try.append(candidate_stripped)

            for candidate in candidates_to_try:
                after_idx = i + 1 + len(candidate)
                if after_idx < len(content) and not _is_mention_boundary_char(content[after_idx]):
                    continue

                resolved_user_id = resolver(candidate)
                if not resolved_user_id:
                    continue

                mention_text = f"@{candidate}"
                out_plain.append(mention_text)
                out_formatted.append(
                    f'<a href="https://matrix.to/#/{html.escape(resolved_user_id, quote=True)}">'
                    f"{html.escape(mention_text)}</a>"
                )
                if resolved_user_id not in mention_seen:
                    mention_seen.add(resolved_user_id)
                    mention_user_ids.append(resolved_user_id)
                i = after_idx
                replaced = True
                replaced_any = True
                break

            if replaced:
                break

        if not replaced:
            out_plain.append(ch)
            out_formatted.append(html.escape(ch))
            i += 1

    formatted = "".join(out_formatted) if replaced_any else None
    return "".join(out_plain), formatted, mention_user_ids


class MatrixBot:
    pendingMessages: dict[str, List[dict[str, Any]]]
    pendingSwipes: dict[str, list[tuple[str, str]]]
    randomChats: dict[str, RandomChat]
    recentMessages: dict[str, deque[dict[str, Any]]]
    roomMentionIndexCache: dict[str, tuple[float, dict[str, str], dict[str, str]]]
    roomsWithMissingOlmSessions: set[str]

    def __init__(self):
        load_dotenv()
        if not NIO_AVAILABLE:
            raise RuntimeError(
                "matrix-nio is not installed. Install dependencies from requirements.txt first."
            )

        self.homeserver = self._required_env("MATRIX_HOMESERVER")
        self.user_id = self._required_env("MATRIX_USER_ID")
        self.access_token = self._required_env("MATRIX_ACCESS_TOKEN")
        self.device_id = self._required_env("MATRIX_DEVICE_ID")
        self.store_path = os.getenv("MATRIX_STORE_PATH", "history/matrix_store")
        self.sync_timeout_ms = self._read_positive_int("MATRIX_SYNC_TIMEOUT_MS", 30000)

        os.makedirs(self.store_path, exist_ok=True)

        config = AsyncClientConfig(encryption_enabled=True, store_sync_tokens=True)
        self.client = AsyncClient(
            homeserver=self.homeserver,
            user=self.user_id,
            device_id=self.device_id,
            store_path=self.store_path,
            config=config,
        )
        # Ensure nio restores login + loads crypto store for this device.
        self.client.restore_login(self.user_id, self.device_id, self.access_token)

        self.import_keys_path = os.getenv("MATRIX_IMPORT_KEYS_PATH", "").strip()
        self.import_keys_passphrase = os.getenv("MATRIX_IMPORT_KEYS_PASSWORD", "")

        self.pendingMessages = {}
        self.pendingSwipes = {}
        self.randomChats = {}
        self.recentMessages = {}
        self.roomMentionIndexCache = {}
        self.roomCache: dict[str, MatrixRoom] = {}
        self.requestedMissingMegolmSessions: set[tuple[str, str, str]] = set()
        self.roomsWithMissingOlmSessions = set()
        self.lastE2EERefreshAt: float = 0.0
        self.bg_task: asyncio.Task | None = None

        self.client.add_event_callback(self.on_room_message, RoomMessage)
        self.client.add_event_callback(self.on_reaction, ReactionEvent)
        self.client.add_event_callback(self.on_undecrypted_event, MegolmEvent)
        self.client.add_event_callback(self.on_undecrypted_event, UnknownEncryptedEvent)

    def _required_env(self, key: str) -> str:
        value = os.getenv(key)
        if not value:
            raise RuntimeError(f"Missing required environment variable: {key}")
        return value

    def _read_positive_int(self, key: str, default: int) -> int:
        raw = os.getenv(key)
        if not raw:
            return default
        try:
            parsed = int(raw)
        except Exception:
            return default
        return parsed if parsed > 0 else default

    async def run(self):
        await self._verify_whoami()
        await self._import_keys_if_configured()
        print(f"[Matrix] Starting sync as {self.user_id} on {self.homeserver}")
        await self.client.sync(timeout=self.sync_timeout_ms, full_state=True)
        print("[Matrix] Initial sync complete")
        await self._refresh_e2ee_state(force=True, reason="startup")
        await self._diagnose_own_device_identity()
        self.bg_task = asyncio.create_task(self.inference_loop_task())
        try:
            await self.client.sync_forever(timeout=self.sync_timeout_ms, full_state=False)
        finally:
            if self.bg_task:
                self.bg_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self.bg_task
            await self.client.close()

    async def _verify_whoami(self) -> None:
        try:
            response = await self.client.whoami()
        except Exception as e:
            print(f"[Matrix] Failed to call whoami: {e}")
            return

        if response.__class__.__name__.endswith("Error"):
            print(f"[Matrix] whoami returned error: {response}")
            return

        actual_user_id = str(getattr(response, "user_id", "") or "")
        actual_device_id = str(getattr(response, "device_id", "") or "")
        if actual_user_id and actual_user_id != self.user_id:
            print(
                f"[Matrix] Warning: MATRIX_USER_ID ({self.user_id}) does not match token user "
                f"({actual_user_id})."
            )
            self.user_id = actual_user_id
        if actual_device_id and actual_device_id != self.device_id:
            print(
                f"[Matrix] Warning: MATRIX_DEVICE_ID ({self.device_id}) does not match token device "
                f"({actual_device_id})."
            )
            self.device_id = actual_device_id

        if (actual_user_id and actual_user_id != self.client.user_id) or (
            actual_device_id and actual_device_id != self.client.device_id
        ):
            self.client.restore_login(self.user_id, self.device_id, self.access_token)
            print(
                f"[Matrix] Using token-bound identity user={self.user_id} device={self.device_id}"
            )

    async def _import_keys_if_configured(self) -> None:
        if not self.import_keys_path:
            return
        if not os.path.exists(self.import_keys_path):
            print(
                f"[Matrix] MATRIX_IMPORT_KEYS_PATH does not exist: {self.import_keys_path}"
            )
            return
        if not self.import_keys_passphrase:
            print(
                "[Matrix] MATRIX_IMPORT_KEYS_PATH is set but MATRIX_IMPORT_KEYS_PASSWORD is empty."
            )
            return
        try:
            await self.client.import_keys(
                self.import_keys_path, self.import_keys_passphrase
            )
            print(f"[Matrix] Imported room keys from {self.import_keys_path}")
        except Exception as e:
            print(f"[Matrix] Failed to import room keys: {e}")

    async def _refresh_e2ee_state(self, force: bool = False, reason: str = "") -> None:
        now = time.time()
        if not force and (now - self.lastE2EERefreshAt) < 10:
            return
        self.lastE2EERefreshAt = now

        suffix = f" ({reason})" if reason else ""

        try:
            if getattr(self.client, "should_upload_keys", False):
                response = await self.client.keys_upload()
                if response.__class__.__name__.endswith("Error"):
                    print(f"[Matrix] keys_upload failed{suffix}: {response}")
                else:
                    print(f"[Matrix] keys_upload complete{suffix}")
        except Exception as e:
            print(f"[Matrix] keys_upload exception{suffix}: {e}")

        try:
            if getattr(self.client, "should_query_keys", False):
                response = await self.client.keys_query()
                if response.__class__.__name__.endswith("Error"):
                    print(f"[Matrix] keys_query failed{suffix}: {response}")
                else:
                    print(f"[Matrix] keys_query complete{suffix}")
        except Exception as e:
            print(f"[Matrix] keys_query exception{suffix}: {e}")

        claim_targets: dict[str, list[str]] = {}
        try:
            claim_targets = self.client.get_users_for_key_claiming()
        except LocalProtocolError:
            claim_targets = {}
        except Exception as e:
            print(f"[Matrix] get_users_for_key_claiming exception{suffix}: {e}")

        if claim_targets:
            try:
                response = await self.client.keys_claim(claim_targets)
                if response.__class__.__name__.endswith("Error"):
                    print(f"[Matrix] keys_claim failed{suffix}: {response}")
                else:
                    print(f"[Matrix] keys_claim complete{suffix}")
            except LocalProtocolError:
                pass
            except Exception as e:
                print(f"[Matrix] keys_claim exception{suffix}: {e}")

        try:
            await self.client.send_to_device_messages()
        except Exception as e:
            print(f"[Matrix] send_to_device_messages exception{suffix}: {e}")

    def _summarize_missing_sessions(self, missing: dict[str, list[str]], max_items: int = 6) -> str:
        pairs: list[str] = []
        for user_id, device_ids in missing.items():
            for device_id in device_ids:
                pairs.append(f"{user_id}:{device_id}")
                if len(pairs) >= max_items:
                    break
            if len(pairs) >= max_items:
                break
        extra = ""
        total = sum(len(v) for v in missing.values())
        if total > len(pairs):
            extra = f" (+{total - len(pairs)} more)"
        return ", ".join(pairs) + extra if pairs else "none"

    def _missing_pairs_set(self, missing: dict[str, list[str]]) -> set[tuple[str, str]]:
        out: set[tuple[str, str]] = set()
        for user_id, device_ids in missing.items():
            if not isinstance(user_id, str):
                continue
            if not isinstance(device_ids, list):
                continue
            for device_id in device_ids:
                if isinstance(device_id, str) and device_id:
                    out.add((user_id, device_id))
        return out

    def _extract_claimed_pairs(self, claim_response: Any) -> set[tuple[str, str]]:
        claimed: set[tuple[str, str]] = set()
        one_time_keys = getattr(claim_response, "one_time_keys", None)
        if not isinstance(one_time_keys, dict):
            return claimed

        for user_id, per_user in one_time_keys.items():
            if not isinstance(user_id, str) or not isinstance(per_user, dict):
                continue
            for device_id, per_device in per_user.items():
                if (
                    isinstance(device_id, str)
                    and device_id
                    and isinstance(per_device, dict)
                    and per_device
                ):
                    claimed.add((user_id, device_id))
        return claimed

    def _summarize_pair_set(
        self, pairs: set[tuple[str, str]], max_items: int = 6
    ) -> str:
        if not pairs:
            return "none"
        ordered = sorted(f"{user_id}:{device_id}" for user_id, device_id in pairs)
        preview = ordered[:max_items]
        extra = ""
        if len(ordered) > len(preview):
            extra = f" (+{len(ordered) - len(preview)} more)"
        return ", ".join(preview) + extra

    def _invalidate_outbound_group_session(self, room_id: str, reason: str) -> None:
        invalidate = getattr(self.client, "invalidate_outbound_session", None)
        if not callable(invalidate):
            return
        try:
            invalidate(room_id)
            print(f"[Matrix:{room_id}] Rotated outbound Megolm session ({reason}).")
        except Exception as e:
            print(f"[Matrix:{room_id}] Failed to rotate outbound Megolm session: {e}")

    async def _query_server_device_keys_for_self(self) -> tuple[str | None, str | None]:
        """
        Query the homeserver directly for this bot user's own device keys.
        This avoids relying on device-store cache/query heuristics.
        """
        try:
            method, path, data = Api.keys_query(self.access_token, {self.user_id})
            response = await self.client._send(KeysQueryResponse, method, path, data)
        except Exception as e:
            print(f"[Matrix] Failed explicit self keys_query request: {e}")
            return None, None

        if response.__class__.__name__.endswith("Error"):
            print(f"[Matrix] Self keys_query returned error: {response}")
            return None, None

        try:
            all_device_keys = getattr(response, "device_keys", {})
            if not isinstance(all_device_keys, dict):
                return None, None
            per_user = all_device_keys.get(self.user_id, {})
            if not isinstance(per_user, dict):
                return None, None
            per_device = per_user.get(self.device_id, {})
            if not isinstance(per_device, dict):
                return None, None
            keys = per_device.get("keys", {})
            if not isinstance(keys, dict):
                return None, None

            curve = keys.get(f"curve25519:{self.device_id}")
            if not isinstance(curve, str) or not curve:
                curve = keys.get("curve25519")
            ed = keys.get(f"ed25519:{self.device_id}")
            if not isinstance(ed, str) or not ed:
                ed = keys.get("ed25519")

            return (
                curve if isinstance(curve, str) and curve else None,
                ed if isinstance(ed, str) and ed else None,
            )
        except Exception as e:
            print(f"[Matrix] Failed to parse self keys_query response: {e}")
            return None, None

    async def _diagnose_own_device_identity(self) -> None:
        local_curve = None
        local_ed = None
        try:
            olm = getattr(self.client, "olm", None)
            account = getattr(olm, "account", None)
            identity_keys = getattr(account, "identity_keys", None)
            if isinstance(identity_keys, dict):
                maybe_curve = identity_keys.get("curve25519")
                if isinstance(maybe_curve, str) and maybe_curve:
                    local_curve = maybe_curve
                maybe_ed = identity_keys.get("ed25519")
                if isinstance(maybe_ed, str) and maybe_ed:
                    local_ed = maybe_ed
        except Exception:
            local_curve = None
            local_ed = None

        remote_curve, remote_ed = await self._query_server_device_keys_for_self()

        if remote_curve is None and remote_ed is None:
            print(
                "[Matrix] Could not verify own device keys via explicit /keys/query. "
                "This may be a homeserver response/caching issue; if clients show "
                "'unknown/deleted device', local vs server device keys may still be mismatched."
            )
            return

        if local_curve and remote_curve and local_curve != remote_curve:
            print(
                "[Matrix] Warning: local crypto identity key does not match homeserver "
                f"device key for {self.device_id}. Encrypted sends may appear as "
                "'unknown/deleted device'. Reset MATRIX_STORE_PATH and restart."
            )
            return
        if local_ed and remote_ed and local_ed != remote_ed:
            print(
                "[Matrix] Warning: local ed25519 signing key does not match homeserver "
                f"device key for {self.device_id}. Encrypted sends may appear as "
                "'unknown/deleted device'. Reset MATRIX_STORE_PATH and restart."
            )
            return

        print(f"[Matrix] Verified local device keys match homeserver for {self.device_id}.")

    async def _prepare_room_encryption(self, room_id: str) -> None:
        room = self._room_for_id(room_id)
        if room is None:
            return
        if not getattr(room, "encrypted", False):
            return
        if not getattr(self.client, "olm", None):
            return

        try:
            if not getattr(room, "members_synced", False):
                response = await self.client.joined_members(room_id)
                if response.__class__.__name__.endswith("Error"):
                    print(f"[Matrix:{room_id}] joined_members failed: {response}")

            # Retry key query + claim a few times before sending.
            missing_after: dict[str, list[str]] = {}
            had_missing = False
            for attempt in range(1, 4):
                try:
                    if getattr(self.client, "should_query_keys", False):
                        qresp = await self.client.keys_query()
                        if qresp.__class__.__name__.endswith("Error"):
                            print(f"[Matrix:{room_id}] keys_query failed (attempt {attempt}): {qresp}")
                except Exception as e:
                    print(f"[Matrix:{room_id}] keys_query exception (attempt {attempt}): {e}")

                missing = self.client.get_missing_sessions(room_id)
                if not missing:
                    missing_after = {}
                    break

                had_missing = True
                total_missing = sum(len(v) for v in missing.values())
                print(
                    f"[Matrix:{room_id}] Missing Olm sessions for {total_missing} device(s) "
                    f"(attempt {attempt}); claiming keys."
                )

                try:
                    cresp = await self.client.keys_claim(missing)
                    if cresp.__class__.__name__.endswith("Error"):
                        print(f"[Matrix:{room_id}] keys_claim failed (attempt {attempt}): {cresp}")
                    else:
                        missing_pairs = self._missing_pairs_set(missing)
                        claimed_pairs = self._extract_claimed_pairs(cresp)
                        claimed_for_devices = len(claimed_pairs)
                        unresolved_pairs = missing_pairs - claimed_pairs

                        failures = getattr(cresp, "failures", {})
                        if claimed_for_devices == 0:
                            print(
                                f"[Matrix:{room_id}] keys_claim returned no one-time keys "
                                f"(attempt {attempt})."
                            )
                        else:
                            print(
                                f"[Matrix:{room_id}] keys_claim returned one-time keys for "
                                f"{claimed_for_devices} device(s) (attempt {attempt})."
                            )
                        if unresolved_pairs:
                            print(
                                f"[Matrix:{room_id}] No one-time/fallback key available for: "
                                f"{self._summarize_pair_set(unresolved_pairs)} (attempt {attempt})."
                            )
                        if isinstance(failures, dict) and failures:
                            print(
                                f"[Matrix:{room_id}] keys_claim failures (attempt {attempt}): "
                                f"{failures}"
                            )
                except Exception as e:
                    print(f"[Matrix:{room_id}] keys_claim exception (attempt {attempt}): {e}")

                try:
                    await self.client.send_to_device_messages()
                except Exception as e:
                    print(
                        f"[Matrix:{room_id}] send_to_device_messages exception "
                        f"(attempt {attempt}): {e}"
                    )

                missing_after = self.client.get_missing_sessions(room_id)
                if not missing_after:
                    break
                await asyncio.sleep(0.2 * attempt)

            if missing_after:
                self.roomsWithMissingOlmSessions.add(room_id)
                print(
                    f"[Matrix:{room_id}] Still missing Olm sessions after retries: "
                    f"{self._summarize_missing_sessions(missing_after)}"
                )
                # Force fresh Megolm sessions while we still have missing devices so
                # newly-established Olm sessions can receive keys on later sends.
                self._invalidate_outbound_group_session(room_id, "missing Olm sessions")
            elif had_missing:
                if room_id in self.roomsWithMissingOlmSessions:
                    print(
                        f"[Matrix:{room_id}] Missing Olm sessions recovered; re-sharing via "
                        "a fresh outbound Megolm session."
                    )
                self.roomsWithMissingOlmSessions.discard(room_id)
                self._invalidate_outbound_group_session(room_id, "Olm sessions recovered")

            # Pre-share outbound group session when needed.
            try:
                assert self.client.olm is not None
                if self.client.olm.should_share_group_session(room_id):
                    sresp = await self.client.share_group_session(
                        room_id,
                        ignore_unverified_devices=True,
                    )
                    if sresp.__class__.__name__.endswith("Error"):
                        print(f"[Matrix:{room_id}] share_group_session failed: {sresp}")
            except LocalProtocolError:
                pass
            except Exception as e:
                print(f"[Matrix:{room_id}] share_group_session exception: {e}")
        except LocalProtocolError as e:
            print(f"[Matrix:{room_id}] Encryption prep failed: {e}")
        except Exception as e:
            print(f"[Matrix:{room_id}] Unexpected encryption prep error: {e}")

    def _content_from_event(self, event: Any) -> dict[str, Any]:
        source = getattr(event, "source", None)
        if not isinstance(source, dict):
            return {}
        content = source.get("content")
        if not isinstance(content, dict):
            return {}
        return content

    def _event_id(self, event: Any) -> str | None:
        event_id = getattr(event, "event_id", None)
        if isinstance(event_id, str) and event_id:
            return event_id
        source = getattr(event, "source", None)
        if isinstance(source, dict):
            sid = source.get("event_id")
            if isinstance(sid, str) and sid:
                return sid
        return None

    def _sender(self, event: Any) -> str | None:
        sender = getattr(event, "sender", None)
        if isinstance(sender, str) and sender:
            return sender
        source = getattr(event, "source", None)
        if isinstance(source, dict):
            ss = source.get("sender")
            if isinstance(ss, str) and ss:
                return ss
        return None

    def _room_id(self, room: MatrixRoom) -> str | None:
        rid = getattr(room, "room_id", None)
        if isinstance(rid, str) and rid:
            return rid
        return None

    def _room_for_id(self, room_id: str) -> MatrixRoom | None:
        room_cache = getattr(self, "roomCache", None)
        if not isinstance(room_cache, dict):
            room_cache = {}
            self.roomCache = room_cache

        cached = room_cache.get(room_id)
        if cached is not None:
            return cached
        rooms = getattr(self.client, "rooms", None)
        if isinstance(rooms, dict):
            room = rooms.get(room_id)
            if room is not None:
                room_cache[room_id] = room
                return room
        return None

    async def on_room_message(self, room: MatrixRoom, event: Any):
        room_id = self._room_id(room)
        if not room_id:
            return
        self.roomCache[room_id] = room

        sender = self._sender(event)
        if not sender or sender == self.user_id:
            return

        content = self._content_from_event(event)
        if not content:
            return

        processed = await self.process_event_message(room, event, content)
        if not processed:
            return

        self._append_recent_message(room_id, processed)

        if self.is_whitelisted(room_id, "mentions") and self.is_mentioned(content, room):
            await self.constantChat(room_id, processed)
            return

        if self.is_whitelisted(room_id, "always"):
            await self.constantChat(room_id, processed)
            return

        if self.is_whitelisted(room_id, "rand"):
            await self.randomChat(room_id, processed, self.is_mentioned(content, room))

    async def on_reaction(self, room: MatrixRoom, event: Any):
        room_id = self._room_id(room)
        if not room_id:
            return
        self.roomCache[room_id] = room

        sender = self._sender(event)
        if not sender or sender == self.user_id:
            return

        swipes_cfg = get_config().get("swipes", {}) or {}
        if not swipes_cfg.get("enabled", False):
            return

        if not self._swipe_allowed(swipes_cfg, sender, room_id):
            return

        content = self._content_from_event(event)
        relates = content.get("m.relates_to") if isinstance(content, dict) else None
        if not isinstance(relates, dict):
            return

        rel_type = relates.get("rel_type")
        if rel_type != "m.annotation":
            return

        target_event_id = relates.get("event_id")
        emoji = relates.get("key")
        if not isinstance(target_event_id, str) or not target_event_id:
            return
        if not isinstance(emoji, str) or not emoji:
            return

        action = swipe_action_for_emoji(emoji, swipes_cfg)
        if not action:
            return

        print(f"[Matrix:{room_id}] Swipe action detected: {action}")
        queue = self.pendingSwipes.get(room_id, [])
        queue.append((target_event_id, action))
        self.pendingSwipes[room_id] = queue

    async def on_undecrypted_event(self, room: MatrixRoom, event: Any):
        room_id = self._room_id(room) or "<unknown-room>"
        sender = self._sender(event) or "<unknown-sender>"
        event_id = self._event_id(event) or "<unknown-event>"
        print(
            f"[Matrix:{room_id}] Received undecrypted event {event_id} from {sender}. "
            "This usually means missing E2EE room keys for this device/session."
        )

        if not isinstance(event, MegolmEvent):
            return

        session_id = str(getattr(event, "session_id", "") or "")
        sender_key = str(getattr(event, "sender_key", "") or "")
        room_key = str(getattr(event, "room_id", "") or room_id)
        cache_key = (room_key, sender_key, session_id)
        if session_id and sender_key and cache_key in self.requestedMissingMegolmSessions:
            return

        try:
            response = await self.client.request_room_key(event)
            if session_id and sender_key:
                self.requestedMissingMegolmSessions.add(cache_key)
            if response.__class__.__name__.endswith("Error"):
                print(
                    f"[Matrix:{room_id}] Room key request failed for session {session_id}: "
                    f"{response}"
                )
            else:
                print(
                    f"[Matrix:{room_id}] Requested missing room key for session {session_id}."
                )
        except LocalProtocolError:
            if session_id and sender_key:
                self.requestedMissingMegolmSessions.add(cache_key)
        except Exception as e:
            print(
                f"[Matrix:{room_id}] Failed to request room key for session {session_id}: {e}"
            )

        await self._refresh_e2ee_state(reason=f"undecrypted:{room_id}")

    def _append_recent_message(self, room_id: str, message: dict[str, Any]):
        existing = self.recentMessages.get(room_id)
        if existing is None:
            existing = deque(maxlen=200)
            self.recentMessages[room_id] = existing
        existing.append(message)

    def is_whitelisted(self, room_id: str, wtype: str) -> bool:
        cfg = get_config()
        values = cfg.get("whitelist", {}).get(wtype, [])
        return is_whitelisted_id(room_id, values)

    def _self_aliases_for_room(self, room: MatrixRoom | None) -> set[str]:
        aliases: set[str] = set()
        localpart = _extract_localpart(self.user_id)
        if localpart:
            aliases.add(localpart)

        if room is not None:
            user_name = getattr(room, "user_name", None)
            if callable(user_name):
                try:
                    own_display = user_name(self.user_id)
                    if isinstance(own_display, str) and own_display.strip():
                        aliases.add(own_display.strip())
                except Exception:
                    pass

            users = getattr(room, "users", None)
            if isinstance(users, dict):
                me = users.get(self.user_id)
                display = getattr(me, "display_name", None) if me is not None else None
                if isinstance(display, str) and display.strip():
                    aliases.add(display.strip())
        return aliases

    def is_mentioned(self, content: dict[str, Any], room: MatrixRoom | None) -> bool:
        aliases = self._self_aliases_for_room(room)
        return matrix_content_mentions_user(content, self.user_id, aliases)

    def clean_content(self, text: str, room: MatrixRoom | None) -> str:
        content = text.strip()
        aliases = sorted(self._self_aliases_for_room(room), key=len, reverse=True)
        for alias in aliases:
            if not alias:
                continue
            pattern = re.compile(rf"(?<!\w)@{re.escape(alias)}(?!\w)", re.IGNORECASE)
            content = pattern.sub("{{char}}", content)
        content = content.replace(self.user_id, "{{char}}")
        return content

    def _display_name_for_sender(self, room: MatrixRoom | None, sender: str) -> str:
        if room is not None:
            user_name = getattr(room, "user_name", None)
            if callable(user_name):
                try:
                    display = user_name(sender)
                    if isinstance(display, str) and display.strip():
                        return display
                except Exception:
                    pass
            users = getattr(room, "users", None)
            if isinstance(users, dict):
                user_obj = users.get(sender)
                display = getattr(user_obj, "display_name", None) if user_obj else None
                if isinstance(display, str) and display.strip():
                    return display
        localpart = _extract_localpart(sender)
        return localpart or sender

    def _is_image_url(self, url: str) -> bool:
        lower = url.lower().split("?")[0]
        return any(lower.endswith(ext) for ext in IMAGE_EXTENSIONS)

    def _extract_image_urls(self, text: str) -> list[str]:
        urls: list[str] = []
        for match in IMAGE_URL_RE.findall(text or ""):
            url = match.rstrip(").,>")
            if self._is_image_url(url):
                urls.append(url)
        return urls

    def _mxc_to_http(self, mxc_url: str) -> str | None:
        converter = getattr(self.client, "mxc_to_http", None)
        if callable(converter):
            try:
                converted = converter(mxc_url)
                if isinstance(converted, str) and converted:
                    return converted
            except Exception:
                return None
        return None

    def _extract_download_body(self, download_response: Any) -> bytes | None:
        if isinstance(download_response, (bytes, bytearray)):
            return bytes(download_response)
        if isinstance(download_response, tuple):
            for item in download_response:
                body = self._extract_download_body(item)
                if body is not None:
                    return body
            return None
        body = getattr(download_response, "body", None)
        if isinstance(body, (bytes, bytearray)):
            return bytes(body)
        return None

    def _decrypt_attachment(self, ciphertext: bytes, file_obj: dict[str, Any]) -> bytes | None:
        try:
            from nio.crypto.attachments import decrypt_attachment  # type: ignore
        except Exception:
            return None

        key_obj = file_obj.get("key")
        key_value = key_obj.get("k") if isinstance(key_obj, dict) else key_obj
        hashes = file_obj.get("hashes")
        sha256 = hashes.get("sha256") if isinstance(hashes, dict) else None
        iv = file_obj.get("iv")

        candidate_calls = [
            (ciphertext, key_value, sha256, iv),
            (ciphertext, key_obj, sha256, iv),
        ]

        for args in candidate_calls:
            try:
                decrypted = decrypt_attachment(*args)
                if isinstance(decrypted, (bytes, bytearray)):
                    return bytes(decrypted)
            except Exception:
                continue
        return None

    def _write_image_bytes(self, room_id: str, data: bytes, filename: str | None) -> str:
        directory = ensure_channel_image_dir(room_id)
        safe_name = sanitize_filename(filename or "image")
        target = os.path.join(directory, safe_name)
        if os.path.exists(target):
            stem, ext = os.path.splitext(safe_name)
            target = os.path.join(directory, f"{stem}_{uuid.uuid4().hex}{ext}")
        with open(target, "wb") as f:
            f.write(data)
        return target

    async def _download_encrypted_image_to_history(
        self, room_id: str, file_obj: dict[str, Any], filename: str | None
    ) -> str | None:
        mxc_url = file_obj.get("url")
        if not isinstance(mxc_url, str) or not mxc_url:
            return None
        try:
            download_response = await self.client.download(mxc_url)
        except Exception as e:
            print(f"[Matrix:{room_id}] Failed encrypted media download: {e}")
            return None

        encrypted = self._extract_download_body(download_response)
        if encrypted is None:
            return None

        decrypted = self._decrypt_attachment(encrypted, file_obj)
        if decrypted is None:
            return None

        try:
            return self._write_image_bytes(room_id, decrypted, filename)
        except Exception as e:
            print(f"[Matrix:{room_id}] Failed to write decrypted image: {e}")
            return None

    def _is_image_message(self, content: dict[str, Any]) -> bool:
        msgtype = str(content.get("msgtype", "") or "")
        if msgtype == "m.image":
            return True
        if msgtype != "m.file":
            return False
        info = content.get("info")
        if isinstance(info, dict):
            mimetype = info.get("mimetype")
            if isinstance(mimetype, str) and mimetype.startswith("image/"):
                return True
        body = content.get("body")
        if isinstance(body, str):
            return self._is_image_url(body)
        return False

    async def extract_images(self, room_id: str, content: dict[str, Any]) -> list[dict[str, Any]]:
        images: list[dict[str, Any]] = []
        seen_urls: set[str] = set()

        body = content.get("body")
        if isinstance(body, str):
            for url in self._extract_image_urls(body):
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                images.append(build_remote_image_record(url, "matrix_link"))

        if not self._is_image_message(content):
            return images

        filename = content.get("body")
        filename_str = filename if isinstance(filename, str) else None
        file_obj = content.get("file") if isinstance(content.get("file"), dict) else None
        mxc_url = content.get("url")
        if file_obj and isinstance(file_obj.get("url"), str):
            mxc_url = file_obj.get("url")

        url_for_model: str | None = None
        if isinstance(mxc_url, str) and mxc_url:
            url_for_model = self._mxc_to_http(mxc_url) or mxc_url

        record: dict[str, Any] = {"source": "matrix_image"}
        if filename_str:
            record["filename"] = filename_str
        if url_for_model:
            record["url"] = url_for_model

        local_path: str | None = None
        if file_obj:
            local_path = await self._download_encrypted_image_to_history(room_id, file_obj, filename_str)
        elif url_for_model and url_for_model.startswith("http"):
            try:
                local_path = await download_image_to_history(room_id, url_for_model, filename_str)
            except Exception as e:
                print(f"[Matrix:{room_id}] Failed to cache image attachment: {e}")

        if local_path:
            record["local_path"] = os.path.relpath(local_path).replace("\\", "/")

        if "url" in record or "local_path" in record:
            record_url = str(record.get("url", ""))
            if record_url and record_url not in seen_urls:
                seen_urls.add(record_url)
                images.append(record)
            elif "local_path" in record:
                images.append(record)

        return images

    async def process_event_message(
        self, room: MatrixRoom, event: Any, content: dict[str, Any]
    ) -> dict[str, Any] | None:
        sender = self._sender(event)
        if not sender:
            return None
        event_id = self._event_id(event)
        if not event_id:
            return None

        room_id = self._room_id(room)
        if not room_id:
            return None

        raw_body = str(content.get("body", "") or "").strip()
        images = await self.extract_images(room_id, content)

        if not raw_body and not images:
            return None

        if not raw_body and images:
            raw_body = "(shared an image)"

        cleaned_body = self.clean_content(raw_body, room)
        username = self._display_name_for_sender(room, sender)

        processed: dict[str, Any] = {
            "user": username,
            "message": cleaned_body,
            "messageId": event_id,
        }
        if images:
            processed["images"] = images
        return processed

    async def constantChat(self, room_id: str, message: dict[str, Any]):
        self.addToQueue(room_id, message)

    async def randomChat(self, room_id: str, message: dict[str, Any], is_mentioned: bool):
        cfg = get_config().get("randomChat", {})
        now = datetime.datetime.now()
        current = self.randomChats.get(room_id)

        if current:
            if now < current.endTime:
                self.randomChats[room_id].lastMessageId = str(message.get("messageId", ""))
                self.addToQueue(room_id, message)
                return
            if now < current.nextChatTime:
                print(f"[Matrix:{room_id}] Not starting random chat until {current.nextChatTime}")
                return

        engagement_chance = cfg.get("engagement_chance", 10)
        if random.randint(0, engagement_chance) != 0:
            if not (is_mentioned and cfg.get("respond_to_mentions", False)):
                print(f"[Matrix:{room_id}] Random chat roll failed")
                return

        history_limit = int(cfg.get("message_history_limit", 10))
        recent = list(self.recentMessages.get(room_id, deque()))
        msgs = recent[-history_limit:] if history_limit > 0 else []
        if not msgs:
            return

        if current:
            for msg in msgs:
                if str(msg.get("messageId", "")) == current.lastMessageId:
                    print(f"[Matrix:{room_id}] Insufficient new messages for random chat")
                    return

        if current != self.randomChats.get(room_id):
            return

        min_duration = int(cfg.get("min_chat_duration_seconds", 40))
        max_duration = int(cfg.get("max_chat_duration_seconds", 500))
        end_time = now + datetime.timedelta(seconds=random.randint(min_duration, max_duration))

        min_down = int(cfg.get("min_downtime_minutes", 5))
        max_down = int(cfg.get("max_downtime_minutes", 20))
        next_chat_time = end_time + datetime.timedelta(minutes=random.randint(min_down, max_down))

        self.randomChats[room_id] = RandomChat(
            endTime=end_time,
            nextChatTime=next_chat_time,
            lastMessageId=str(message.get("messageId", "")),
        )

        for queued in msgs:
            self.addToQueue(room_id, queued)

        print(
            f"[Matrix:{room_id}] Starting random chat from {now} to {end_time}. "
            f"Next earliest {next_chat_time}"
        )

    def addToQueue(self, room_id: str, message: dict[str, Any]):
        extra = ""
        if message.get("images"):
            extra = f" [{len(message['images'])} image(s)]"
        print(f"[Matrix:{room_id}] {message['user']}: {message['message']}{extra}")
        queue = self.pendingMessages.get(room_id, [])
        queue.append(message)
        self.pendingMessages[room_id] = queue

    def _coerce_id_set(self, values: Any) -> set[str]:
        return normalize_id_set(values)

    def _swipe_allowed(self, swipes_cfg: dict[str, Any], user_id: str, room_id: str) -> bool:
        user_whitelist = self._coerce_id_set(swipes_cfg.get("user_whitelist"))
        channel_whitelist = self._coerce_id_set(swipes_cfg.get("channel_whitelist"))
        user_key = normalize_id(user_id)
        room_key = normalize_id(room_id)

        if user_key is None or room_key is None:
            return False
        if not user_whitelist and not channel_whitelist:
            return True
        if not user_whitelist:
            return room_key in channel_whitelist
        if not channel_whitelist:
            return user_key in user_whitelist
        return (user_key in user_whitelist) or (room_key in channel_whitelist)

    def _control_reactions_path(self, room_id: str) -> str:
        safe_room_id = sanitize_filename(room_id, fallback_prefix="room")
        return os.path.join("history", "control-reactions", f"{safe_room_id}.json")

    def _load_control_reactions(self, room_id: str) -> list[dict[str, str]]:
        path = self._control_reactions_path(room_id)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("reactions"), list):
                out: list[dict[str, str]] = []
                for item in data["reactions"]:
                    if not isinstance(item, dict):
                        continue
                    message_id = item.get("messageId")
                    emoji = item.get("emoji")
                    reaction_event_id = item.get("reactionEventId")
                    if not isinstance(message_id, str) or not message_id:
                        continue
                    if not isinstance(emoji, str) or not emoji:
                        continue
                    if not isinstance(reaction_event_id, str) or not reaction_event_id:
                        continue
                    out.append(
                        {
                            "messageId": message_id,
                            "emoji": emoji,
                            "reactionEventId": reaction_event_id,
                        }
                    )
                return out
        except Exception:
            pass
        return []

    def _save_control_reactions(self, room_id: str, reactions: list[dict[str, str]]) -> None:
        path = self._control_reactions_path(room_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"reactions": reactions}, f, indent=2)

    def _record_control_reaction(
        self, room_id: str, message_id: str, emoji: str, reaction_event_id: str
    ) -> None:
        existing = self._load_control_reactions(room_id)
        key = (message_id, emoji, reaction_event_id)
        existing_set = {
            (r.get("messageId"), r.get("emoji"), r.get("reactionEventId")) for r in existing
        }
        if key in existing_set:
            return
        existing.append(
            {"messageId": message_id, "emoji": emoji, "reactionEventId": reaction_event_id}
        )
        self._save_control_reactions(room_id, existing)

    async def _remove_control_reaction(self, room_id: str, reaction_event_id: str) -> bool:
        try:
            await self.client.room_redact(room_id, reaction_event_id, reason="remove swipe control")
            return True
        except Exception as e:
            print(f"[Matrix:{room_id}] Failed to redact reaction {reaction_event_id}: {e}")
            return False

    async def _set_control_reaction(
        self,
        room_id: str,
        message_id: str,
        emoji: str,
        should_have: bool,
    ) -> None:
        existing = self._load_control_reactions(room_id)
        matching = [
            r
            for r in existing
            if r.get("messageId") == message_id and r.get("emoji") == emoji
        ]

        if should_have:
            if matching:
                return
            reaction_event_id = await self.send_reaction(room_id, message_id, emoji)
            if reaction_event_id:
                self._record_control_reaction(room_id, message_id, emoji, reaction_event_id)
            return

        if not matching:
            return

        remaining = [
            r
            for r in existing
            if not (r.get("messageId") == message_id and r.get("emoji") == emoji)
        ]
        for rec in matching:
            rid = rec.get("reactionEventId")
            if not isinstance(rid, str):
                continue
            ok = await self._remove_control_reaction(room_id, rid)
            if not ok:
                remaining.append(rec)
        self._save_control_reactions(room_id, remaining)

    async def _cleanup_recorded_control_reactions(self, room_id: str) -> None:
        records = self._load_control_reactions(room_id)
        if not records:
            return
        remaining: list[dict[str, str]] = []
        for rec in records:
            rid = rec.get("reactionEventId")
            if not isinstance(rid, str):
                continue
            ok = await self._remove_control_reaction(room_id, rid)
            if not ok:
                remaining.append(rec)
        self._save_control_reactions(room_id, remaining)

    def _event_id_from_send_response(self, response: Any) -> str | None:
        event_id = getattr(response, "event_id", None)
        if isinstance(event_id, str) and event_id:
            return event_id
        if isinstance(response, tuple):
            for item in response:
                event_id = self._event_id_from_send_response(item)
                if event_id:
                    return event_id
        return None

    async def send_message(
        self,
        room_id: str,
        body: str,
        formatted_body: str | None = None,
        mention_user_ids: Sequence[str] | None = None,
    ) -> str | None:
        await self._prepare_room_encryption(room_id)
        content = build_text_content(body, formatted_body, mention_user_ids)
        try:
            response = await self.client.room_send(
                room_id=room_id,
                message_type="m.room.message",
                content=content,
                ignore_unverified_devices=True,
            )
            return self._event_id_from_send_response(response)
        except Exception as e:
            print(f"[Matrix:{room_id}] Failed to send message: {e}")
            return None

    async def send_edit(
        self,
        room_id: str,
        target_event_id: str,
        body: str,
        formatted_body: str | None = None,
        mention_user_ids: Sequence[str] | None = None,
    ) -> str | None:
        await self._prepare_room_encryption(room_id)
        content = build_edit_content(body, target_event_id, formatted_body, mention_user_ids)
        try:
            response = await self.client.room_send(
                room_id=room_id,
                message_type="m.room.message",
                content=content,
                ignore_unverified_devices=True,
            )
            return self._event_id_from_send_response(response)
        except Exception as e:
            print(f"[Matrix:{room_id}] Failed to send edit for {target_event_id}: {e}")
            return None

    async def send_reaction(self, room_id: str, target_event_id: str, emoji: str) -> str | None:
        content = build_reaction_content(target_event_id, emoji)
        try:
            response = await self.client.room_send(
                room_id=room_id,
                message_type="m.reaction",
                content=content,
                ignore_unverified_devices=True,
            )
            return self._event_id_from_send_response(response)
        except Exception as e:
            print(f"[Matrix:{room_id}] Failed to react with {emoji}: {e}")
            return None

    async def _set_typing(self, room_id: str, typing: bool) -> None:
        room_typing = getattr(self.client, "room_typing", None)
        if not callable(room_typing):
            return
        try:
            if typing:
                await room_typing(room_id, True, timeout=self.sync_timeout_ms)
            else:
                await room_typing(room_id, False)
        except Exception:
            try:
                await room_typing(room_id, typing)
            except Exception:
                return

    def _get_mention_indexes_for_room(self, room_id: str) -> tuple[dict[str, str], dict[str, str]]:
        now = time.time()
        cached = self.roomMentionIndexCache.get(room_id)
        if cached and now - cached[0] < 300:
            return cached[1], cached[2]

        room = self._room_for_id(room_id)
        users = getattr(room, "users", None) if room else None
        if not isinstance(users, dict):
            return {}, {}

        display_counts: dict[str, int] = {}
        local_counts: dict[str, int] = {}
        entries: list[tuple[str, str | None, str | None]] = []

        for uid, user_obj in users.items():
            if not isinstance(uid, str) or not uid:
                continue
            display = getattr(user_obj, "display_name", None)
            display_name = display.strip() if isinstance(display, str) and display.strip() else None
            localpart = _extract_localpart(uid) or None
            entries.append((uid, display_name, localpart))
            if display_name:
                key = display_name.casefold()
                display_counts[key] = display_counts.get(key, 0) + 1
            if localpart:
                key = localpart.casefold()
                local_counts[key] = local_counts.get(key, 0) + 1

        display_index: dict[str, str] = {}
        local_index: dict[str, str] = {}
        for uid, display_name, localpart in entries:
            if display_name:
                key = display_name.casefold()
                if display_counts.get(key) == 1:
                    display_index[key] = uid
            if localpart:
                key = localpart.casefold()
                if local_counts.get(key) == 1:
                    local_index[key] = uid

        self.roomMentionIndexCache[room_id] = (now, display_index, local_index)
        return display_index, local_index

    def _resolve_user_id_for_generated_at_mention(self, room_id: str, name: str) -> str | None:
        stripped = name.strip()
        if not stripped:
            return None
        if stripped.casefold() in {"everyone", "here"}:
            return None
        display_index, local_index = self._get_mention_indexes_for_room(room_id)
        key = stripped.casefold()
        return display_index.get(key) or local_index.get(key)

    async def resolve_generated_at_mentions(
        self, room_id: str, content: str
    ) -> tuple[str, str | None, list[str]]:
        if "@" not in content:
            return content, None, []

        def resolver(name: str) -> str | None:
            return self._resolve_user_id_for_generated_at_mention(room_id, name)

        return apply_generated_at_mentions(content, resolver)

    async def process_messages(self):
        room_ids = list(set(self.pendingMessages.keys()) | set(self.pendingSwipes.keys()))
        tasks = []

        for room_id in room_ids:
            messages = self.pendingMessages.get(room_id, [])
            swipe_jobs = self.pendingSwipes.get(room_id, [])
            if not messages and not swipe_jobs:
                continue

            self.pendingMessages[room_id] = []
            self.pendingSwipes[room_id] = []

            async def process_room(
                room_id: str,
                messages: List[dict[str, Any]],
                swipe_jobs: list[tuple[str, str]],
            ):
                if swipe_jobs:
                    swipes_cfg = get_config().get("swipes", {}) or {}
                    prev_emoji = str(swipes_cfg.get("prev_emoji", "◀️"))
                    next_emoji = str(swipes_cfg.get("next_emoji", "▶️"))
                    for target_event_id, action in dedupe_swipe_jobs(swipe_jobs):
                        new_text: str | None = None
                        try:
                            if action == "regen":
                                new_text = await swipe_regenerate(room_id, target_event_id)
                            elif action == "prev":
                                new_text = swipe_prev(room_id, target_event_id)
                            elif action == "next":
                                new_text = swipe_next(room_id, target_event_id)
                        except Exception as e:
                            print(f"[Matrix:{room_id}] Swipe action failed for {target_event_id}: {e}")
                            continue

                        if new_text:
                            cleaned = clean_response(new_text)
                            plain, formatted, mentions = await self.resolve_generated_at_mentions(
                                room_id, cleaned
                            )
                            await self.send_edit(room_id, target_event_id, plain, formatted, mentions)

                        try:
                            nav = get_swipe_nav_state(room_id, target_event_id)
                            if nav is not None:
                                has_prev, has_next = nav
                                await self._set_control_reaction(
                                    room_id, target_event_id, prev_emoji, has_prev
                                )
                                await self._set_control_reaction(
                                    room_id, target_event_id, next_emoji, has_next
                                )
                        except Exception as e:
                            print(
                                f"[Matrix:{room_id}] Swipe reaction update failed for "
                                f"{target_event_id}: {e}"
                            )

                if not messages:
                    return

                await self._set_typing(room_id, True)
                try:
                    inference_result = await chat_inference(room_id, messages)
                except Exception as e:
                    print(f"[Matrix:{room_id}] chat_inference error: {e}")
                    inference_result = None
                finally:
                    await self._set_typing(room_id, False)

                response: str | None = None
                pending_message_id: str | None = None
                if isinstance(inference_result, tuple) and len(inference_result) == 2:
                    response = inference_result[0]
                    pending_message_id = inference_result[1]
                elif isinstance(inference_result, str):
                    response = inference_result

                if not response:
                    print(f"[Matrix:{room_id}] No response")
                    return

                cleaned = clean_response(response)
                plain, formatted, mentions = await self.resolve_generated_at_mentions(room_id, cleaned)
                sent_event_id = await self.send_message(room_id, plain, formatted, mentions)
                if sent_event_id:
                    try:
                        if pending_message_id:
                            finalize_assistant_message_id(room_id, pending_message_id, sent_event_id)
                        else:
                            finalize_last_assistant_message_id(room_id, sent_event_id)
                    except Exception as e:
                        print(f"[Matrix:{room_id}] Failed to store assistant messageId: {e}")

                swipes_cfg = get_config().get("swipes", {}) or {}
                try:
                    await self._cleanup_recorded_control_reactions(room_id)
                except Exception as e:
                    print(f"[Matrix:{room_id}] Swipe control cleanup failed: {e}")

                if not sent_event_id:
                    return

                if swipes_cfg.get("auto_react_controls", False):
                    allowed_rooms = self._coerce_id_set(swipes_cfg.get("auto_react_channel_whitelist"))
                    should_auto_react = (not allowed_rooms) or (room_id in allowed_rooms)
                    if should_auto_react:
                        regen_emoji = str(swipes_cfg.get("regen_emoji", "🔄"))
                        reaction_event_id = await self.send_reaction(room_id, sent_event_id, regen_emoji)
                        if reaction_event_id:
                            self._record_control_reaction(
                                room_id, sent_event_id, regen_emoji, reaction_event_id
                            )

            tasks.append(process_room(room_id, messages, swipe_jobs))

        await asyncio.gather(*tasks)

    async def inference_loop_task(self):
        while True:
            await self.process_messages()
            await asyncio.sleep(0.5)


def clean_response(resp: str) -> str:
    resp = html.unescape(resp)
    resp = resp.strip()
    return dequote(resp)


async def run_bot():
    bot = MatrixBot()
    await bot.run()


if __name__ == "__main__":
    asyncio.run(run_bot())
