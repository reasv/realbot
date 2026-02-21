import re
import random
import datetime
import time
from typing import Any, List
import html
import os
import json
from dataclasses import dataclass

import asyncio
import discord
from discord import Guild, Member, Message

from .chat_image_utils import (
    build_remote_image_record,
    download_image_to_history,
    is_supported_image_mime,
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
from .utils import get_config, dequote, is_whitelisted_id, normalize_id_set

@dataclass
class RandomChat:
    endTime: datetime.datetime
    nextChatTime: datetime.datetime
    lastMessage: Message

class Bot(discord.Client):
    pendingMessages: dict[int, List[dict[str, Any]]] = {}
    pendingSwipes: dict[int, list[tuple[int, str]]] = {}

    randomChats: dict[int, RandomChat] = {}

    # guild_id -> (timestamp, display_name_index, username_index)
    mentionIndexCache: dict[int, tuple[float, dict[str, int], dict[str, int]]] = {}

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.pendingMessages = {}
        self.pendingSwipes = {}
        self.randomChats = {}
        self.mentionIndexCache = {}
        self._inflight_channels: set[int] = set()
        self._inflight_tasks: dict[int, asyncio.Task[None]] = {}

    async def on_ready(self):
        print(f'Logged on as {self.user}!')

    async def setup_hook(self) -> None:
        # create the background task and run it in the background
        self.bg_task = self.loop.create_task(self.inference_loop_task())

    async def on_message(self, message: Message):
        #print(f'Message in {message.channel.id} from {message.author}: {message.content}')
        assert self.user is not None, "User is None"
        if message.author.id == self.user.id:
            return

        if len(message.content.strip()) == 0:
            return
        
        if self.isMentioned(message) and is_whitelisted(message.channel.id, 'mentions'):
            return await self.constantChat(message)
        
        if is_whitelisted(message.channel.id, 'always'):
            return await self.constantChat(message)

        if is_whitelisted(message.channel.id, 'rand'):
            return await self.randomChat(message)

    async def constantChat(self, message: Message):
        msg = await self.processMessage(message)
        self.addToQueue(message.channel.id, msg)

    async def randomChat(self, message: Message):
        config = get_config()["randomChat"]
        currentChat = self.randomChats.get(message.channel.id)
        if currentChat:
            # Continue existing randomchat if it's still valid
            if datetime.datetime.now() < currentChat.endTime:
                # Set new last message
                self.randomChats[message.channel.id].lastMessage = message
                # Continuing a chat means doing the normal single message flow
                msg = await self.processMessage(message)
                self.addToQueue(message.channel.id, msg)
                return
            else: # if the chat already ended
                # Wait minutes before starting a new conversation
                if datetime.datetime.now() < currentChat.nextChatTime:
                    print(f"Not starting a new conversation in {message.channel.id} until {currentChat.nextChatTime}")

        # Only start chat with 1 in 10 chance
        if random.randint(0, config["engagement_chance"]) != 0:
            print("Roll failed")
            if not self.isMentioned(message):
                return
            if not config["respond_to_mentions"]:
                return
            # If we were mentioned, start a new chat regardless of odds

        # Start new randomchat
        msgs = []
        async for historyMessage in message.channel.history(limit=config["message_history_limit"]):
            if currentChat and historyMessage.id == currentChat.lastMessage.id:
                # Don't create new chat if any message in this message history was already in the previous discussion
                print("Insufficient new messages")
                return
            msgs.append(await self.processMessage(historyMessage))

        msgs.reverse()

        if (len(msgs) == 0):
            return

        # Another chat has already been started
        if currentChat != self.randomChats.get(message.channel.id):
            return
        
        # Add this new randomchat to the registry
        endTime = datetime.datetime.now() + datetime.timedelta(seconds=random.randint(config["min_chat_duration_seconds"], config["max_chat_duration_seconds"]))
        self.randomChats[message.channel.id] = RandomChat(
            endTime,
            nextChatTime=endTime + datetime.timedelta(minutes=random.randint(config["min_downtime_minutes"], config["max_downtime_minutes"])),
            lastMessage=message,
        )
        for msg in msgs:
            self.addToQueue(message.channel.id, msg)

        print(f"Starting random chat in {message.channel.id} from {datetime.datetime.now()} to {self.randomChats[message.channel.id].endTime}. Will not chat again until {self.randomChats[message.channel.id].nextChatTime}")

    def isMentioned(self, message: Message):
        assert self.user is not None, "User is None"
        for mention in message.mentions:
            if (self.user.id == mention.id):
                return True
        return False

    async def processMessage(self, message: Message):
        if (message.guild):
            assert isinstance(message.author, Member), "Message author is not a Member"
            username = message.author.nick or await self.getName(message.author.id, message.guild)
            content = await self.cleanContent(message.content, message.guild)
        else:
            username = str(message.author)
            content = await self.cleanContent(message.content, None)

        processed: dict[str, Any] = {"user": username, "message": content}
        processed["messageId"] = str(message.id)
        images = await self.extract_images(message)
        if images:
            processed["images"] = images
        return processed

    async def extract_images(self, message: Message) -> List[dict[str, Any]]:
        """
        Gather all image attachments or preview images for the provided message.
        Downloads attachments to history/images for replayable context.
        """
        images: List[dict[str, Any]] = []
        channel_id = message.channel.id

        for attachment in getattr(message, "attachments", []):
            if not is_supported_image_mime(attachment.content_type, attachment.filename):
                continue
            record: dict[str, Any] = {
                "source": "discord_attachment",
                "url": attachment.url,
                "filename": attachment.filename,
            }
            try:
                local_path = await download_image_to_history(channel_id, attachment.url, attachment.filename)
                record["local_path"] = os.path.relpath(local_path).replace("\\", "/")
            except Exception as e:
                print(f"Failed to cache attachment {attachment.filename}: {e}")
            images.append(record)

        for embed in getattr(message, "embeds", []):
            urls = []
            image_obj = getattr(embed, "image", None)
            if image_obj and image_obj.url:
                urls.append(image_obj.url)
            thumbnail = getattr(embed, "thumbnail", None)
            if thumbnail and thumbnail.url:
                urls.append(thumbnail.url)
            if embed.type == "image" and embed.url:
                urls.append(embed.url)
            for url in urls:
                if not url:
                    continue
                record = build_remote_image_record(
                    url,
                    source="discord_embed",
                    description=getattr(embed, "title", None) or getattr(embed, "description", None),
                )
                images.append(record)

        return images

    async def getName(self, id: int, guild: Guild) -> str | None:
        assert self.user is not None, "User is None"
        if id == self.user.id:
            return "{{char}}"
        try:
            member = await guild.fetch_member(id)
            if member.nick:
                return member.nick
        except:
            pass
        try:
            profile = await self.fetch_user_profile(id)
            if profile.name:
                return profile.name
        except:
            pass
        user = await self.fetch_user(id)
        if user.name:
            return user.name

    async def cleanContent(self, content: str, guild: Guild | None) -> str:
        mentions = re.compile(r'<@([\d]+)>')
        result = content
        if guild:
            for match in mentions.finditer(content):
                user_id = match.group(1)
                username: str | None = await self.getName(int(user_id), guild)
                if username:
                    result = result.replace(match.group(0), f"@{username}")
        emoji = re.compile(r'<[\w]*:([\w]+):[\d]+>')
        for match in emoji.finditer(content):
            result = result.replace(match.group(0), f":{match.group(1)}:")

        return result

    def _is_mention_boundary_char(self, ch: str) -> bool:
        return ch.isspace() or ch in "\"'`,!?;:()[]{}<>"

    def _get_mention_indexes_for_guild(self, guild: Guild) -> tuple[dict[str, int], dict[str, int]]:
        now = time.time()
        cached = self.mentionIndexCache.get(guild.id)
        if cached and now - cached[0] < 300:
            return cached[1], cached[2]

        members = list(getattr(guild, "members", []) or [])
        display_counts: dict[str, int] = {}
        username_counts: dict[str, int] = {}
        for member in members:
            display = getattr(member, "display_name", None)
            if isinstance(display, str) and display:
                key = display.casefold()
                display_counts[key] = display_counts.get(key, 0) + 1
            username = getattr(member, "name", None)
            if isinstance(username, str) and username:
                key = username.casefold()
                username_counts[key] = username_counts.get(key, 0) + 1

        display_index: dict[str, int] = {}
        username_index: dict[str, int] = {}
        for member in members:
            display = getattr(member, "display_name", None)
            if isinstance(display, str) and display:
                key = display.casefold()
                if display_counts.get(key) == 1:
                    display_index[key] = member.id
            username = getattr(member, "name", None)
            if isinstance(username, str) and username:
                key = username.casefold()
                if username_counts.get(key) == 1:
                    username_index[key] = member.id

        self.mentionIndexCache[guild.id] = (now, display_index, username_index)
        return display_index, username_index

    async def _resolve_user_id_for_generated_at_mention(self, guild: Guild, name: str) -> int | None:
        name = name.strip()
        if not name:
            return None
        if name.casefold() in {"everyone", "here"}:
            return None

        display_index, username_index = self._get_mention_indexes_for_guild(guild)
        key = name.casefold()
        user_id = display_index.get(key)
        if user_id:
            return user_id

        user_id = username_index.get(key)
        if user_id:
            return user_id

        query_members = getattr(guild, "query_members", None)
        if not callable(query_members):
            return None

        try:
            results = await query_members(name, limit=10) # type: ignore
        except Exception:
            return None

        exact_display = [m for m in results if getattr(m, "display_name", "").casefold() == key]
        if len(exact_display) == 1:
            return exact_display[0].id

        exact_username = [m for m in results if getattr(m, "name", "").casefold() == key]
        if len(exact_username) == 1:
            return exact_username[0].id

        return None

    async def resolve_generated_at_mentions(self, content: str, guild: Guild | None) -> str:
        if not guild or "@" not in content:
            return content

        out: list[str] = []
        i = 0
        while i < len(content):
            ch = content[i]
            if ch != "@":
                out.append(ch)
                i += 1
                continue

            # Skip real Discord mentions like <@123> / <@!123>
            if i > 0 and content[i - 1] == "<":
                out.append(ch)
                i += 1
                continue

            if i > 0 and not self._is_mention_boundary_char(content[i - 1]):
                out.append(ch)
                i += 1
                continue

            if i + 1 >= len(content) or content[i + 1].isspace():
                out.append(ch)
                i += 1
                continue

            tail = content[i + 1 :]
            stop_match = re.match(r"^[^\r\n]+", tail)
            raw_segment = (stop_match.group(0) if stop_match else tail)[:64].strip()
            if not raw_segment:
                out.append(ch)
                i += 1
                continue

            words = raw_segment.split()
            if not words:
                out.append(ch)
                i += 1
                continue

            # Prefer single-token mentions (e.g. @Alice). If that doesn't resolve, fall back to
            # capturing one more word to support two-word names (e.g. @Alice Smith).
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
                    if after_idx < len(content) and not self._is_mention_boundary_char(content[after_idx]):
                        continue

                    user_id = await self._resolve_user_id_for_generated_at_mention(guild, candidate)
                    if not user_id:
                        continue

                    out.append(f"<@{user_id}>")
                    i = after_idx
                    replaced = True
                    break

                if replaced:
                    break

            if not replaced:
                out.append(ch)
                i += 1

        return "".join(out)

    def _control_reactions_path(self, channel_id: int) -> str:
        return os.path.join("history", "control-reactions", f"{channel_id}.json")

    def _load_control_reactions(self, channel_id: int) -> list[dict[str, str]]:
        path = self._control_reactions_path(channel_id)
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
                    if not isinstance(message_id, str) or not message_id:
                        continue
                    if not isinstance(emoji, str) or not emoji:
                        continue
                    out.append({"messageId": message_id, "emoji": emoji})
                return out
        except Exception:
            pass
        return []

    def _save_control_reactions(self, channel_id: int, reactions: list[dict[str, str]]) -> None:
        path = self._control_reactions_path(channel_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"reactions": reactions}, f, indent=2)

    def _unrecord_control_reaction(self, channel_id: int, message_id: int, emoji: str) -> None:
        existing = self._load_control_reactions(channel_id)
        message_id_str = str(message_id)
        emoji_str = str(emoji)
        filtered = [
            r
            for r in existing
            if not (r.get("messageId") == message_id_str and r.get("emoji") == emoji_str)
        ]
        if len(filtered) != len(existing):
            self._save_control_reactions(channel_id, filtered)

    def _record_control_reactions(self, channel_id: int, message_id: int, emojis: list[str]) -> None:
        if not emojis:
            return
        existing = self._load_control_reactions(channel_id)
        existing_set = {(r.get("messageId"), r.get("emoji")) for r in existing}
        message_id_str = str(message_id)
        for emoji in emojis:
            emoji_str = str(emoji)
            key = (message_id_str, emoji_str)
            if key in existing_set:
                continue
            existing.append({"messageId": message_id_str, "emoji": emoji_str})
            existing_set.add(key)
        self._save_control_reactions(channel_id, existing)

    async def _set_control_reaction(
        self,
        channel_id: int,
        msg: Message,
        emoji: str,
        should_have: bool,
    ) -> None:
        assert self.user is not None, "User is None"
        emoji_str = str(emoji)
        if not emoji_str:
            return

        if should_have:
            try:
                await msg.add_reaction(emoji_str)
                self._record_control_reactions(channel_id, msg.id, [emoji_str])
            except Exception:
                return
            return

        try:
            await msg.remove_reaction(emoji_str, self.user)
            self._unrecord_control_reaction(channel_id, msg.id, emoji_str)
        except Exception as e:
            not_found = getattr(discord, "NotFound", None)
            if not_found and isinstance(e, not_found):
                self._unrecord_control_reaction(channel_id, msg.id, emoji_str)
            return

    async def _cleanup_recorded_control_reactions(
        self,
        channel: discord.abc.Messageable,
        channel_id: int,
    ) -> None:
        """
        Removes any reactions previously recorded as added by the bot in this channel,
        and clears those records from disk.
        """
        assert self.user is not None, "User is None"

        records = self._load_control_reactions(channel_id)
        if not records:
            return

        fetch_message = getattr(channel, "fetch_message", None)
        if not callable(fetch_message):
            return

        by_message: dict[str, list[str]] = {}
        for rec in records:
            by_message.setdefault(rec["messageId"], []).append(rec["emoji"])

        remaining: list[dict[str, str]] = []
        for message_id_str, emojis in by_message.items():
            try:
                message_id = int(message_id_str)
            except Exception:
                continue

            try:
                msg: Message = await channel.fetch_message(message_id)  # type: ignore
            except Exception:
                for emoji in emojis:
                    remaining.append({"messageId": message_id_str, "emoji": emoji})
                continue

            for emoji in emojis:
                try:
                    await msg.remove_reaction(emoji, self.user)
                except Exception as e:
                    print(f"[{channel_id}] Failed to remove recorded reaction {emoji} from {message_id}: {e}")
                    remaining.append({"messageId": message_id_str, "emoji": emoji})

        self._save_control_reactions(channel_id, remaining)
    
    def addToQueue(self, channelID: int, message: dict[str, Any]):
        extra = ""
        if message.get("images"):
            extra = f" [{len(message['images'])} image(s)]"
        print(f"[{channelID}] {message['user']}: {message['message']}{extra}")
        queue = self.pendingMessages.get(channelID, [])
        queue.append(message)
        self.pendingMessages[channelID] = queue

    def _coerce_id_set(self, values: Any) -> set[str]:
        return normalize_id_set(values)

    def _swipe_allowed(self, swipes_cfg: dict[str, Any], user_id: int, channel_id: int) -> bool:
        user_whitelist = self._coerce_id_set(swipes_cfg.get("user_whitelist"))
        channel_whitelist = self._coerce_id_set(swipes_cfg.get("channel_whitelist"))
        user_key = str(user_id)
        channel_key = str(channel_id)

        if not user_whitelist and not channel_whitelist:
            return True
        if not user_whitelist:
            return channel_key in channel_whitelist
        if not channel_whitelist:
            return user_key in user_whitelist
        return (user_key in user_whitelist) or (channel_key in channel_whitelist)

    def _on_channel_task_done(self, channelID: int, task: asyncio.Task[None]) -> None:
        self._inflight_channels.discard(channelID)
        self._inflight_tasks.pop(channelID, None)
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[{channelID}] Background channel task failed: {e}")
    
    async def process_messages(self):
        channelIDs = list(set(self.pendingMessages.keys()) | set(self.pendingSwipes.keys()))
        
        for channelID in channelIDs:
            if channelID in self._inflight_channels:
                continue
            pending = self.pendingMessages.get(channelID, [])
            swipe_jobs = self.pendingSwipes.get(channelID, [])
            if len(pending) == 0 and len(swipe_jobs) == 0:
                continue
                
            self.pendingMessages[channelID] = []
            self.pendingSwipes[channelID] = []
            channel = self.get_channel(channelID)
            if not isinstance(channel, discord.abc.Messageable):
                print(f"Channel {channelID} is not a Messageable")
                continue
                
            # Create coroutine but don't await it yet
            async def process_channel(
                channel: discord.abc.Messageable,
                channelID: int,
                messages: List[dict[str, Any]],
                swipe_jobs: list[tuple[int, str]],
            ):
                # DMs/group DMs have no guild.
                is_dm_channel = getattr(channel, "guild", None) is None
                async with channel.typing():
                    if swipe_jobs:
                        swipes_cfg = get_config().get("swipes", {}) or {}
                        prev_emoji = str(swipes_cfg.get("prev_emoji", "◀️"))
                        next_emoji = str(swipes_cfg.get("next_emoji", "▶️"))

                        last_by_message: dict[int, tuple[int, str]] = {}
                        for i, (message_id, action) in enumerate(swipe_jobs):
                            last_by_message[message_id] = (i, action)

                        jobs_to_run = [
                            (message_id, action)
                            for message_id, (i, action) in sorted(
                                last_by_message.items(), key=lambda kv: kv[1][0]
                            )
                        ]

                        fetch_message = getattr(channel, "fetch_message", None)
                        if callable(fetch_message):
                            for message_id, action in jobs_to_run:
                                try:
                                    target_message: Message = await channel.fetch_message(message_id) # type: ignore
                                except Exception as e:
                                    print(f"[{channelID}] Swipe fetch_message failed for {message_id}: {e}")
                                    continue

                                assert self.user is not None, "User is None"
                                if target_message.author.id != self.user.id:
                                    continue

                                new_text: str | None = None
                                try:
                                    if action == "regen":
                                        new_text = await swipe_regenerate(
                                            channelID,
                                            str(message_id),
                                            is_dm=is_dm_channel,
                                        )
                                    elif action == "prev":
                                        new_text = swipe_prev(channelID, str(message_id))
                                    elif action == "next":
                                        new_text = swipe_next(channelID, str(message_id))
                                except Exception as e:
                                    print(f"[{channelID}] Swipe action failed for {message_id}: {e}")
                                    continue

                                if new_text:
                                    cleaned = clean_response(new_text)
                                    guild = getattr(target_message, "guild", None)
                                    resolved = await self.resolve_generated_at_mentions(
                                        cleaned,
                                        guild if isinstance(guild, Guild) else None,
                                    )
                                    try:
                                        await target_message.edit(content=resolved)
                                    except Exception as e:
                                        print(f"[{channelID}] Swipe edit failed for {message_id}: {e}")

                                try:
                                    nav = get_swipe_nav_state(channelID, str(message_id))
                                    if nav is not None:
                                        has_prev, has_next = nav
                                        await self._set_control_reaction(channelID, target_message, prev_emoji, has_prev)
                                        await self._set_control_reaction(channelID, target_message, next_emoji, has_next)
                                except Exception as e:
                                    print(f"[{channelID}] Swipe reaction update failed for {message_id}: {e}")
                        else:
                            print(f"[{channelID}] Channel does not support fetch_message; skipping swipe jobs")

                    if len(messages) == 0:
                        # Nothing more to do
                        return
                    try:
                        inference_result = await chat_inference(
                            channelID,
                            messages,
                            is_dm=is_dm_channel,
                        )
                    except Exception as e:
                        print(e)
                        inference_result = None

                response: str | None = None
                pending_message_id: str | None = None
                if isinstance(inference_result, tuple) and len(inference_result) == 2:
                    response = inference_result[0]
                    pending_message_id = inference_result[1]
                elif isinstance(inference_result, str):
                    response = inference_result

                if response and len(response) > 0:
                    cleaned = clean_response(response)
                    guild = getattr(channel, "guild", None)
                    resolved = await self.resolve_generated_at_mentions(
                        cleaned,
                        guild if isinstance(guild, Guild) else None,
                    )
                    sent = await channel.send(resolved)
                    try:
                        if pending_message_id:
                            finalize_assistant_message_id(channelID, pending_message_id, str(sent.id))
                        else:
                            finalize_last_assistant_message_id(channelID, str(sent.id))
                    except Exception as e:
                        print(f"[{channelID}] Failed to store assistant messageId: {e}")

                    swipes_cfg = get_config().get("swipes", {}) or {}
                    try:
                        await self._cleanup_recorded_control_reactions(channel, channelID)
                    except Exception as e:
                        print(f"[{channelID}] Swipe reaction cleanup failed: {e}")

                    if swipes_cfg.get("auto_react_controls", False):
                        auto_react_channel_whitelist = self._coerce_id_set(
                            swipes_cfg.get("auto_react_channel_whitelist")
                        )
                        should_auto_react = (not auto_react_channel_whitelist) or (
                            str(channelID) in auto_react_channel_whitelist
                        )
                        if should_auto_react:
                            added: list[str] = []
                            try:
                                regen_emoji = str(swipes_cfg.get("regen_emoji", "🔄"))
                                await sent.add_reaction(regen_emoji)
                                added.append(regen_emoji)
                            except Exception as e:
                                print(f"[{channelID}] Failed to auto-react swipe controls: {e}")

                            try:
                                self._record_control_reactions(channelID, sent.id, added)
                            except Exception as e:
                                print(f"[{channelID}] Failed to record control reactions: {e}")
                else:
                    print("No response")
                    
            task = asyncio.create_task(
                process_channel(channel, channelID, pending, swipe_jobs)
            )
            self._inflight_channels.add(channelID)
            self._inflight_tasks[channelID] = task
            task.add_done_callback(
                lambda done_task, cid=channelID: self._on_channel_task_done(
                    cid, done_task
                )
            )

    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent):
        assert self.user is not None, "User is None"
        if payload.user_id == self.user.id:
            return

        config = get_config().get("swipes", {}) or {}
        if not config.get("enabled", False):
            return

        if not self._swipe_allowed(config, payload.user_id, payload.channel_id):
            return

        emoji_text = str(payload.emoji)
        regen_emoji = config.get("regen_emoji", "🔄")
        prev_emoji = config.get("prev_emoji", "◀️")
        next_emoji = config.get("next_emoji", "▶️")

        action: str | None = None
        if emoji_text == regen_emoji:
            action = "regen"
        elif emoji_text == prev_emoji:
            action = "prev"
        elif emoji_text == next_emoji:
            action = "next"
        else:
            return
        print(f"Swipe action detected: {action}")
        queue = self.pendingSwipes.get(payload.channel_id, [])
        queue.append((payload.message_id, action))
        self.pendingSwipes[payload.channel_id] = queue
    
    async def inference_loop_task(self):
        await self.wait_until_ready()
        try:
            while not self.is_closed():
                await self.process_messages()
                await asyncio.sleep(0.5)
        finally:
            for task in list(self._inflight_tasks.values()):
                task.cancel()
            if self._inflight_tasks:
                await asyncio.gather(*self._inflight_tasks.values(), return_exceptions=True)
            self._inflight_tasks.clear()
            self._inflight_channels.clear()

def is_whitelisted(channelID: int, wtype: str = 'always'):
    config = get_config()
    whitelist = config["whitelist"][wtype]
    return is_whitelisted_id(channelID, whitelist)

def clean_response(resp: str) -> str:
    resp = html.unescape(resp)
    resp = resp.strip()
    return dequote(resp)
