import re
import random
import datetime
import time
from typing import Any, List
import html
import os
from dataclasses import dataclass

import asyncio
import discord
from discord import Guild, Member, Message

from .chat_image_utils import (
    build_remote_image_record,
    download_image_to_history,
    is_supported_image_mime,
)
from .openai_inference import chat_inference
from .utils import get_config, dequote

@dataclass
class RandomChat:
    endTime: datetime.datetime
    nextChatTime: datetime.datetime
    lastMessage: Message

class Bot(discord.Client):
    pendingMessages: dict[int, List[dict[str, Any]]] = {}

    randomChats: dict[int, RandomChat] = {}

    # guild_id -> (timestamp, display_name_index, username_index)
    mentionIndexCache: dict[int, tuple[float, dict[str, int], dict[str, int]]] = {}

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
        return ch.isspace() or ch in "\"'`.,!?;:()[]{}<>"

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
            stop_match = re.match(r"^[^\r\n,!.?:;]+", tail)
            raw_segment = (stop_match.group(0) if stop_match else tail).strip()
            if not raw_segment:
                out.append(ch)
                i += 1
                continue

            # Limit how far we try to interpret a name if no punctuation ends it.
            words = raw_segment.split()
            if len(words) > 4:
                words = words[:4]

            replaced = False
            for k in range(len(words), 0, -1):
                candidate = " ".join(words[:k]).rstrip("\"'`.,!?;:()[]{}<>")
                if not candidate:
                    continue

                if not tail[: len(candidate)].casefold() == candidate.casefold():
                    continue

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

            if not replaced:
                out.append(ch)
                i += 1

        return "".join(out)
    
    def addToQueue(self, channelID: int, message: dict[str, Any]):
        extra = ""
        if message.get("images"):
            extra = f" [{len(message['images'])} image(s)]"
        print(f"[{channelID}] {message['user']}: {message['message']}{extra}")
        queue = self.pendingMessages.get(channelID, [])
        queue.append(message)
        self.pendingMessages[channelID] = queue
    
    async def process_messages(self):
        channelIDs = list(self.pendingMessages.keys())
        tasks = []
        
        for channelID in channelIDs:
            pending = self.pendingMessages.get(channelID, [])
            if len(pending) == 0:
                continue
                
            self.pendingMessages[channelID] = []
            channel = self.get_channel(channelID)
            if not isinstance(channel, discord.abc.Messageable):
                print(f"Channel {channelID} is not a Messageable")
                continue
                
            # Create coroutine but don't await it yet
            async def process_channel(channel: discord.abc.Messageable, channelID: int, messages: List[dict[str, Any]]):
                async with channel.typing():
                    try:
                        response = await chat_inference(channelID, messages)
                    except Exception as e:
                        print(e)
                        response = None

                if response and len(response) > 0:
                    cleaned = clean_response(response)
                    guild = getattr(channel, "guild", None)
                    resolved = await self.resolve_generated_at_mentions(
                        cleaned,
                        guild if isinstance(guild, Guild) else None,
                    )
                    await channel.send(resolved)
                else:
                    print("No response")
                    
            tasks.append(process_channel(channel, channelID, pending))
        
        # Run all tasks concurrently
        await asyncio.gather(*tasks)
    
    async def inference_loop_task(self):
        await self.wait_until_ready()
        while not self.is_closed():
            await self.process_messages()
            await asyncio.sleep(0.5)

def is_whitelisted(channelID: int, wtype: str = 'always'):
    config = get_config()
    whitelist: list[int] = config["whitelist"][wtype]

    return channelID in whitelist

def clean_response(resp: str) -> str:
    resp = html.unescape(resp)
    resp = resp.strip()
    return dequote(resp)
