import re
import random
import datetime
from typing import List
import html
from dataclasses import dataclass

import asyncio
import discord
from discord import Guild, Member, Message

from .openai_inference import chat_inference
from .utils import get_config, dequote

@dataclass
class RandomChat:
    endTime: datetime.datetime
    nextChatTime: datetime.datetime
    lastMessage: Message

class Bot(discord.Client):
    pendingMessages: dict[int, List[dict[str, str]]] = {}

    randomChats: dict[int, RandomChat] = {}

    async def on_ready(self):
        print(f'Logged on as {self.user}!')

    async def setup_hook(self) -> None:
        # create the background task and run it in the background
        self.bg_task = self.loop.create_task(self.inference_loop_task())

    async def on_message(self, message: Message):
        print(f'Message in {message.channel.id} from {message.author}: {message.content}')
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

        return {"user": username, "message": content}

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
                if (username):
                    result = result.replace(match.group(0), f"@{username}")
        emoji = re.compile(r'<[\w]*:([\w]+):[\d]+>')
        for match in emoji.finditer(content):
            result = result.replace(match.group(0), f":{match.group(1)}:")

        return result
    
    def addToQueue(self, channelID: int, message: dict[str, str]):
        queue = self.pendingMessages.get(channelID, [])
        queue.append(message)
        self.pendingMessages[channelID] = queue
    
    async def process_messages(self):
        channelIDs = list(self.pendingMessages.keys())
        for channelID in channelIDs:
            pending = self.pendingMessages.get(channelID, [])
            if len(pending) == 0:
                continue
            self.pendingMessages[channelID] = []
            channel = self.get_channel(channelID)
            if not isinstance(channel, discord.abc.Messageable):
                print(f"Channel {channelID} is not a Messageable")
                continue
            async with channel.typing():
                try:
                    response = await chat_inference(channelID, pending)
                except Exception as e:
                    print(e)
                    response = ""
            if response and len(response) > 0:
                await channel.send(clean_response(response))
            else:
                print("No response")
    
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