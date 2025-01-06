import asyncio
import os
import random
import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

from dotenv import load_dotenv
import irc.client
import irc.client_aio
import irc.connection

from .openai_inference import chat_inference
from .utils import get_config, dequote

@dataclass
class RandomChat:
    """
    Tracks a "random chat" session in a channel.
    """
    endTime: datetime.datetime
    nextChatTime: datetime.datetime
    lastMessage: dict  # e.g. {"nick": str, "content": str}


class AioIrcBot(irc.client_aio.AioSimpleIRCClient):
    def __init__(self, nickname: str, channels: List[str], use_ssl: bool = True):
        super().__init__()
        self.nickname = nickname
        self.channels = channels
        self.use_ssl = use_ssl

        # Holds inbound messages waiting for inference:
        #   { "#channel": [ {"user": "...", "message": "..."}, ... ] }
        self.pendingMessages: Dict[str, List[dict]] = {}

        # Tracks random chat sessions: { "#channel": RandomChat(...) }
        self.randomChats: Dict[str, RandomChat] = {}

        # A handle for our background task (launched on_welcome).
        self.bg_task: Optional[asyncio.Task] = None

    # ----------------------------------------------------------------------
    # Overridden IRC handlers
    # ----------------------------------------------------------------------
    async def connect(self, server: str, port: int, nickname: str): # type: ignore
        """
        Connect using AioFactory with SSL if enabled
        """
        factory = irc.connection.AioFactory(ssl=self.use_ssl)
        try:
            await self.connection.connect(
                server=server,
                port=port,
                nickname=nickname,
                connect_factory=factory # type: ignore
            ) # type: ignore
        except Exception as e:
            print(f"[Bot] Connection failed: {e}")
            raise

    def on_welcome(self, connection, event):
        """
        Triggered upon successful connection. Let's join channels, and
        start the background task for inference.
        """
        print(f"[Bot] Connected as {self.nickname}, joining channels...")

        # Join each channel
        for ch in self.channels:
            connection.join(ch)

        # Launch background loop (only once).
        if not self.bg_task:
            # self.reactor.loop is the asyncio event loop used by this library.
            print("[Bot] Starting background inference loop.")
            self.bg_task = asyncio.ensure_future(
                self._background_loop(),
                loop=self.reactor.loop # type: ignore
            )

    def on_join(self, connection, event):
        """
        Called when a user (including us) joins a channel.
        """
        nick = event.source.nick
        channel = event.target
        if nick == self.nickname:
            print(f"[Bot] Joined channel {channel}")

    def on_disconnect(self, connection, event):
        """
        Called when the bot is disconnected from the server.
        """
        print("[Bot] Disconnected.")
        if self.bg_task:
            self.bg_task.cancel()
            self.bg_task = None

        # Stop the reactor loop so `start()` returns.
        try:
            self.reactor.loop.stop() # type: ignore
        except Exception:
            pass

    def on_pubmsg(self, connection, event):
        """
        Called when a public message is posted in a channel.
        """
        channel = event.target
        nick = event.source.nick
        text = event.arguments[0].strip()

        if not text or nick == self.nickname:
            return  # ignore empty or self

        # If channel isn't whitelisted for any category, do nothing
        if not any(self.is_whitelisted(channel, wtype) for wtype in ("always", "mentions", "rand")):
            return

        # Build a message-like dict
        msg_dict = {"nick": nick, "content": text}

        # Mentions
        if self.is_whitelisted(channel, "mentions") and self.is_mentioned(text):
            return self.constant_chat(channel, msg_dict)

        # Always
        if self.is_whitelisted(channel, "always"):
            return self.constant_chat(channel, msg_dict)

        # Random
        if self.is_whitelisted(channel, "rand"):
            return self.random_chat(channel, msg_dict)

    # ----------------------------------------------------------------------
    # Whitelist / Mentions
    # ----------------------------------------------------------------------

    def is_whitelisted(self, channel: str, wtype: str) -> bool:
        """
        Check if channel is in the config's whitelist for the given type.
        """
        cfg = get_config()
        channels = cfg.get("whitelist", {}).get(wtype, [])
        return channel in channels

    def is_mentioned(self, text: str) -> bool:
        """Check if our nickname is in the text."""
        return self.nickname.lower() in text.lower()

    # ----------------------------------------------------------------------
    # Chat Behaviors
    # ----------------------------------------------------------------------

    def constant_chat(self, channel: str, message: dict):
        """
        Always respond to this message.
        """
        processed = self.process_incoming_message(message)
        self.add_to_queue(channel, processed)

    def random_chat(self, channel: str, message: dict):
        """
        Possibly respond (start or continue a random chat) per the randomChat config.
        """
        cfg = get_config().get("randomChat", {})
        now = datetime.datetime.now()
        current = self.randomChats.get(channel)

        if current:
            # If a session is ongoing
            if now < current.endTime:
                # Continue session
                current.lastMessage = message
                processed = self.process_incoming_message(message)
                self.add_to_queue(channel, processed)
                return
            else:
                # Session ended; see if we can start a new one yet
                if now < current.nextChatTime:
                    print(f"[{channel}] Not starting new random chat until {current.nextChatTime}")
                    return

        # Possibly start a new session with some probability
        if random.randint(0, cfg.get("engagement_chance", 10)) != 0:
            # If roll fails, see if user mentioned us and config allows responding anyway
            if self.is_mentioned(message["content"]) and cfg.get("respond_to_mentions", False):
                pass
            else:
                print(f"[{channel}] Random chat roll failed.")
                return

        # Start a new random chat
        processed = self.process_incoming_message(message)
        self.add_to_queue(channel, processed)

        # Mark session bounds
        min_duration = cfg.get("min_chat_duration_seconds", 60)
        max_duration = cfg.get("max_chat_duration_seconds", 300)
        endTime = now + datetime.timedelta(seconds=random.randint(min_duration, max_duration))

        min_down = cfg.get("min_downtime_minutes", 5)
        max_down = cfg.get("max_downtime_minutes", 15)
        nextChatTime = endTime + datetime.timedelta(minutes=random.randint(min_down, max_down))

        self.randomChats[channel] = RandomChat(
            endTime=endTime,
            nextChatTime=nextChatTime,
            lastMessage=message,
        )
        print(f"[{channel}] Started random chat until {endTime}. Next earliest: {nextChatTime}")

    # ----------------------------------------------------------------------
    # Queue & Inference
    # ----------------------------------------------------------------------

    def process_incoming_message(self, message: dict) -> dict:
        """
        Convert an IRC message dict into the structure expected by chat_inference.
        """
        user = message["nick"]
        content = self.clean_content(message["content"])
        return {"user": user, "message": content}

    def clean_content(self, content: str) -> str:
        """
        If you want to emulate Discord-style mention/emoji conversions, do so here.
        """
        return content.strip()

    def add_to_queue(self, channel: str, msg: dict):
        """
        Store message in the channel's queue to be handled by the background loop.
        """
        self.pendingMessages.setdefault(channel, []).append(msg)

    async def _background_loop(self):
        """
        Continuously processes queued messages every 0.5s.  
        We do this in the same event loop used by the IRC client.
        """
        print("[Bot] Background inference loop started.")
        while True:
            await self.process_queued_messages_once()
            await asyncio.sleep(0.5)

    async def process_queued_messages_once(self):
        """
        For each channel's queued messages, call chat_inference and send a response.
        """
        # We'll snapshot channel list so we don't modify the dictionary while iterating
        channels = list(self.pendingMessages.keys())
        for channel in channels:
            pending = self.pendingMessages.get(channel, [])
            if not pending:
                continue

            # Clear out the queue
            self.pendingMessages[channel] = []

            try:
                # Await your async inference call
                response = await chat_inference(channel, pending)
            except Exception as e:
                print(f"[{channel}] chat_inference error: {e}")
                response = ""
            
            if response:
                response = self.clean_response(response)
                # Send message back to IRC. Note that this is synchronous, but it's safe
                # in this library as we are still within the same event loop context.
                self.connection.privmsg(channel, response)

    def clean_response(self, resp: str) -> str:
        """
        Final cleanup for the model's response (e.g. HTML unescape, dequote).
        """
        return dequote(resp).strip()


# ------------------------------------------------------------------------------
# Running the Bot
# ------------------------------------------------------------------------------

def run_bot():
    config = get_config_from_env()
    print(f"Connecting to {config['server']}:{config['port']} as {config['nickname']} in {config['channel']}")
    
    use_ssl = config.get('use_ssl', True)
    bot = AioIrcBot(
        nickname=config["nickname"], 
        channels=[config["channel"]], 
        use_ssl=use_ssl
    )
    loop = None
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            bot.connect(config["server"], config["port"], config["nickname"])
        )
        bot.start()
    except KeyboardInterrupt:
        print("[Bot] Received keyboard interrupt, shutting down...")
    except Exception as e:
        print(f"[Bot] Error: {e}")
    finally:
        if hasattr(bot, 'connection') and bot.connection:
            bot.connection.disconnect()
        if loop and loop.is_running():
            loop.stop()

def get_config_from_env():
    load_dotenv()
    channels = os.getenv("IRC_CHANNELS", "#ai").split(",")
    return {
        "server": os.getenv("IRC_SERVER", "irc.freenode.net"),
        "port": int(os.getenv("IRC_PORT", "6697")),
        "channel": channels[0],
        "nickname": os.getenv("IRC_NICKNAME", "SimpleBot124")
    }