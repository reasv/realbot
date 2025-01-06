import os
from dotenv import load_dotenv
import multiprocessing

from src.discord_chatbot import Bot
from src.irc_chatbot import run_bot

def start_discord_bot():
    load_dotenv()
    client = Bot()
    client.run(os.environ['DISCORD_TOKEN'])

def start_irc_bot():
    load_dotenv()
    run_bot()

if __name__ == '__main__':
    # Create processes for each bot
    discord_process = multiprocessing.Process(target=start_discord_bot)
    irc_process = multiprocessing.Process(target=start_irc_bot)
    
    # Start both processes
    discord_process.start()
    irc_process.start()
    # Wait for both processes to complete (they won't normally complete)
    discord_process.join()
    irc_process.join()