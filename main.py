from dotenv import load_dotenv
import os

from src.discord_chatbot import Bot

if __name__ == '__main__':
    load_dotenv()
    client = Bot()
    client.run(os.environ['DISCORD_TOKEN'])