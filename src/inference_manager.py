# inference_manager.py

import json
import os
import asyncio
import multiprocessing
from multiprocessing import Queue, Manager
from typing import List, Dict, Tuple
from openai_inference import InferenceClient

class InferenceManager:
    def __init__(self, input_queue: Queue, output_queues: Dict[Tuple[str, str], Queue]):
        self.input_queue = input_queue
        self.output_queues = output_queues  # Key: (bot_name, channel_id)
        self.client = InferenceClient()

    async def process_messages(self, grouped_messages: Dict[Tuple[str, str], List[dict]]):
        tasks = []
        for (bot_name, channel_id), messages in grouped_messages.items():
            task = asyncio.create_task(self.handle_channel(bot_name, channel_id, messages))
            tasks.append(task)
        await asyncio.gather(*tasks)

    async def handle_channel(self, bot_name: str, channel_id: str, formatted_history: List[dict]):
        # Run inference
        reply = await self.client.run_inference(formatted_history)
        if reply is None:
            return
        
        self.client.append_history_reply(channel_id, botname=bot_name, reply=reply)

        # Clean the message
        cleaned_message = self.client.clean_generated_message(reply)

        # Dispatch the response
        key = (bot_name, channel_id)
        if key in self.output_queues:
            self.output_queues[key].put(cleaned_message)
        else:
            print(f"No output queue found for {key}")

    async def prepare_channel_history(self, bot_name: str, channel_id: str, new_messages: List[dict]):
        # Load history and append new messages
        history = self.client.load_history(channel_id, botname=bot_name)
        history.extend(new_messages)
        self.client.save_history(channel_id, botname=bot_name, history=history)
        # Format chat
        return self.client.format_chat(history)

    async def main_loop(self):
        while True:
            # Gather all messages currently in the queue
            messages = []
            while not self.input_queue.empty():
                item = self.input_queue.get()
                if item is None:
                    continue  # Skip None items
                messages.append(item)

            if messages:
                # Group messages by (bot_name, channel_id)
                grouped = {}
                for msg in messages:
                    bot_name = msg['bot_name']
                    channel_id = str(msg['channel_id'])
                    key = (bot_name, channel_id)
                    if key not in grouped:
                        grouped[key] = []
                    grouped[key].append({
                        'user': msg['user'],
                        'message': msg['message']
                    })
                grouped_with_history = {}
                for key, messages in grouped.items():
                    bot_name, channel_id = key
                    formatted_messages = await self.prepare_channel_history(bot_name, channel_id, messages)
                    grouped_with_history[key] = formatted_messages
                    
                # Process all grouped messages
                await self.process_messages(grouped_with_history)

            # Wait for 0.5 seconds before next check
            await asyncio.sleep(0.5)

def inference_manager_process(input_queue: Queue, output_queues: Dict[Tuple[str, str], Queue]):
    manager = InferenceManager(input_queue, output_queues)
    asyncio.run(manager.main_loop())
