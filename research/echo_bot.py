import logging
import os
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load environment variables from .env file
load_dotenv()

# Get the Telegram API token from the .env file
TELEGRAM_API_TOKEN = os.getenv("TELEGRAM_API_TOKEN")

# Debugging step: check if the token is loaded
if TELEGRAM_API_TOKEN is None:
    print("Error: TELEGRAM_API_TOKEN is not set in the .env file.")
else:
    print(f"Loaded Telegram API Token: {TELEGRAM_API_TOKEN}")

# Set up logging to help debug if needed
logging.basicConfig(level=logging.INFO)

# Initialize the Telegram bot
bot = Bot(token=TELEGRAM_API_TOKEN)

# Initialize the Dispatcher
dp = Dispatcher()

# Load GPT-2 model
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Create a dictionary to store the conversation history for each user
user_contexts = {}

# Function to generate a response using GPT-2
def generate_text(user_id, prompt):
    # Retrieve the user's conversation history or initialize it
    context = user_contexts.get(user_id, "")
    
    # Combine the conversation history with the new prompt
    combined_input = context + prompt
    
    # Encode the input with context
    inputs = tokenizer.encode(combined_input, return_tensors="pt")
    
    # Generate the response from the model
    outputs = model.generate(inputs, max_length=500, num_return_sequences=1, no_repeat_ngram_size=2)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Update the user's conversation history with the prompt (not the response)
    user_contexts[user_id] = combined_input + response + "\n"  # Store history
    
    # Return only the response part (not the entire context)
    return response[len(combined_input):].strip()


# /start command handler using the filters.Command
@dp.message(Command("start"))
async def send_welcome(message: types.Message):
    await message.reply("Hello! I am your AI-powered Telegram bot. Type anything, and I will reply.")

# /help command handler using the filters.Command
@dp.message(Command("help"))
async def send_help(message: types.Message):
    await message.reply("Send me a message, and I will reply with an intelligent response powered by GPT-2!")

# Echo handler that will interact with GPT-2
@dp.message()
async def handle_message(message: types.Message):
    user_message = message.text
    await message.answer("Thinking...")  # Let the user know the bot is processing
    gpt2_reply = generate_text(message.from_user.id, user_message)
    await message.answer(gpt2_reply)

# Start polling the Telegram server for messages
async def main():
    # Start polling
    await dp.start_polling(bot)

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
