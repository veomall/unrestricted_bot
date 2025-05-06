# Telegram AI Bot

A Telegram bot that provides access to advanced AI models using the g4f library.

## Features

- Access to multiple AI models: GPT-4, Bing, and You.com
- Conversation history management
- Easy model switching

## Setup

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root with your Telegram bot token:
   ```
   TELEGRAM_TOKEN=your_telegram_token_here
   ```
   To get a token, message [@BotFather](https://t.me/BotFather) on Telegram and follow the instructions.

4. Run the bot:
   ```
   python main.py
   ```

## Usage

Once the bot is running, you can interact with it through Telegram:

- `/start` - Start the bot and select an AI model
- `/model` - Change the AI model
- `/reset` - Clear conversation history
- `/help` - Show help information

## Available Models

- GPT-4 - OpenAI's GPT-4 model
- Bing - Microsoft's Bing AI
- You - You.com AI

## Note

This bot uses g4f (GPT4Free) to access AI models without requiring API keys. The availability and stability of these models may vary. 