import os
import json
import logging
import asyncio
import multiprocessing
import concurrent.futures
import time
from functools import partial
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, MenuButton, MenuButtonCommands
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, ConversationHandler, CallbackQueryHandler
import g4f
from g4f.client import Client
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# States for conversation
CHOOSING_CATEGORY, CHOOSING_MODEL, CHATTING, CHOOSING_LANGUAGE, TRANSLATING, SETTING_PROMPT = range(6)

# Global variable to track active processes
active_processes = {}

# Load models from JSON file
def load_models():
    try:
        with open('models.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading models.json: {e}")
        return {"text": {}, "image": {}}

# Global models dictionary
MODELS = load_models()

# Function to run in a separate process for text generation
def generate_text(model_name, messages, provider_name):
    try:
        provider = getattr(g4f.Provider, provider_name, None)
        if not provider:
            return {"error": f"Provider {provider_name} not found"}
        
        response = g4f.ChatCompletion.create(
            model=model_name,
            messages=messages,
            provider=provider,
        )
        return {"result": response}
    except Exception as e:
        return {"error": str(e)}

# Function to run in a separate process for image generation
def generate_image(model_name, prompt):
    try:
        client = Client()
        response = client.images.generate(
            model=model_name,
            prompt=prompt,
            response_format="url"
        )
        if response and response.data and len(response.data) > 0:
            return {"result": response.data[0].url}
        else:
            return {"error": "No image was generated"}
    except Exception as e:
        return {"error": str(e)}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Send message on `/start`."""
    user = update.effective_user
    context.user_data["messages"] = []
    
    # Show available commands
    commands_text = (
        f"Привет, {user.first_name}! Я бот для доступа к ИИ-моделям.\n\n"
        "Доступные команды:\n"
        "/translate - Перевести текст\n"
        "/model - Выбрать модель AI\n"
        "/reset - Очистить историю разговора\n"
        "/help - Показать справку\n"
    )
    
    await update.message.reply_text(commands_text)
    
    return CHATTING

async def select_category(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles the selection of model category (text or image)."""
    query = update.callback_query
    await query.answer()
    
    category = query.data.split('_')[1]
    context.user_data["current_category"] = category
    
    keyboard = []
    for model_name in MODELS.get(category, {}):
        keyboard.append([InlineKeyboardButton(model_name, callback_data=f"model_{model_name}")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(
        f"Please select a {category} model:",
        reply_markup=reply_markup
    )
    return CHOOSING_MODEL

async def select_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Selects the AI model based on user choice."""
    query = update.callback_query
    await query.answer()
    
    model_name = query.data.split('_')[1]
    category = context.user_data.get("current_category", "text")
    provider_name = MODELS.get(category, {}).get(model_name)
    
    context.user_data["current_model"] = model_name
    context.user_data["current_provider"] = provider_name
    
    # Clear message history when changing models
    context.user_data["messages"] = []
    
    # Create keyboard for prompt choice
    keyboard = [
        [InlineKeyboardButton("Да, установить промпт", callback_data="prompt_yes")],
        [InlineKeyboardButton("Нет, продолжить без промпта", callback_data="prompt_no")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(
        f"Вы выбрали модель {model_name} с провайдером {provider_name}.\n"
        f"Теперь вы можете начать {'общение' if category == 'text' else 'генерацию изображений'}!\n"
        "Отправьте /model для смены модели или /reset для очистки истории разговора.\n\n"
        "Хотите установить системный промпт для этой модели? (Это повлияет на все последующие ответы до сброса чата)",
        reply_markup=reply_markup
    )
    return SETTING_PROMPT

async def handle_prompt_choice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle user's choice about setting a system prompt."""
    query = update.callback_query
    await query.answer()
    
    choice = query.data.split('_')[1]
    
    if choice == 'yes':
        await query.edit_message_text(
            "Пожалуйста, отправьте системный промпт, который будет использоваться для этой модели."
        )
        return SETTING_PROMPT
    else:  # choice == 'no'
        context.user_data["system_prompt"] = None
        await query.edit_message_text(
            "Системный промпт не установлен. Вы можете начать общение с моделью."
        )
        return CHATTING

async def set_system_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Set the system prompt for the current model."""
    prompt = update.message.text
    context.user_data["system_prompt"] = prompt
    
    await update.message.reply_text(
        f"Системный промпт установлен:\n{prompt}\n\n"
        "Вы можете начать общение с моделью. Промпт будет использоваться до сброса чата."
    )
    return CHATTING

async def translate_text(text, target_language="English"):
    """Translates text using GPT-4o with a system prompt."""
    try:
        system_prompt = f"You are an expert translator. Translate the following text to {target_language}. Maintain the original meaning, tone, and context as much as possible. Only return the translated text, without any explanations, comments, or notes."
        
        response = await g4f.ChatCompletion.create_async(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            provider=getattr(g4f.Provider, "PollinationsAI", None),
        )
        
        return response
    except Exception as e:
        logging.error(f"Translation error: {e}")
        return text  # Return original text if translation fails

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle user messages and get AI response."""
    user_message = update.message.text
    category = context.user_data.get("current_category", "text")
    model_name = context.user_data.get("current_model")
    provider_name = context.user_data.get("current_provider")
    user_id = update.effective_user.id
    
    if not model_name or not provider_name:
        await update.message.reply_text("Пожалуйста, выберите модель с помощью /start")
        return CHATTING
    
    # Add the user message to history
    if "messages" not in context.user_data:
        context.user_data["messages"] = []
    
    context.user_data["messages"].append({"role": "user", "content": user_message})
    
    # Create keyboard with cancel button
    keyboard = [[InlineKeyboardButton("Отменить запрос", callback_data="cancel_request")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # Send "waiting" message with cancel button
    waiting_message = await update.message.reply_text(
        "Запрос отправлен, ожидаю ответа от модели...",
        reply_markup=reply_markup
    )
    
    # Store message ID for later reference
    context.user_data["waiting_message_id"] = waiting_message.message_id
    
    try:
        # Create a multiprocessing pool with 1 worker
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            if category == "text":
                # Prepare messages with system prompt if available
                messages = context.user_data["messages"].copy()
                system_prompt = context.user_data.get("system_prompt")
                if system_prompt:
                    messages.insert(0, {"role": "system", "content": system_prompt})
                
                # Start the generation process
                future = executor.submit(generate_text, model_name, messages, provider_name)
                
                # Store the future and executor in context for cancellation
                process_key = f"{user_id}_{int(time.time())}"
                active_processes[process_key] = {
                    "future": future,
                    "executor": executor,
                    "start_time": time.time()
                }
                context.user_data["process_key"] = process_key
                
                try:
                    # Wait for result or cancellation
                    result = None
                    while not future.done():
                        # Check if canceled every 0.1 seconds
                        await asyncio.sleep(0.1)
                        if context.user_data.get("request_cancelled", False):
                            # Force shutdown the executor to kill the process
                            if process_key in active_processes:
                                executor._processes.clear()
                                executor.shutdown(wait=False)
                                del active_processes[process_key]
                            raise asyncio.CancelledError("Request was cancelled")
                    
                    # Get the result
                    result = future.result()
                    
                    # Check for errors
                    if "error" in result:
                        raise Exception(result["error"])
                    
                    response = result["result"]
                    
                    # Add AI response to history
                    context.user_data["messages"].append({"role": "assistant", "content": response})
                    
                    # Delete waiting message and send response
                    await waiting_message.delete()
                    await update.message.reply_text(response)
                    
                except concurrent.futures.TimeoutError:
                    # Handle timeout
                    await waiting_message.delete()
                    await update.message.reply_text("Превышено время ожидания ответа. Пожалуйста, попробуйте еще раз.")
                finally:
                    # Clean up
                    if process_key in active_processes:
                        del active_processes[process_key]
                    context.user_data["request_cancelled"] = False
                
            else:  # category == "image"
                # Translate the prompt to English for better image generation results
                translated_prompt = await translate_text(user_message, "English")
                
                # Send feedback that translation occurred
                if translated_prompt != user_message:
                    await update.message.reply_text(f"Переведенный запрос: {translated_prompt}")
                
                # Start the image generation process
                future = executor.submit(generate_image, model_name, translated_prompt)
                
                # Store the future and executor in context for cancellation
                process_key = f"{user_id}_{int(time.time())}"
                active_processes[process_key] = {
                    "future": future,
                    "executor": executor,
                    "start_time": time.time()
                }
                context.user_data["process_key"] = process_key
                
                try:
                    # Wait for result or cancellation
                    result = None
                    while not future.done():
                        # Check if canceled every 0.1 seconds
                        await asyncio.sleep(0.1)
                        if context.user_data.get("request_cancelled", False):
                            # Force shutdown the executor to kill the process
                            if process_key in active_processes:
                                executor._processes.clear()
                                executor.shutdown(wait=False)
                                del active_processes[process_key]
                            raise asyncio.CancelledError("Request was cancelled")
                    
                    # Get the result
                    result = future.result()
                    
                    # Check for errors
                    if "error" in result:
                        raise Exception(result["error"])
                    
                    image_url = result["result"]
                    
                    # Delete waiting message
                    await waiting_message.delete()
                    
                    # Send image
                    await update.message.reply_photo(image_url)
                    
                except concurrent.futures.TimeoutError:
                    # Handle timeout
                    await waiting_message.delete()
                    await update.message.reply_text("Превышено время ожидания ответа. Пожалуйста, попробуйте еще раз.")
                finally:
                    # Clean up
                    if process_key in active_processes:
                        del active_processes[process_key]
                    context.user_data["request_cancelled"] = False
    
    except asyncio.CancelledError:
        # Request was cancelled
        logger.info(f"Request cancelled for user {user_id}")
        # Remove the last user message from history
        if context.user_data["messages"]:
            context.user_data["messages"].pop()
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        # Delete waiting message
        try:
            await waiting_message.delete()
        except:
            pass
        
        # Send error message and retry prompt
        await update.message.reply_text(
            f"Произошла ошибка при обработке запроса: {str(e)}\n"
            "Пожалуйста, попробуйте отправить запрос еще раз."
        )
    
    return CHATTING

async def reset_conversation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Reset the conversation history."""
    context.user_data["messages"] = []
    await update.message.reply_text("История разговора была очищена. Вы можете продолжить общение!")
    return CHATTING

async def change_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Allow user to change the AI model."""
    keyboard = [
        [InlineKeyboardButton("Text Models", callback_data="category_text")],
        [InlineKeyboardButton("Image Models", callback_data="category_image")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "Выберите тип модели, которую вы хотите использовать:",
        reply_markup=reply_markup
    )
    return CHOOSING_CATEGORY

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text(
        "Доступные команды:\n"
        "/translate - Перевести текст\n"
        "/model - Выбрать модель AI\n"
        "/reset - Очистить историю разговора\n"
        "/help - Показать справку\n"
    )
    return CHATTING

async def translate_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle /translate command to request language for translation."""
    await update.message.reply_text(
        "На какой язык вы хотите перевести текст? Напишите название языка (например, 'английский', 'французский', 'испанский' и т.д.)."
    )
    return CHOOSING_LANGUAGE

async def select_language(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Process the language selection for translation."""
    language = update.message.text
    
    # Store the language in user_data
    context.user_data["target_language"] = language
    
    await update.message.reply_text(
        f"Выбран язык: {language}\n"
        "Теперь отправьте текст, который хотите перевести."
    )
    return TRANSLATING

async def process_translation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Process the text message received after language selection."""
    message_text = update.message.text
    
    if not message_text:
        await update.message.reply_text("Пожалуйста, отправьте текст для перевода.")
        return TRANSLATING
    
    await update.message.chat.send_action(action="typing")
    
    # Get the target language from user_data
    target_language = context.user_data.get("target_language", "английский")
    
    translated_text = await translate_text(message_text, target_language)
    await update.message.reply_text(translated_text)
    
    return CHATTING

async def cancel_request(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancel the current request."""
    query = update.callback_query
    await query.answer()
    user_id = update.effective_user.id
    
    # Mark request as cancelled
    context.user_data["request_cancelled"] = True
    
    # Try to cancel the process if it exists
    process_key = context.user_data.get("process_key")
    if process_key and process_key in active_processes:
        process_info = active_processes[process_key]
        # Force shutdown executor to kill the process
        process_info["executor"]._processes.clear()
        process_info["executor"].shutdown(wait=False)
        del active_processes[process_key]
    
    # Delete the waiting message
    try:
        await query.message.delete()
    except:
        pass
    
    await query.message.reply_text(
        "Запрос отменен. Вы можете отправить новый запрос."
    )
    
    return CHATTING

def main() -> None:
    """Run the bot."""
    # Get token from environment variable
    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        logger.error("No TELEGRAM_TOKEN found in environment variables!")
        return
    
    # Enable process forking (needed for ProcessPoolExecutor on Windows)
    if os.name == 'nt':  # Windows
        multiprocessing.set_start_method('spawn', force=True)
    
    # Create the Application
    application = Application.builder().token(token).build()
    
    # Set up commands menu
    commands = [
        ("start", "Запустить бота"),
        ("model", "Изменить модель AI"),
        ("reset", "Очистить историю разговора"),
        ("help", "Показать справку"),
        ("translate", "Перевести текст")
    ]
    
    # Set commands in menu
    application.bot.set_my_commands(commands)
    
    # Set up conversation handler
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            CHOOSING_CATEGORY: [CallbackQueryHandler(select_category, pattern=r"^category_")],
            CHOOSING_MODEL: [CallbackQueryHandler(select_model, pattern=r"^model_")],
            CHATTING: [
                CommandHandler("model", change_model),
                CommandHandler("reset", reset_conversation),
                CommandHandler("help", help_command),
                CommandHandler("translate", translate_command),
                CallbackQueryHandler(cancel_request, pattern=r"^cancel_request$"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message),
            ],
            CHOOSING_LANGUAGE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, select_language),
                CommandHandler("cancel", lambda u, c: (u.message.reply_text("Перевод отменен."), CHATTING)[-1]),
            ],
            TRANSLATING: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, process_translation),
                CommandHandler("cancel", lambda u, c: (u.message.reply_text("Перевод отменен."), CHATTING)[-1]),
            ],
            SETTING_PROMPT: [
                CallbackQueryHandler(handle_prompt_choice, pattern=r"^prompt_(yes|no)$"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, set_system_prompt),
                CommandHandler("cancel", lambda u, c: (u.message.reply_text("Установка промпта отменена."), CHATTING)[-1]),
            ],
        },
        fallbacks=[CommandHandler("start", start)],
    )
    
    application.add_handler(conv_handler)
    
    # Run the bot until the user presses Ctrl-C
    application.run_polling()

if __name__ == "__main__":
    main()
