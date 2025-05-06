import os
import json
import logging
import asyncio
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
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
CHOOSING_CATEGORY, CHOOSING_MODEL, CHATTING, CHOOSING_LANGUAGE, TRANSLATING = range(5)

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

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Send message on `/start`."""
    user = update.effective_user
    context.user_data["messages"] = []
    
    # Show available commands
    commands_text = (
        f"Привет, {user.first_name}! Я робот-долбоёб.\n\n"
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
    
    await query.edit_message_text(
        f"Выбрана модель {model_name}\n"
        f"Теперь вы можете начать {'чат' if category == 'text' else 'генерировать изображения'}!\n"
        "Отправь /model чтобы изменить модель или /reset чтобы очистить историю разговора."
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
    
    if not model_name or not provider_name:
        await update.message.reply_text("Модель не выбрана, нажмите /start")
        return CHATTING
    
    # Add the user message to history
    if "messages" not in context.user_data:
        context.user_data["messages"] = []
    
    context.user_data["messages"].append({"role": "user", "content": user_message})
    
    # Send "typing" action
    await update.message.chat.send_action(action="typing")
    
    try:
        if category == "text":
            # Handle text generation
            # Get the provider class dynamically
            provider = getattr(g4f.Provider, provider_name, None)
            if not provider:
                raise ValueError(f"Provider {provider_name} not found")
                
            # Generate AI response
            response = await g4f.ChatCompletion.create_async(
                model=model_name,
                messages=context.user_data["messages"],
                provider=provider,
            )
            
            # Add AI response to history
            context.user_data["messages"].append({"role": "assistant", "content": response})
            
            # Send response
            await update.message.reply_text(response)
        else:
            # Translate the prompt to English for better image generation results
            translated_prompt = await translate_text(user_message, "English")
            
            # Send feedback that translation occurred
            if translated_prompt != user_message:
                await update.message.reply_text(f"Запрос: {translated_prompt}")
            
            # Handle image generation using Client API
            # Using asyncio.to_thread to run the synchronous method in a non-blocking way
            client = Client()
            
            # Define a function to run synchronously
            def generate_image():
                return client.images.generate(
                    model=model_name,
                    prompt=translated_prompt,  # Use translated prompt
                    response_format="url"
                )
            
            # Run the synchronous function in a separate thread
            response = await asyncio.to_thread(generate_image)
            
            # Send image
            if response and response.data and len(response.data) > 0:
                image_url = response.data[0].url
                await update.message.reply_photo(image_url)
            else:
                await update.message.reply_text("Не удалось сгенерировать изображение. Пожалуйста, попробуйте снова.")
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        await update.message.reply_text(
            f"Извините, я не смог сгенерировать ответ с {model_name}. "
            "Пожалуйста, попробуйте снова или выберите другую модель с /model."
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
        "Чё, уже забыл что я делаю? Я напомню:\n"
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

def main() -> None:
    """Run the bot."""
    # Get token from environment variable
    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        logger.error("No TELEGRAM_TOKEN found in environment variables!")
        return
    
    # Create the Application
    application = Application.builder().token(token).build()
    
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
        },
        fallbacks=[CommandHandler("start", start)],
    )
    
    application.add_handler(conv_handler)
    
    # Run the bot until the user presses Ctrl-C
    application.run_polling()

if __name__ == "__main__":
    main()
