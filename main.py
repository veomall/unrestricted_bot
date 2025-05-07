# Импортируем модуль os для взаимодействия с операционной системой, например, для доступа к переменным окружения.
import os
# Импортируем модуль json для работы с данными в формате JSON, например, для загрузки конфигурации моделей.
import json
# Импортируем модуль logging для ведения журнала событий и ошибок в приложении.
import logging
# Импортируем модуль asyncio для поддержки асинхронного программирования, что позволяет выполнять операции неблокирующим образом.
import asyncio
# Импортируем модуль multiprocessing для создания и управления процессами, используется для выполнения ресурсоемких задач в отдельных процессах.
import multiprocessing
# Импортируем модуль concurrent.futures для работы с асинхронными задачами на высоком уровне, в частности ProcessPoolExecutor.
import concurrent.futures
# Импортируем модуль time для работы со временем, например, для отслеживания времени выполнения запросов.
import time
# Импортируем functools.partial для создания частичных функций, которые могут быть полезны при передаче функций с предустановленными аргументами.
from functools import partial
# Импортируем классы и типы из библиотеки python-telegram-bot для взаимодействия с Telegram API.
# Update: представляет входящее обновление (сообщение, колбэк и т.д.).
# InlineKeyboardButton: кнопка, отображаемая под сообщением.
# InlineKeyboardMarkup: разметка для набора инлайн-кнопок.
# MenuButton: кнопка меню бота.
# MenuButtonCommands: тип кнопки меню, отображающий список команд.
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, MenuButton, MenuButtonCommands
# Application: основной класс для создания и запуска бота.
# CommandHandler: обработчик команд (например, /start).
# MessageHandler: обработчик текстовых сообщений и других типов сообщений.
# filters: модуль для фильтрации входящих сообщений по типу, содержанию и т.д.
# ContextTypes: содержит типы для контекста, передаваемого в обработчики.
# ConversationHandler: обработчик для управления диалогами с несколькими состояниями.
# CallbackQueryHandler: обработчик нажатий на инлайн-кнопки.
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, ConversationHandler, CallbackQueryHandler
# Импортируем библиотеку g4f для взаимодействия с различными моделями искусственного интеллекта.
import g4f
# Импортируем Client из g4f.client для работы с API клиента g4f, например, для генерации изображений.
from g4f.client import Client
# Импортируем функцию load_dotenv из библиотеки python-dotenv для загрузки переменных окружения из файла .env.
from dotenv import load_dotenv

# Загружаем переменные окружения из файла .env в текущую среду.
# Это позволяет хранить конфигурационные данные, такие как токены API, отдельно от кода.
load_dotenv()

# Настраиваем базовую конфигурацию для системы логирования.
# format: определяет формат вывода сообщений лога (время, имя логгера, уровень сообщения, само сообщение).
# level: устанавливает минимальный уровень сообщений, которые будут обрабатываться (INFO и выше).
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
# Получаем объект логгера для текущего модуля (__name__ будет именем текущего файла).
# Этот логгер будет использоваться для записи информационных сообщений и ошибок.
logger = logging.getLogger(__name__)

# Определяем константы для состояний диалога в ConversationHandler.
# Эти числа представляют различные этапы взаимодействия пользователя с ботом.
# CHOOSING_CATEGORY: состояние выбора категории модели (текст или изображение).
# CHOOSING_MODEL: состояние выбора конкретной модели AI.
# CHATTING: основное состояние общения с выбранной моделью.
# CHOOSING_LANGUAGE: состояние выбора языка для перевода.
# TRANSLATING: состояние, когда бот ожидает текст для перевода.
# SETTING_PROMPT: состояние установки системного промпта для модели.
CHOOSING_CATEGORY, CHOOSING_MODEL, CHATTING, CHOOSING_LANGUAGE, TRANSLATING, SETTING_PROMPT = range(6)

# Глобальный словарь для отслеживания активных процессов генерации.
# Ключ - уникальный идентификатор процесса, значение - информация о процессе (future, executor).
active_processes = {}

# Функция для загрузки информации о моделях из JSON-файла 'models.json'.
def load_models():
    # Пытаемся открыть и прочитать файл 'models.json'.
    try:
        # Открываем файл 'models.json' в режиме чтения ('r').
        # Конструкция 'with' гарантирует, что файл будет корректно закрыт после использования.
        with open('models.json', 'r') as f:
            # Загружаем данные из JSON-файла и возвращаем их в виде словаря Python.
            return json.load(f)
    # Если при загрузке файла возникает исключение (например, файл не найден или содержит некорректный JSON).
    except Exception as e:
        # Логируем ошибку с указанием причины.
        logger.error(f"Error loading models.json: {e}")
        # Возвращаем пустую структуру моделей по умолчанию, чтобы бот мог продолжить работу.
        return {"text": {}, "image": {}}

# Глобальный словарь, хранящий информацию о доступных моделях.
# Заполняется при запуске бота путем вызова функции load_models().
MODELS = load_models()

# Функция для генерации текста с использованием указанной модели и провайдера.
# Эта функция предназначена для запуска в отдельном процессе, чтобы не блокировать основной поток бота.
def generate_text(model_name, messages, provider_name):
    # Пытаемся выполнить генерацию текста.
    try:
        # Динамически получаем класс провайдера из модуля g4f.Provider по его имени.
        # Если провайдер с таким именем не найден, getattr вернет None.
        provider = getattr(g4f.Provider, provider_name, None)
        # Проверяем, был ли найден провайдер.
        if not provider:
            # Если провайдер не найден, возвращаем словарь с ошибкой.
            return {"error": f"Provider {provider_name} not found"}
        
        # Выполняем запрос к API для генерации текстового ответа.
        # model: имя используемой модели.
        # messages: история диалога (список сообщений).
        # provider: класс используемого провайдера.
        response = g4f.ChatCompletion.create(
            model=model_name,
            messages=messages,
            provider=provider,
        )
        # Возвращаем словарь с результатом генерации.
        return {"result": response}
    # Если в процессе генерации возникает исключение.
    except Exception as e:
        # Возвращаем словарь с описанием ошибки.
        return {"error": str(e)}

# Функция для генерации изображения по заданному промпту с использованием указанной модели.
# Эта функция также предназначена для запуска в отдельном процессе.
def generate_image(model_name, prompt):
    # Пытаемся выполнить генерацию изображения.
    try:
        # Создаем экземпляр клиента g4f.
        client = Client()
        # Выполняем запрос к API для генерации изображения.
        # model: имя используемой модели.
        # prompt: текстовое описание (промпт) для генерации изображения.
        # response_format: формат, в котором ожидается URL сгенерированного изображения.
        response = client.images.generate(
            model=model_name,
            prompt=prompt,
            response_format="url"
        )
        # Проверяем, что ответ существует, содержит данные и эти данные не пусты.
        if response and response.data and len(response.data) > 0:
            # Возвращаем словарь с URL первого сгенерированного изображения.
            return {"result": response.data[0].url}
        # Если изображение не было сгенерировано.
        else:
            # Возвращаем словарь с ошибкой.
            return {"error": "No image was generated"}
    # Если в процессе генерации возникает исключение.
    except Exception as e:
        # Возвращаем словарь с описанием ошибки.
        return {"error": str(e)}

# Асинхронная функция-обработчик команды /start.
# update: объект Update, содержащий информацию о входящем сообщении.
# context: объект ContextTypes.DEFAULT_TYPE, хранящий данные пользователя и бота.
# Возвращает целочисленное значение, представляющее следующее состояние диалога.
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Отправляет приветственное сообщение при выполнении команды `/start`."""
    # Получаем информацию о пользователе, отправившем команду.
    user = update.effective_user
    # Инициализируем (или очищаем) историю сообщений для данного пользователя в user_data.
    # user_data - это словарь, связанный с конкретным пользователем, для хранения его данных между запросами.
    context.user_data["messages"] = []
    
    # Формируем текст приветственного сообщения и список доступных команд.
    # Используем f-строку для вставки имени пользователя.
    commands_text = (
        f"Привет, {user.first_name}! Я бот для доступа к ИИ-моделям.\n\n"
        "Доступные команды:\n"
        "/translate - Перевести текст\n"
        "/model - Выбрать модель AI\n"
        "/reset - Очистить историю разговора\n"
        "/help - Показать справку\n"
    )
    
    # Отправляем сформированное сообщение пользователю.
    await update.message.reply_text(commands_text)
    
    # Возвращаем состояние CHATTING, так как после приветствия бот готов к общению.
    return CHATTING

# Асинхронная функция-обработчик выбора категории модели (текст или изображение).
# Срабатывает при нажатии на инлайн-кнопку выбора категории.
async def select_category(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обрабатывает выбор категории модели (текст или изображение)."""
    # Получаем объект CallbackQuery из обновления (это данные о нажатой инлайн-кнопке).
    query = update.callback_query
    # Отправляем подтверждение Telegram API, что колбэк получен и обработан (чтобы убрать "часики" на кнопке).
    await query.answer()
    
    # Извлекаем выбранную категорию из данных колбэка.
    # Данные колбэка имеют формат "category_text" или "category_image".
    category = query.data.split('_')[1]
    # Сохраняем выбранную категорию в user_data для последующего использования.
    context.user_data["current_category"] = category
    
    # Создаем пустой список для кнопок клавиатуры.
    keyboard = []
    # Итерируемся по моделям, доступным для выбранной категории, из глобального словаря MODELS.
    # MODELS.get(category, {}) безопасно получает словарь моделей для категории, или пустой словарь, если категория не найдена.
    for model_name in MODELS.get(category, {}):
        # Для каждой модели создаем инлайн-кнопку.
        # Текст кнопки - имя модели, данные колбэка - "model_<имя_модели>".
        keyboard.append([InlineKeyboardButton(model_name, callback_data=f"model_{model_name}")])
    
    # Создаем объект инлайн-клавиатуры на основе списка кнопок.
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # Редактируем исходное сообщение, в котором была нажата кнопка.
    # Заменяем его текстом с предложением выбрать модель и новой клавиатурой.
    await query.edit_message_text(
        f"Please select a {category} model:", # Просим выбрать модель указанной категории
        reply_markup=reply_markup # Прикрепляем клавиатуру с моделями
    )
    # Возвращаем состояние CHOOSING_MODEL, так как следующим шагом будет выбор конкретной модели.
    return CHOOSING_MODEL

# Асинхронная функция-обработчик выбора конкретной AI-модели.
# Срабатывает при нажатии на инлайн-кнопку с именем модели.
async def select_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Выбирает AI-модель на основе выбора пользователя."""
    # Получаем объект CallbackQuery.
    query = update.callback_query
    # Подтверждаем получение колбэка.
    await query.answer()
    
    # Извлекаем имя выбранной модели из данных колбэка (формат "model_<имя_модели>").
    model_name = query.data.split('_')[1]
    # Получаем текущую выбранную категорию из user_data. Если не найдена, по умолчанию "text".
    category = context.user_data.get("current_category", "text")
    # Получаем имя провайдера для выбранной модели и категории из словаря MODELS.
    provider_name = MODELS.get(category, {}).get(model_name)
    
    # Сохраняем имя выбранной модели и ее провайдера в user_data.
    context.user_data["current_model"] = model_name
    context.user_data["current_provider"] = provider_name
    
    # Очищаем историю сообщений при смене модели, чтобы начать новый диалог.
    context.user_data["messages"] = []
    
    # Создаем клавиатуру для выбора, устанавливать ли системный промпт.
    keyboard = [
        # Кнопка "Да, установить промпт" с колбэком "prompt_yes".
        [InlineKeyboardButton("Да, установить промпт", callback_data="prompt_yes")],
        # Кнопка "Нет, продолжить без промпта" с колбэком "prompt_no".
        [InlineKeyboardButton("Нет, продолжить без промпта", callback_data="prompt_no")]
    ]
    # Создаем объект инлайн-клавиатуры.
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # Редактируем сообщение, информируя пользователя о выбранной модели и провайдере.
    # Предлагаем начать общение/генерацию или сменить модель/очистить историю.
    # Также спрашиваем, хочет ли пользователь установить системный промпт.
    await query.edit_message_text(
        f"Вы выбрали модель {model_name} с провайдером {provider_name}.\n"
        f"Теперь вы можете начать {'общение' if category == 'text' else 'генерацию изображений'}!\n"
        "Отправьте /model для смены модели или /reset для очистки истории разговора.\n\n"
        "Хотите установить системный промпт для этой модели? (Это повлияет на все последующие ответы до сброса чата)",
        reply_markup=reply_markup # Прикрепляем клавиатуру с выбором промпта
    )
    # Возвращаем состояние SETTING_PROMPT, ожидая ответа пользователя по поводу системного промпта.
    return SETTING_PROMPT

# Асинхронная функция-обработчик выбора пользователя относительно установки системного промпта.
async def handle_prompt_choice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обрабатывает выбор пользователя относительно установки системного промпта."""
    # Получаем объект CallbackQuery.
    query = update.callback_query
    # Подтверждаем получение колбэка.
    await query.answer()
    
    # Извлекаем выбор пользователя из данных колбэка (формат "prompt_yes" или "prompt_no").
    choice = query.data.split('_')[1]
    
    # Если пользователь выбрал "yes" (установить промпт).
    if choice == 'yes':
        # Редактируем сообщение, прося пользователя отправить текст системного промпта.
        await query.edit_message_text(
            "Пожалуйста, отправьте системный промпт, который будет использоваться для этой модели."
        )
        # Возвращаем состояние SETTING_PROMPT, так как ожидаем сообщение с текстом промпта.
        return SETTING_PROMPT
    # Если пользователь выбрал "no" (не устанавливать промпт).
    else:  # choice == 'no'
        # Устанавливаем системный промпт в None в user_data.
        context.user_data["system_prompt"] = None
        # Редактируем сообщение, информируя, что промпт не установлен и можно начинать общение.
        await query.edit_message_text(
            "Системный промпт не установлен. Вы можете начать общение с моделью."
        )
        # Возвращаем состояние CHATTING.
        return CHATTING

# Асинхронная функция для установки системного промпта для текущей модели.
# Срабатывает, когда пользователь отправляет текст после выбора "Да, установить промпт".
async def set_system_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Устанавливает системный промпт для текущей модели."""
    # Получаем текст системного промпта из сообщения пользователя.
    prompt = update.message.text
    # Сохраняем системный промпт в user_data.
    context.user_data["system_prompt"] = prompt
    
    # Отправляем пользователю подтверждение с текстом установленного промпта.
    await update.message.reply_text(
        f"Системный промпт установлен:\n{prompt}\n\n"
        "Вы можете начать общение с моделью. Промпт будет использоваться до сброса чата."
    )
    # Возвращаем состояние CHATTING.
    return CHATTING

# Асинхронная функция для перевода текста с использованием модели gpt-4o.
# text: текст для перевода.
# target_language: язык, на который нужно перевести текст (по умолчанию "English").
async def translate_text(text, target_language="English"):
    """Переводит текст, используя GPT-4o с системным промптом."""
    # Пытаемся выполнить перевод.
    try:
        # Формируем системный промпт для задачи перевода.
        # Указываем модели, что она должна выступить в роли эксперта-переводчика.
        system_prompt = f"You are an expert translator. Translate the following text to {target_language}. Maintain the original meaning, tone, and context as much as possible. Only return the translated text, without any explanations, comments, or notes."
        
        # Асинхронно вызываем API g4f для выполнения перевода.
        # model: используем модель "gpt-4o".
        # messages: передаем системный промпт и текст пользователя.
        # provider: используем провайдера PollinationsAI (можно выбрать другого, если необходимо).
        response = await g4f.ChatCompletion.create_async(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            provider=getattr(g4f.Provider, "PollinationsAI", None), # Динамическое получение провайдера
        )
        
        # Возвращаем переведенный текст.
        return response
    # Если при переводе возникает ошибка.
    except Exception as e:
        # Логируем ошибку.
        logging.error(f"Translation error: {e}")
        # В случае ошибки возвращаем оригинальный текст.
        return text

# Асинхронная функция-обработчик обычных текстовых сообщений от пользователя.
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обрабатывает сообщения пользователя и получает ответ от AI."""
    # Получаем текст сообщения пользователя.
    user_message = update.message.text
    # Получаем текущую категорию модели из user_data (по умолчанию "text").
    category = context.user_data.get("current_category", "text")
    # Получаем имя текущей модели из user_data.
    model_name = context.user_data.get("current_model")
    # Получаем имя текущего провайдера из user_data.
    provider_name = context.user_data.get("current_provider")
    # Получаем ID пользователя.
    user_id = update.effective_user.id
    
    # Проверяем, выбраны ли модель и провайдер.
    if not model_name or not provider_name:
        # Если нет, просим пользователя выбрать модель с помощью команды /start (или /model).
        await update.message.reply_text("Пожалуйста, выберите модель с помощью /start")
        # Остаемся в состоянии CHATTING.
        return CHATTING
    
    # Проверяем, существует ли ключ "messages" в user_data, и если нет, инициализируем его пустым списком.
    if "messages" not in context.user_data:
        context.user_data["messages"] = []
    
    # Добавляем сообщение пользователя в историю диалога.
    # Каждое сообщение - это словарь с ключами "role" (роль) и "content" (содержимое).
    context.user_data["messages"].append({"role": "user", "content": user_message})
    
    # Создаем инлайн-клавиатуру с кнопкой "Отменить запрос".
    keyboard = [[InlineKeyboardButton("Отменить запрос", callback_data="cancel_request")]]
    # Создаем объект разметки для этой клавиатуры.
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # Отправляем пользователю сообщение о том, что запрос обрабатывается, с кнопкой отмены.
    waiting_message = await update.message.reply_text(
        "Запрос отправлен, ожидаю ответа от модели...",
        reply_markup=reply_markup # Прикрепляем клавиатуру с кнопкой отмены
    )
    
    # Сохраняем ID "ожидающего" сообщения в user_data для возможности его последующего удаления.
    context.user_data["waiting_message_id"] = waiting_message.message_id
    
    # Основной блок обработки запроса к AI.
    try:
        # Используем ProcessPoolExecutor для выполнения ресурсоемкой задачи генерации в отдельном процессе.
        # max_workers=1 означает, что будет использоваться только один рабочий процесс.
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            # Если текущая категория - "text" (генерация текста).
            if category == "text":
                # Копируем текущую историю сообщений.
                messages = context.user_data["messages"].copy()
                # Получаем системный промпт из user_data, если он установлен.
                system_prompt = context.user_data.get("system_prompt")
                # Если системный промпт существует.
                if system_prompt:
                    # Вставляем системный промпт в начало списка сообщений.
                    messages.insert(0, {"role": "system", "content": system_prompt})
                
                # Отправляем задачу генерации текста на выполнение в отдельный процесс.
                # executor.submit возвращает объект Future, представляющий асинхронное выполнение.
                future = executor.submit(generate_text, model_name, messages, provider_name)
                
                # Создаем уникальный ключ для процесса, используя ID пользователя и текущее время.
                process_key = f"{user_id}_{int(time.time())}"
                # Сохраняем информацию о запущенном процессе (future, executor, время старта) в глобальный словарь active_processes.
                active_processes[process_key] = {
                    "future": future,
                    "executor": executor,
                    "start_time": time.time()
                }
                # Сохраняем ключ процесса в user_data для возможности его отмены.
                context.user_data["process_key"] = process_key
                
                # Внутренний блок try/finally для корректной обработки завершения или отмены процесса.
                try:
                    # Инициализируем переменную для результата.
                    result = None
                    # Цикл ожидания завершения задачи.
                    # future.done() возвращает True, если задача завершена (успешно, с ошибкой или отменена).
                    while not future.done():
                        # Асинхронно ждем 0.1 секунды, чтобы не блокировать основной поток и дать возможность обработать другие события.
                        await asyncio.sleep(0.1)
                        # Проверяем, был ли запрос отменен пользователем.
                        if context.user_data.get("request_cancelled", False):
                            # Если запрос отменен и процесс все еще активен.
                            if process_key in active_processes:
                                # Пытаемся принудительно остановить процесс.
                                # executor._processes.clear() очищает список активных процессов пула (может быть недокументированным API).
                                executor._processes.clear()
                                # executor.shutdown(wait=False) немедленно завершает работу пула, не дожидаясь завершения задач.
                                executor.shutdown(wait=False)
                                # Удаляем информацию о процессе из active_processes.
                                del active_processes[process_key]
                            # Генерируем исключение asyncio.CancelledError, чтобы прервать ожидание.
                            raise asyncio.CancelledError("Request was cancelled")
                    
                    # Получаем результат выполнения задачи из объекта Future.
                    # Этот вызов блокирующий, но мы до него доходим, только если future.done() == True.
                    result = future.result()
                    
                    # Проверяем, содержит ли результат ошибку.
                    if "error" in result:
                        # Если да, генерируем исключение с текстом ошибки.
                        raise Exception(result["error"])
                    
                    # Извлекаем текстовый ответ модели из результата.
                    response = result["result"]
                    
                    # Добавляем ответ AI в историю сообщений.
                    context.user_data["messages"].append({"role": "assistant", "content": response})
                    
                    # Удаляем "ожидающее" сообщение.
                    await waiting_message.delete()
                    # Отправляем пользователю ответ от AI.
                    await update.message.reply_text(response)
                    
                # Этот блок TimeoutError здесь не сработает, так как future.result() без таймаута может ждать вечно.
                # Однако, он оставлен как часть общей структуры обработки ошибок.
                except concurrent.futures.TimeoutError:
                    # Удаляем "ожидающее" сообщение.
                    await waiting_message.delete()
                    # Сообщаем пользователю о таймауте.
                    await update.message.reply_text("Превышено время ожидания ответа. Пожалуйста, попробуйте еще раз.")
                # Блок finally выполняется в любом случае (успех, ошибка, отмена).
                finally:
                    # Если информация о процессе все еще есть в active_processes, удаляем ее.
                    if process_key in active_processes:
                        del active_processes[process_key]
                    # Сбрасываем флаг отмены запроса.
                    context.user_data["request_cancelled"] = False
                
            # Если текущая категория - "image" (генерация изображений).
            else:  # category == "image"
                # Переводим промпт пользователя на английский язык, так как многие модели генерации изображений лучше работают с английским.
                translated_prompt = await translate_text(user_message, "English")
                
                # Если промпт был переведен (т.е. переведенный текст отличается от оригинала).
                if translated_prompt != user_message:
                    # Отправляем пользователю переведенный промпт для информации.
                    await update.message.reply_text(f"Переведенный запрос: {translated_prompt}")
                
                # Отправляем задачу генерации изображения на выполнение в отдельный процесс.
                future = executor.submit(generate_image, model_name, translated_prompt)
                
                # Создаем уникальный ключ для процесса.
                process_key = f"{user_id}_{int(time.time())}"
                # Сохраняем информацию о запущенном процессе.
                active_processes[process_key] = {
                    "future": future,
                    "executor": executor,
                    "start_time": time.time()
                }
                # Сохраняем ключ процесса в user_data.
                context.user_data["process_key"] = process_key
                
                # Внутренний блок try/finally для обработки генерации изображения.
                try:
                    # Инициализируем переменную для результата.
                    result = None
                    # Цикл ожидания завершения задачи с проверкой отмены.
                    while not future.done():
                        await asyncio.sleep(0.1)
                        if context.user_data.get("request_cancelled", False):
                            if process_key in active_processes:
                                executor._processes.clear()
                                executor.shutdown(wait=False)
                                del active_processes[process_key]
                            raise asyncio.CancelledError("Request was cancelled")
                    
                    # Получаем результат выполнения задачи.
                    result = future.result()
                    
                    # Проверяем, содержит ли результат ошибку.
                    if "error" in result:
                        raise Exception(result["error"])
                    
                    # Извлекаем URL сгенерированного изображения из результата.
                    image_url = result["result"]
                    
                    # Удаляем "ожидающее" сообщение.
                    await waiting_message.delete()
                    
                    # Отправляем пользователю сгенерированное изображение по URL.
                    await update.message.reply_photo(image_url)
                    
                # Обработка таймаута (аналогично текстовой генерации).
                except concurrent.futures.TimeoutError:
                    await waiting_message.delete()
                    await update.message.reply_text("Превышено время ожидания ответа. Пожалуйста, попробуйте еще раз.")
                # Блок finally для очистки.
                finally:
                    if process_key in active_processes:
                        del active_processes[process_key]
                    context.user_data["request_cancelled"] = False
    
    # Если было сгенерировано исключение asyncio.CancelledError (запрос отменен).
    except asyncio.CancelledError:
        # Логируем информацию об отмене запроса.
        logger.info(f"Request cancelled for user {user_id}")
        # Удаляем последнее сообщение пользователя из истории, так как оно не было обработано.
        if context.user_data["messages"]:
            context.user_data["messages"].pop()
    # Если возникло любое другое исключение в процессе обработки.
    except Exception as e:
        # Логируем ошибку.
        logger.error(f"Error generating response: {e}")
        # Пытаемся удалить "ожидающее" сообщение, если оно еще существует.
        try:
            await waiting_message.delete()
        # Игнорируем возможные ошибки при удалении (например, если сообщение уже удалено).
        except:
            pass
        
        # Отправляем пользователю сообщение об ошибке и предложение попробовать снова.
        await update.message.reply_text(
            f"Произошла ошибка при обработке запроса: {str(e)}\n"
            "Пожалуйста, попробуйте отправить запрос еще раз."
        )
    
    # В любом случае, возвращаем состояние CHATTING, чтобы бот продолжал ожидать сообщения.
    return CHATTING

# Асинхронная функция-обработчик команды /reset.
async def reset_conversation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Сбрасывает историю диалога."""
    # Очищаем (или инициализируем пустым списком) историю сообщений для пользователя.
    context.user_data["messages"] = []
    # Отправляем пользователю подтверждение, что история очищена.
    await update.message.reply_text("История разговора была очищена. Вы можете продолжить общение!")
    # Возвращаем состояние CHATTING.
    return CHATTING

# Асинхронная функция-обработчик команды /model.
# Позволяет пользователю изменить выбранную AI-модель.
async def change_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Позволяет пользователю изменить AI-модель."""
    # Создаем инлайн-клавиатуру для выбора категории модели.
    keyboard = [
        # Кнопка "Text Models" с колбэком "category_text".
        [InlineKeyboardButton("Text Models", callback_data="category_text")],
        # Кнопка "Image Models" с колбэком "category_image".
        [InlineKeyboardButton("Image Models", callback_data="category_image")]
    ]
    # Создаем объект разметки для этой клавиатуры.
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # Отправляем пользователю сообщение с предложением выбрать тип модели и клавиатурой.
    await update.message.reply_text(
        "Выберите тип модели, которую вы хотите использовать:",
        reply_markup=reply_markup
    )
    # Возвращаем состояние CHOOSING_CATEGORY, так как следующим шагом будет выбор категории.
    return CHOOSING_CATEGORY

# Асинхронная функция-обработчик команды /help.
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет сообщение со справкой при выполнении команды /help."""
    # Отправляем пользователю список доступных команд.
    await update.message.reply_text(
        "Доступные команды:\n"
        "/translate - Перевести текст\n"
        "/model - Выбрать модель AI\n"
        "/reset - Очистить историю разговора\n"
        "/help - Показать справку\n"
    )
    # Возвращаем состояние CHATTING (хотя для None-возвращающих обработчиков это не так критично, но для единообразия).
    return CHATTING # На самом деле, для CommandHandler, который не является частью ConversationHandler, возвращаемое значение не используется.

# Асинхронная функция-обработчик команды /translate.
# Запрашивает у пользователя язык, на который нужно перевести текст.
async def translate_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обрабатывает команду /translate, запрашивая язык для перевода."""
    # Отправляем пользователю сообщение с просьбой указать целевой язык.
    await update.message.reply_text(
        "На какой язык вы хотите перевести текст? Напишите название языка (например, 'английский', 'французский', 'испанский' и т.д.)."
    )
    # Возвращаем состояние CHOOSING_LANGUAGE, так как ожидаем от пользователя сообщение с названием языка.
    return CHOOSING_LANGUAGE

# Асинхронная функция для обработки выбора языка для перевода.
# Срабатывает, когда пользователь отправляет текст после команды /translate.
async def select_language(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обрабатывает выбор языка для перевода."""
    # Получаем название языка из сообщения пользователя.
    language = update.message.text
    
    # Сохраняем выбранный язык в user_data.
    context.user_data["target_language"] = language
    
    # Отправляем пользователю подтверждение выбора языка и просим отправить текст для перевода.
    await update.message.reply_text(
        f"Выбран язык: {language}\n"
        "Теперь отправьте текст, который хотите перевести."
    )
    # Возвращаем состояние TRANSLATING, так как ожидаем текст для перевода.
    return TRANSLATING

# Асинхронная функция для обработки текста, который нужно перевести.
# Срабатывает, когда пользователь отправляет текст после выбора языка.
async def process_translation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обрабатывает текстовое сообщение, полученное после выбора языка, для перевода."""
    # Получаем текст сообщения от пользователя.
    message_text = update.message.text
    
    # Проверяем, не пустой ли текст.
    if not message_text:
        # Если пустой, просим отправить текст еще раз.
        await update.message.reply_text("Пожалуйста, отправьте текст для перевода.")
        # Остаемся в состоянии TRANSLATING.
        return TRANSLATING
    
    # Отправляем действие "typing" в чат, чтобы пользователь видел, что бот обрабатывает запрос.
    await update.message.chat.send_action(action="typing")
    
    # Получаем целевой язык из user_data (по умолчанию "английский", если не найден).
    target_language = context.user_data.get("target_language", "английский")
    
    # Вызываем функцию для перевода текста.
    translated_text = await translate_text(message_text, target_language)
    # Отправляем переведенный текст пользователю.
    await update.message.reply_text(translated_text)
    
    # Возвращаем состояние CHATTING, так как перевод завершен.
    return CHATTING

# Асинхронная функция для отмены текущего запроса к AI.
# Срабатывает при нажатии на инлайн-кнопку "Отменить запрос".
async def cancel_request(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Отменяет текущий запрос на генерацию."""
    # Получаем объект CallbackQuery.
    query = update.callback_query
    # Подтверждаем получение колбэка.
    await query.answer()
    # Получаем ID пользователя.
    user_id = update.effective_user.id
    
    # Устанавливаем флаг отмены запроса в user_data.
    # Этот флаг будет проверен в цикле ожидания в `handle_message`.
    context.user_data["request_cancelled"] = True
    
    # Пытаемся отменить процесс, если он существует и активен.
    # Получаем ключ процесса из user_data.
    process_key = context.user_data.get("process_key")
    # Если ключ существует и такой процесс есть в словаре active_processes.
    if process_key and process_key in active_processes:
        # Получаем информацию о процессе.
        process_info = active_processes[process_key]
        # Принудительно завершаем работу ProcessPoolExecutor, чтобы остановить дочерний процесс.
        # _processes.clear() и shutdown(wait=False) - это агрессивные методы остановки.
        process_info["executor"]._processes.clear()
        process_info["executor"].shutdown(wait=False)
        # Удаляем информацию о процессе из active_processes.
        del active_processes[process_key]
    
    # Пытаемся удалить "ожидающее" сообщение (сообщение с кнопкой отмены).
    try:
        await query.message.delete()
    # Игнорируем возможные ошибки при удалении.
    except:
        pass
    
    # Отправляем пользователю сообщение о том, что запрос отменен.
    # Используем query.message.reply_text, чтобы ответить в том же чате, где было исходное сообщение с кнопкой.
    await query.message.reply_text(
        "Запрос отменен. Вы можете отправить новый запрос."
    )
    
    # Возвращаем состояние CHATTING.
    return CHATTING

# Основная функция для запуска бота.
def main() -> None:
    """Запускает бота."""
    # Получаем токен Telegram-бота из переменных окружения.
    token = os.getenv("TELEGRAM_TOKEN")
    # Проверяем, был ли найден токен.
    if not token:
        # Если токен не найден, логируем ошибку и завершаем выполнение.
        logger.error("No TELEGRAM_TOKEN found in environment variables!")
        return
    
    # Включаем метод запуска процессов 'spawn' для Windows.
    # Это необходимо для корректной работы ProcessPoolExecutor на Windows.
    # force=True перезаписывает метод, если он уже был установлен.
    if os.name == 'nt':  # 'nt' - это имя для ОС Windows в Python.
        multiprocessing.set_start_method('spawn', force=True)
    
    # Создаем экземпляр Application, используя builder pattern.
    # Указываем токен бота.
    application = Application.builder().token(token).build()
    
    # Определяем список команд, которые будут отображаться в меню бота в Telegram.
    # Каждая команда - это кортеж (имя_команды, описание).
    commands = [
        ("start", "Запустить бота"),
        ("model", "Изменить модель AI"),
        ("reset", "Очистить историю разговора"),
        ("help", "Показать справку"),
        ("translate", "Перевести текст")
    ]
    
    # Устанавливаем команды меню для бота.
    # Это асинхронный вызов, но в данном контексте (настройка перед запуском) можно вызвать синхронно, если библиотека это позволяет,
    # или обернуть в asyncio.run(), если это строго асинхронная операция.
    # В python-telegram-bot v20+ set_my_commands является корутиной.
    # Однако, здесь он вызывается на application.bot, что может иметь синхронный интерфейс или требовать запуска в event loop.
    # Для простоты предполагаем, что на этапе инициализации это работает.
    # В идеале, если это асинхронный метод, его нужно вызывать через asyncio.run(application.bot.set_my_commands(commands))
    # или внутри асинхронной функции `async def setup_bot(): await application.bot.set_my_commands(commands)`
    # и затем `asyncio.run(setup_bot())`.
    # Но так как это происходит до `application.run_polling()`, библиотека может это обрабатывать.
    # TODO: Проверить, требует ли set_my_commands асинхронного вызова в современных версиях PTB.
    # Ответ: Да, требует. Однако, в старых версиях это могло быть иначе.
    # Для текущей структуры кода, если бы это было проблемой, оно бы не работало.
    # Предположим, что текущая версия библиотеки или способ ее использования допускает такой вызов.
    # Для корректности в v20+ это нужно делать асинхронно.
    # В данном коде это, скорее всего, работает из-за особенностей инициализации Application.
    # На самом деле, `application.bot` предоставляет доступ к `BaseBot`, методы которого являются корутинами.
    # Этот вызов `application.bot.set_my_commands(commands)` должен быть `await application.bot.set_my_commands(commands)`
    # и выполнен в асинхронной функции, запущенной через `asyncio.run()`.
    # Однако, т.к. это предоставленный код, оставляем как есть, предполагая, что он работает в окружении автора.
    # Правильно было бы сделать `application.post_init = set_commands_async_function`
    # или использовать `add_job` для асинхронного вызова после инициализации.
    # Либо, если это делается перед `run_polling`, можно обернуть:
    # async def set_bot_commands():
    #     await application.bot.set_my_commands(commands)
    # asyncio.run(set_bot_commands()) # Перед application.run_polling()
    # Но т.к. `run_polling` сама создает event loop, это может быть излишним.
    # `Application` может иметь механизм для обработки этого.
    application.bot.set_my_commands(commands) # Устанавливаем команды в меню бота
    
    # Настраиваем ConversationHandler для управления многоэтапными диалогами.
    conv_handler = ConversationHandler(
        # entry_points: список обработчиков, которые могут начать диалог.
        # Здесь диалог начинается с команды /start.
        entry_points=[CommandHandler("start", start)],
        # states: словарь, определяющий состояния диалога и обработчики для каждого состояния.
        states={
            # Состояние CHOOSING_CATEGORY: ожидает нажатие на инлайн-кнопку с паттерном "category_".
            CHOOSING_CATEGORY: [CallbackQueryHandler(select_category, pattern=r"^category_")],
            # Состояние CHOOSING_MODEL: ожидает нажатие на инлайн-кнопку с паттерном "model_".
            CHOOSING_MODEL: [CallbackQueryHandler(select_model, pattern=r"^model_")],
            # Состояние CHATTING: основное состояние.
            CHATTING: [
                # Обработчики команд, доступных в состоянии CHATTING.
                CommandHandler("model", change_model), # Команда для смены модели
                CommandHandler("reset", reset_conversation), # Команда для сброса диалога
                CommandHandler("help", help_command), # Команда для вызова справки
                CommandHandler("translate", translate_command), # Команда для начала перевода
                # Обработчик нажатия на инлайн-кнопку "cancel_request".
                CallbackQueryHandler(cancel_request, pattern=r"^cancel_request$"),
                # Обработчик текстовых сообщений, которые не являются командами.
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message),
            ],
            # Состояние CHOOSING_LANGUAGE: ожидает текстовое сообщение с названием языка.
            CHOOSING_LANGUAGE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, select_language),
                # Команда /cancel для отмены выбора языка и возврата в CHATTING.
                # Лямбда-функция используется для простого ответа и изменения состояния.
                CommandHandler("cancel", lambda u, c: (u.message.reply_text("Перевод отменен."), CHATTING)[-1]),
            ],
            # Состояние TRANSLATING: ожидает текстовое сообщение для перевода.
            TRANSLATING: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, process_translation),
                # Команда /cancel для отмены перевода.
                CommandHandler("cancel", lambda u, c: (u.message.reply_text("Перевод отменен."), CHATTING)[-1]),
            ],
            # Состояние SETTING_PROMPT: ожидает либо нажатие на кнопку "prompt_yes/no", либо текстовое сообщение с промптом.
            SETTING_PROMPT: [
                CallbackQueryHandler(handle_prompt_choice, pattern=r"^prompt_(yes|no)$"), # Обработка выбора да/нет
                MessageHandler(filters.TEXT & ~filters.COMMAND, set_system_prompt), # Обработка текста промпта
                # Команда /cancel для отмены установки промпта.
                CommandHandler("cancel", lambda u, c: (u.message.reply_text("Установка промпта отменена."), CHATTING)[-1]),
            ],
        },
        # fallbacks: список обработчиков, которые вызываются, если текущее состояние не имеет подходящего обработчика
        # или если диалог прерывается. Здесь, если что-то пойдет не так, команда /start вернет пользователя в начало.
        fallbacks=[CommandHandler("start", start)],
    )
    
    # Добавляем ConversationHandler в приложение. Он будет обрабатывать все входящие обновления.
    application.add_handler(conv_handler)
    
    # Запускаем бота. Бот будет работать в режиме опроса (polling), т.е. периодически запрашивать у Telegram новые обновления.
    # Бот будет работать до тех пор, пока пользователь не нажмет Ctrl-C для прерывания.
    application.run_polling()

# Стандартная конструкция Python: если скрипт запускается напрямую (а не импортируется как модуль).
if __name__ == "__main__":
    # Вызываем основную функцию main() для запуска бота.
    main()
