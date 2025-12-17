#!/usr/bin/env python3
"""
Omni-Hear AI v2.1 - Hugging Face Spaces Edition
"""
import os
import logging
import base64
from typing import Optional, Tuple
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters,
)
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ===== CONFIG =====
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

MODEL_PRIORITY = [
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.0-flash",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
]

MAX_FILE_SIZE = 20 * 1024 * 1024

PROMPTS = {
    "lecture": """You are a University Professor teaching in Persian (Farsi).
Listen to this audio carefully. Do NOT summarize.
Write a comprehensive Textbook Chapter in Persian.
Cover every detail, example, and nuance mentioned.
Use bold headers to organize sections.
Write in fluent, academic Persian. زبان خروجی حتماً فارسی باشد.""",
    "soap": """You are a Chief Resident at a teaching hospital.
Listen to this medical dictation audio.
Write a professional SOAP Note in English.
Format: Subjective, Objective, Assessment, Plan.
Output MUST be in English only.""",
    "summary": """Listen to this audio carefully.
Summarize the content into clear, concise Persian bullet points.
Use • for bullet points. Write in fluent Persian.
زبان خروجی حتماً فارسی باشد.""",
    "lyrics": """Listen to this audio.
If music: Extract complete lyrics in original language.
If speech: Provide verbatim transcription."""
}

MESSAGES = {
    "welcome": "🎧 **Omni-Hear AI**\n\nیک فایل صوتی ارسال کنید.",
    "audio_received": "🎵 فایل دریافت شد! نوع پردازش را انتخاب کنید:",
    "processing": "⏳ در حال پردازش...",
    "error": "❌ خطا در پردازش.",
    "all_failed": "❌ همه مدل‌ها با خطا مواجه شدند.",
    "no_audio": "⚠️ ابتدا فایل صوتی ارسال کنید.",
    "file_too_large": "⚠️ حجم فایل بیشتر از ۲۰ مگابایت است.",
    "not_audio": "⚠️ لطفاً یک فایل صوتی ارسال کنید.",
}

user_audio_cache = {}

def get_menu_keyboard():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📚 درسنامه کامل", callback_data="lecture"),
            InlineKeyboardButton("🩺 شرح‌حال پزشکی", callback_data="soap"),
        ],
        [
            InlineKeyboardButton("📝 خلاصه متن", callback_data="summary"),
            InlineKeyboardButton("🎵 متن آهنگ", callback_data="lyrics"),
        ],
    ])

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(MESSAGES["welcome"], parse_mode="Markdown")

async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    msg = update.message
    
    audio_file = None
    file_type = "audio"
    
    if msg.voice:
        audio_file = msg.voice
        file_type = "voice"
    elif msg.audio:
        audio_file = msg.audio
    elif msg.document:
        if msg.document.mime_type and msg.document.mime_type.startswith("audio/"):
            audio_file = msg.document
        else:
            await msg.reply_text(MESSAGES["not_audio"])
            return
    else:
        await msg.reply_text(MESSAGES["not_audio"])
        return
    
    file_size = getattr(audio_file, 'file_size', 0)
    if file_size and file_size > MAX_FILE_SIZE:
        await msg.reply_text(MESSAGES["file_too_large"])
        return
    
    try:
        file = await context.bot.get_file(audio_file.file_id)
        if file.file_size and file.file_size > MAX_FILE_SIZE:
            await msg.reply_text(MESSAGES["file_too_large"])
            return
        
        audio_bytes = await file.download_as_bytearray()
        
        if file_type == "voice":
            mime_type = "audio/ogg"
        elif hasattr(audio_file, 'mime_type') and audio_file.mime_type:
            mime_type = audio_file.mime_type
        else:
            mime_type = "audio/mpeg"
        
        user_audio_cache[user_id] = {
            "data": bytes(audio_bytes),
            "mime_type": mime_type
        }
        
        logger.info(f"Audio cached: user={user_id}, size={len(audio_bytes)}")
        await msg.reply_text(MESSAGES["audio_received"], reply_markup=get_menu_keyboard())
        
    except Exception as e:
        logger.error(f"Error: {e}")
        await msg.reply_text(MESSAGES["error"])

async def process_with_cascade(
    audio_data: bytes, mime_type: str, mode: str
) -> Tuple[Optional[str], Optional[str]]:
    audio_b64 = base64.standard_b64encode(audio_data).decode("utf-8")
    
    for model_name in MODEL_PRIORITY:
        try:
            logger.info(f"Trying: {model_name}")
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=PROMPTS.get(mode, PROMPTS["summary"])
            )
            response = model.generate_content(
                [
                    {"inline_data": {"mime_type": mime_type, "data": audio_b64}},
                    "Process this audio according to your instructions."
                ],
                generation_config={"temperature": 0.7, "max_output_tokens": 8192}
            )
            logger.info(f"Success: {model_name}")
            return response.text, model_name
        except Exception as e:
            logger.warning(f"{model_name} failed: {e}")
            continue
    
    return None, None

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    mode = query.data
    
    if user_id not in user_audio_cache:
        await query.edit_message_text(MESSAGES["no_audio"])
        return
    
    try:
        audio_info = user_audio_cache[user_id]
        await query.edit_message_text(MESSAGES['processing'])
        
        result, model_used = await process_with_cascade(
            audio_info["data"], audio_info["mime_type"], mode
        )
        
        if result:
            full_text = f"✅ پردازش کامل شد\n\n{result}\n\n---\n🤖 {model_used}"
            
            if len(full_text) > 4000:
                await query.edit_message_text(full_text[:4000], parse_mode="Markdown")
                remaining = full_text[4000:]
                while remaining:
                    chunk = remaining[:4000]
                    remaining = remaining[4000:]
                    await context.bot.send_message(
                        chat_id=update.effective_chat.id, text=chunk
                    )
            else:
                try:
                    await query.edit_message_text(full_text, parse_mode="Markdown")
                except:
                    await query.edit_message_text(full_text)
        else:
            await query.edit_message_text(MESSAGES["all_failed"])
    
    except Exception as e:
        logger.error(f"Error: {e}")
        try:
            await query.edit_message_text(MESSAGES["error"])
        except:
            pass
    finally:
        if user_id in user_audio_cache:
            del user_audio_cache[user_id]
            logger.info(f"Cache cleaned: user={user_id}")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Error: {context.error}")

def main():
    if not TELEGRAM_BOT_TOKEN:
        print("❌ TELEGRAM_BOT_TOKEN not set in Secrets!")
        return
    if not GEMINI_API_KEY:
        print("❌ GEMINI_API_KEY not set in Secrets!")
        return
    
    print("🚀 Starting Omni-Hear AI...")
    print(f"🔄 Models: {MODEL_PRIORITY}")
    
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", start_command))
    app.add_handler(MessageHandler(
        filters.VOICE | filters.AUDIO | filters.Document.AUDIO,
        handle_audio
    ))
    app.add_handler(CallbackQueryHandler(button_callback))
    app.add_error_handler(error_handler)
    
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == "__main__":
    main()
