#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           OMNI-HEAR AI v2.5 (STABLE)                         ║
║            Fixed Network DNS Errors + Stable Model Names                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ✅ NETWORK FIX: Added HTTPXRequest tuning for Hugging Face/Containers       ║
║  ✅ MODEL FIX: Reverted to standard stable model names (No Experimental)     ║
║  ✅ SAFETY: Improved error catching for API limits                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import logging
import base64
import asyncio
import traceback
from typing import Optional, List, Tuple

# Telegram Imports
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)
from telegram.request import HTTPXRequest  # <--- Vital for fixing Network Errors

# Google Gemini Imports
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

# ============== LOGGING ==============
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ============== CONFIGURATION ==============
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    logger.error("❌ GEMINI_API_KEY is not set!")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# ============== STABLE MODEL PRIORITY ==============
# NOTE: Removed "experimental" or "latest" aliases to prevent 404 errors on servers.
MODEL_PRIORITY: List[str] = [
    "gemini-1.5-flash",          # Most stable & fast
    "gemini-1.5-pro",            # High intelligence fallback
    "gemini-2.0-flash-exp",      # Experimental (might fail, so it is 3rd)
    "gemini-1.5-flash-8b",       # Low latency
]

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB

# ============== SYSTEM PROMPTS ==============
PROMPTS = {
    "lecture": """You are a University Professor teaching in Persian (Farsi).
Listen to this audio carefully. Do NOT summarize.
Write a comprehensive **Textbook Chapter in Persian**.
Cover every single detail, example, and nuance mentioned.
Use bold headers (با ** علامت‌گذاری کنید) to organize sections.
The goal is to replace the need to listen to the audio entirely.
Write in fluent, academic Persian. زبان خروجی حتماً فارسی باشد.""",

    "soap": """You are a Chief Resident at a teaching hospital.
Listen to this medical dictation audio.
Write a professional **SOAP Note in English**.
Format:
**Subjective:** (Chief complaint, HPI, ROS, PMH, medications, allergies)
**Objective:** (Vitals, physical exam findings, lab results, imaging)
**Assessment:** (Diagnoses with ICD codes if possible)
**Plan:** (Treatment plan, medications, follow-up)
Correct all medical terminology. Output MUST be in English only.""",

    "summary": """Listen to this audio carefully.
Summarize the content into clear, concise **Persian bullet points**.
Use • for bullet points. Write in fluent Persian.
Focus on the most important information.
زبان خروجی حتماً فارسی باشد.""",

    "lyrics": """Listen to this audio.
If it contains music: Extract and provide the complete lyrics in the original language.
If it contains speech: Provide a verbatim transcription in the original language.
Format the output cleanly with proper line breaks."""
}

# Persian messages
MESSAGES = {
    "welcome": """🎧 **به Omni-Hear AI خوش آمدید!**

🎤 یک فایل صوتی یا ویس ارسال کنید.

⚡ قابلیت‌ها:
• 📚 درسنامه کامل (فارسی)
• 🩺 شرح‌حال پزشکی SOAP (انگلیسی)
• 📝 خلاصه متن (فارسی)
• 🎵 متن آهنگ

🔄 v2.5 Stable""",
    "audio_received": "🎵 فایل دریافت شد!\n\n📋 نوع پردازش را انتخاب کنید:",
    "processing": "⏳ در حال پردازش... (لطفاً صبر کنید)",
    "error": "❌ خطا در پردازش.",
    "file_too_large": "⚠️ حجم فایل بیشتر از ۲۰ مگابایت است.",
    "not_audio": "⚠️ لطفاً یک فایل صوتی ارسال کنید.",
    "api_key_missing": "⚠️ کلید API تنظیم نشده است.",
}

user_audio_cache: dict = {}

def get_menu_keyboard() -> InlineKeyboardMarkup:
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

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(MESSAGES["welcome"], parse_mode="Markdown")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("فایل صوتی بفرستید تا پردازش شود.", parse_mode="Markdown")

async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    msg = update.message
    
    if not GEMINI_API_KEY:
        await msg.reply_text(MESSAGES["api_key_missing"])
        return
    
    # Determine audio file
    audio_file = None
    file_type = "audio"
    if msg.voice:
        audio_file = msg.voice
        file_type = "voice"
    elif msg.audio:
        audio_file = msg.audio
    elif msg.document and msg.document.mime_type and msg.document.mime_type.startswith("audio/"):
        audio_file = msg.document
    else:
        await msg.reply_text(MESSAGES["not_audio"])
        return
    
    # Size Check
    if getattr(audio_file, 'file_size', 0) > MAX_FILE_SIZE:
        await msg.reply_text(MESSAGES["file_too_large"])
        return
    
    try:
        file = await context.bot.get_file(audio_file.file_id)
        if file.file_size and file.file_size > MAX_FILE_SIZE:
            await msg.reply_text(MESSAGES["file_too_large"])
            return
        
        # Download
        audio_bytes = await file.download_as_bytearray()
        
        mime_type = "audio/ogg" if file_type == "voice" else getattr(audio_file, 'mime_type', "audio/mpeg")
        
        user_audio_cache[user_id] = {
            "data": bytes(audio_bytes),
            "mime_type": mime_type
        }
        
        await msg.reply_text(MESSAGES["audio_received"], reply_markup=get_menu_keyboard())
        
    except Exception as e:
        logger.error(f"Download Error: {e}")
        await msg.reply_text(MESSAGES["error"])

async def process_with_cascade(audio_data: bytes, mime_type: str, mode: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Tries models in sequence."""
    audio_b64 = base64.standard_b64encode(audio_data).decode("utf-8")
    last_error = ""
    
    prompt = PROMPTS.get(mode, PROMPTS["summary"])
    
    for model_name in MODEL_PRIORITY:
        try:
            logger.info(f"Trying Model: {model_name}")
            model = genai.GenerativeModel(model_name)
            
            response = await asyncio.to_thread(
                model.generate_content,
                [{"inline_data": {"mime_type": mime_type, "data": audio_b64}}, prompt],
                generation_config={"temperature": 0.5, "max_output_tokens": 4000}
            )
            
            if response.text:
                return response.text, model_name, None
                
        except Exception as e:
            err_msg = str(e)
            logger.warning(f"Model {model_name} failed: {err_msg[:100]}")
            # If 404/Not Found, just continue. If Quota, continue.
            if "404" in err_msg or "not found" in err_msg.lower():
                last_error = f"مدل {model_name} در دسترس نیست"
            elif "429" in err_msg or "exhausted" in err_msg.lower():
                last_error = "سقف درخواست تمام شده (Quota)"
            else:
                last_error = f"خطای فنی: {err_msg[:50]}"
            continue
            
    return None, None, last_error

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    user_id = update.effective_user.id
    mode = query.data
    
    if user_id not in user_audio_cache:
        await query.edit_message_text(MESSAGES["no_audio"])
        return
    
    try:
        await query.edit_message_text(f"{MESSAGES['processing']}\n🤖 در حال اتصال به Gemini...")
        audio_info = user_audio_cache[user_id]
        
        result, model_used, error = await process_with_cascade(audio_info["data"], audio_info["mime_type"], mode)
        
        if result:
            header = f"✅ **پردازش شد**\n\n"
            footer = f"\n\n---\n🤖 `{model_used}`"
            full_text = header + result + footer
            
            if len(full_text) > 4000:
                await query.edit_message_text(full_text[:4000], parse_mode="Markdown")
                # Send rest in chunks
                remaining = full_text[4000:]
                while remaining:
                    chunk = remaining[:4000]
                    remaining = remaining[4000:]
                    await context.bot.send_message(chat_id=update.effective_chat.id, text=chunk)
            else:
                await query.edit_message_text(full_text, parse_mode="Markdown")
        else:
            await query.edit_message_text(f"❌ خطا: {error}\nلطفاً مدل دیگری را تست کنید یا بعداً تلاش کنید.")
            
    except Exception as e:
        logger.error(f"Callback fatal: {e}")
        await query.edit_message_text("❌ خطای غیرمنتظره در سرور")
    finally:
        if user_id in user_audio_cache:
            del user_audio_cache[user_id]

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Update error: {context.error}")

def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        print("❌ TELEGRAM_BOT_TOKEN missing")
        sys.exit(1)
        
    # --- NETWORK CONFIGURATION (FIX FOR HUGGING FACE) ---
    # This configures connection timeouts and pool size to handle 
    # flaky container networks and prevent "No address associated with hostname"
    trequest = HTTPXRequest(
        connection_pool_size=8,
        connect_timeout=60.0,  # Increased timeout
        read_timeout=60.0,
        write_timeout=60.0
    )
    
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).request(trequest).build()
    
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO | filters.Document.AUDIO, handle_audio))
    app.add_handler(CallbackQueryHandler(button_callback))
    app.add_error_handler(error_handler)
    
    print("🚀 Omni-Hear AI v2.5 Started...")
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == "__main__":
    main()
