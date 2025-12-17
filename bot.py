#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           OMNI-HEAR AI v2.2                                  ║
║         Bilingual Telegram Bot with Proxy Support & Smart Cascade            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  🌐 PROXY SUPPORT: HTTP/SOCKS5 proxy for restricted networks                 ║
║  🔄 SMART CASCADE: Auto-switches between models on failure                   ║
║  🧹 MEMORY SAFE: Audio cache cleaned properly                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import logging
import base64
import asyncio
from typing import Optional, List, Tuple

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)
from telegram.request import HTTPXRequest

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

# ============== LOGGING ==============
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ============== CONFIGURATION ==============
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ============== PROXY CONFIGURATION ==============
# Supports: http://host:port or socks5://user:pass@host:port
PROXY_URL = os.getenv("PROXY_URL")  # Example: "socks5://127.0.0.1:1080"

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ============== MODEL CASCADE ==============
MODEL_PRIORITY: List[str] = [
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.0-flash",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
]

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB

# ============== SYSTEM PROMPTS ==============
PROMPTS = {
    "lecture": """You are a University Professor teaching in Persian (Farsi).
Listen to this audio carefully. Do NOT summarize.
Write a comprehensive **Textbook Chapter in Persian**.
Cover every single detail, example, and nuance mentioned.
Use bold headers to organize sections.
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
If it contains music: Extract and provide the complete lyrics in original language.
If it contains speech: Provide a verbatim transcription in original language.
Format the output cleanly with proper line breaks."""
}

# ============== PERSIAN MESSAGES ==============
MESSAGES = {
    "welcome": """🎧 **به Omni-Hear AI خوش آمدید!**

🎤 یک فایل صوتی یا ویس ارسال کنید.

⚡ قابلیت‌ها:
• 📚 درسنامه کامل (فارسی)
• 🩺 شرح‌حال پزشکی SOAP (انگلیسی)
• 📝 خلاصه متن (فارسی)
• 🎵 متن آهنگ

🔄 مجهز به Smart Model Cascade
🌐 پشتیبانی از Proxy""",
    "audio_received": "🎵 فایل دریافت شد!\n\n📋 نوع پردازش را انتخاب کنید:",
    "processing": "⏳ در حال پردازش...",
    "error": "❌ خطا در پردازش. لطفاً دوباره تلاش کنید.",
    "all_failed": "❌ تمام مدل‌ها با خطا مواجه شدند. لطفاً بعداً تلاش کنید.",
    "no_audio": "⚠️ لطفاً ابتدا یک فایل صوتی ارسال کنید.",
    "file_too_large": "⚠️ حجم فایل بیشتر از ۲۰ مگابایت است.",
    "not_audio": "⚠️ لطفاً یک فایل صوتی ارسال کنید.",
    "network_error": "🌐 خطای اتصال به شبکه. لطفاً Proxy را بررسی کنید.",
}

# Audio cache
user_audio_cache: dict = {}


def get_menu_keyboard() -> InlineKeyboardMarkup:
    """Create Persian inline keyboard menu."""
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
    """Handle /start command."""
    await update.message.reply_text(MESSAGES["welcome"], parse_mode="Markdown")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    help_text = """📖 **راهنمای استفاده**

1️⃣ یک فایل صوتی یا ویس ارسال کنید
2️⃣ از منو نوع پردازش را انتخاب کنید
3️⃣ منتظر نتیجه بمانید

**حالت‌های پردازش:**
📚 **درسنامه کامل** - متن درسی کامل (فارسی)
🩺 **شرح‌حال پزشکی** - SOAP Note (انگلیسی)
📝 **خلاصه متن** - خلاصه نکات (فارسی)
🎵 **متن آهنگ** - استخراج لیریک

💡 حداکثر حجم: ۲۰ مگابایت"""
    await update.message.reply_text(help_text, parse_mode="Markdown")


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming audio files and voice messages."""
    user_id = update.effective_user.id
    msg = update.message
    
    # Detect audio type
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
    
    # File size check
    file_size = getattr(audio_file, 'file_size', 0)
    if file_size and file_size > MAX_FILE_SIZE:
        logger.warning(f"User {user_id}: file too large ({file_size} bytes)")
        await msg.reply_text(MESSAGES["file_too_large"])
        return
    
    try:
        file = await context.bot.get_file(audio_file.file_id)
        
        if file.file_size and file.file_size > MAX_FILE_SIZE:
            await msg.reply_text(MESSAGES["file_too_large"])
            return
        
        audio_bytes = await file.download_as_bytearray()
        
        # Determine mime type
        if file_type == "voice":
            mime_type = "audio/ogg"
        elif hasattr(audio_file, 'mime_type') and audio_file.mime_type:
            mime_type = audio_file.mime_type
        else:
            mime_type = "audio/mpeg"
        
        # Cache audio
        user_audio_cache[user_id] = {
            "data": bytes(audio_bytes),
            "mime_type": mime_type
        }
        
        logger.info(f"Audio cached: user={user_id}, size={len(audio_bytes)}, mime={mime_type}")
        await msg.reply_text(MESSAGES["audio_received"], reply_markup=get_menu_keyboard())
        
    except Exception as e:
        logger.error(f"Error downloading audio for user {user_id}: {e}")
        await msg.reply_text(MESSAGES["error"])


async def process_with_cascade(
    audio_data: bytes,
    mime_type: str,
    mode: str
) -> Tuple[Optional[str], Optional[str]]:
    """
    Smart Model Cascade - tries each model until one succeeds.
    """
    audio_b64 = base64.standard_b64encode(audio_data).decode("utf-8")
    
    for i, model_name in enumerate(MODEL_PRIORITY):
        try:
            logger.info(f"Trying model {i+1}/{len(MODEL_PRIORITY)}: {model_name}")
            
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=PROMPTS.get(mode, PROMPTS["summary"])
            )
            
            response = await asyncio.to_thread(
                model.generate_content,
                [
                    {"inline_data": {"mime_type": mime_type, "data": audio_b64}},
                    "Process this audio according to your instructions."
                ],
                generation_config={"temperature": 0.7, "max_output_tokens": 8192}
            )
            
            logger.info(f"✅ Success with: {model_name}")
            return response.text, model_name
            
        except (google_exceptions.NotFound,
                google_exceptions.ResourceExhausted,
                google_exceptions.ServiceUnavailable,
                google_exceptions.InvalidArgument) as e:
            logger.warning(f"Model {model_name} failed: {e}")
            continue
        except Exception as e:
            logger.error(f"Model {model_name} unexpected error: {e}")
            continue
    
    logger.error("All models failed!")
    return None, None


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle button callbacks."""
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    mode = query.data
    
    if user_id not in user_audio_cache:
        await query.edit_message_text(MESSAGES["no_audio"])
        return
    
    try:
        audio_info = user_audio_cache[user_id]
        
        mode_names = {
            "lecture": "📚 درسنامه",
            "soap": "🩺 پزشکی",
            "summary": "📝 خلاصه",
            "lyrics": "🎵 متن آهنگ"
        }
        
        await query.edit_message_text(
            f"{MESSAGES['processing']}\n🔄 {mode_names.get(mode, mode)}"
        )
        
        result, model_used = await process_with_cascade(
            audio_info["data"],
            audio_info["mime_type"],
            mode
        )
        
        if result:
            full_text = f"✅ **پردازش کامل شد**\n\n{result}\n\n---\n🤖 `{model_used}`"
            
            # Handle long messages
            if len(full_text) > 4000:
                try:
                    await query.edit_message_text(full_text[:4000], parse_mode="Markdown")
                except Exception:
                    await query.edit_message_text(full_text[:4000])
                
                remaining = full_text[4000:]
                while remaining:
                    chunk = remaining[:4000]
                    remaining = remaining[4000:]
                    try:
                        await context.bot.send_message(
                            chat_id=update.effective_chat.id,
                            text=chunk,
                            parse_mode="Markdown"
                        )
                    except Exception:
                        await context.bot.send_message(
                            chat_id=update.effective_chat.id,
                            text=chunk
                        )
            else:
                try:
                    await query.edit_message_text(full_text, parse_mode="Markdown")
                except Exception:
                    await query.edit_message_text(full_text)
        else:
            await query.edit_message_text(MESSAGES["all_failed"])
    
    except Exception as e:
        logger.error(f"Callback error for user {user_id}: {e}")
        try:
            await query.edit_message_text(MESSAGES["error"])
        except Exception:
            pass
    
    finally:
        # Cleanup cache
        if user_id in user_audio_cache:
            del user_audio_cache[user_id]
            logger.info(f"🧹 Cache cleaned: user={user_id}")


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle errors."""
    logger.error(f"Update {update} caused error: {context.error}")


def main() -> None:
    """Start the bot."""
    # Validate tokens
    if not TELEGRAM_BOT_TOKEN:
        logger.error("❌ TELEGRAM_BOT_TOKEN not set!")
        print("\n⚠️  Please set environment variables:")
        print("   export TELEGRAM_BOT_TOKEN='your_bot_token'")
        print("   export GEMINI_API_KEY='your_gemini_key'")
        print("   export PROXY_URL='socks5://host:port'  # Optional\n")
        sys.exit(1)
    
    if not GEMINI_API_KEY:
        logger.error("❌ GEMINI_API_KEY not set!")
        sys.exit(1)
    
    logger.info("🔄 Model Cascade: " + " → ".join(MODEL_PRIORITY))
    
    # ============== BUILD WITH PROXY SUPPORT ==============
    builder = Application.builder().token(TELEGRAM_BOT_TOKEN)
    
    if PROXY_URL:
        logger.info(f"🌐 Using proxy: {PROXY_URL[:20]}...")
        # Create custom request with proxy
        request = HTTPXRequest(
            proxy=PROXY_URL,
            connect_timeout=30.0,
            read_timeout=60.0,
            write_timeout=60.0,
            pool_timeout=60.0,
        )
        builder = builder.request(request)
        builder = builder.get_updates_request(HTTPXRequest(proxy=PROXY_URL))
    else:
        logger.info("🌐 No proxy configured. Direct connection.")
    
    app = builder.build()
    
    # Add handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(
        filters.VOICE | filters.AUDIO | filters.Document.AUDIO,
        handle_audio
    ))
    app.add_handler(CallbackQueryHandler(button_callback))
    app.add_error_handler(error_handler)
    
    # Run
    logger.info("🚀 Starting Omni-Hear AI v2.2...")
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
