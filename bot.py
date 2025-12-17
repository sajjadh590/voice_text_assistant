#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           OMNI-HEAR AI v2.3                                  ║
║              Bilingual Telegram Bot - Fixed & Stable Version                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  🔄 STABLE MODELS: Uses proven Gemini models                                 ║
║  📝 BETTER LOGGING: Shows exact error messages                               ║
║  🧹 MEMORY SAFE: Audio cache cleaned properly                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import logging
import base64
import asyncio
import traceback
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

# ============== VALIDATE API KEY ==============
if not GEMINI_API_KEY:
    logger.error("❌ GEMINI_API_KEY is not set!")
    print("⚠️  Please set GEMINI_API_KEY environment variable")
else:
    logger.info(f"✅ GEMINI_API_KEY found: {GEMINI_API_KEY[:10]}...")
    genai.configure(api_key=GEMINI_API_KEY)

# ============== STABLE MODEL CASCADE ==============
# Using proven models that work reliably
MODEL_PRIORITY: List[str] = [
    "gemini-1.5-flash",      # Most stable, fast
    "gemini-1.5-pro",        # More capable
    "gemini-2.0-flash-exp",  # Experimental but good
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

🔄 نسخه 2.3 - پایدار و سریع""",
    "audio_received": "🎵 فایل دریافت شد!\n\n📋 نوع پردازش را انتخاب کنید:",
    "processing": "⏳ در حال پردازش با هوش مصنوعی...\n\n⏱ لطفاً صبر کنید (۱۰-۳۰ ثانیه)",
    "error": "❌ خطا در پردازش. لطفاً دوباره تلاش کنید.",
    "error_detail": "❌ خطا: {error}\n\nلطفاً دوباره تلاش کنید.",
    "all_failed": "❌ تمام مدل‌ها با خطا مواجه شدند.\n\n🔍 جزئیات: {details}\n\nلطفاً بعداً تلاش کنید.",
    "no_audio": "⚠️ لطفاً ابتدا یک فایل صوتی ارسال کنید.",
    "file_too_large": "⚠️ حجم فایل بیشتر از ۲۰ مگابایت است.",
    "not_audio": "⚠️ لطفاً یک فایل صوتی ارسال کنید.",
    "api_key_missing": "⚠️ تنظیمات سرور ناقص است. لطفاً با ادمین تماس بگیرید.",
}

# Store user audio files temporarily
user_audio_cache: dict = {}


def get_menu_keyboard() -> InlineKeyboardMarkup:
    """Create the Persian inline keyboard menu."""
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
    """Handle the /start command."""
    await update.message.reply_text(MESSAGES["welcome"], parse_mode="Markdown")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /help command."""
    help_text = """📖 **راهنمای استفاده**

1️⃣ یک فایل صوتی یا ویس ارسال کنید
2️⃣ از منو نوع پردازش را انتخاب کنید
3️⃣ منتظر نتیجه بمانید (۱۰-۳۰ ثانیه)

**حالت‌های پردازش:**
📚 **درسنامه کامل** - متن درسی کامل به فارسی
🩺 **شرح‌حال پزشکی** - SOAP Note به انگلیسی
📝 **خلاصه متن** - خلاصه نکات به فارسی
🎵 **متن آهنگ** - استخراج متن/لیریک

💡 حداکثر حجم: ۲۰ مگابایت
🤖 مدل: Gemini 1.5"""
    await update.message.reply_text(help_text, parse_mode="Markdown")


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Check bot status and API connectivity."""
    status_parts = ["🔍 **وضعیت سیستم:**\n"]
    
    # Check Telegram Token
    if TELEGRAM_BOT_TOKEN:
        status_parts.append("✅ Telegram Token: تنظیم شده")
    else:
        status_parts.append("❌ Telegram Token: تنظیم نشده!")
    
    # Check Gemini API Key
    if GEMINI_API_KEY:
        status_parts.append(f"✅ Gemini API Key: {GEMINI_API_KEY[:8]}...")
        
        # Test Gemini connection
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content("Say 'OK' in one word.")
            status_parts.append("✅ Gemini API: متصل و فعال")
        except Exception as e:
            status_parts.append(f"❌ Gemini API Error: {str(e)[:50]}")
    else:
        status_parts.append("❌ Gemini API Key: تنظیم نشده!")
    
    status_parts.append(f"\n🔄 مدل‌ها: {', '.join(MODEL_PRIORITY)}")
    
    await update.message.reply_text("\n".join(status_parts), parse_mode="Markdown")


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming audio files and voice messages."""
    user_id = update.effective_user.id
    msg = update.message
    
    # Check API Key first
    if not GEMINI_API_KEY:
        await msg.reply_text(MESSAGES["api_key_missing"])
        return
    
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
        
        logger.info(f"✅ Audio cached: user={user_id}, size={len(audio_bytes)}, mime={mime_type}")
        await msg.reply_text(MESSAGES["audio_received"], reply_markup=get_menu_keyboard())
        
    except Exception as e:
        logger.error(f"Error downloading audio for user {user_id}: {e}")
        await msg.reply_text(MESSAGES["error"])


async def process_with_cascade(
    audio_data: bytes,
    mime_type: str,
    mode: str
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Process audio with model cascade.
    Returns: (result_text, model_used, error_message)
    """
    audio_b64 = base64.standard_b64encode(audio_data).decode("utf-8")
    last_error = None
    
    for i, model_name in enumerate(MODEL_PRIORITY):
        try:
            logger.info(f"🔄 Trying model {i+1}/{len(MODEL_PRIORITY)}: {model_name}")
            
            model = genai.GenerativeModel(model_name)
            
            # Create content with audio
            response = await asyncio.to_thread(
                model.generate_content,
                [
                    {"inline_data": {"mime_type": mime_type, "data": audio_b64}},
                    PROMPTS.get(mode, PROMPTS["summary"])
                ],
                generation_config={
                    "temperature": 0.7,
                    "max_output_tokens": 8192
                }
            )
            
            if response.text:
                logger.info(f"✅ Success with: {model_name}")
                return response.text, model_name, None
            else:
                logger.warning(f"⚠️ Empty response from {model_name}")
                last_error = "Empty response"
                continue
                
        except google_exceptions.InvalidArgument as e:
            error_msg = str(e)
            logger.warning(f"❌ {model_name} - Invalid argument: {error_msg[:100]}")
            last_error = f"Model doesn't support audio: {error_msg[:50]}"
            continue
            
        except google_exceptions.NotFound as e:
            logger.warning(f"❌ {model_name} - Model not found: {e}")
            last_error = f"Model not found: {model_name}"
            continue
            
        except google_exceptions.ResourceExhausted as e:
            logger.warning(f"❌ {model_name} - Quota exhausted: {e}")
            last_error = "API quota exhausted"
            continue
            
        except google_exceptions.PermissionDenied as e:
            logger.error(f"❌ {model_name} - Permission denied: {e}")
            last_error = "Invalid API key or no access"
            continue
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ {model_name} - Unexpected error: {error_msg}")
            logger.error(traceback.format_exc())
            last_error = error_msg[:100]
            continue
    
    logger.error(f"❌ All models failed! Last error: {last_error}")
    return None, None, last_error


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
            "lecture": "📚 درسنامه کامل",
            "soap": "🩺 شرح‌حال پزشکی",
            "summary": "📝 خلاصه متن",
            "lyrics": "🎵 متن آهنگ"
        }
        
        await query.edit_message_text(
            f"{MESSAGES['processing']}\n\n🎯 حالت: {mode_names.get(mode, mode)}"
        )
        
        result, model_used, error = await process_with_cascade(
            audio_info["data"],
            audio_info["mime_type"],
            mode
        )
        
        if result:
            # Success
            full_text = f"✅ **{mode_names.get(mode, 'پردازش')} کامل شد**\n\n{result}\n\n---\n🤖 `{model_used}`"
            
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
            # All models failed - show error details
            error_msg = MESSAGES["all_failed"].format(details=error or "Unknown error")
            await query.edit_message_text(error_msg)
    
    except Exception as e:
        logger.error(f"Callback error for user {user_id}: {e}")
        logger.error(traceback.format_exc())
        try:
            await query.edit_message_text(MESSAGES["error_detail"].format(error=str(e)[:100]))
        except Exception:
            pass
    
    finally:
        # Always cleanup cache
        if user_id in user_audio_cache:
            del user_audio_cache[user_id]
            logger.info(f"🧹 Cache cleaned: user={user_id}")


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle errors."""
    logger.error(f"Update {update} caused error: {context.error}")
    logger.error(traceback.format_exc())


def main() -> None:
    """Start the bot."""
    print("\n" + "="*50)
    print("🎧 OMNI-HEAR AI v2.3")
    print("="*50)
    
    # Validate tokens
    if not TELEGRAM_BOT_TOKEN:
        logger.error("❌ TELEGRAM_BOT_TOKEN not set!")
        print("\n⚠️  Set environment variables:")
        print("   TELEGRAM_BOT_TOKEN=your_bot_token")
        print("   GEMINI_API_KEY=your_gemini_key\n")
        sys.exit(1)
    
    if not GEMINI_API_KEY:
        logger.error("❌ GEMINI_API_KEY not set!")
        print("\n⚠️  Get your API key from:")
        print("   https://aistudio.google.com/app/apikey\n")
        sys.exit(1)
    
    print(f"✅ Telegram Token: {TELEGRAM_BOT_TOKEN[:10]}...")
    print(f"✅ Gemini API Key: {GEMINI_API_KEY[:10]}...")
    print(f"🔄 Models: {' → '.join(MODEL_PRIORITY)}")
    print("="*50 + "\n")
    
    # Build application
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Add handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(MessageHandler(
        filters.VOICE | filters.AUDIO | filters.Document.AUDIO,
        handle_audio
    ))
    app.add_handler(CallbackQueryHandler(button_callback))
    app.add_error_handler(error_handler)
    
    # Run
    logger.info("🚀 Starting Omni-Hear AI v2.3...")
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
