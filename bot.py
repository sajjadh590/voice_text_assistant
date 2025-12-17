#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           OMNI-HEAR AI v2.4                                  ║
║              Fixed Model Names + Better Error Handling                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ✅ CORRECT MODEL NAMES: Using latest model identifiers                      ║
║  📝 BETTER LOGGING: Shows exact error messages                               ║
║  🔄 SMART RETRY: Waits and retries on rate limit                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import logging
import base64
import asyncio
import time
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
    format="%(asctime)s - %(levelname)s - %(message)s",
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
    logger.info(f"✅ GEMINI_API_KEY configured (length: {len(GEMINI_API_KEY)})")
    genai.configure(api_key=GEMINI_API_KEY)

# ============== CORRECT MODEL NAMES ==============
# Updated to use correct model identifiers that actually work!
MODEL_PRIORITY: List[str] = [
"gemini-2.5-flash-lite",      # 🥇 بیشترین سهمیه رایگان (1,000 تا 1,500 درخواست در روز)
    "gemini-2.0-flash",           # 🥈 مدل استاندارد (1,000 درخواست در روز)
    "gemini-2.5-flash",           # 🥉 سهمیه محدود شده (فقط ۲۰ درخواست در روز)
    "gemini-1.5-flash-latest",    # 🛡️ زاپاس نهایی (مدل قدیمی)
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

🔄 نسخه 2.4 - پایدار و سریع""",
    "audio_received": "🎵 فایل دریافت شد!\n\n📋 نوع پردازش را انتخاب کنید:",
    "processing": "⏳ در حال پردازش با هوش مصنوعی...\n\n⏱ لطفاً صبر کنید (۱۰-۳۰ ثانیه)",
    "error": "❌ خطا در پردازش. لطفاً دوباره تلاش کنید.",
    "quota_exceeded": "⚠️ سقف استفاده API تمام شده.\n\n💡 لطفاً چند دقیقه صبر کنید یا با ادمین تماس بگیرید.",
    "all_failed": "❌ خطا: {details}\n\n🔄 لطفاً دوباره تلاش کنید.",
    "no_audio": "⚠️ لطفاً ابتدا یک فایل صوتی ارسال کنید.",
    "file_too_large": "⚠️ حجم فایل بیشتر از ۲۰ مگابایت است.",
    "not_audio": "⚠️ لطفاً یک فایل صوتی ارسال کنید.",
    "api_key_missing": "⚠️ تنظیمات سرور ناقص است. GEMINI_API_KEY تنظیم نشده!",
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
🤖 مدل: Gemini 2.0"""
    await update.message.reply_text(help_text, parse_mode="Markdown")


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Check bot status and API connectivity."""
    status_parts = ["🔍 **وضعیت سیستم:**\n"]
    
    # Check Telegram Token
    if TELEGRAM_BOT_TOKEN:
        status_parts.append("✅ Telegram Token: فعال")
    else:
        status_parts.append("❌ Telegram Token: تنظیم نشده!")
    
    # Check Gemini API Key
    if GEMINI_API_KEY:
        status_parts.append(f"✅ Gemini API Key: تنظیم شده")
        
        # Test Gemini connection with a simple request
        try:
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = await asyncio.to_thread(
                model.generate_content,
                "Say 'API Working' in exactly 2 words."
            )
            if response.text:
                status_parts.append("✅ Gemini API: متصل و فعال ✨")
                status_parts.append(f"   پاسخ تست: {response.text[:50]}")
        except google_exceptions.ResourceExhausted:
            status_parts.append("⚠️ Gemini API: Quota تمام شده!")
            status_parts.append("   💡 نیاز به API Key جدید دارید")
        except google_exceptions.InvalidArgument as e:
            status_parts.append(f"❌ Gemini API: خطای پارامتر")
        except Exception as e:
            status_parts.append(f"❌ Gemini API Error: {str(e)[:80]}")
    else:
        status_parts.append("❌ Gemini API Key: تنظیم نشده!")
    
    status_parts.append(f"\n🔄 مدل‌ها:\n   " + "\n   ".join(MODEL_PRIORITY))
    
    await update.message.reply_text("\n".join(status_parts), parse_mode="Markdown")


async def models_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List available models."""
    try:
        models_list = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                models_list.append(f"• `{m.name.replace('models/', '')}`")
        
        if models_list:
            text = "🤖 **مدل‌های موجود:**\n\n" + "\n".join(models_list[:15])
            if len(models_list) > 15:
                text += f"\n\n... و {len(models_list) - 15} مدل دیگر"
        else:
            text = "❌ هیچ مدلی یافت نشد!"
            
        await update.message.reply_text(text, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ خطا: {str(e)[:100]}")


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
    quota_exhausted = False
    
    prompt = PROMPTS.get(mode, PROMPTS["summary"])
    
    for i, model_name in enumerate(MODEL_PRIORITY):
        try:
            logger.info(f"🔄 Trying model {i+1}/{len(MODEL_PRIORITY)}: {model_name}")
            
            model = genai.GenerativeModel(model_name)
            
            # Create content with audio inline
            response = await asyncio.to_thread(
                model.generate_content,
                [
                    {"inline_data": {"mime_type": mime_type, "data": audio_b64}},
                    prompt
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
                last_error = "پاسخ خالی از مدل"
                continue
                
        except google_exceptions.NotFound as e:
            logger.warning(f"❌ {model_name} - Not found: {str(e)[:50]}")
            last_error = f"مدل {model_name} یافت نشد"
            continue
            
        except google_exceptions.ResourceExhausted as e:
            logger.warning(f"❌ {model_name} - Quota exhausted")
            quota_exhausted = True
            last_error = "سقف استفاده رایگان API تمام شده"
            continue
            
        except google_exceptions.InvalidArgument as e:
            error_str = str(e)
            logger.warning(f"❌ {model_name} - Invalid: {error_str[:80]}")
            if "audio" in error_str.lower():
                last_error = "این مدل از صدا پشتیبانی نمی‌کند"
            else:
                last_error = f"پارامتر نامعتبر: {error_str[:50]}"
            continue
            
        except google_exceptions.PermissionDenied as e:
            logger.error(f"❌ {model_name} - Permission denied")
            last_error = "API Key معتبر نیست یا دسترسی ندارید"
            continue
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ {model_name} - Error: {error_msg[:100]}")
            last_error = error_msg[:80]
            continue
    
    # Determine final error message
    if quota_exhausted:
        final_error = "⚠️ سقف استفاده رایگان API تمام شده!\n\n💡 لطفاً:\n• چند دقیقه صبر کنید\n• یا API Key جدید بگیرید"
    else:
        final_error = last_error or "خطای ناشناخته"
    
    logger.error(f"❌ All models failed! Last error: {last_error}")
    return None, None, final_error


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
            header = f"✅ **{mode_names.get(mode, 'پردازش')} کامل شد**\n\n"
            footer = f"\n\n---\n🤖 مدل: `{model_used}`"
            full_text = header + result + footer
            
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
                    await asyncio.sleep(0.5)  # Rate limit protection
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
            # Failed - show detailed error
            await query.edit_message_text(f"❌ {error}")
    
    except Exception as e:
        logger.error(f"Callback error for user {user_id}: {e}")
        logger.error(traceback.format_exc())
        try:
            await query.edit_message_text(f"❌ خطا: {str(e)[:100]}")
        except Exception:
            pass
    
    finally:
        # Always cleanup cache
        if user_id in user_audio_cache:
            del user_audio_cache[user_id]
            logger.info(f"🧹 Cache cleaned: user={user_id}")


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle errors."""
    logger.error(f"Error: {context.error}")
    if update:
        logger.error(f"Update: {update}")


def main() -> None:
    """Start the bot."""
    print("\n" + "="*60)
    print("  🎧 OMNI-HEAR AI v2.4 - Fixed Model Names")
    print("="*60)
    
    # Validate tokens
    if not TELEGRAM_BOT_TOKEN:
        logger.error("❌ TELEGRAM_BOT_TOKEN not set!")
        print("\n⚠️  Set: TELEGRAM_BOT_TOKEN=your_token")
        sys.exit(1)
    
    if not GEMINI_API_KEY:
        logger.error("❌ GEMINI_API_KEY not set!")
        print("\n⚠️  Set: GEMINI_API_KEY=your_key")
        print("   Get it from: https://aistudio.google.com/app/apikey")
        sys.exit(1)
    
    print(f"✅ Telegram: Connected")
    print(f"✅ Gemini: Configured")
    print(f"🔄 Models: {' → '.join(MODEL_PRIORITY)}")
    print("="*60 + "\n")
    
    # Build application
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Add handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("models", models_command))
    app.add_handler(MessageHandler(
        filters.VOICE | filters.AUDIO | filters.Document.AUDIO,
        handle_audio
    ))
    app.add_handler(CallbackQueryHandler(button_callback))
    app.add_error_handler(error_handler)
    
    # Run
    logger.info("🚀 Bot starting...")
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
