#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        OMNI-HEAR AI v3.0 (GROQ Edition)                      ║
║                    High-Performance Audio Processing Bot                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  🚀 STT: Groq Whisper Large V3 (14,400 requests/day)                         ║
║  🧠 LLM: Llama 3.3 70B Versatile                                             ║
║  🌍 Auto Language Detection (Persian/English)                                 ║
║  ⚡ Lightning Fast: ~3-5 seconds processing time                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import logging
import asyncio
import tempfile
import traceback
from pathlib import Path
from typing import Optional, Tuple

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

from groq import Groq
from pydub import AudioSegment

# ============== LOGGING ==============
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ============== CONFIGURATION ==============
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ============== GROQ CLIENT ==============
groq_client: Optional[Groq] = None

if not GROQ_API_KEY:
    logger.error("❌ GROQ_API_KEY is not set!")
else:
    logger.info(f"✅ GROQ_API_KEY configured (length: {len(GROQ_API_KEY)})")
    groq_client = Groq(api_key=GROQ_API_KEY)

# ============== MODEL CONFIGURATION ==============
# Groq Models - Super fast with generous limits!
WHISPER_MODEL = "whisper-large-v3"  # STT: 14,400 requests/day
LLM_MODEL_PRIMARY = "llama-3.3-70b-versatile"  # Primary LLM
LLM_MODEL_FALLBACK = "llama-3.1-70b-versatile"  # Fallback LLM
LLM_MODEL_FAST = "llama-3.1-8b-instant"  # Fast fallback

MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB (Groq supports larger files)

# ============== SYSTEM PROMPTS ==============
PROMPTS = {
    "transcript": """You are an expert transcription formatter.

**INPUT:** Raw transcription text from audio.

**YOUR TASK:** Clean and format this transcription perfectly.

**RULES:**
1. Fix any obvious transcription errors
2. Add proper punctuation (. , ? !)
3. Break into logical paragraphs
4. If multiple speakers detected, mark as [Speaker 1], [Speaker 2]
5. Keep the EXACT language - if Persian, output Persian. If English, output English.
6. If mixed (Persian with English words), preserve English words in Latin script
   Example: "من یک meeting داشتم" stays exactly like this
7. Mark unclear parts as [نامفهوم] or [unclear]

**OUTPUT:** Clean, formatted transcription in the original language.""",

    "lecture": """You are a distinguished University Professor writing educational content.

**INPUT:** Transcription of an educational audio/lecture.

**YOUR TASK:** Transform this into a comprehensive **Textbook Chapter in Persian (Farsi)**.

**REQUIREMENTS:**
1. زبان خروجی: فارسی روان و آکادمیک
2. Use **bold headers** (با ** علامت‌گذاری) for sections
3. Cover EVERY detail, example, and nuance from the audio
4. Add explanations where helpful
5. Structure logically with clear sections
6. The reader should NOT need to listen to the audio after reading this
7. Keep technical terms in their original form (English terms stay English)

**FORMAT:**
## عنوان اصلی
### بخش اول
متن کامل...
### بخش دوم
متن کامل...

**OUTPUT LANGUAGE: PERSIAN (FARSI) ONLY - فارسی**""",

    "soap": """You are a Chief Resident physician at a major teaching hospital.

**INPUT:** Transcription of a medical dictation or patient encounter.

**YOUR TASK:** Create a professional **SOAP Note in English**.

**FORMAT (STRICT):**

**SUBJECTIVE:**
- Chief Complaint (CC):
- History of Present Illness (HPI):
- Review of Systems (ROS):
- Past Medical History (PMH):
- Medications:
- Allergies:
- Social/Family History:

**OBJECTIVE:**
- Vital Signs:
- Physical Examination:
- Laboratory Results:
- Imaging/Diagnostics:

**ASSESSMENT:**
- Primary Diagnosis:
- Differential Diagnoses:
- ICD-10 Codes (if applicable):

**PLAN:**
- Treatment:
- Medications Prescribed:
- Follow-up:
- Patient Education:
- Referrals:

**RULES:**
1. OUTPUT MUST BE IN ENGLISH ONLY
2. Use proper medical terminology
3. Correct any medical term errors in transcription
4. Be thorough but concise
5. Include all relevant clinical details

**OUTPUT LANGUAGE: ENGLISH ONLY**""",

    "summary": """You are an expert summarizer.

**INPUT:** Transcription of audio content.

**YOUR TASK:** Create a clear, concise summary in **Persian (Farsi)**.

**FORMAT:**
• استفاده از bullet points با علامت •
• هر نکته در یک خط
• تمرکز بر مهم‌ترین اطلاعات
• حذف جزئیات غیرضروری
• زبان ساده و روان

**STRUCTURE:**
📌 **خلاصه کلی:**
یک پاراگراف کوتاه

📋 **نکات اصلی:**
• نکته اول
• نکته دوم
• نکته سوم
...

🎯 **نتیجه‌گیری:**
جمع‌بندی نهایی

**OUTPUT LANGUAGE: PERSIAN (FARSI) ONLY - فارسی**""",

    "lyrics": """You are a music and speech transcription specialist.

**INPUT:** Transcription of audio (music or speech).

**YOUR TASK:** 
- If MUSIC: Extract complete lyrics with proper formatting
- If SPEECH: Provide clean verbatim transcription

**FORMAT FOR LYRICS:**
[Verse 1]
Line 1
Line 2

[Chorus]
Line 1
Line 2

[Verse 2]
...

**RULES:**
1. Keep original language (don't translate)
2. Proper line breaks for readability
3. Mark instrumental sections as [Instrumental], [Music], etc.
4. For speech: use paragraphs and punctuation

**OUTPUT LANGUAGE: SAME AS INPUT (preserve original)**"""
}

# ============== PERSIAN MESSAGES ==============
MESSAGES = {
    "welcome": """🎧 **به Omni-Hear AI خوش آمدید!**
    
🚀 **نسخه 3.0 - موتور Groq**

🎤 یک فایل صوتی یا ویس ارسال کنید.

⚡ **قابلیت‌ها:**
• 📜 رونویسی کامل (Transcript)
• 📚 درسنامه کامل (فارسی)
• 🩺 شرح‌حال پزشکی SOAP (انگلیسی)  
• 📝 خلاصه متن (فارسی)
• 🎵 متن آهنگ (Lyrics)

🌟 **ویژگی‌های جدید:**
• ⚡ سرعت فوق‌العاده (~۵ ثانیه)
• 🎯 تشخیص خودکار زبان
• 🔄 ۱۴,۴۰۰ درخواست در روز!""",

    "audio_received": "🎵 **فایل دریافت شد!**\n\n📋 نوع پردازش را انتخاب کنید:",
    
    "processing_stt": "🎤 **مرحله ۱/۲:** تبدیل صدا به متن...\n⏱ چند ثانیه صبر کنید",
    
    "processing_llm": "🧠 **مرحله ۲/۲:** پردازش هوشمند متن...\n⏱ تقریباً آماده است",
    
    "error": "❌ خطا در پردازش. لطفاً دوباره تلاش کنید.",
    
    "quota_exceeded": "⚠️ سقف درخواست‌ها پر شده.\n💡 لطفاً کمی صبر کنید.",
    
    "no_audio": "⚠️ لطفاً ابتدا یک فایل صوتی ارسال کنید.",
    
    "file_too_large": "⚠️ حجم فایل بیشتر از ۲۵ مگابایت است.",
    
    "not_audio": "⚠️ لطفاً یک فایل صوتی ارسال کنید.",
    
    "api_key_missing": "⚠️ تنظیمات سرور ناقص است.\n\n🔑 GROQ_API_KEY تنظیم نشده!",
    
    "transcription_failed": "❌ خطا در تبدیل صدا به متن.\n\n💡 مطمئن شوید فایل صوتی سالم است.",
}

# ============== USER CACHE ==============
user_audio_cache: dict = {}


# ============== KEYBOARD ==============
def get_menu_keyboard() -> InlineKeyboardMarkup:
    """Create the Persian inline keyboard menu."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📜 رونویسی کامل", callback_data="transcript"),
        ],
        [
            InlineKeyboardButton("📚 درسنامه کامل", callback_data="lecture"),
            InlineKeyboardButton("🩺 شرح‌حال SOAP", callback_data="soap"),
        ],
        [
            InlineKeyboardButton("📝 خلاصه متن", callback_data="summary"),
            InlineKeyboardButton("🎵 متن آهنگ", callback_data="lyrics"),
        ],
    ])


# ============== AUDIO CONVERSION ==============
async def convert_audio_to_mp3(audio_data: bytes, original_format: str = "ogg") -> Tuple[Optional[bytes], Optional[str]]:
    """
    Convert audio to MP3 format for Groq Whisper compatibility.
    Returns: (mp3_bytes, error_message)
    """
    try:
        def _convert():
            # Create temp file for input
            with tempfile.NamedTemporaryFile(suffix=f".{original_format}", delete=False) as input_file:
                input_file.write(audio_data)
                input_path = input_file.name
            
            try:
                # Load audio with pydub
                if original_format in ["ogg", "oga"]:
                    audio = AudioSegment.from_ogg(input_path)
                elif original_format == "mp3":
                    audio = AudioSegment.from_mp3(input_path)
                elif original_format == "wav":
                    audio = AudioSegment.from_wav(input_path)
                elif original_format == "m4a":
                    audio = AudioSegment.from_file(input_path, format="m4a")
                else:
                    audio = AudioSegment.from_file(input_path)
                
                # Export as MP3
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as output_file:
                    output_path = output_file.name
                
                audio.export(output_path, format="mp3", bitrate="128k")
                
                with open(output_path, "rb") as f:
                    mp3_data = f.read()
                
                # Cleanup
                os.unlink(output_path)
                
                return mp3_data, None
                
            finally:
                # Cleanup input file
                if os.path.exists(input_path):
                    os.unlink(input_path)
        
        return await asyncio.to_thread(_convert)
        
    except Exception as e:
        logger.error(f"Audio conversion error: {e}")
        return None, str(e)


# ============== GROQ STT (WHISPER) ==============
async def transcribe_audio(audio_data: bytes, filename: str = "audio.mp3") -> Tuple[Optional[str], Optional[str]]:
    """
    Transcribe audio using Groq Whisper.
    Returns: (transcription_text, error_message)
    """
    if not groq_client:
        return None, "Groq client not initialized"
    
    try:
        def _transcribe():
            # Create temp file for Groq
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name
            
            try:
                with open(temp_path, "rb") as audio_file:
                    transcription = groq_client.audio.transcriptions.create(
                        model=WHISPER_MODEL,
                        file=audio_file,
                        response_format="text",
                        language=None,  # Auto-detect language
                        temperature=0.0,  # More accurate
                    )
                return transcription, None
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        
        result, error = await asyncio.to_thread(_transcribe)
        
        if error:
            return None, error
        
        if result and len(result.strip()) > 0:
            logger.info(f"✅ Transcription successful: {len(result)} chars")
            return result.strip(), None
        else:
            return None, "Empty transcription result"
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Transcription error: {error_msg}")
        
        if "rate_limit" in error_msg.lower():
            return None, "rate_limit"
        elif "invalid_api_key" in error_msg.lower():
            return None, "invalid_api_key"
        else:
            return None, error_msg[:100]


# ============== GROQ LLM PROCESSING ==============
async def process_with_llm(
    transcription: str,
    mode: str
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Process transcription with Groq LLM.
    Returns: (result_text, model_used, error_message)
    """
    if not groq_client:
        return None, None, "Groq client not initialized"
    
    prompt = PROMPTS.get(mode, PROMPTS["summary"])
    
    # Model cascade
    models = [LLM_MODEL_PRIMARY, LLM_MODEL_FALLBACK, LLM_MODEL_FAST]
    
    for model_name in models:
        try:
            logger.info(f"🔄 Trying LLM: {model_name}")
            
            def _generate():
                return groq_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": prompt
                        },
                        {
                            "role": "user",
                            "content": f"Here is the transcription to process:\n\n{transcription}"
                        }
                    ],
                    temperature=0.7,
                    max_tokens=8000,
                    top_p=0.9,
                )
            
            response = await asyncio.to_thread(_generate)
            
            if response.choices and response.choices[0].message.content:
                result = response.choices[0].message.content.strip()
                logger.info(f"✅ LLM success with {model_name}: {len(result)} chars")
                return result, model_name, None
            else:
                logger.warning(f"⚠️ Empty response from {model_name}")
                continue
                
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"❌ {model_name} error: {error_msg[:80]}")
            
            if "rate_limit" in error_msg.lower():
                continue
            else:
                continue
    
    return None, None, "All LLM models failed"


# ============== FULL PROCESSING PIPELINE ==============
async def process_audio_full(
    audio_data: bytes,
    mime_type: str,
    mode: str,
    update_callback=None
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Full audio processing pipeline.
    Returns: (result_text, transcription, model_used, error_message)
    """
    # Step 1: Determine format and convert if needed
    format_map = {
        "audio/ogg": "ogg",
        "audio/oga": "ogg",
        "audio/opus": "ogg",
        "audio/mp3": "mp3",
        "audio/mpeg": "mp3",
        "audio/wav": "wav",
        "audio/x-wav": "wav",
        "audio/m4a": "m4a",
        "audio/mp4": "m4a",
    }
    
    original_format = format_map.get(mime_type, "ogg")
    
    # Convert to MP3 for best compatibility
    if original_format != "mp3":
        logger.info(f"🔄 Converting {original_format} to MP3...")
        mp3_data, conv_error = await convert_audio_to_mp3(audio_data, original_format)
        if conv_error:
            logger.warning(f"Conversion warning: {conv_error}, trying original format")
            mp3_data = audio_data
    else:
        mp3_data = audio_data
    
    # Step 2: Transcribe with Whisper
    logger.info("🎤 Starting transcription with Whisper...")
    transcription, stt_error = await transcribe_audio(mp3_data)
    
    if stt_error:
        if stt_error == "rate_limit":
            return None, None, None, "⚠️ سقف درخواست‌های تبدیل صدا پر شده. چند دقیقه صبر کنید."
        elif stt_error == "invalid_api_key":
            return None, None, None, "❌ کلید API نامعتبر است."
        else:
            return None, None, None, f"❌ خطا در تبدیل صدا: {stt_error}"
    
    if not transcription:
        return None, None, None, "❌ متنی از صدا استخراج نشد. مطمئن شوید فایل صوتی واضح است."
    
    # For transcript mode, just clean up the transcription
    if mode == "transcript":
        result, model_used, llm_error = await process_with_llm(transcription, mode)
        if llm_error:
            # If LLM fails, return raw transcription
            return transcription, transcription, WHISPER_MODEL, None
        return result, transcription, model_used, None
    
    # Step 3: Process with LLM for other modes
    logger.info(f"🧠 Processing with LLM for mode: {mode}")
    result, model_used, llm_error = await process_with_llm(transcription, mode)
    
    if llm_error:
        return None, transcription, None, f"❌ خطا در پردازش متن: {llm_error}"
    
    return result, transcription, model_used, None


# ============== TELEGRAM HANDLERS ==============
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    await update.message.reply_text(MESSAGES["welcome"], parse_mode="Markdown")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    help_text = """📖 **راهنمای استفاده از Omni-Hear AI**

**🔹 مراحل:**
1️⃣ یک فایل صوتی یا ویس ارسال کنید
2️⃣ از منو نوع پردازش را انتخاب کنید
3️⃣ منتظر نتیجه بمانید (۳-۱۰ ثانیه)

**🔹 حالت‌های پردازش:**

📜 **رونویسی کامل**
متن کامل کلمه به کلمه با تشخیص خودکار زبان

📚 **درسنامه کامل** 
تبدیل به متن درسی جامع به فارسی

🩺 **شرح‌حال پزشکی**
SOAP Note استاندارد به انگلیسی

📝 **خلاصه متن**
خلاصه نکات کلیدی به فارسی

🎵 **متن آهنگ**
استخراج لیریک یا متن گفتار

**🔹 نکات:**
• حداکثر حجم: ۲۵ مگابایت
• فرمت‌های پشتیبانی: MP3, OGG, WAV, M4A
• زبان: فارسی و انگلیسی (خودکار)

**🔹 دستورات:**
/start - شروع مجدد
/help - این راهنما
/status - وضعیت سرویس"""
    
    await update.message.reply_text(help_text, parse_mode="Markdown")


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /status command."""
    status_parts = ["🔍 **وضعیت سیستم Omni-Hear AI**\n"]
    
    # Telegram
    status_parts.append("✅ **Telegram:** متصل")
    
    # Groq API
    if GROQ_API_KEY:
        status_parts.append(f"✅ **Groq API Key:** تنظیم شده")
        
        # Test Groq connection
        if groq_client:
            try:
                test_response = await asyncio.to_thread(
                    lambda: groq_client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[{"role": "user", "content": "Say OK"}],
                        max_tokens=5
                    )
                )
                if test_response.choices:
                    status_parts.append("✅ **Groq API:** فعال و متصل ⚡")
            except Exception as e:
                status_parts.append(f"⚠️ **Groq API:** خطا - {str(e)[:50]}")
        else:
            status_parts.append("❌ **Groq Client:** مقداردهی نشده")
    else:
        status_parts.append("❌ **Groq API Key:** تنظیم نشده!")
    
    # Models info
    status_parts.append(f"\n**🤖 مدل‌ها:**")
    status_parts.append(f"• STT: `{WHISPER_MODEL}`")
    status_parts.append(f"• LLM: `{LLM_MODEL_PRIMARY}`")
    
    # Limits
    status_parts.append(f"\n**📊 محدودیت‌ها:**")
    status_parts.append(f"• Whisper: 14,400 req/day")
    status_parts.append(f"• LLM: 14,400 req/day")
    status_parts.append(f"• حجم فایل: 25MB")
    
    await update.message.reply_text("\n".join(status_parts), parse_mode="Markdown")


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming audio files and voice messages."""
    user_id = update.effective_user.id
    msg = update.message
    
    # Check API
    if not GROQ_API_KEY or not groq_client:
        await msg.reply_text(MESSAGES["api_key_missing"])
        return
    
    # Determine audio source
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
        # Download audio
        file = await context.bot.get_file(audio_file.file_id)
        
        if file.file_size and file.file_size > MAX_FILE_SIZE:
            await msg.reply_text(MESSAGES["file_too_large"])
            return
        
        audio_bytes = await file.download_as_bytearray()
        
        # Determine MIME type
        if file_type == "voice":
            mime_type = "audio/ogg"
        elif hasattr(audio_file, 'mime_type') and audio_file.mime_type:
            mime_type = audio_file.mime_type
        else:
            mime_type = "audio/mpeg"
        
        # Cache audio for this user
        user_audio_cache[user_id] = {
            "data": bytes(audio_bytes),
            "mime_type": mime_type,
            "size": len(audio_bytes)
        }
        
        logger.info(f"✅ Audio cached: user={user_id}, size={len(audio_bytes)}, mime={mime_type}")
        
        # Show menu
        await msg.reply_text(
            MESSAGES["audio_received"],
            reply_markup=get_menu_keyboard(),
            parse_mode="Markdown"
        )
        
    except Exception as e:
        logger.error(f"Error downloading audio for user {user_id}: {e}")
        await msg.reply_text(MESSAGES["error"])


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle button callbacks."""
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    mode = query.data
    
    # Check if user has audio cached
    if user_id not in user_audio_cache:
        await query.edit_message_text(MESSAGES["no_audio"])
        return
    
    audio_info = user_audio_cache[user_id]
    
    # Mode names for display
    mode_names = {
        "transcript": "📜 رونویسی کامل",
        "lecture": "📚 درسنامه کامل",
        "soap": "🩺 شرح‌حال پزشکی",
        "summary": "📝 خلاصه متن",
        "lyrics": "🎵 متن آهنگ"
    }
    
    try:
        # Update: Processing STT
        await query.edit_message_text(
            f"🎯 **حالت:** {mode_names.get(mode, mode)}\n\n"
            f"{MESSAGES['processing_stt']}",
            parse_mode="Markdown"
        )
        
        # Process audio
        result, transcription, model_used, error = await process_audio_full(
            audio_info["data"],
            audio_info["mime_type"],
            mode
        )
        
        if error:
            await query.edit_message_text(error)
            return
        
        if not result:
            await query.edit_message_text(MESSAGES["error"])
            return
        
        # Format response
        header = f"✅ **{mode_names.get(mode, 'پردازش')}**\n\n"
        
        # Add transcription preview for non-transcript modes
        if mode != "transcript" and transcription:
            trans_preview = transcription[:200] + "..." if len(transcription) > 200 else transcription
            header += f"📝 **متن اصلی:**\n_{trans_preview}_\n\n---\n\n"
        
        footer = f"\n\n---\n🤖 مدل: `{model_used}`\n⚡ Powered by Groq"
        
        full_text = header + result + footer
        
        # Handle long messages (Telegram limit: 4096)
        if len(full_text) > 4000:
            # Send first chunk
            try:
                await query.edit_message_text(full_text[:4000], parse_mode="Markdown")
            except Exception:
                await query.edit_message_text(full_text[:4000])
            
            # Send remaining chunks
            remaining = full_text[4000:]
            while remaining:
                chunk = remaining[:4000]
                remaining = remaining[4000:]
                await asyncio.sleep(0.3)
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
                # Fallback without markdown if parsing fails
                await query.edit_message_text(full_text)
    
    except Exception as e:
        logger.error(f"Callback error for user {user_id}: {e}")
        logger.error(traceback.format_exc())
        try:
            await query.edit_message_text(f"❌ خطا: {str(e)[:100]}")
        except Exception:
            pass
    
    finally:
        # Cleanup cache
        if user_id in user_audio_cache:
            del user_audio_cache[user_id]
            logger.info(f"🧹 Cache cleaned: user={user_id}")


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle errors globally."""
    logger.error(f"Error: {context.error}")
    if update:
        logger.error(f"Update that caused error: {update}")


# ============== MAIN ==============
def main() -> None:
    """Start the bot."""
    print("\n" + "="*65)
    print("  🎧 OMNI-HEAR AI v3.0 - GROQ EDITION")
    print("  ⚡ Ultra-fast Speech-to-Text + LLM Processing")
    print("="*65)
    
    # Validate tokens
    if not TELEGRAM_BOT_TOKEN:
        logger.error("❌ TELEGRAM_BOT_TOKEN not set!")
        print("\n⚠️  Set environment variable: TELEGRAM_BOT_TOKEN")
        sys.exit(1)
    
    if not GROQ_API_KEY:
        logger.error("❌ GROQ_API_KEY not set!")
        print("\n⚠️  Set environment variable: GROQ_API_KEY")
        print("   Get it from: https://console.groq.com/keys")
        sys.exit(1)
    
    print(f"✅ Telegram Bot: Ready")
    print(f"✅ Groq API: Configured")
    print(f"🎤 STT Model: {WHISPER_MODEL}")
    print(f"🧠 LLM Model: {LLM_MODEL_PRIMARY}")
    print("="*65 + "\n")
    
    # Build application
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Register handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("status", status_command))
    
    app.add_handler(MessageHandler(
        filters.VOICE | filters.AUDIO | filters.Document.AUDIO,
        handle_audio
    ))
    
    app.add_handler(CallbackQueryHandler(button_callback))
    
    app.add_error_handler(error_handler)
    
    # Start polling
    logger.info("🚀 Bot starting with Groq backend...")
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
