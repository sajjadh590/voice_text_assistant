#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                OMNI-HEAR AI v7.0 (AssemblyAI + Groq Edition)                 ║
║         🎤 AssemblyAI STT | 🧠 Groq Dual-LLM | 🔄 Persistent Sessions        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  🎤 STT: AssemblyAI (Best-in-class accuracy + Language Detection)            ║
║  ⚡ Fast LLM: Groq Llama 3.1 8B Instant                                      ║
║  🧠 Complex LLM: Groq Llama 3.3 70B Versatile                                ║
║  🔄 Persistent Audio: Process same file multiple times                       ║
║  🌍 7 Languages | Auto Language Detection | Progress Tracking                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import logging
import asyncio
import tempfile
import traceback
import time
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

import assemblyai as aai
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
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ============== API CLIENTS ==============
groq_client: Optional[Groq] = None
aai_transcriber = None

# Initialize AssemblyAI
if ASSEMBLYAI_API_KEY:
    aai.settings.api_key = ASSEMBLYAI_API_KEY
    logger.info("✅ AssemblyAI configured")
else:
    logger.error("❌ ASSEMBLYAI_API_KEY not set!")

# Initialize Groq
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
    logger.info("✅ Groq client initialized")
else:
    logger.error("❌ GROQ_API_KEY not set!")

# ============== MODEL CONFIGURATION ==============
# Groq Models
GROQ_MODEL_FAST = "llama-3.1-8b-instant"        # Fast: Transcript, Lyrics, Quick tasks
GROQ_MODEL_COMPLEX = "llama-3.3-70b-versatile"  # Complex: Lecture, SOAP, Detailed tasks

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB (Telegram limit)


# ============== TASK COMPLEXITY ==============
class TaskComplexity(Enum):
    FAST = "fast"       # Simple tasks - 8B model
    COMPLEX = "complex"  # Heavy tasks - 70B model


# Mode to complexity mapping
MODE_COMPLEXITY = {
    "transcript": TaskComplexity.FAST,
    "lyrics": TaskComplexity.FAST,
    "summary_quick": TaskComplexity.FAST,
    "translate_quick": TaskComplexity.FAST,
    "lecture": TaskComplexity.COMPLEX,
    "soap": TaskComplexity.COMPLEX,
    "summary_detailed": TaskComplexity.COMPLEX,
    "translate_detailed": TaskComplexity.COMPLEX,
}


# ============== LANGUAGES ==============
@dataclass
class Language:
    code: str
    name_en: str
    name_native: str
    flag: str
    assemblyai_code: str  # AssemblyAI language code


LANGUAGES: Dict[str, Language] = {
    "fa": Language("fa", "Persian", "فارسی", "🇮🇷", "fa"),
    "en": Language("en", "English", "English", "🇬🇧", "en"),
    "fr": Language("fr", "French", "Français", "🇫🇷", "fr"),
    "es": Language("es", "Spanish", "Español", "🇪🇸", "es"),
    "ru": Language("ru", "Russian", "Русский", "🇷🇺", "ru"),
    "de": Language("de", "German", "Deutsch", "🇩🇪", "de"),
    "ar": Language("ar", "Arabic", "العربية", "🇸🇦", "ar"),
}

# AssemblyAI language code to our code mapping
AAI_LANG_MAP = {
    "fa": "fa", "en": "en", "en_us": "en", "en_uk": "en", "en_au": "en",
    "fr": "fr", "es": "es", "ru": "ru", "de": "de", "ar": "ar",
}


# ============== USER STATE (PERSISTENT) ==============
user_audio_cache: Dict[int, dict] = {}  # Stores audio data
user_state: Dict[int, dict] = {}        # Stores workflow state


def get_cached_audio(user_id: int) -> Optional[dict]:
    """Get cached audio for user."""
    return user_audio_cache.get(user_id)


def clear_user_cache(user_id: int):
    """Clear all cached data for user."""
    user_audio_cache.pop(user_id, None)
    user_state.pop(user_id, None)


# ============== SYSTEM PROMPTS ==============

def get_transcript_prompt(detected_lang: str) -> str:
    """Simple transcript formatting prompt."""
    lang = LANGUAGES.get(detected_lang, LANGUAGES["en"])
    return f"""You are a professional transcription editor.

TASK: Clean and format this raw transcription.

RULES:
1. Fix obvious errors while preserving meaning
2. Add proper punctuation (. , ? ! :)
3. Create logical paragraphs
4. Mark multiple speakers as [Speaker 1], [Speaker 2]
5. Keep the ORIGINAL language ({lang.name_en})
6. Preserve mixed-language words as-is

OUTPUT: Formatted transcription in {lang.name_en}."""


def get_lecture_prompt(detected_lang: str) -> str:
    """Academic lecture prompt - outputs in detected language."""
    lang = LANGUAGES.get(detected_lang, LANGUAGES["fa"])
    
    if detected_lang == "fa":
        return """نقش: استاد برجسته دانشگاه با تجربه ۲۰ ساله در تدریس و نگارش کتب مرجع.

وظیفه: تبدیل رونویسی این صوت آموزشی به یک **فصل جامع کتاب درسی** به زبان فارسی.

═══════════════════════════════════════════════════════════════
                      📚 فصل درسی آکادمیک
═══════════════════════════════════════════════════════════════

ساختار الزامی:

**۱. مقدمه علمی**
━━━━━━━━━━━━━━━━━━━
- تعریف موضوع
- اهمیت علمی/بالینی
- اهداف یادگیری

**۲. متن اصلی**
━━━━━━━━━━━━━━━━━━━
- تقسیم‌بندی با **عناوین بولد**
- توضیح گام‌به‌گام از ساده به پیچیده
- مثال‌های کاربردی

**۳. نکات کلیدی (Clinical Pearls) 💎**
━━━━━━━━━━━━━━━━━━━
- نکات مهم برای حفظ کردن
- اشتباهات رایج

**۴. جدول خلاصه 📊**
━━━━━━━━━━━━━━━━━━━
| موضوع | توضیح |
|-------|-------|

**۵. خلاصه فصل**
━━━━━━━━━━━━━━━━━━━
- مرور نکات کلیدی

**۶. سؤالات مروری**
━━━━━━━━━━━━━━━━━━━
- ۳ سؤال خودآزمایی

═══════════════════════════════════════════════════════════════

الزامات نگارشی:
• زبان فارسی رسمی و آکادمیک
• اصطلاحات تخصصی فارسی + (معادل انگلیسی)
• بدون کلمات عامیانه

زبان خروجی: فقط فارسی"""

    elif detected_lang == "ar":
        return """الدور: أستاذ جامعي متميز.
المهمة: تحويل هذا النص إلى فصل كتاب أكاديمي شامل باللغة العربية.
الهيكل: مقدمة، محتوى رئيسي مع عناوين، نقاط رئيسية، جدول، ملخص، أسئلة.
لغة الإخراج: العربية فقط"""

    else:
        return f"""Role: Distinguished University Professor with 20+ years of teaching experience.

Task: Transform this transcription into a comprehensive **Textbook Chapter** in {lang.name_en}.

STRUCTURE:

## 1. Introduction
- Topic definition
- Scientific importance
- Learning objectives

## 2. Main Content
- Organized with **bold headers**
- Step-by-step explanations
- Practical examples

## 3. Clinical Pearls 💎
- Key points to remember
- Common mistakes

## 4. Summary Table 📊
| Topic | Description |
|-------|-------------|

## 5. Chapter Summary
- Key points review

## 6. Review Questions
- 3 self-assessment questions

OUTPUT LANGUAGE: {lang.name_en} ONLY"""


def get_soap_prompt() -> str:
    """Medical SOAP note - always English."""
    return """Role: Senior Board-Certified Attending Physician.

Task: Transform this medical dictation into a US Medical Standard SOAP Note.

FORMAT:

═══════════════════════════════════════════════════════════════
                         SOAP NOTE
═══════════════════════════════════════════════════════════════

📋 **SUBJECTIVE**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Chief Complaint (CC):** [Patient's words]

**History of Present Illness (HPI):**
- Onset:
- Location:
- Duration:
- Character:
- Aggravating/Alleviating:
- Severity (1-10):

**Review of Systems (ROS):**
□ Constitutional | □ HEENT | □ Cardiovascular | □ Respiratory
□ GI | □ GU | □ MSK | □ Neuro | □ Psych | □ Skin

**PMH:** | **PSH:** | **Medications:** | **Allergies:**

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔬 **OBJECTIVE**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Vitals:** BP: /mmHg | HR: bpm | RR: /min | Temp: °F | SpO2: %

**Physical Exam:**
- General:
- HEENT:
- Cardiovascular:
- Pulmonary:
- Abdomen:
- Extremities:
- Neuro:

**Labs/Imaging:**

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 **ASSESSMENT**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Primary Diagnosis:** [Diagnosis] — ICD-10: [Code]

**Differential:**
1. [DDx 1]
2. [DDx 2]
3. [DDx 3]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 **PLAN**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Diagnostics:** [ ]
**Treatment:** [ ]
**Medications:** [ ]
**Patient Education:** [ ]
**Follow-up:** [ ]
**Referrals:** [ ]

═══════════════════════════════════════════════════════════════

RULES:
1. OUTPUT IN ENGLISH ONLY
2. Correct medical terminology errors
3. Include ICD-10 codes
4. Mark missing info as "Not documented"

OUTPUT LANGUAGE: ENGLISH ONLY"""


def get_summary_prompt(detected_lang: str, detailed: bool = False) -> str:
    """Summary prompt."""
    lang = LANGUAGES.get(detected_lang, LANGUAGES["fa"])
    
    if detailed:
        return f"""Role: Expert Content Analyst.

Task: Create a comprehensive summary in {lang.name_en}.

FORMAT:

📌 **Executive Summary**
[3-4 sentences]

📋 **Key Points**
• [Point 1]
• [Point 2]
• [Point 3]
...

💡 **Important Details**
[Names, numbers, specifics]

🎯 **Conclusions**
[Main takeaways]

✅ **Action Items** (if any)

OUTPUT: {lang.name_en} only"""
    else:
        return f"""Summarize this content in {lang.name_en}.

Format:
• Overview (2 sentences)
• Key points (bullets)
• Conclusion

OUTPUT: {lang.name_en} only."""


def get_lyrics_prompt() -> str:
    """Lyrics extraction prompt."""
    return """Extract and format lyrics OR speech from this transcription.

FOR MUSIC:
🎵 **Song Info** (if identifiable)
- Title:
- Artist:

[Verse 1]
Lines...

[Chorus]
Lines...

[Verse 2]
...

FOR SPEECH:
Clean paragraphs with speaker identification.

RULES:
1. Keep ORIGINAL language
2. Mark unclear: [...]
3. Mark instrumental: [🎸 Instrumental]

OUTPUT: Original language, formatted."""


def get_translation_prompt(source_lang: str, target_lang: str, detailed: bool = False) -> str:
    """Translation prompt."""
    source = LANGUAGES.get(source_lang, LANGUAGES["en"])
    target = LANGUAGES.get(target_lang, LANGUAGES["fa"])
    
    if detailed:
        return f"""Role: Expert Translator fluent in {source.name_en} and {target.name_en}.

Task: Translate from {source.name_en} to {target.name_en}.

PRINCIPLES:
1. Preserve complete meaning
2. Use natural, idiomatic {target.name_en}
3. Maintain tone and style
4. Keep proper nouns
5. Translate idioms to equivalents

OUTPUT FORMAT:

📝 **Translation:**
[Full translation]

---

📌 **Summary:**
[2 sentences about content]

OUTPUT: {target.name_en} only"""
    else:
        return f"""Translate from {source.name_en} to {target.name_en}.
Keep meaning, use natural language.
OUTPUT: {target.name_en} only."""


# ============== UI MESSAGES ==============
MESSAGES = {
    "welcome": """🎧 **به Omni-Hear AI خوش آمدید!**

🚀 **نسخه 7.0 - AssemblyAI + Groq**

**🎤 موتور رونویسی:** AssemblyAI (دقت بالا)
**⚡ پردازش سریع:** Llama 8B
**🧠 پردازش پیشرفته:** Llama 70B

📤 **یک فایل صوتی ارسال کنید**

🔄 **قابلیت جدید:** پردازش چندباره روی یک فایل!

🌐 **زبان‌ها:**
🇮🇷 فارسی | 🇬🇧 English | 🇫🇷 Français
🇪🇸 Español | 🇩🇪 Deutsch | 🇷🇺 Русский | 🇸🇦 العربية""",

    "audio_received": """🎵 **فایل دریافت شد!** ({size})

⚡ **سریع** = پاسخ فوری (8B)
🧠 **پیشرفته** = کیفیت بالا (70B)

🔄 می‌توانید چند عملیات روی همین فایل انجام دهید!

📋 نوع پردازش را انتخاب کنید:""",

    "processing_stt": "🎤 **مرحله ۱/۲:** رونویسی با AssemblyAI...\n\n⏳ پیشرفت: {progress}%",
    "processing_llm_fast": "🧠 **مرحله ۲/۲:** پردازش سریع با Llama 8B...\n\n⏳ پیشرفت: {progress}%",
    "processing_llm_complex": "🧠 **مرحله ۲/۲:** پردازش پیشرفته با Llama 70B...\n\n⏳ پیشرفت: {progress}%",
    
    "operation_complete": "✅ **عملیات {mode} کامل شد!**\n\n🔄 می‌توانید عملیات دیگری روی همین فایل انجام دهید.",
    
    "select_language": "🌍 **زبان خروجی را انتخاب کنید:**",
    "select_source_lang": "🗣 **زبان صوت (مبدا):**",
    "select_target_lang": "🎯 **زبان ترجمه (مقصد):**",
    
    "detected_language": "🔍 **زبان تشخیص داده شده:** {lang}",
    
    "error": "❌ خطا در پردازش. لطفاً دوباره تلاش کنید.",
    "error_detail": "❌ خطا: {detail}",
    "no_audio": "⚠️ لطفاً ابتدا یک فایل صوتی ارسال کنید.",
    "file_too_large": "⚠️ حجم فایل بیشتر از ۲۰ مگابایت است.",
    "not_audio": "⚠️ لطفاً فایل صوتی ارسال کنید (MP3, OGG, WAV, M4A).",
    "api_missing": "⚠️ کلید API تنظیم نشده: {missing}",
    "session_expired": "⚠️ فایل صوتی منقضی شده. لطفاً دوباره ارسال کنید.",
}


# ============== KEYBOARDS ==============
def get_main_menu_keyboard() -> InlineKeyboardMarkup:
    """Main menu with dual options."""
    return InlineKeyboardMarkup([
        # Transcript
        [
            InlineKeyboardButton("📜 رونویسی ⚡", callback_data="mode:transcript:fast"),
        ],
        # Lecture
        [
            InlineKeyboardButton("📚 درسنامه 🧠", callback_data="mode:lecture:complex"),
        ],
        # Medical SOAP
        [
            InlineKeyboardButton("🩺 SOAP پزشکی 🧠", callback_data="mode:soap:complex"),
        ],
        # Summary
        [
            InlineKeyboardButton("📝 خلاصه ⚡", callback_data="mode:summary_quick:fast"),
            InlineKeyboardButton("📝 خلاصه جامع 🧠", callback_data="mode:summary_detailed:complex"),
        ],
        # Lyrics
        [
            InlineKeyboardButton("🎵 متن آهنگ ⚡", callback_data="mode:lyrics:fast"),
        ],
        # Translation
        [
            InlineKeyboardButton("🌍 ترجمه ⚡", callback_data="mode:translate_quick:fast"),
            InlineKeyboardButton("🌍 ترجمه دقیق 🧠", callback_data="mode:translate_detailed:complex"),
        ],
        # Clear session
        [
            InlineKeyboardButton("🗑 پاک کردن فایل", callback_data="clear:session"),
        ],
    ])


def get_back_to_menu_keyboard() -> InlineKeyboardMarkup:
    """Back to menu button after operation."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔙 بازگشت به منوی اصلی", callback_data="back:main")],
        [InlineKeyboardButton("🗑 پاک کردن و خروج", callback_data="clear:session")],
    ])


def get_language_keyboard(callback_prefix: str) -> InlineKeyboardMarkup:
    """Language selection keyboard."""
    buttons = []
    row = []
    
    for code, lang in LANGUAGES.items():
        btn = InlineKeyboardButton(
            f"{lang.flag} {lang.name_native}",
            callback_data=f"{callback_prefix}:{code}"
        )
        row.append(btn)
        if len(row) == 3:
            buttons.append(row)
            row = []
    
    if row:
        buttons.append(row)
    
    buttons.append([InlineKeyboardButton("🔙 بازگشت", callback_data="back:main")])
    return InlineKeyboardMarkup(buttons)


def get_target_language_keyboard(source_lang: str, callback_prefix: str) -> InlineKeyboardMarkup:
    """Target language keyboard excluding source."""
    buttons = []
    row = []
    
    for code, lang in LANGUAGES.items():
        if code == source_lang:
            continue
        btn = InlineKeyboardButton(
            f"{lang.flag} {lang.name_native}",
            callback_data=f"{callback_prefix}:{code}"
        )
        row.append(btn)
        if len(row) == 3:
            buttons.append(row)
            row = []
    
    if row:
        buttons.append(row)
    
    buttons.append([InlineKeyboardButton("🔙 بازگشت", callback_data="back:main")])
    return InlineKeyboardMarkup(buttons)


# ============== AUDIO PROCESSING ==============
async def convert_audio_to_mp3(audio_data: bytes, original_format: str = "ogg") -> Tuple[Optional[bytes], Optional[str]]:
    """Convert audio to MP3."""
    try:
        def _convert():
            with tempfile.NamedTemporaryFile(suffix=f".{original_format}", delete=False) as f:
                f.write(audio_data)
                input_path = f.name
            
            try:
                if original_format in ["ogg", "oga", "opus"]:
                    audio = AudioSegment.from_ogg(input_path)
                elif original_format == "mp3":
                    return audio_data, None
                elif original_format == "wav":
                    audio = AudioSegment.from_wav(input_path)
                elif original_format in ["m4a", "mp4"]:
                    audio = AudioSegment.from_file(input_path, format="m4a")
                else:
                    audio = AudioSegment.from_file(input_path)
                
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as out:
                    output_path = out.name
                
                audio.export(output_path, format="mp3", bitrate="128k")
                
                with open(output_path, "rb") as f:
                    mp3_data = f.read()
                
                os.unlink(output_path)
                return mp3_data, None
            finally:
                if os.path.exists(input_path):
                    os.unlink(input_path)
        
        return await asyncio.to_thread(_convert)
    except Exception as e:
        logger.error(f"Audio conversion error: {e}")
        return None, str(e)


# ============== ASSEMBLYAI STT ==============
async def transcribe_with_assemblyai(
    audio_data: bytes,
    progress_callback=None
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Transcribe with AssemblyAI using async polling.
    Returns: (transcription, detected_language, error)
    """
    if not ASSEMBLYAI_API_KEY:
        return None, None, "AssemblyAI not configured"
    
    try:
        # Save audio to temp file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(audio_data)
            temp_path = f.name
        
        try:
            # Configure transcriber with language detection
            config = aai.TranscriptionConfig(
                language_detection=True,  # Auto-detect language
                punctuate=True,
                format_text=True,
            )
            
            transcriber = aai.Transcriber()
            
            # Submit for transcription (async polling internally)
            if progress_callback:
                await progress_callback(10)
            
            def _transcribe():
                return transcriber.transcribe(temp_path, config=config)
            
            # Poll with progress updates
            if progress_callback:
                await progress_callback(20)
            
            transcript = await asyncio.to_thread(_transcribe)
            
            if progress_callback:
                await progress_callback(80)
            
            if transcript.status == aai.TranscriptStatus.error:
                return None, None, f"AssemblyAI error: {transcript.error}"
            
            if transcript.status == aai.TranscriptStatus.completed:
                text = transcript.text
                
                # Get detected language
                detected_lang = "en"  # Default
                if hasattr(transcript, 'language_code') and transcript.language_code:
                    detected_lang = AAI_LANG_MAP.get(transcript.language_code, "en")
                
                if progress_callback:
                    await progress_callback(100)
                
                logger.info(f"✅ AssemblyAI: {len(text)} chars, lang={detected_lang}")
                return text, detected_lang, None
            
            return None, None, f"Unexpected status: {transcript.status}"
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    except Exception as e:
        logger.error(f"AssemblyAI error: {e}")
        return None, None, str(e)[:100]


# ============== GROQ LLM ==============
async def process_with_groq(
    text: str,
    system_prompt: str,
    complexity: TaskComplexity,
    progress_callback=None
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Process with Groq LLM based on complexity."""
    if not groq_client:
        return None, None, "Groq not configured"
    
    # Select model based on complexity
    if complexity == TaskComplexity.FAST:
        models = [GROQ_MODEL_FAST, GROQ_MODEL_COMPLEX]
    else:
        models = [GROQ_MODEL_COMPLEX, GROQ_MODEL_FAST]
    
    for model in models:
        try:
            logger.info(f"🧠 Groq: {model}")
            
            if progress_callback:
                await progress_callback(30)
            
            def _generate():
                return groq_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Process this transcription:\n\n{text}"}
                    ],
                    temperature=0.7,
                    max_tokens=8000,
                )
            
            if progress_callback:
                await progress_callback(60)
            
            response = await asyncio.to_thread(_generate)
            
            if progress_callback:
                await progress_callback(90)
            
            if response.choices and response.choices[0].message.content:
                result = response.choices[0].message.content.strip()
                
                if progress_callback:
                    await progress_callback(100)
                
                model_label = "⚡ 8B" if model == GROQ_MODEL_FAST else "🧠 70B"
                logger.info(f"✅ Groq success: {len(result)} chars")
                return result, f"{model_label} ({model})", None
        
        except Exception as e:
            logger.warning(f"❌ Groq {model}: {str(e)[:50]}")
            continue
    
    return None, None, "All Groq models failed"


# ============== FULL PIPELINE ==============
async def process_audio_complete(
    audio_data: bytes,
    mime_type: str,
    mode: str,
    complexity: TaskComplexity,
    target_lang: Optional[str] = None,
    source_lang: Optional[str] = None,
    progress_callback=None,
) -> Dict:
    """Complete audio processing pipeline."""
    result = {
        "text": None,
        "transcription": None,
        "detected_lang": None,
        "model": None,
        "error": None,
    }
    
    # Format detection
    format_map = {
        "audio/ogg": "ogg", "audio/oga": "ogg", "audio/opus": "opus",
        "audio/mp3": "mp3", "audio/mpeg": "mp3",
        "audio/wav": "wav", "audio/x-wav": "wav",
        "audio/m4a": "m4a", "audio/mp4": "m4a",
    }
    original_format = format_map.get(mime_type, "ogg")
    
    # Convert to MP3
    if original_format != "mp3":
        mp3_data, _ = await convert_audio_to_mp3(audio_data, original_format)
        if not mp3_data:
            mp3_data = audio_data
    else:
        mp3_data = audio_data
    
    # Step 1: Transcribe with AssemblyAI
    async def stt_progress(p):
        if progress_callback:
            await progress_callback("stt", p)
    
    transcription, detected_lang, stt_error = await transcribe_with_assemblyai(
        mp3_data, stt_progress
    )
    
    if stt_error:
        result["error"] = f"❌ خطای AssemblyAI: {stt_error}"
        return result
    
    if not transcription:
        result["error"] = "❌ متنی استخراج نشد."
        return result
    
    result["transcription"] = transcription
    result["detected_lang"] = detected_lang
    
    # Step 2: Get appropriate prompt
    if mode == "transcript":
        prompt = get_transcript_prompt(detected_lang)
    elif mode == "lecture":
        prompt = get_lecture_prompt(detected_lang)
    elif mode == "soap":
        prompt = get_soap_prompt()
    elif mode in ["summary_quick", "summary_detailed"]:
        detailed = mode == "summary_detailed"
        prompt = get_summary_prompt(detected_lang, detailed)
    elif mode == "lyrics":
        prompt = get_lyrics_prompt()
    elif mode in ["translate_quick", "translate_detailed"]:
        if not source_lang:
            source_lang = detected_lang
        if not target_lang:
            result["error"] = "❌ زبان مقصد مشخص نشده"
            return result
        detailed = mode == "translate_detailed"
        prompt = get_translation_prompt(source_lang, target_lang, detailed)
    else:
        prompt = get_transcript_prompt(detected_lang)
    
    # Step 3: Process with Groq
    async def llm_progress(p):
        if progress_callback:
            await progress_callback("llm", p)
    
    text, model, llm_error = await process_with_groq(
        transcription, prompt, complexity, llm_progress
    )
    
    result["text"] = text
    result["model"] = model
    
    if llm_error and not text:
        result["error"] = f"❌ {llm_error}"
    
    return result


# ============== TELEGRAM HANDLERS ==============
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    clear_user_cache(user_id)
    await update.message.reply_text(MESSAGES["welcome"], parse_mode="Markdown")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = """📖 **راهنمای Omni-Hear AI v7.0**

**🔹 نحوه استفاده:**
1️⃣ فایل صوتی ارسال کنید
2️⃣ نوع پردازش را انتخاب کنید
3️⃣ می‌توانید چند عملیات روی همین فایل انجام دهید!

**🔹 موتورها:**
• ⚡ **سریع (8B):** رونویسی، لیریک، ترجمه سریع
• 🧠 **پیشرفته (70B):** درسنامه، SOAP، خلاصه جامع

**🔹 قابلیت‌ها:**
📜 رونویسی | 📚 درسنامه | 🩺 SOAP
📝 خلاصه | 🎵 لیریک | 🌍 ترجمه

**🔹 ویژگی جدید:**
🔄 پردازش چندباره روی یک فایل!

**🔹 دستورات:**
/start - شروع مجدد
/help - راهنما
/status - وضعیت"""
    
    await update.message.reply_text(help_text, parse_mode="Markdown")


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    has_audio = user_id in user_audio_cache
    
    status = ["🔍 **وضعیت سیستم v7.0**\n"]
    
    if ASSEMBLYAI_API_KEY:
        status.append("✅ **AssemblyAI (STT):** فعال")
    else:
        status.append("❌ **AssemblyAI:** غیرفعال")
    
    if groq_client:
        status.append("✅ **Groq (LLM):** فعال")
    else:
        status.append("❌ **Groq:** غیرفعال")
    
    status.append(f"\n**🤖 مدل‌ها:**")
    status.append(f"• Fast: `{GROQ_MODEL_FAST}`")
    status.append(f"• Complex: `{GROQ_MODEL_COMPLEX}`")
    
    status.append(f"\n**📁 وضعیت فایل شما:**")
    if has_audio:
        size = user_audio_cache[user_id].get("size", 0) / 1024
        status.append(f"✅ فایل موجود ({size:.1f} KB)")
    else:
        status.append("❌ فایلی ندارید")
    
    flags = " ".join([l.flag for l in LANGUAGES.values()])
    status.append(f"\n**🌍 زبان‌ها:** {flags}")
    
    await update.message.reply_text("\n".join(status), parse_mode="Markdown")


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle audio files."""
    user_id = update.effective_user.id
    msg = update.message
    
    # Check APIs
    missing = []
    if not ASSEMBLYAI_API_KEY:
        missing.append("ASSEMBLYAI_API_KEY")
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    
    if missing:
        await msg.reply_text(MESSAGES["api_missing"].format(missing=", ".join(missing)))
        return
    
    # Get audio
    audio_file = None
    if msg.voice:
        audio_file = msg.voice
    elif msg.audio:
        audio_file = msg.audio
    elif msg.document and msg.document.mime_type and msg.document.mime_type.startswith("audio/"):
        audio_file = msg.document
    else:
        await msg.reply_text(MESSAGES["not_audio"])
        return
    
    # Size check
    file_size = getattr(audio_file, 'file_size', 0)
    if file_size and file_size > MAX_FILE_SIZE:
        await msg.reply_text(MESSAGES["file_too_large"])
        return
    
    try:
        file = await context.bot.get_file(audio_file.file_id)
        audio_bytes = await file.download_as_bytearray()
        
        mime_type = "audio/ogg" if msg.voice else getattr(audio_file, 'mime_type', 'audio/mpeg')
        
        # Store in persistent cache
        user_audio_cache[user_id] = {
            "data": bytes(audio_bytes),
            "mime_type": mime_type,
            "size": len(audio_bytes),
            "timestamp": time.time(),
        }
        
        # Clear old state
        user_state.pop(user_id, None)
        
        size_kb = len(audio_bytes) / 1024
        size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb/1024:.1f} MB"
        
        logger.info(f"✅ Audio cached: user={user_id}, size={len(audio_bytes)}")
        
        await msg.reply_text(
            MESSAGES["audio_received"].format(size=size_str),
            reply_markup=get_main_menu_keyboard(),
            parse_mode="Markdown"
        )
    
    except Exception as e:
        logger.error(f"Audio error: {e}")
        await msg.reply_text(MESSAGES["error"])


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle button callbacks."""
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    data = query.data
    parts = data.split(":")
    action = parts[0]
    
    # Clear session
    if action == "clear":
        clear_user_cache(user_id)
        await query.edit_message_text(
            "🗑 **فایل پاک شد.**\n\n📤 برای شروع مجدد، یک فایل صوتی ارسال کنید.",
            parse_mode="Markdown"
        )
        return
    
    # Back to main menu
    if action == "back":
        if user_id in user_audio_cache:
            size_kb = user_audio_cache[user_id]["size"] / 1024
            size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb/1024:.1f} MB"
            await query.edit_message_text(
                MESSAGES["audio_received"].format(size=size_str),
                reply_markup=get_main_menu_keyboard(),
                parse_mode="Markdown"
            )
        else:
            await query.edit_message_text(MESSAGES["session_expired"])
        user_state.pop(user_id, None)
        return
    
    # Mode selection: mode:type:complexity
    if action == "mode":
        mode = parts[1]
        complexity_str = parts[2]
        complexity = TaskComplexity.COMPLEX if complexity_str == "complex" else TaskComplexity.FAST
        
        if user_id not in user_audio_cache:
            await query.edit_message_text(MESSAGES["session_expired"])
            return
        
        # Store state
        user_state[user_id] = {
            "mode": mode,
            "complexity": complexity,
        }
        
        # Translation needs target language selection
        if mode in ["translate_quick", "translate_detailed"]:
            await query.edit_message_text(
                MESSAGES["select_target_lang"],
                reply_markup=get_language_keyboard(f"target:{complexity_str}"),
                parse_mode="Markdown"
            )
            return
        
        # Process directly for other modes
        await process_and_respond(query, context, user_id, mode, complexity)
        return
    
    # Target language for translation: target:complexity:code
    if action == "target":
        complexity_str = parts[1]
        target_lang = parts[2]
        complexity = TaskComplexity.COMPLEX if complexity_str == "complex" else TaskComplexity.FAST
        
        state = user_state.get(user_id, {})
        mode = state.get("mode", "translate_quick")
        
        await process_and_respond(
            query, context, user_id, mode, complexity,
            target_lang=target_lang
        )
        return


async def process_and_respond(
    query,
    context,
    user_id: int,
    mode: str,
    complexity: TaskComplexity,
    target_lang: Optional[str] = None,
) -> None:
    """Process and send response with progress updates."""
    
    if user_id not in user_audio_cache:
        await query.edit_message_text(MESSAGES["session_expired"])
        return
    
    audio_info = user_audio_cache[user_id]
    
    mode_names = {
        "transcript": "📜 رونویسی",
        "lecture": "📚 درسنامه",
        "soap": "🩺 SOAP پزشکی",
        "summary_quick": "📝 خلاصه سریع",
        "summary_detailed": "📝 خلاصه جامع",
        "lyrics": "🎵 متن آهنگ",
        "translate_quick": "🌍 ترجمه سریع",
        "translate_detailed": "🌍 ترجمه دقیق",
    }
    
    current_stage = "stt"
    
    async def update_progress(stage: str, progress: int):
        nonlocal current_stage
        current_stage = stage
        
        if stage == "stt":
            msg = MESSAGES["processing_stt"].format(progress=progress)
        elif stage == "llm":
            if complexity == TaskComplexity.FAST:
                msg = MESSAGES["processing_llm_fast"].format(progress=progress)
            else:
                msg = MESSAGES["processing_llm_complex"].format(progress=progress)
        else:
            return
        
        try:
            await query.edit_message_text(
                f"🎯 **{mode_names.get(mode)}**\n\n{msg}",
                parse_mode="Markdown"
            )
        except Exception:
            pass  # Ignore rate limit errors
    
    try:
        # Initial progress
        await update_progress("stt", 0)
        
        # Process
        result = await process_audio_complete(
            audio_info["data"],
            audio_info["mime_type"],
            mode,
            complexity,
            target_lang=target_lang,
            progress_callback=update_progress,
        )
        
        if result["error"]:
            await query.edit_message_text(result["error"])
            return
        
        if not result["text"]:
            await query.edit_message_text(MESSAGES["error"])
            return
        
        # Build response
        detected_lang = result.get("detected_lang", "en")
        lang_info = LANGUAGES.get(detected_lang, LANGUAGES["en"])
        
        header = f"✅ **{mode_names.get(mode)}**\n"
        header += f"🔍 زبان تشخیص داده شده: {lang_info.flag} {lang_info.name_native}\n"
        
        if target_lang:
            target = LANGUAGES.get(target_lang)
            header += f"🎯 ترجمه به: {target.flag} {target.name_native}\n"
        
        header += "\n"
        
        # Footer
        footer = f"\n\n---\n🤖 مدل: `{result['model']}`"
        
        full_text = header + result["text"] + footer
        
        # Send main response
        if len(full_text) > 4000:
            # First chunk
            await query.edit_message_text(full_text[:4000], parse_mode="Markdown")
            
            # Remaining chunks
            remaining = full_text[4000:]
            while remaining:
                chunk = remaining[:4000]
                remaining = remaining[4000:]
                await asyncio.sleep(0.3)
                await context.bot.send_message(
                    chat_id=query.message.chat_id,
                    text=chunk,
                    parse_mode="Markdown"
                )
            
            # Send back button separately
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text=MESSAGES["operation_complete"].format(mode=mode_names.get(mode)),
                reply_markup=get_back_to_menu_keyboard(),
                parse_mode="Markdown"
            )
        else:
            await query.edit_message_text(full_text, parse_mode="Markdown")
            
            # Send back button
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text=MESSAGES["operation_complete"].format(mode=mode_names.get(mode)),
                reply_markup=get_back_to_menu_keyboard(),
                parse_mode="Markdown"
            )
    
    except Exception as e:
        logger.error(f"Process error: {e}")
        logger.error(traceback.format_exc())
        await query.edit_message_text(f"❌ خطا: {str(e)[:100]}")
    
    finally:
        # Clear state but KEEP audio cache!
        user_state.pop(user_id, None)


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(f"Error: {context.error}")
    logger.error(traceback.format_exc())


# ============== MAIN ==============
def main() -> None:
    print("\n" + "=" * 70)
    print("  🎧 OMNI-HEAR AI v7.0 - AssemblyAI + Groq Edition")
    print("  🎤 AssemblyAI STT | ⚡ Llama 8B | 🧠 Llama 70B")
    print("=" * 70)
    
    if not TELEGRAM_BOT_TOKEN:
        print("❌ TELEGRAM_BOT_TOKEN not set!")
        sys.exit(1)
    
    if not ASSEMBLYAI_API_KEY:
        print("❌ ASSEMBLYAI_API_KEY not set!")
        sys.exit(1)
    
    if not GROQ_API_KEY:
        print("❌ GROQ_API_KEY not set!")
        sys.exit(1)
    
    print(f"✅ Telegram: Ready")
    print(f"✅ AssemblyAI: Ready")
    print(f"✅ Groq: Ready")
    print(f"\n🤖 Models:")
    print(f"   • Fast: {GROQ_MODEL_FAST}")
    print(f"   • Complex: {GROQ_MODEL_COMPLEX}")
    print(f"\n🌍 Languages: {', '.join([l.flag for l in LANGUAGES.values()])}")
    print("=" * 70 + "\n")
    
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(MessageHandler(
        filters.VOICE | filters.AUDIO | filters.Document.AUDIO,
        handle_audio
    ))
    app.add_handler(CallbackQueryHandler(button_callback))
    app.add_error_handler(error_handler)
    
    logger.info("🚀 Starting bot...")
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
