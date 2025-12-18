#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      OMNI-HEAR AI v4.0 (Multilingual)                        ║
║              Advanced Audio Processing + Translation + Lyrics                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  🌍 Languages: Persian, English, French, Spanish, Russian, German, Arabic    ║
║  🎵 Smart Lyrics: Genius API + AI Fallback                                   ║
║  🔄 Translation: Any language to any language                                 ║
║  ⚡ Powered by Groq (Whisper + Llama 3.3)                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import logging
import asyncio
import tempfile
import traceback
import re
import urllib.parse
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

import httpx
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
GENIUS_API_KEY = os.getenv("GENIUS_API_KEY", "")  # Optional for lyrics

# ============== GROQ CLIENT ==============
groq_client: Optional[Groq] = None

if not GROQ_API_KEY:
    logger.error("❌ GROQ_API_KEY is not set!")
else:
    logger.info(f"✅ GROQ_API_KEY configured")
    groq_client = Groq(api_key=GROQ_API_KEY)

# ============== MODEL CONFIGURATION ==============
WHISPER_MODEL = "whisper-large-v3"
LLM_MODEL_PRIMARY = "llama-3.3-70b-versatile"
LLM_MODEL_FALLBACK = "llama-3.1-70b-versatile"
LLM_MODEL_FAST = "llama-3.1-8b-instant"

MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB

# ============== SUPPORTED LANGUAGES ==============
@dataclass
class Language:
    code: str
    name_en: str
    name_native: str
    flag: str

LANGUAGES: Dict[str, Language] = {
    "fa": Language("fa", "Persian", "فارسی", "🇮🇷"),
    "en": Language("en", "English", "English", "🇬🇧"),
    "fr": Language("fr", "French", "Français", "🇫🇷"),
    "es": Language("es", "Spanish", "Español", "🇪🇸"),
    "ru": Language("ru", "Russian", "Русский", "🇷🇺"),
    "de": Language("de", "German", "Deutsch", "🇩🇪"),
    "ar": Language("ar", "Arabic", "العربية", "🇸🇦"),
}

# ============== SYSTEM PROMPTS ==============
def get_transcript_prompt(target_lang: Optional[str] = None) -> str:
    """Get transcript prompt, optionally for a specific language."""
    base = """You are an expert transcription formatter.

**INPUT:** Raw transcription text from audio.

**YOUR TASK:** Clean and format this transcription perfectly.

**RULES:**
1. Fix any obvious transcription errors
2. Add proper punctuation
3. Break into logical paragraphs
4. If multiple speakers, mark as [Speaker 1], [Speaker 2]
5. Keep mixed language words in their original script
6. Mark unclear parts appropriately

**OUTPUT:** Clean, formatted transcription."""
    
    if target_lang and target_lang in LANGUAGES:
        lang = LANGUAGES[target_lang]
        base += f"\n\n**OUTPUT LANGUAGE:** {lang.name_en} ({lang.name_native})"
    
    return base


def get_translation_prompt(source_lang: str, target_lang: str) -> str:
    """Generate translation prompt for any language pair."""
    source = LANGUAGES.get(source_lang, LANGUAGES["en"])
    target = LANGUAGES.get(target_lang, LANGUAGES["en"])
    
    return f"""You are an expert translator specializing in {source.name_en} to {target.name_en} translation.

**INPUT:** Text transcribed from audio in {source.name_en}.

**YOUR TASK:** Translate the text to {target.name_en} ({target.name_native}).

**TRANSLATION RULES:**
1. Maintain the original meaning and tone
2. Use natural, fluent {target.name_en}
3. Preserve proper nouns and names
4. Keep technical terms accurate
5. Maintain paragraph structure
6. For idioms, use equivalent expressions in {target.name_en}

**OUTPUT FORMAT:**
First, provide a brief summary (1-2 sentences) of what the audio is about.
Then provide the full translation.

**OUTPUT LANGUAGE:** {target.name_en} ({target.name_native}) ONLY"""


def get_lecture_prompt(lang: str = "fa") -> str:
    """Get lecture prompt for specific language."""
    target = LANGUAGES.get(lang, LANGUAGES["fa"])
    
    prompts = {
        "fa": """You are a distinguished University Professor.

**INPUT:** Transcription of an educational audio/lecture.

**YOUR TASK:** Transform this into a comprehensive **Textbook Chapter in Persian (Farsi)**.

**REQUIREMENTS:**
1. زبان خروجی: فارسی روان و آکادمیک
2. Use **bold headers** for sections
3. Cover EVERY detail from the audio
4. Add helpful explanations
5. Structure logically
6. Keep technical terms in original form

**OUTPUT LANGUAGE: PERSIAN (فارسی) ONLY**""",
        
        "en": """You are a distinguished University Professor.

**INPUT:** Transcription of an educational audio/lecture.

**YOUR TASK:** Transform this into a comprehensive **Textbook Chapter in English**.

**REQUIREMENTS:**
1. Academic, clear English
2. Use **bold headers** for sections
3. Cover EVERY detail from the audio
4. Add helpful explanations
5. Structure logically

**OUTPUT LANGUAGE: ENGLISH ONLY**""",

        "fr": """Vous êtes un professeur d'université distingué.

**INPUT:** Transcription d'un audio/cours éducatif.

**VOTRE TÂCHE:** Transformez ceci en un **Chapitre de Manuel Complet en Français**.

**EXIGENCES:**
1. Français académique et fluide
2. Utilisez des **en-têtes en gras** pour les sections
3. Couvrez TOUS les détails
4. Structure logique

**LANGUE DE SORTIE: FRANÇAIS UNIQUEMENT**""",

        "es": """Usted es un distinguido Profesor Universitario.

**INPUT:** Transcripción de un audio/clase educativa.

**SU TAREA:** Transforme esto en un **Capítulo de Libro de Texto Completo en Español**.

**REQUISITOS:**
1. Español académico y fluido
2. Use **encabezados en negrita** para las secciones
3. Cubra TODOS los detalles
4. Estructura lógica

**IDIOMA DE SALIDA: ESPAÑOL ÚNICAMENTE**""",

        "de": """Sie sind ein angesehener Universitätsprofessor.

**INPUT:** Transkription eines Bildungsaudios/Vorlesung.

**IHRE AUFGABE:** Verwandeln Sie dies in ein umfassendes **Lehrbuchkapitel auf Deutsch**.

**ANFORDERUNGEN:**
1. Akademisches, flüssiges Deutsch
2. Verwenden Sie **fette Überschriften** für Abschnitte
3. Decken Sie ALLE Details ab
4. Logische Struktur

**AUSGABESPRACHE: NUR DEUTSCH**""",

        "ru": """Вы — выдающийся университетский профессор.

**INPUT:** Транскрипция образовательного аудио/лекции.

**ВАША ЗАДАЧА:** Преобразуйте это в полноценную **Главу Учебника на Русском языке**.

**ТРЕБОВАНИЯ:**
1. Академический, грамотный русский
2. Используйте **жирные заголовки** для разделов
3. Охватите ВСЕ детали
4. Логичная структура

**ЯЗЫК ВЫВОДА: ТОЛЬКО РУССКИЙ**""",

        "ar": """أنت أستاذ جامعي متميز.

**INPUT:** نص مكتوب من صوت/محاضرة تعليمية.

**مهمتك:** حوّل هذا إلى **فصل كتاب دراسي شامل باللغة العربية**.

**المتطلبات:**
1. لغة عربية أكاديمية وسلسة
2. استخدم **عناوين غامقة** للأقسام
3. غطِّ جميع التفاصيل
4. هيكل منطقي

**لغة الإخراج: العربية فقط**"""
    }
    
    return prompts.get(lang, prompts["en"])


def get_soap_prompt() -> str:
    """SOAP note prompt - always English."""
    return """You are a Chief Resident physician at a major teaching hospital.

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

**OBJECTIVE:**
- Vital Signs:
- Physical Examination:
- Laboratory Results:
- Imaging:

**ASSESSMENT:**
- Primary Diagnosis:
- Differential Diagnoses:
- ICD-10 Codes:

**PLAN:**
- Treatment:
- Medications:
- Follow-up:
- Referrals:

**OUTPUT LANGUAGE: ENGLISH ONLY**"""


def get_summary_prompt(lang: str = "fa") -> str:
    """Get summary prompt for specific language."""
    target = LANGUAGES.get(lang, LANGUAGES["fa"])
    
    return f"""You are an expert summarizer.

**INPUT:** Transcription of audio content.

**YOUR TASK:** Create a clear, concise summary in **{target.name_en} ({target.name_native})**.

**FORMAT:**
• Use bullet points
• Focus on key information
• Remove unnecessary details
• Clear and fluent language

**STRUCTURE:**
📌 **Overview:** Brief summary (1-2 sentences)

📋 **Key Points:**
• Point 1
• Point 2
• Point 3

🎯 **Conclusion:** Final takeaway

**OUTPUT LANGUAGE: {target.name_en.upper()} ({target.name_native}) ONLY**"""


def get_lyrics_prompt() -> str:
    """Lyrics extraction prompt."""
    return """You are a music transcription specialist with expertise in lyrics.

**INPUT:** Transcription of audio (likely music).

**YOUR TASK:** Format this as proper song lyrics.

**FORMAT:**
[Verse 1]
Line 1
Line 2

[Chorus]
Line 1
Line 2

[Verse 2]
...

[Bridge] (if applicable)
...

[Outro] (if applicable)
...

**RULES:**
1. Keep the original language
2. Proper line breaks
3. Identify song structure (verse, chorus, bridge, etc.)
4. Mark instrumental sections as [Instrumental]
5. If you can identify the song/artist, mention it at the top

**OUTPUT:** Formatted lyrics in original language"""


# ============== PERSIAN MESSAGES ==============
MESSAGES = {
    "welcome": """🎧 **به Omni-Hear AI خوش آمدید!**
    
🚀 **نسخه 4.0 - چندزبانه**

🎤 یک فایل صوتی یا ویس ارسال کنید.

⚡ **قابلیت‌ها:**
• 📜 رونویسی کامل
• 📚 درسنامه (۷ زبان)
• 🩺 شرح‌حال پزشکی SOAP
• 📝 خلاصه متن
• 🎵 متن آهنگ (با جستجوی هوشمند)
• 🌍 ترجمه به ۷ زبان

🌐 **زبان‌ها:**
🇮🇷 فارسی | 🇬🇧 English | 🇫🇷 Français
🇪🇸 Español | 🇩🇪 Deutsch | 🇷🇺 Русский | 🇸🇦 العربية""",

    "audio_received": "🎵 **فایل دریافت شد!**\n\n📋 نوع پردازش را انتخاب کنید:",
    "select_language": "🌍 **زبان خروجی را انتخاب کنید:**",
    "select_source_lang": "🗣 **زبان صوت (مبدا) را انتخاب کنید:**",
    "select_target_lang": "🎯 **زبان مقصد ترجمه را انتخاب کنید:**",
    "processing_stt": "🎤 **مرحله ۱/۲:** تبدیل صدا به متن...",
    "processing_llm": "🧠 **مرحله ۲/۲:** پردازش هوشمند...",
    "processing_lyrics": "🎵 **جستجوی متن آهنگ...**",
    "lyrics_found": "✅ **متن آهنگ از دیتابیس پیدا شد!**",
    "lyrics_ai": "🤖 **متن با هوش مصنوعی استخراج شد**",
    "error": "❌ خطا در پردازش. لطفاً دوباره تلاش کنید.",
    "no_audio": "⚠️ لطفاً ابتدا یک فایل صوتی ارسال کنید.",
    "file_too_large": "⚠️ حجم فایل بیشتر از ۲۵ مگابایت است.",
    "not_audio": "⚠️ لطفاً یک فایل صوتی ارسال کنید.",
    "api_key_missing": "⚠️ GROQ_API_KEY تنظیم نشده!",
}

# ============== USER CACHE ==============
user_audio_cache: dict = {}
user_state: dict = {}  # Track user's current operation state


# ============== KEYBOARDS ==============
def get_main_menu_keyboard() -> InlineKeyboardMarkup:
    """Main processing menu."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📜 رونویسی کامل", callback_data="mode:transcript"),
        ],
        [
            InlineKeyboardButton("📚 درسنامه", callback_data="mode:lecture"),
            InlineKeyboardButton("🩺 SOAP پزشکی", callback_data="mode:soap"),
        ],
        [
            InlineKeyboardButton("📝 خلاصه", callback_data="mode:summary"),
            InlineKeyboardButton("🎵 متن آهنگ", callback_data="mode:lyrics"),
        ],
        [
            InlineKeyboardButton("🌍 ترجمه صوت", callback_data="mode:translate"),
        ],
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
    
    # Add back button
    buttons.append([InlineKeyboardButton("🔙 بازگشت", callback_data="back:main")])
    
    return InlineKeyboardMarkup(buttons)


def get_target_language_keyboard(source_lang: str) -> InlineKeyboardMarkup:
    """Target language keyboard (excludes source language)."""
    buttons = []
    row = []
    
    for code, lang in LANGUAGES.items():
        if code == source_lang:
            continue
        btn = InlineKeyboardButton(
            f"{lang.flag} {lang.name_native}",
            callback_data=f"target_lang:{code}"
        )
        row.append(btn)
        if len(row) == 3:
            buttons.append(row)
            row = []
    
    if row:
        buttons.append(row)
    
    buttons.append([InlineKeyboardButton("🔙 بازگشت", callback_data="back:main")])
    
    return InlineKeyboardMarkup(buttons)


# ============== GENIUS LYRICS API ==============
async def search_genius_lyrics(query: str) -> Optional[Dict]:
    """Search for lyrics on Genius."""
    if not GENIUS_API_KEY:
        logger.info("Genius API key not configured, skipping lyrics search")
        return None
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Search for song
            search_url = "https://api.genius.com/search"
            headers = {"Authorization": f"Bearer {GENIUS_API_KEY}"}
            params = {"q": query}
            
            response = await client.get(search_url, headers=headers, params=params)
            
            if response.status_code != 200:
                logger.warning(f"Genius search failed: {response.status_code}")
                return None
            
            data = response.json()
            hits = data.get("response", {}).get("hits", [])
            
            if not hits:
                return None
            
            # Get first result
            song = hits[0]["result"]
            
            return {
                "title": song.get("title", ""),
                "artist": song.get("primary_artist", {}).get("name", ""),
                "url": song.get("url", ""),
                "thumbnail": song.get("song_art_image_thumbnail_url", ""),
            }
            
    except Exception as e:
        logger.error(f"Genius search error: {e}")
        return None


async def get_lyrics_from_url(url: str) -> Optional[str]:
    """Scrape lyrics from Genius URL."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(url)
            
            if response.status_code != 200:
                return None
            
            html = response.text
            
            # Extract lyrics using regex (Genius stores lyrics in specific divs)
            # This is a simplified extraction
            lyrics_pattern = r'<div[^>]*class="[^"]*Lyrics__Container[^"]*"[^>]*>(.*?)</div>'
            matches = re.findall(lyrics_pattern, html, re.DOTALL)
            
            if matches:
                lyrics = ""
                for match in matches:
                    # Clean HTML tags
                    clean = re.sub(r'<br\s*/?>', '\n', match)
                    clean = re.sub(r'<[^>]+>', '', clean)
                    clean = clean.strip()
                    lyrics += clean + "\n\n"
                
                return lyrics.strip() if lyrics.strip() else None
            
            return None
            
    except Exception as e:
        logger.error(f"Lyrics scraping error: {e}")
        return None


async def identify_song_from_transcription(transcription: str) -> Optional[Dict]:
    """Use LLM to identify song from transcription."""
    if not groq_client:
        return None
    
    try:
        prompt = """Analyze this transcription which may be from a song.

**YOUR TASK:**
1. Determine if this is likely song lyrics or spoken content
2. If it's a song, try to identify:
   - Song title
   - Artist name
   - Any recognizable lyrics phrases

**OUTPUT FORMAT (JSON):**
{
    "is_song": true/false,
    "confidence": "high/medium/low",
    "title": "song title or null",
    "artist": "artist name or null", 
    "search_query": "best search query for this song"
}

**TRANSCRIPTION:**
""" + transcription[:1500]

        response = await asyncio.to_thread(
            lambda: groq_client.chat.completions.create(
                model=LLM_MODEL_FAST,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3,
            )
        )
        
        if response.choices:
            result_text = response.choices[0].message.content
            
            # Try to parse JSON from response
            json_match = re.search(r'\{[^}]+\}', result_text, re.DOTALL)
            if json_match:
                import json
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
        
        return None
        
    except Exception as e:
        logger.error(f"Song identification error: {e}")
        return None


# ============== AUDIO CONVERSION ==============
async def convert_audio_to_mp3(audio_data: bytes, original_format: str = "ogg") -> Tuple[Optional[bytes], Optional[str]]:
    """Convert audio to MP3 format."""
    try:
        def _convert():
            with tempfile.NamedTemporaryFile(suffix=f".{original_format}", delete=False) as input_file:
                input_file.write(audio_data)
                input_path = input_file.name
            
            try:
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
                
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as output_file:
                    output_path = output_file.name
                
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


# ============== GROQ STT ==============
async def transcribe_audio(audio_data: bytes) -> Tuple[Optional[str], Optional[str]]:
    """Transcribe audio using Groq Whisper."""
    if not groq_client:
        return None, "Groq client not initialized"
    
    try:
        def _transcribe():
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name
            
            try:
                with open(temp_path, "rb") as audio_file:
                    transcription = groq_client.audio.transcriptions.create(
                        model=WHISPER_MODEL,
                        file=audio_file,
                        response_format="text",
                        language=None,  # Auto-detect
                        temperature=0.0,
                    )
                return transcription, None
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        
        result, error = await asyncio.to_thread(_transcribe)
        
        if error:
            return None, error
        
        if result and len(result.strip()) > 0:
            logger.info(f"✅ Transcription: {len(result)} chars")
            return result.strip(), None
        else:
            return None, "Empty transcription"
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Transcription error: {error_msg}")
        return None, error_msg[:100]


# ============== GROQ LLM ==============
async def process_with_llm(
    text: str,
    system_prompt: str
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Process text with Groq LLM."""
    if not groq_client:
        return None, None, "Groq client not initialized"
    
    models = [LLM_MODEL_PRIMARY, LLM_MODEL_FALLBACK, LLM_MODEL_FAST]
    
    for model_name in models:
        try:
            logger.info(f"🔄 Trying LLM: {model_name}")
            
            response = await asyncio.to_thread(
                lambda m=model_name: groq_client.chat.completions.create(
                    model=m,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text}
                    ],
                    temperature=0.7,
                    max_tokens=8000,
                )
            )
            
            if response.choices and response.choices[0].message.content:
                result = response.choices[0].message.content.strip()
                logger.info(f"✅ LLM success: {model_name}")
                return result, model_name, None
                
        except Exception as e:
            logger.warning(f"❌ {model_name}: {str(e)[:50]}")
            continue
    
    return None, None, "All models failed"


# ============== FULL PROCESSING ==============
async def process_audio_full(
    audio_data: bytes,
    mime_type: str,
    mode: str,
    lang: str = "fa",
    source_lang: Optional[str] = None,
    target_lang: Optional[str] = None,
) -> Dict:
    """
    Full audio processing pipeline.
    Returns dict with: result, transcription, model, lyrics_source, error
    """
    result = {
        "text": None,
        "transcription": None,
        "model": None,
        "lyrics_source": None,
        "song_info": None,
        "error": None,
    }
    
    # Determine format and convert
    format_map = {
        "audio/ogg": "ogg", "audio/oga": "ogg", "audio/opus": "ogg",
        "audio/mp3": "mp3", "audio/mpeg": "mp3",
        "audio/wav": "wav", "audio/x-wav": "wav",
        "audio/m4a": "m4a", "audio/mp4": "m4a",
    }
    
    original_format = format_map.get(mime_type, "ogg")
    
    if original_format != "mp3":
        mp3_data, _ = await convert_audio_to_mp3(audio_data, original_format)
        if not mp3_data:
            mp3_data = audio_data
    else:
        mp3_data = audio_data
    
    # Step 1: Transcribe
    transcription, stt_error = await transcribe_audio(mp3_data)
    
    if stt_error:
        result["error"] = f"❌ خطا در تبدیل صدا: {stt_error}"
        return result
    
    if not transcription:
        result["error"] = "❌ متنی استخراج نشد"
        return result
    
    result["transcription"] = transcription
    
    # Step 2: Process based on mode
    if mode == "transcript":
        prompt = get_transcript_prompt(lang)
        text, model, err = await process_with_llm(transcription, prompt)
        result["text"] = text or transcription
        result["model"] = model or WHISPER_MODEL
        
    elif mode == "lecture":
        prompt = get_lecture_prompt(lang)
        text, model, err = await process_with_llm(transcription, prompt)
        result["text"] = text
        result["model"] = model
        if err:
            result["error"] = err
            
    elif mode == "soap":
        prompt = get_soap_prompt()
        text, model, err = await process_with_llm(transcription, prompt)
        result["text"] = text
        result["model"] = model
        if err:
            result["error"] = err
            
    elif mode == "summary":
        prompt = get_summary_prompt(lang)
        text, model, err = await process_with_llm(transcription, prompt)
        result["text"] = text
        result["model"] = model
        if err:
            result["error"] = err
            
    elif mode == "lyrics":
        # Smart lyrics: Try to identify song and search Genius first
        song_info = await identify_song_from_transcription(transcription)
        
        genius_lyrics = None
        if song_info and song_info.get("is_song"):
            result["song_info"] = song_info
            
            # Search Genius
            search_query = song_info.get("search_query") or f"{song_info.get('title', '')} {song_info.get('artist', '')}"
            genius_result = await search_genius_lyrics(search_query)
            
            if genius_result:
                result["song_info"]["genius"] = genius_result
                genius_lyrics = await get_lyrics_from_url(genius_result["url"])
                
                if genius_lyrics:
                    result["text"] = f"""🎵 **{genius_result['title']}**
🎤 **{genius_result['artist']}**
🔗 [Genius]({genius_result['url']})

---

{genius_lyrics}"""
                    result["lyrics_source"] = "genius"
                    result["model"] = "Genius API"
                    return result
        
        # Fallback to AI lyrics extraction
        prompt = get_lyrics_prompt()
        text, model, err = await process_with_llm(transcription, prompt)
        result["text"] = text
        result["model"] = model
        result["lyrics_source"] = "ai"
        if err:
            result["error"] = err
            
    elif mode == "translate":
        if not source_lang or not target_lang:
            result["error"] = "زبان مبدا و مقصد مشخص نشده"
            return result
        
        prompt = get_translation_prompt(source_lang, target_lang)
        text, model, err = await process_with_llm(transcription, prompt)
        result["text"] = text
        result["model"] = model
        if err:
            result["error"] = err
    
    return result


# ============== TELEGRAM HANDLERS ==============
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(MESSAGES["welcome"], parse_mode="Markdown")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = """📖 **راهنمای Omni-Hear AI v4.0**

**🔹 استفاده:**
1️⃣ فایل صوتی ارسال کنید
2️⃣ نوع پردازش را انتخاب کنید
3️⃣ زبان خروجی را انتخاب کنید (در صورت نیاز)

**🔹 حالت‌ها:**
• 📜 رونویسی - متن کامل صوت
• 📚 درسنامه - تبدیل به متن آموزشی
• 🩺 SOAP - شرح‌حال پزشکی
• 📝 خلاصه - خلاصه نکات مهم
• 🎵 متن آهنگ - استخراج لیریک با جستجو در Genius
• 🌍 ترجمه - ترجمه به ۷ زبان

**🔹 زبان‌ها:**
🇮🇷 فارسی | 🇬🇧 English | 🇫🇷 Français
🇪🇸 Español | 🇩🇪 Deutsch | 🇷🇺 Русский | 🇸🇦 العربية

**🔹 دستورات:**
/start - شروع
/help - راهنما
/status - وضعیت سرویس"""
    
    await update.message.reply_text(help_text, parse_mode="Markdown")


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    status = ["🔍 **وضعیت سیستم**\n"]
    status.append("✅ **Telegram:** متصل")
    
    if GROQ_API_KEY and groq_client:
        status.append("✅ **Groq API:** فعال")
    else:
        status.append("❌ **Groq API:** غیرفعال")
    
    if GENIUS_API_KEY:
        status.append("✅ **Genius API:** فعال (جستجوی لیریک)")
    else:
        status.append("⚠️ **Genius API:** غیرفعال (اختیاری)")
    
    status.append(f"\n**🤖 مدل‌ها:**")
    status.append(f"• STT: `{WHISPER_MODEL}`")
    status.append(f"• LLM: `{LLM_MODEL_PRIMARY}`")
    
    status.append(f"\n**🌍 زبان‌ها:** {len(LANGUAGES)} زبان پشتیبانی")
    
    await update.message.reply_text("\n".join(status), parse_mode="Markdown")


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    msg = update.message
    
    if not GROQ_API_KEY or not groq_client:
        await msg.reply_text(MESSAGES["api_key_missing"])
        return
    
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
    
    file_size = getattr(audio_file, 'file_size', 0)
    if file_size and file_size > MAX_FILE_SIZE:
        await msg.reply_text(MESSAGES["file_too_large"])
        return
    
    try:
        file = await context.bot.get_file(audio_file.file_id)
        audio_bytes = await file.download_as_bytearray()
        
        if file_type == "voice":
            mime_type = "audio/ogg"
        elif hasattr(audio_file, 'mime_type') and audio_file.mime_type:
            mime_type = audio_file.mime_type
        else:
            mime_type = "audio/mpeg"
        
        user_audio_cache[user_id] = {
            "data": bytes(audio_bytes),
            "mime_type": mime_type,
        }
        
        # Clear any previous state
        user_state.pop(user_id, None)
        
        logger.info(f"✅ Audio cached: user={user_id}, size={len(audio_bytes)}")
        
        await msg.reply_text(
            MESSAGES["audio_received"],
            reply_markup=get_main_menu_keyboard(),
            parse_mode="Markdown"
        )
        
    except Exception as e:
        logger.error(f"Audio download error: {e}")
        await msg.reply_text(MESSAGES["error"])


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    data = query.data
    
    # Parse callback data
    parts = data.split(":")
    action = parts[0]
    value = parts[1] if len(parts) > 1 else None
    
    # Handle back button
    if action == "back":
        if user_id in user_audio_cache:
            await query.edit_message_text(
                MESSAGES["audio_received"],
                reply_markup=get_main_menu_keyboard(),
                parse_mode="Markdown"
            )
        else:
            await query.edit_message_text(MESSAGES["no_audio"])
        user_state.pop(user_id, None)
        return
    
    # Handle mode selection
    if action == "mode":
        mode = value
        
        if user_id not in user_audio_cache:
            await query.edit_message_text(MESSAGES["no_audio"])
            return
        
        # Modes that need language selection
        if mode in ["transcript", "lecture", "summary"]:
            user_state[user_id] = {"mode": mode, "step": "select_lang"}
            await query.edit_message_text(
                MESSAGES["select_language"],
                reply_markup=get_language_keyboard(f"lang_{mode}"),
                parse_mode="Markdown"
            )
            return
        
        # Translation needs source and target language
        if mode == "translate":
            user_state[user_id] = {"mode": mode, "step": "select_source"}
            await query.edit_message_text(
                MESSAGES["select_source_lang"],
                reply_markup=get_language_keyboard("source_lang"),
                parse_mode="Markdown"
            )
            return
        
        # SOAP and Lyrics - process directly
        await process_and_respond(query, context, user_id, mode)
        return
    
    # Handle language selection for transcript/lecture/summary
    if action.startswith("lang_"):
        mode = action.replace("lang_", "")
        lang = value
        await process_and_respond(query, context, user_id, mode, lang=lang)
        return
    
    # Handle source language for translation
    if action == "source_lang":
        source = value
        user_state[user_id] = {
            "mode": "translate",
            "step": "select_target",
            "source_lang": source
        }
        await query.edit_message_text(
            MESSAGES["select_target_lang"],
            reply_markup=get_target_language_keyboard(source),
            parse_mode="Markdown"
        )
        return
    
    # Handle target language for translation
    if action == "target_lang":
        target = value
        state = user_state.get(user_id, {})
        source = state.get("source_lang", "en")
        await process_and_respond(
            query, context, user_id, "translate",
            source_lang=source, target_lang=target
        )
        return


async def process_and_respond(
    query,
    context,
    user_id: int,
    mode: str,
    lang: str = "fa",
    source_lang: Optional[str] = None,
    target_lang: Optional[str] = None,
) -> None:
    """Process audio and send response."""
    
    if user_id not in user_audio_cache:
        await query.edit_message_text(MESSAGES["no_audio"])
        return
    
    audio_info = user_audio_cache[user_id]
    
    # Mode display names
    mode_names = {
        "transcript": "📜 رونویسی",
        "lecture": "📚 درسنامه",
        "soap": "🩺 SOAP",
        "summary": "📝 خلاصه",
        "lyrics": "🎵 متن آهنگ",
        "translate": "🌍 ترجمه",
    }
    
    try:
        # Show processing message
        processing_msg = f"🎯 **{mode_names.get(mode, mode)}**\n\n{MESSAGES['processing_stt']}"
        if mode == "translate" and source_lang and target_lang:
            src = LANGUAGES.get(source_lang, LANGUAGES["en"])
            tgt = LANGUAGES.get(target_lang, LANGUAGES["fa"])
            processing_msg += f"\n\n{src.flag} {src.name_native} → {tgt.flag} {tgt.name_native}"
        
        await query.edit_message_text(processing_msg, parse_mode="Markdown")
        
        # Process audio
        result = await process_audio_full(
            audio_info["data"],
            audio_info["mime_type"],
            mode,
            lang=lang,
            source_lang=source_lang,
            target_lang=target_lang,
        )
        
        if result["error"]:
            await query.edit_message_text(result["error"])
            return
        
        if not result["text"]:
            await query.edit_message_text(MESSAGES["error"])
            return
        
        # Build response
        header = f"✅ **{mode_names.get(mode, mode)}**\n\n"
        
        # Add lyrics source info
        if mode == "lyrics":
            if result["lyrics_source"] == "genius":
                header = f"✅ **{MESSAGES['lyrics_found']}**\n\n"
            else:
                header = f"✅ **{MESSAGES['lyrics_ai']}**\n\n"
        
        # Add translation info
        if mode == "translate" and source_lang and target_lang:
            src = LANGUAGES.get(source_lang)
            tgt = LANGUAGES.get(target_lang)
            header += f"{src.flag} → {tgt.flag}\n\n"
        
        footer = f"\n\n---\n🤖 `{result['model']}`"
        
        full_text = header + result["text"] + footer
        
        # Send response (handle long messages)
        if len(full_text) > 4000:
            await query.edit_message_text(full_text[:4000], parse_mode="Markdown")
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
        else:
            try:
                await query.edit_message_text(full_text, parse_mode="Markdown")
            except Exception:
                await query.edit_message_text(full_text)
    
    except Exception as e:
        logger.error(f"Process error: {e}")
        logger.error(traceback.format_exc())
        await query.edit_message_text(f"❌ خطا: {str(e)[:100]}")
    
    finally:
        # Cleanup
        user_audio_cache.pop(user_id, None)
        user_state.pop(user_id, None)


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(f"Error: {context.error}")


# ============== MAIN ==============
def main() -> None:
    print("\n" + "="*65)
    print("  🎧 OMNI-HEAR AI v4.0 - MULTILINGUAL EDITION")
    print("  🌍 7 Languages | 🎵 Smart Lyrics | 🔄 Translation")
    print("="*65)
    
    if not TELEGRAM_BOT_TOKEN:
        print("❌ TELEGRAM_BOT_TOKEN not set!")
        sys.exit(1)
    
    if not GROQ_API_KEY:
        print("❌ GROQ_API_KEY not set!")
        sys.exit(1)
    
    print(f"✅ Telegram: Ready")
    print(f"✅ Groq: Configured")
    print(f"{'✅' if GENIUS_API_KEY else '⚠️'} Genius: {'Configured' if GENIUS_API_KEY else 'Not configured (optional)'}")
    print(f"🌍 Languages: {', '.join([l.flag for l in LANGUAGES.values()])}")
    print("="*65 + "\n")
    
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
    
    logger.info("🚀 Bot starting...")
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
