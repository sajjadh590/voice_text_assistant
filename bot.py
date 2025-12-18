#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   OMNI-HEAR AI v6.0 (Dual-Engine Edition)                    ║
║            ⚡ Fast Mode (Groq 70B) | 🚀 Pro Mode (SambaNova 405B)            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  🎤 STT: Groq Whisper Large V3 (with context prompting)                      ║
║  ⚡ Fast LLM: Groq Llama 3.3 70B (~3 seconds)                                ║
║  🚀 Pro LLM: SambaNova Llama 3.1 405B (Maximum Accuracy)                     ║
║  🌍 7 Languages | 🔄 Auto-Fallback | 📊 Dual-Button UI                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import logging
import asyncio
import tempfile
import traceback
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

from groq import Groq
from openai import OpenAI
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
SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY")

# ============== API CLIENTS ==============
groq_client: Optional[Groq] = None
sambanova_client: Optional[OpenAI] = None

if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
    logger.info("✅ Groq client initialized")
else:
    logger.error("❌ GROQ_API_KEY not set!")

if SAMBANOVA_API_KEY:
    sambanova_client = OpenAI(
        api_key=SAMBANOVA_API_KEY,
        base_url="https://api.sambanova.ai/v1"
    )
    logger.info("✅ SambaNova client initialized")
else:
    logger.warning("⚠️ SAMBANOVA_API_KEY not set - Pro mode unavailable")

# ============== MODEL CONFIGURATION ==============
WHISPER_MODEL = "whisper-large-v3"
GROQ_LLM_PRIMARY = "llama-3.3-70b-versatile"
GROQ_LLM_FALLBACK = "llama-3.1-8b-instant"
SAMBANOVA_MODEL_PRO = "Meta-Llama-3.1-405B-Instruct"
SAMBANOVA_MODEL_FALLBACK = "Meta-Llama-3.1-70B-Instruct"

MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB

# Whisper context prompt for better accuracy
WHISPER_CONTEXT_PROMPT = """Medical terminology: SOAP, diagnosis, patient, symptoms, treatment, prescription, 
blood pressure, cardiac, respiratory, neurological, assessment, differential diagnosis.
Academic terms: professor, lecture, university, chapter, introduction, conclusion, methodology.
Persian academic: درس، استاد، دانشگاه، فصل، مقدمه، نتیجه‌گیری، تشخیص، بیمار، درمان."""


# ============== ENGINE TYPES ==============
class Engine(Enum):
    FAST = "fast"   # Groq 70B
    PRO = "pro"     # SambaNova 405B


# ============== LANGUAGES ==============
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

# ============== USER STATE ==============
user_audio_cache: Dict[int, dict] = {}
user_state: Dict[int, dict] = {}


# ============== ADVANCED SYSTEM PROMPTS ==============

def get_soap_prompt_pro() -> str:
    """Advanced Medical SOAP prompt for 405B Pro Engine."""
    return """Role: Senior Board-Certified Attending Physician with 20+ years of clinical experience at a major academic medical center.

Task: Transform the provided medical dictation into a comprehensive, US Medical Standard clinical SOAP Note that meets Joint Commission (JCAHO) documentation requirements.

DOCUMENTATION STANDARDS:
- Follow CMS Documentation Guidelines
- Include all medically necessary information
- Use standard medical abbreviations appropriately
- Maintain HIPAA-compliant language

FORMAT:

═══════════════════════════════════════════════════════════════
                         SOAP NOTE
═══════════════════════════════════════════════════════════════

📋 SUBJECTIVE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Chief Complaint (CC):**
[Primary reason for visit - patient's own words in quotes]

**History of Present Illness (HPI):**
Capture with chronological precision using OLDCARTS:
- Onset: [When did it start?]
- Location: [Where is the problem?]
- Duration: [How long does it last?]
- Character: [What does it feel like?]
- Aggravating Factors: [What makes it worse?]
- Relieving Factors: [What makes it better?]
- Timing: [When does it occur?]
- Severity: [Rate 1-10]
- Associated Symptoms: [Related symptoms]

**Review of Systems (ROS):**
□ Constitutional: [Fever, weight changes, fatigue]
□ HEENT: [Head, eyes, ears, nose, throat]
□ Cardiovascular: [Chest pain, palpitations, edema]
□ Respiratory: [Dyspnea, cough, wheezing]
□ Gastrointestinal: [Nausea, vomiting, abdominal pain]
□ Genitourinary: [Dysuria, frequency, hematuria]
□ Musculoskeletal: [Joint pain, stiffness, swelling]
□ Neurological: [Headache, dizziness, weakness]
□ Psychiatric: [Mood, anxiety, sleep]
□ Integumentary: [Rash, lesions, changes]

**Past Medical History (PMH):**
**Past Surgical History (PSH):**
**Medications:** [Include dose, frequency, route]
**Allergies:** [Drug allergies with reaction type - NKDA if none]
**Family History (FHx):**
**Social History (SHx):**
- Tobacco: [Pack-years or never]
- Alcohol: [Drinks per week]
- Illicit drugs: [Yes/No, type if yes]
- Occupation:
- Living situation:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔬 OBJECTIVE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Vital Signs:**
| Parameter | Value | Reference |
|-----------|-------|-----------|
| BP | /mmHg | <120/80 |
| HR | bpm | 60-100 |
| RR | /min | 12-20 |
| Temp | °F (°C) | 97.8-99.1°F |
| SpO2 | % | >95% |
| Weight | kg/lbs | |
| Height | cm/in | |
| BMI | kg/m² | 18.5-24.9 |

**Physical Examination:**

*General:* [Appearance, distress level, cooperation]

*HEENT:*
- Head: [Normocephalic, atraumatic]
- Eyes: [PERRLA, EOM intact, conjunctivae]
- Ears: [TMs, canals]
- Nose: [Patency, mucosa]
- Throat: [Oropharynx, tonsils, uvula]

*Neck:* [Supple, lymphadenopathy, thyroid, JVD]

*Cardiovascular:* [Rate, rhythm, murmurs, S1/S2, peripheral pulses, edema]

*Pulmonary:* [Effort, breath sounds, wheezes, rhonchi, rales]

*Abdomen:* [Soft, tenderness, distension, bowel sounds, organomegaly]

*Extremities:* [Edema, cyanosis, clubbing, ROM]

*Neurological:* [Mental status, cranial nerves, motor, sensory, reflexes, gait]

*Skin:* [Color, turgor, lesions, rashes]

**Diagnostic Results:**

*Laboratory:*
| Test | Result | Reference Range | Flag |
|------|--------|-----------------|------|
| | | | |

*Imaging:*
[Modality, findings, impression]

*Other Studies:*
[EKG, PFTs, etc.]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 ASSESSMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Primary Diagnosis:**
[Diagnosis] — ICD-10: [Code]

**Differential Diagnoses (Prioritized):**
1. [Most likely] — ICD-10: [Code]
   - Supporting evidence:
   - Against:
2. [Second likely] — ICD-10: [Code]
3. [Third likely] — ICD-10: [Code]

**Clinical Reasoning:**
[Brief explanation of diagnostic thought process]

**Risk Stratification:**
[Low/Moderate/High risk with justification]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 PLAN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Diagnostic Plan:**
- [ ] [Tests to order with rationale]

**Therapeutic Plan:**
- [ ] [Medications: Drug, Dose, Route, Frequency, Duration]
- [ ] [Procedures]
- [ ] [Therapies]

**Patient Education:**
- [ ] [Key points discussed]
- [ ] [Warning signs to watch for]
- [ ] [Lifestyle modifications]

**Disposition:**
☐ Discharge home
☐ Admit to: [Unit]
☐ Transfer to: [Facility]
☐ Observation

**Follow-up:**
- [Timeframe]: [Provider/Specialty]
- Return precautions: [Specific symptoms]

**Referrals:**
- [ ] [Specialty]: [Reason]

═══════════════════════════════════════════════════════════════

CRITICAL INSTRUCTIONS:
1. OUTPUT MUST BE IN ENGLISH ONLY
2. Correct any medical mispronunciations from transcription
3. Use standard medical terminology and abbreviations
4. Include ICD-10 codes for all diagnoses
5. If information not provided, mark as "Not documented" or "Not assessed"
6. Flag any critical or concerning findings with ⚠️
7. Maintain formal, objective clinical tone throughout"""


def get_soap_prompt_fast() -> str:
    """Simplified SOAP prompt for Fast Engine."""
    return """You are an experienced physician. Create a SOAP Note from this medical dictation.

FORMAT:
## SUBJECTIVE
- CC, HPI, ROS, PMH, Medications, Allergies

## OBJECTIVE  
- Vitals, Physical Exam, Labs/Imaging

## ASSESSMENT
- Diagnosis with ICD-10 codes
- Differential diagnoses

## PLAN
- Treatment, medications, follow-up

OUTPUT: English only. Correct medical terminology errors."""


def get_lecture_prompt_pro(lang: str = "fa") -> str:
    """Advanced Academic Lecture prompt for 405B Pro Engine."""
    prompts = {
        "fa": """نقش: استاد برجسته دانشگاه با سابقه ۲۰ ساله تدریس و تألیف کتب مرجع دانشگاهی.

وظیفه: تبدیل رونویسی صوت به یک فصل جامع کتاب مرجع دانشگاهی (در سطح کتاب‌های مرجع مانند هاریسون، گایتون، یا رابینز).

═══════════════════════════════════════════════════════════════
                      فصل درسی آکادمیک
═══════════════════════════════════════════════════════════════

📚 ساختار الزامی:

**۱. مقدمه علمی (Introduction)**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- تعریف دقیق موضوع با ارجاع به مفاهیم پایه
- اهمیت بالینی/علمی موضوع
- اهداف یادگیری این فصل
- پیش‌نیازهای مطالعه

**۲. متن اصلی (Main Content)**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- تقسیم‌بندی منطقی با **عناوین درشت**
- توضیح گام‌به‌گام مفاهیم از ساده به پیچیده
- استفاده از مثال‌های بالینی/کاربردی
- ارتباط بین مفاهیم مختلف

**۳. نکات کلیدی (Clinical Pearls) 💎**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- نکات مهم برای به‌خاطر سپردن
- اشتباهات رایج و نحوه اجتناب
- نکات امتحانی (High-Yield Points)

**۴. جداول آموزشی (Educational Tables) 📊**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
| عنوان | توضیح | مثال |
|-------|-------|------|
| | | |

**۵. خلاصه فصل (Summary)**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- مرور نکات کلیدی
- نقشه مفهومی (Concept Map)

**۶. سؤالات مروری (Review Questions)**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- ۳-۵ سؤال برای خودآزمایی

═══════════════════════════════════════════════════════════════

📝 الزامات نگارشی:

۱. **زبان:** فارسی رسمی و آکادمیک - از کلمات عامیانه استفاده نشود
۲. **اصطلاحات تخصصی:** به فارسی با معادل انگلیسی در پرانتز
   مثال: فشار خون (Blood Pressure)
۳. **ساختار جملات:** روان، علمی، بدون پیچیدگی غیرضروری
۴. **پاراگراف‌بندی:** هر پاراگراف یک ایده اصلی
۵. **تأکید:** استفاده از **بولد** برای نکات مهم

🎯 هدف نهایی: خواننده پس از مطالعه این فصل، نیازی به گوش دادن به صوت اصلی نداشته باشد و درک کاملی از موضوع پیدا کند.

زبان خروجی: فقط فارسی آکادمیک""",

        "en": """Role: Distinguished University Professor with 20+ years of teaching and textbook authoring experience.

Task: Transform the audio transcription into a comprehensive Reference Textbook Chapter (similar to Harrison's, Guyton's, or Robbins' standards).

STRUCTURE:

## 1. Introduction
- Scientific definition with foundational concepts
- Clinical/scientific significance
- Learning objectives
- Prerequisites

## 2. Main Content
- Logical organization with **bold headers**
- Step-by-step explanation from simple to complex
- Clinical/practical examples
- Concept interconnections

## 3. Clinical Pearls 💎
- Key points to remember
- Common mistakes to avoid
- High-yield examination points

## 4. Educational Tables 📊
| Topic | Description | Example |
|-------|-------------|---------|

## 5. Chapter Summary
- Key points review
- Concept map

## 6. Review Questions
- 3-5 self-assessment questions

OUTPUT LANGUAGE: English only, formal academic tone.""",

        "fr": """Rôle: Professeur d'université distingué.
Tâche: Transformer la transcription en un chapitre de manuel académique complet en français.
Structure: Introduction, Contenu principal avec en-têtes, Points clés, Tableaux, Résumé, Questions.
LANGUE DE SORTIE: Français académique uniquement.""",

        "es": """Rol: Profesor universitario distinguido.
Tarea: Transformar la transcripción en un capítulo de libro de texto académico completo en español.
Estructura: Introducción, Contenido principal con encabezados, Puntos clave, Tablas, Resumen, Preguntas.
IDIOMA DE SALIDA: Español académico únicamente.""",

        "de": """Rolle: Angesehener Universitätsprofessor.
Aufgabe: Die Transkription in ein umfassendes akademisches Lehrbuchkapitel auf Deutsch umwandeln.
Struktur: Einleitung, Hauptinhalt mit Überschriften, Kernpunkte, Tabellen, Zusammenfassung, Fragen.
AUSGABESPRACHE: Nur akademisches Deutsch.""",

        "ru": """Роль: Выдающийся профессор университета.
Задача: Преобразовать транскрипцию в полноценную главу академического учебника на русском языке.
Структура: Введение, Основное содержание с заголовками, Ключевые моменты, Таблицы, Резюме, Вопросы.
ЯЗЫК ВЫВОДА: Только академический русский.""",

        "ar": """الدور: أستاذ جامعي متميز.
المهمة: تحويل النص المكتوب إلى فصل كتاب أكاديمي شامل باللغة العربية.
الهيكل: مقدمة، محتوى رئيسي مع عناوين، نقاط رئيسية، جداول، ملخص، أسئلة.
لغة الإخراج: العربية الأكاديمية فقط."""
    }
    return prompts.get(lang, prompts["en"])


def get_lecture_prompt_fast(lang: str = "fa") -> str:
    """Simplified Lecture prompt for Fast Engine."""
    target = LANGUAGES.get(lang, LANGUAGES["fa"])
    return f"""You are a university professor. Create a comprehensive lecture notes document.

Include:
1. Introduction
2. Main content with bold headers
3. Key points
4. Summary

OUTPUT LANGUAGE: {target.name_en} ({target.name_native}) only."""


def get_summary_prompt(lang: str, engine: Engine) -> str:
    """Summary prompt for specified language and engine."""
    target = LANGUAGES.get(lang, LANGUAGES["fa"])
    
    if engine == Engine.PRO:
        return f"""Role: Expert Content Analyst and Academic Summarizer.

Task: Create a comprehensive, structured summary in {target.name_en} ({target.name_native}).

FORMAT:

📌 **نمای کلی / Executive Summary**
[3-4 sentences capturing the essence]

📋 **نکات کلیدی / Key Points**
• [Point 1 - most important]
• [Point 2]
• [Point 3]
• [Continue as needed...]

💡 **جزئیات مهم / Critical Details**
[Names, numbers, dates, specific information]

📊 **ساختار محتوا / Content Structure**
[How the original content was organized]

🎯 **نتیجه‌گیری / Conclusions**
[Main takeaways and implications]

✅ **اقدامات پیشنهادی / Recommended Actions** (if applicable)
[Any action items mentioned]

OUTPUT LANGUAGE: {target.name_en.upper()} ({target.name_native}) ONLY"""
    else:
        return f"""Summarize this content in {target.name_en}.

Include:
• Overview (2-3 sentences)
• Key points (bullet list)
• Conclusion

OUTPUT: {target.name_en} only."""


def get_transcript_prompt(lang: str, engine: Engine) -> str:
    """Transcript formatting prompt."""
    target = LANGUAGES.get(lang, LANGUAGES["fa"])
    
    if engine == Engine.PRO:
        return f"""Role: Professional Transcription Specialist.

Task: Format and clean the raw transcription with expert precision.

RULES:
1. Fix transcription errors while preserving original meaning
2. Add proper punctuation (. , ? ! : ; —)
3. Create logical paragraph breaks
4. Mark speakers as [Speaker 1], [Speaker 2] if multiple
5. Preserve mixed-language content:
   - Keep English words in Latin script within {target.name_en} text
   - Example: "من یک meeting داشتم" stays as-is
6. Mark unclear audio as [نامفهوم] or [unclear]
7. Add timestamps for long content: [00:00]
8. Preserve technical terms, names, and numbers exactly

FORMAT:
Clean, professional paragraphs with proper formatting.

OUTPUT LANGUAGE: Preserve original language, format in {target.name_en}."""
    else:
        return f"""Clean and format this transcription.
- Fix errors, add punctuation
- Create paragraphs
- Mark unclear parts as [unclear]
- Keep original language
OUTPUT: Formatted text."""


def get_lyrics_prompt(engine: Engine) -> str:
    """Lyrics extraction prompt."""
    if engine == Engine.PRO:
        return """Role: Professional Music Transcriptionist and Lyrics Analyst.

Task: Extract and format lyrics OR speech transcription with expert precision.

FOR MUSIC:
🎵 **Song Information** (if identifiable)
- Title:
- Artist:
- Album:
- Genre:
- Language:

---

[Intro] (if applicable)

[Verse 1]
Line 1
Line 2
...

[Pre-Chorus]
...

[Chorus]
...

[Verse 2]
...

[Bridge]
...

[Outro]
...

---

📝 **Notes:**
- Describe the mood/tone
- Note any background vocals
- Identify instruments if notable

FOR SPEECH:
Format as clean paragraphs with speaker identification.

RULES:
1. Keep ORIGINAL language - never translate
2. Mark instrumental: [🎸 Guitar Solo], [🎹 Piano], [🥁 Drums]
3. Mark unclear lyrics: [...]
4. Note harmonies: (harmony) or [Background: ...]
5. Include ad-libs in parentheses

OUTPUT: Original language, professionally formatted."""
    else:
        return """Extract lyrics or transcribe speech.

Format:
[Verse 1]
Lines...

[Chorus]
Lines...

Keep original language. Mark unclear parts as [...].
OUTPUT: Formatted lyrics/transcription."""


def get_translation_prompt(source_lang: str, target_lang: str, engine: Engine) -> str:
    """Translation prompt between languages."""
    source = LANGUAGES.get(source_lang, LANGUAGES["en"])
    target = LANGUAGES.get(target_lang, LANGUAGES["fa"])
    
    if engine == Engine.PRO:
        return f"""Role: Expert Translator with native fluency in both {source.name_en} and {target.name_en}.

Task: Translate the content from {source.name_en} to {target.name_en} with professional quality.

TRANSLATION PRINCIPLES:

1. **Semantic Accuracy:** Preserve complete meaning
2. **Natural Fluency:** Use idiomatic {target.name_en}
3. **Tone Preservation:** Maintain speaker's style
4. **Cultural Adaptation:** Adapt cultural references appropriately
5. **Technical Precision:** Keep specialized terms accurate

SPECIAL HANDLING:
- **Proper nouns:** Keep original or use standard transliteration
- **Idioms:** Use equivalent expressions, not literal translation
- **Numbers/Dates:** Convert to target locale if appropriate
- **Quotes:** Preserve with appropriate quotation marks
- **Technical terms:** Translate with original in parentheses

OUTPUT FORMAT:

📝 **{target.name_native} Translation:**

[Full translated text]

---

📌 **خلاصه / Summary:**
[2-3 sentence summary of content]

🔤 **کلمات کلیدی / Keywords:**
[Key terms from the text]

OUTPUT LANGUAGE: {target.name_en.upper()} ({target.name_native}) ONLY"""
    else:
        return f"""Translate from {source.name_en} to {target.name_en}.

Maintain:
- Original meaning
- Natural expression
- Proper nouns

OUTPUT: {target.name_en} translation only."""


# ============== UI MESSAGES ==============
MESSAGES = {
    "welcome": """🎧 **به Omni-Hear AI خوش آمدید!**

🚀 **نسخه 6.0 - موتور دوگانه**

**⚡ حالت سریع:** پاسخ در ۳ ثانیه (Llama 70B)
**🚀 حالت دقیق:** حداکثر کیفیت (Llama 405B)

📤 **یک فایل صوتی یا ویس ارسال کنید**

🌐 **زبان‌ها:**
🇮🇷 فارسی | 🇬🇧 English | 🇫🇷 Français
🇪🇸 Español | 🇩🇪 Deutsch | 🇷🇺 Русский | 🇸🇦 العربية""",

    "audio_received": """🎵 **فایل دریافت شد!** ({size})

⚡ **سریع** = پاسخ سریع (~۳ ثانیه)
🚀 **دقیق** = کیفیت حرفه‌ای (405B)

📋 نوع پردازش را انتخاب کنید:""",

    "select_language": "🌍 **زبان خروجی را انتخاب کنید:**",
    "select_source_lang": "🗣 **زبان صوت (مبدا):**",
    "select_target_lang": "🎯 **زبان ترجمه (مقصد):**",
    
    "processing_stt": "🎤 **مرحله ۱/۲:** تبدیل صدا به متن...",
    "processing_fast": "⚡ **مرحله ۲/۲:** پردازش سریع با Llama 70B...",
    "processing_pro": "🚀 **مرحله ۲/۲:** پردازش حرفه‌ای با Llama 405B...",
    "fallback_notice": "⚠️ سرویس 405B در دسترس نیست. استفاده از حالت سریع...",
    
    "error": "❌ خطا در پردازش. لطفاً دوباره تلاش کنید.",
    "no_audio": "⚠️ لطفاً ابتدا یک فایل صوتی ارسال کنید.",
    "file_too_large": "⚠️ حجم فایل بیشتر از ۲۵ مگابایت است.",
    "not_audio": "⚠️ لطفاً فایل صوتی ارسال کنید (MP3, OGG, WAV, M4A).",
    "api_missing": "⚠️ کلید API تنظیم نشده: {missing}",
    "pro_unavailable": "⚠️ حالت Pro در دسترس نیست. از حالت سریع استفاده شد.",
}


# ============== KEYBOARDS ==============
def get_main_menu_keyboard() -> InlineKeyboardMarkup:
    """Main menu with dual buttons for each feature."""
    return InlineKeyboardMarkup([
        # Transcript
        [
            InlineKeyboardButton("📜 رونویسی ⚡", callback_data="mode:transcript:fast"),
            InlineKeyboardButton("📜 رونویسی 🚀", callback_data="mode:transcript:pro"),
        ],
        # Lecture
        [
            InlineKeyboardButton("📚 درسنامه ⚡", callback_data="mode:lecture:fast"),
            InlineKeyboardButton("📚 درسنامه 🚀", callback_data="mode:lecture:pro"),
        ],
        # Medical SOAP
        [
            InlineKeyboardButton("🩺 پزشکی ⚡", callback_data="mode:soap:fast"),
            InlineKeyboardButton("🩺 پزشکی 🚀", callback_data="mode:soap:pro"),
        ],
        # Summary
        [
            InlineKeyboardButton("📝 خلاصه ⚡", callback_data="mode:summary:fast"),
            InlineKeyboardButton("📝 خلاصه 🚀", callback_data="mode:summary:pro"),
        ],
        # Lyrics
        [
            InlineKeyboardButton("🎵 متن آهنگ ⚡", callback_data="mode:lyrics:fast"),
            InlineKeyboardButton("🎵 متن آهنگ 🚀", callback_data="mode:lyrics:pro"),
        ],
        # Translation
        [
            InlineKeyboardButton("🌍 ترجمه ⚡", callback_data="mode:translate:fast"),
            InlineKeyboardButton("🌍 ترجمه 🚀", callback_data="mode:translate:pro"),
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


# ============== GROQ WHISPER STT ==============
async def transcribe_with_whisper(audio_data: bytes) -> Tuple[Optional[str], Optional[str]]:
    """Transcribe with Groq Whisper including context prompt."""
    if not groq_client:
        return None, "Groq client not initialized"
    
    try:
        def _transcribe():
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(audio_data)
                temp_path = f.name
            
            try:
                with open(temp_path, "rb") as audio_file:
                    result = groq_client.audio.transcriptions.create(
                        model=WHISPER_MODEL,
                        file=audio_file,
                        response_format="text",
                        language=None,  # Auto-detect
                        temperature=0.0,
                        prompt=WHISPER_CONTEXT_PROMPT,  # Context for better accuracy
                    )
                return result, None
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        
        result, error = await asyncio.to_thread(_transcribe)
        
        if error:
            return None, error
        
        if result and len(result.strip()) > 0:
            logger.info(f"✅ Whisper: {len(result)} chars")
            return result.strip(), None
        
        return None, "Empty transcription"
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Whisper error: {error_msg}")
        if "rate_limit" in error_msg.lower():
            return None, "rate_limit"
        return None, error_msg[:100]


# ============== GROQ LLM (FAST) ==============
async def process_with_groq(text: str, system_prompt: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Process with Groq LLM (Fast mode)."""
    if not groq_client:
        return None, None, "Groq client not initialized"
    
    models = [GROQ_LLM_PRIMARY, GROQ_LLM_FALLBACK]
    
    for model in models:
        try:
            logger.info(f"⚡ Groq: {model}")
            
            def _generate():
                return groq_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Process:\n\n{text}"}
                    ],
                    temperature=0.7,
                    max_tokens=8000,
                )
            
            response = await asyncio.to_thread(_generate)
            
            if response.choices and response.choices[0].message.content:
                result = response.choices[0].message.content.strip()
                logger.info(f"✅ Groq success: {len(result)} chars")
                return result, f"⚡ {model}", None
        
        except Exception as e:
            logger.warning(f"❌ Groq {model}: {str(e)[:50]}")
            continue
    
    return None, None, "All Groq models failed"


# ============== SAMBANOVA LLM (PRO) ==============
async def process_with_sambanova(text: str, system_prompt: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Process with SambaNova LLM (Pro mode)."""
    if not sambanova_client:
        return None, None, "SambaNova not available"
    
    models = [SAMBANOVA_MODEL_PRO, SAMBANOVA_MODEL_FALLBACK]
    
    for model in models:
        try:
            logger.info(f"🚀 SambaNova: {model}")
            
            def _generate():
                return sambanova_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Process this transcription:\n\n{text}"}
                    ],
                    temperature=0.7,
                    max_tokens=8000,
                )
            
            response = await asyncio.to_thread(_generate)
            
            if response.choices and response.choices[0].message.content:
                result = response.choices[0].message.content.strip()
                logger.info(f"✅ SambaNova success: {len(result)} chars")
                return result, f"🚀 {model}", None
        
        except Exception as e:
            logger.warning(f"❌ SambaNova {model}: {str(e)[:50]}")
            continue
    
    return None, None, "SambaNova failed"


# ============== UNIFIED PROCESSOR ==============
async def process_with_llm(
    text: str,
    system_prompt: str,
    engine: Engine
) -> Tuple[Optional[str], Optional[str], Optional[str], bool]:
    """
    Process with appropriate engine.
    Returns: (result, model_name, error, used_fallback)
    """
    used_fallback = False
    
    if engine == Engine.PRO:
        # Try SambaNova first
        if sambanova_client:
            result, model, error = await process_with_sambanova(text, system_prompt)
            if result:
                return result, model, None, False
            logger.warning("SambaNova failed, falling back to Groq")
        
        # Fallback to Groq
        used_fallback = True
    
    # Use Groq (Fast mode or fallback)
    result, model, error = await process_with_groq(text, system_prompt)
    return result, model, error, used_fallback


# ============== FULL PIPELINE ==============
async def process_audio_complete(
    audio_data: bytes,
    mime_type: str,
    mode: str,
    engine: Engine,
    lang: str = "fa",
    source_lang: Optional[str] = None,
    target_lang: Optional[str] = None,
) -> Dict:
    """Complete audio processing pipeline."""
    result = {
        "text": None,
        "transcription": None,
        "model": None,
        "error": None,
        "used_fallback": False,
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
    
    # Step 1: Transcribe with Whisper
    transcription, stt_error = await transcribe_with_whisper(mp3_data)
    
    if stt_error:
        if stt_error == "rate_limit":
            result["error"] = "⚠️ محدودیت Whisper. چند دقیقه صبر کنید."
        else:
            result["error"] = f"❌ خطا در STT: {stt_error}"
        return result
    
    if not transcription:
        result["error"] = "❌ متنی استخراج نشد."
        return result
    
    result["transcription"] = transcription
    
    # Step 2: Get appropriate prompt and process
    if mode == "transcript":
        prompt = get_transcript_prompt(lang, engine)
    elif mode == "lecture":
        prompt = get_lecture_prompt_pro(lang) if engine == Engine.PRO else get_lecture_prompt_fast(lang)
    elif mode == "soap":
        prompt = get_soap_prompt_pro() if engine == Engine.PRO else get_soap_prompt_fast()
    elif mode == "summary":
        prompt = get_summary_prompt(lang, engine)
    elif mode == "lyrics":
        prompt = get_lyrics_prompt(engine)
    elif mode == "translate":
        if not source_lang or not target_lang:
            result["error"] = "❌ زبان مشخص نشده"
            return result
        prompt = get_translation_prompt(source_lang, target_lang, engine)
    else:
        prompt = get_transcript_prompt(lang, engine)
    
    # Process with LLM
    text, model, llm_error, used_fallback = await process_with_llm(transcription, prompt, engine)
    
    result["text"] = text
    result["model"] = model
    result["used_fallback"] = used_fallback
    
    if llm_error and not text:
        result["error"] = f"❌ {llm_error}"
    
    return result


# ============== TELEGRAM HANDLERS ==============
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(MESSAGES["welcome"], parse_mode="Markdown")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = """📖 **راهنمای Omni-Hear AI v6.0**

**🔹 نحوه استفاده:**
1️⃣ فایل صوتی ارسال کنید
2️⃣ حالت پردازش را انتخاب کنید:
   • ⚡ سریع = پاسخ فوری
   • 🚀 دقیق = کیفیت حرفه‌ای

**🔹 قابلیت‌ها:**
• 📜 رونویسی - متن کامل صوت
• 📚 درسنامه - فصل کتاب درسی
• 🩺 پزشکی - SOAP Note استاندارد
• 📝 خلاصه - خلاصه هوشمند
• 🎵 متن آهنگ - لیریک
• 🌍 ترجمه - ۷ زبان

**🔹 موتورها:**
• ⚡ Groq Llama 70B (~۳ ثانیه)
• 🚀 SambaNova Llama 405B (حرفه‌ای)

**🔹 دستورات:**
/start - شروع
/help - راهنما
/status - وضعیت"""
    
    await update.message.reply_text(help_text, parse_mode="Markdown")


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    status = ["🔍 **وضعیت سیستم**\n"]
    
    if groq_client:
        status.append("✅ **Groq (STT + Fast):** فعال")
    else:
        status.append("❌ **Groq:** غیرفعال")
    
    if sambanova_client:
        status.append("✅ **SambaNova (Pro 405B):** فعال")
    else:
        status.append("⚠️ **SambaNova:** غیرفعال")
    
    status.append(f"\n**🤖 مدل‌ها:**")
    status.append(f"• STT: `{WHISPER_MODEL}`")
    status.append(f"• Fast: `{GROQ_LLM_PRIMARY}`")
    status.append(f"• Pro: `{SAMBANOVA_MODEL_PRO}`")
    
    flags = " ".join([l.flag for l in LANGUAGES.values()])
    status.append(f"\n**🌍 زبان‌ها:** {flags}")
    
    await update.message.reply_text("\n".join(status), parse_mode="Markdown")


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle audio files."""
    user_id = update.effective_user.id
    msg = update.message
    
    if not groq_client:
        await msg.reply_text(MESSAGES["api_missing"].format(missing="GROQ_API_KEY"))
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
        
        user_audio_cache[user_id] = {
            "data": bytes(audio_bytes),
            "mime_type": mime_type,
            "size": len(audio_bytes),
        }
        
        # Clear state
        user_state.pop(user_id, None)
        
        size_kb = len(audio_bytes) / 1024
        size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb/1024:.1f} MB"
        
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
    
    # Back button
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
            await query.edit_message_text(MESSAGES["no_audio"])
        user_state.pop(user_id, None)
        return
    
    # Mode selection: mode:type:engine
    if action == "mode":
        mode = parts[1]
        engine_str = parts[2]
        engine = Engine.PRO if engine_str == "pro" else Engine.FAST
        
        if user_id not in user_audio_cache:
            await query.edit_message_text(MESSAGES["no_audio"])
            return
        
        # Store engine preference
        user_state[user_id] = {"engine": engine}
        
        # Modes needing language selection
        if mode in ["transcript", "lecture", "summary"]:
            user_state[user_id]["mode"] = mode
            await query.edit_message_text(
                MESSAGES["select_language"],
                reply_markup=get_language_keyboard(f"lang:{mode}:{engine_str}"),
                parse_mode="Markdown"
            )
            return
        
        # Translation needs source + target
        if mode == "translate":
            user_state[user_id]["mode"] = mode
            await query.edit_message_text(
                MESSAGES["select_source_lang"],
                reply_markup=get_language_keyboard(f"source:{engine_str}"),
                parse_mode="Markdown"
            )
            return
        
        # SOAP and Lyrics - process directly
        await process_and_respond(query, context, user_id, mode, engine)
        return
    
    # Language selection: lang:mode:engine:code
    if action == "lang":
        mode = parts[1]
        engine_str = parts[2]
        lang = parts[3]
        engine = Engine.PRO if engine_str == "pro" else Engine.FAST
        await process_and_respond(query, context, user_id, mode, engine, lang=lang)
        return
    
    # Source language: source:engine:code
    if action == "source":
        engine_str = parts[1]
        source_lang = parts[2]
        user_state[user_id]["source_lang"] = source_lang
        await query.edit_message_text(
            MESSAGES["select_target_lang"],
            reply_markup=get_target_language_keyboard(source_lang, f"target:{engine_str}"),
            parse_mode="Markdown"
        )
        return
    
    # Target language: target:engine:code
    if action == "target":
        engine_str = parts[1]
        target_lang = parts[2]
        engine = Engine.PRO if engine_str == "pro" else Engine.FAST
        source_lang = user_state.get(user_id, {}).get("source_lang", "en")
        await process_and_respond(
            query, context, user_id, "translate", engine,
            source_lang=source_lang, target_lang=target_lang
        )
        return


async def process_and_respond(
    query,
    context,
    user_id: int,
    mode: str,
    engine: Engine,
    lang: str = "fa",
    source_lang: Optional[str] = None,
    target_lang: Optional[str] = None,
) -> None:
    """Process and send response."""
    
    if user_id not in user_audio_cache:
        await query.edit_message_text(MESSAGES["no_audio"])
        return
    
    audio_info = user_audio_cache[user_id]
    
    mode_names = {
        "transcript": "📜 رونویسی",
        "lecture": "📚 درسنامه",
        "soap": "🩺 SOAP پزشکی",
        "summary": "📝 خلاصه",
        "lyrics": "🎵 متن آهنگ",
        "translate": "🌍 ترجمه",
    }
    
    engine_name = "🚀 Pro (405B)" if engine == Engine.PRO else "⚡ Fast (70B)"
    
    try:
        # Show STT progress
        await query.edit_message_text(
            f"🎯 **{mode_names.get(mode)}** | {engine_name}\n\n{MESSAGES['processing_stt']}",
            parse_mode="Markdown"
        )
        
        # Process
        result = await process_audio_complete(
            audio_info["data"],
            audio_info["mime_type"],
            mode,
            engine,
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
        header = f"✅ **{mode_names.get(mode)}**\n"
        
        if mode == "translate" and source_lang and target_lang:
            src = LANGUAGES.get(source_lang)
            tgt = LANGUAGES.get(target_lang)
            header += f"{src.flag} → {tgt.flag}\n"
        
        header += "\n"
        
        # Footer with model info
        footer = f"\n\n---\n🤖 `{result['model']}`"
        if result["used_fallback"]:
            footer += f"\n⚠️ {MESSAGES['pro_unavailable']}"
        
        full_text = header + result["text"] + footer
        
        # Send (handle long messages)
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
            except:
                await query.edit_message_text(full_text)
    
    except Exception as e:
        logger.error(f"Process error: {e}")
        logger.error(traceback.format_exc())
        await query.edit_message_text(f"❌ خطا: {str(e)[:100]}")
    
    finally:
        user_audio_cache.pop(user_id, None)
        user_state.pop(user_id, None)


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(f"Error: {context.error}")


# ============== MAIN ==============
def main() -> None:
    print("\n" + "=" * 70)
    print("  🎧 OMNI-HEAR AI v6.0 - DUAL-ENGINE EDITION")
    print("  ⚡ Fast (Groq 70B) | 🚀 Pro (SambaNova 405B)")
    print("=" * 70)
    
    if not TELEGRAM_BOT_TOKEN:
        print("❌ TELEGRAM_BOT_TOKEN not set!")
        sys.exit(1)
    
    if not GROQ_API_KEY:
        print("❌ GROQ_API_KEY not set!")
        sys.exit(1)
    
    print(f"✅ Telegram: Ready")
    print(f"✅ Groq (STT + Fast): Ready")
    print(f"{'✅' if SAMBANOVA_API_KEY else '⚠️'} SambaNova (Pro): {'Ready' if SAMBANOVA_API_KEY else 'Not configured'}")
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
