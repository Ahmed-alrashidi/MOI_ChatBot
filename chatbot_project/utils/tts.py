# =========================================================================
# File Name: utils/tts.py
# Purpose: High-Fidelity Text-to-Speech (TTS) via Microsoft Edge Neural Engine.
# Project: Absher Smart Assistant (MOI ChatBot)
# Version: 5.1 (8-Language Voice Support)
# Features:
# - 8 Languages: Native voices for AR, EN, UR, FR, ES, DE, RU, ZH.
# - Dialect Accuracy: Uses 'ar-SA-HamedNeural' for authentic Saudi dialect output.
# - Sanitization: Strips Markdown/HTML/Brackets to prevent robotic pronunciation.
# - Storage Management: Auto-purges temporary audio files older than 10 minutes (TTL).
# - Async/Sync Bridge: Robust multi-threading to prevent Gradio UI deadlocks.
# =========================================================================

import os
import uuid
import time
import glob
import re
import asyncio
import threading
import edge_tts
from typing import Optional

from config import Config
from utils.logger import setup_logger

logger = setup_logger("TTS_Engine")

# [FIX v5.1] 8-language voice map (was only AR/EN)
VOICE_MAP = {
    'ar': "ar-SA-HamedNeural",       # Saudi Arabic
    'en': "en-US-AriNeural",          # American English
    'ur': "ur-PK-AsadNeural",         # Pakistani Urdu
    'fr': "fr-FR-HenriNeural",        # French
    'es': "es-ES-AlvaroNeural",       # Spanish
    'de': "de-DE-ConradNeural",       # German
    'ru': "ru-RU-DmitryNeural",       # Russian
    'zh': "zh-CN-YunxiNeural",        # Mandarin Chinese
}

DEFAULT_VOICE = VOICE_MAP['ar']


def cleanup_old_audio(directory: str, max_age_seconds: int = 600):
    """
    Periodic Maintenance Utility.
    Deletes expired audio files from the disk to conserve storage on the server.
    """
    try:
        if not os.path.exists(directory):
            return

        now = time.time()
        files = glob.glob(os.path.join(directory, "*.mp3"))

        for f in files:
            if os.stat(f).st_mtime < now - max_age_seconds:
                try:
                    os.remove(f)
                except OSError:
                    pass

    except Exception as e:
        logger.warning(f"Audio cleanup warning: {e}")


def sanitize_for_speech(text: str) -> str:
    """
    Prepares raw LLM text responses for the Neural Voice Engine.
    Removes visual formatting that sounds unnatural when spoken aloud.
    """
    if not text:
        return ""

    # 1. Strip URLs and Links
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

    # 2. Strip HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # 3. Strip Markdown & Brackets
    text = re.sub(r'[\*_`~#\[\]]', '', text)

    # 4. Normalize spacing
    return " ".join(text.split())


def detect_language(text: str) -> str:
    """
    [FIX v5.1] Multi-script language detection for TTS voice routing.
    Supports Arabic, Urdu, Chinese, Russian, and Latin-script languages.
    Falls back to langdetect for French/Spanish/German/English disambiguation.
    """
    # 1. Arabic/Urdu script (U+0600–U+06FF)
    if any("\u0600" <= c <= "\u06FF" for c in text):
        urdu_chars = set('پچڈڑٹںھہےۓکگ')
        if any(c in urdu_chars for c in text):
            return 'ur'
        return 'ar'

    # 2. Chinese script (CJK Unified Ideographs)
    if any("\u4e00" <= c <= "\u9fff" for c in text):
        return 'zh'

    # 3. Cyrillic script (Russian)
    if any("\u0400" <= c <= "\u04FF" for c in text):
        return 'ru'

    # 4. Latin script — use langdetect to distinguish FR/ES/DE/EN
    try:
        from langdetect import detect
        lang = detect(text)
        if lang in VOICE_MAP:
            return lang
    except Exception:
        pass

    return 'en'


async def _run_tts_async(text: str, voice: str, output_path: str):
    """Internal asynchronous task that communicates with the Edge-TTS API."""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


def generate_speech(text: str) -> Optional[str]:
    """
    Public Entry Point for Speech Synthesis.
    Acts as an Async-to-Sync bridge for safe Gradio integration.

    Args:
        text: The response text to be spoken.
    Returns:
        Path to the generated .mp3 file, or None if failed.
    """
    if not text or not text.strip():
        return None

    try:
        # 1. Infrastructure prep
        os.makedirs(Config.AUDIO_DIR, exist_ok=True)
        cleanup_old_audio(Config.AUDIO_DIR)

        # 2. Text preparation and voice routing
        clean_text = sanitize_for_speech(text)
        if not clean_text:
            return None

        lang = detect_language(clean_text)
        selected_voice = VOICE_MAP.get(lang, DEFAULT_VOICE)

        # 3. File generation
        filename = f"voice_{uuid.uuid4().hex[:10]}.mp3"
        output_path = os.path.join(Config.AUDIO_DIR, filename)

        # 4. Async/Sync Bridge (Thread-Safe)
        try:
            asyncio.run(_run_tts_async(clean_text, selected_voice, output_path))
        except RuntimeError:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                def thread_runner():
                    asyncio.run(_run_tts_async(clean_text, selected_voice, output_path))
                t = threading.Thread(target=thread_runner)
                t.start()
                t.join()
            else:
                loop.run_until_complete(_run_tts_async(clean_text, selected_voice, output_path))

        logger.info(f"🎙️ TTS Generated: {filename} | Voice: {selected_voice} | Lang: {lang}")
        return output_path

    except Exception as e:
        logger.error(f"❌ TTS Engine Error: {str(e)}", exc_info=True)
        return None