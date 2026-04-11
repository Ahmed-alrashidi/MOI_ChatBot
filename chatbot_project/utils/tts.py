# =========================================================================
# File Name: utils/tts.py
# Purpose: High-Fidelity Text-to-Speech (TTS) via Microsoft Edge Neural Engine.
# Project: Absher Smart Assistant (MOI ChatBot)
# Features:
# - Dialect Accuracy: Uses 'ar-SA-HamedNeural' for authentic Saudi dialect output.
# - Sanitization: Strips Markdown/HTML/Brackets to prevent robotic/unnatural pronunciation.
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

# Dedicated logger for speech synthesis events
logger = setup_logger("TTS_Engine")

# Voice selection: Authentic Saudi Arabic and Professional English
VOICE_AR = "ar-SA-HamedNeural"
VOICE_EN = "en-US-AriNeural"

def cleanup_old_audio(directory: str, max_age_seconds: int = 600):
    """
    Periodic Maintenance Utility.
    Deletes expired audio files from the disk to conserve storage on the server.
    
    Args:
        directory (str): The directory containing audio outputs.
        max_age_seconds (int): Time-to-Live (TTL). Defaults to 10 minutes.
    """
    try:
        if not os.path.exists(directory):
            return

        now = time.time()
        files = glob.glob(os.path.join(directory, "*.mp3"))
        
        for f in files:
            # Check file modification time against TTL
            if os.stat(f).st_mtime < now - max_age_seconds:
                try:
                    os.remove(f)
                except OSError:
                    pass # Safely skip if the file is currently locked or playing
                    
    except Exception as e:
        logger.warning(f"Audio cleanup warning: {e}")

def sanitize_for_speech(text: str) -> str:
    """
    Prepares raw LLM text responses for the Neural Voice Engine.
    Removes visual formatting (Markdown, Links) that sounds unnatural when spoken aloud.
    
    Args:
        text (str): Raw text from the LLM.
    Returns:
        str: Sanitized text optimized for TTS.
    """
    if not text:
        return ""

    # 1. Strip URLs and Links: Convert [Text](Link) -> Text
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

    # 2. Strip HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # 3. Strip Markdown & Brackets (*, _, #, `, ~, [, ]) to avoid robotic spelling
    text = re.sub(r'[\*_`~#\[\]]', '', text)
    
    # 4. Normalize spacing
    return " ".join(text.split())

def detect_language(text: str) -> str:
    """
    Heuristic language detection to select the appropriate TTS voice model.
    Checks for the presence of Arabic script characters.
    """
    if any("\u0600" <= c <= "\u06FF" for c in text):
        return 'ar'
    return 'en'

async def _run_tts_async(text: str, voice: str, output_path: str):
    """
    Internal asynchronous task that communicates with the Edge-TTS API.
    """
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

def generate_speech(text: str) -> Optional[str]:
    """
    Public Entry Point for Speech Synthesis.
    Crucially acts as an Async-to-Sync bridge, safely allowing synchronous 
    Gradio UI clicks to execute asynchronous network calls without crashing the event loop.
    
    Args:
        text (str): The response text to be spoken.
    Returns:
        str: Path to the generated local .mp3 file, or None if failed.
    """
    if not text or not text.strip():
        return None

    try:
        # 1. Infrastructure prep
        os.makedirs(Config.AUDIO_DIR, exist_ok=True)
        cleanup_old_audio(Config.AUDIO_DIR)
        
        # 2. Text preparation and routing
        clean_text = sanitize_for_speech(text)
        if not clean_text: 
            return None
        
        lang = detect_language(clean_text)
        selected_voice = VOICE_AR if lang == 'ar' else VOICE_EN
        
        # 3. File generation (Unique identifier)
        filename = f"voice_{uuid.uuid4().hex[:10]}.mp3"
        output_path = os.path.join(Config.AUDIO_DIR, filename)

        # 4. Modern Async/Sync Bridge (Thread-Safe)
        # Gradio runs its own FastAPI event loop. Calling asyncio.run() directly 
        # inside an active loop throws a RuntimeError. 
        try:
            asyncio.run(_run_tts_async(clean_text, selected_voice, output_path))
        except RuntimeError:
            # Fallback: We are inside an active loop. Spin up a separate thread 
            # with its own isolated event loop to run the async task safely.
            loop = asyncio.get_event_loop()
            if loop.is_running():
                def thread_runner():
                    asyncio.run(_run_tts_async(clean_text, selected_voice, output_path))
                t = threading.Thread(target=thread_runner)
                t.start()
                t.join() # Block until audio is fully downloaded
            else:
                loop.run_until_complete(_run_tts_async(clean_text, selected_voice, output_path))

        logger.info(f"🎙️ TTS Generated: {filename} | Voice: {selected_voice}")
        return output_path

    except Exception as e:
        logger.error(f"❌ TTS Engine Error: {str(e)}", exc_info=True)
        return None