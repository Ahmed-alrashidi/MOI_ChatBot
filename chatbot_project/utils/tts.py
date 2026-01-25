# =========================================================================
# File Name: utils/tts.py
# Purpose: High-Fidelity Text-to-Speech (TTS) using Microsoft Edge Neural Engine.
# Project: Absher Smart Assistant (MOI ChatBot)
# Features:
# - Localized Dialect: Uses 'ar-SA-HamedNeural' for authentic Saudi male voice.
# - Clean Synthesis: Strips Markdown/HTML to prevent the AI from reading symbols.
# - Async Integration: Uses asyncio to handle API calls without blocking the UI.
# - Storage Optimization: Self-cleaning mechanism for temporary MP3 files.
# =========================================================================

import os
import uuid
import time
import glob
import re
import asyncio
import edge_tts
from config import Config
from utils.logger import setup_logger

# Initialize a dedicated logger for the TTS engine
logger = setup_logger("TTS_Engine")

# --- Voice Configuration ---
# Arabic: "ar-SA-HamedNeural" - Specifically chosen for its natural Saudi tone.
# English: "en-US-AriNeural" - A clear, professional male voice.
VOICE_AR = "ar-SA-HamedNeural"
VOICE_EN = "en-US-AriNeural"

def cleanup_old_audio(directory: str, max_age_seconds: int = 600):
    """
    Automated Maintenance: Purges expired temporary audio files from the cache.
    This prevents the server storage from filling up over time.
    
    Args:
        directory (str): The path where MP3 files are stored.
        max_age_seconds (int): Time-to-Live (TTL) for files. Default is 10 minutes.
    """
    try:
        if not os.path.exists(directory):
            return

        now = time.time()
        files = glob.glob(os.path.join(directory, "*.mp3"))
        
        purged_count = 0
        for f in files:
            # Check if the file's last modification time is older than the threshold
            if os.stat(f).st_mtime < now - max_age_seconds:
                try:
                    os.remove(f)
                    purged_count += 1
                except OSError:
                    pass # Ignore if file is currently being accessed
        
        if purged_count > 0:
            logger.debug(f"🧹 Storage Cleanup: Removed {purged_count} expired audio files.")
                    
    except Exception as e:
        logger.warning(f"⚠️ Audio maintenance warning: {e}")

def clean_text_for_tts(text: str) -> str:
    """
    Sanitizes raw AI responses for the TTS engine.
    It removes Markdown symbols and HTML tags so the voice reads only the 
    intended words, not the formatting.

    Args:
        text (str): Raw string from the LLM.

    Returns:
        str: Pure text ready for speech synthesis.
    """
    if not text:
        return ""

    # 1. Strip HTML tags (e.g., <br>, <b>) using regex
    text = re.sub(r'<[^>]+>', '', text)
    
    # 2. Handle Markdown Links: [Absher](https://...) -> Becomes "Absher"
    # We extract the readable text and discard the URL.
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

    # 3. Strip Markdown Formatting (Stars, underscores, code ticks, headers)
    # This prevents the TTS from trying to pronounce symbols like "star star".
    text = re.sub(r'[\*_`~]', '', text)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    
    # 4. Normalize Whitespace: Collapse multiple tabs/newlines into single spaces.
    return " ".join(text.split())

def detect_tts_language(text: str) -> str:
    """
    Uses character heuristics to select the appropriate Neural Voice.
    
    Args:
        text (str): The sanitized text.
        
    Returns:
        str: 'ar' for Arabic script, 'en' otherwise.
    """
    # Check if text contains characters within the Arabic Unicode block
    if any("\u0600" <= c <= "\u06FF" for c in text):
        return 'ar'
    return 'en'

async def _generate_edge_tts(text: str, voice: str, output_path: str):
    """
    Low-level Async function to communicate with the Edge TTS WebSocket API.
    
    Args:
        text: Sanity-checked text.
        voice: The specific neural voice ID.
        output_path: Destination MP3 path.
    """
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

def generate_speech(text: str) -> str:
    """
    The High-Level Entry Point for Speech Synthesis.
    It bridges the synchronous Gradio UI with the asynchronous TTS engine.
    
    Returns:
        str: The absolute local path to the generated MP3 file.
    """
    if not text or not text.strip():
        return None

    try:
        # Step 1: Run Maintenance (Garbage Collection)
        cleanup_old_audio(Config.AUDIO_DIR)
        
        # Step 2: Pre-process Text (Strip Markdown/HTML)
        speech_text = clean_text_for_tts(text)
        if not speech_text.strip():
            return None

        # Step 3: Select Voice (Saudi Hamed vs. American Ari)
        lang = detect_tts_language(speech_text)
        selected_voice = VOICE_AR if lang == 'ar' else VOICE_EN
        
        # Step 4: Generate Unique Filename to prevent cache collisions
        unique_id = uuid.uuid4().hex[:8]
        filename = f"res_{unique_id}.mp3"
        output_path = os.path.join(Config.AUDIO_DIR, filename)
        
        # Ensure the output directory exists on the server
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Step 5: Execute Async task in a synchronous context
        # This allows Gradio (Sync) to wait for the Edge TTS (Async) result.
        asyncio.run(_generate_edge_tts(speech_text, selected_voice, output_path))
        
        logger.info(f"🎙️ Neural TTS Generated | Voice: {selected_voice} | File: {filename}")
        
        return output_path

    except Exception as e:
        logger.error(f"❌ Speech Synthesis Failed: {str(e)}")
        return None