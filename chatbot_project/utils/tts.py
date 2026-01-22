# =========================================================================
# File Name: utils/tts.py
# Project: Absher Smart Assistant (MOI ChatBot)
# Architecture: Cross-Lingual Hybrid RAG (BGE-M3 + BM25 + ALLaM-7B)
#
# Affiliation: King Abdullah University of Science and Technology (KAUST)
# Team: Ahmed AlRashidi, Sultan Alshaibani, Fahad Alqahtani, 
#       Rakan Alharbi, Sultan Alotaibi, Abdulaziz Almutairi.
# Advisors: Prof. Naeemullah Khan & Dr. Salman Khan
# =========================================================================

import os
import uuid
import time
import glob
import re
from gtts import gTTS
from langdetect import detect
from config import Config
from utils.logger import setup_logger
from utils.text_utils import soft_clean

# Initialize module logger
logger = setup_logger(__name__)

def cleanup_old_audio(directory: str, max_age_seconds: int = 600):
    """
    Maintenance function: Removes audio files older than 'max_age_seconds'.
    Prevents disk space issues in long-running sessions.
    """
    try:
        if not os.path.exists(directory):
            return

        now = time.time()
        # Find all mp3 files
        files = glob.glob(os.path.join(directory, "*.mp3"))
        
        for f in files:
            # Check file modification time
            if os.stat(f).st_mtime < now - max_age_seconds:
                try:
                    os.remove(f)
                    logger.debug(f"🧹 Deleted old audio file: {os.path.basename(f)}")
                except OSError as e:
                    logger.warning(f"⚠️ Could not delete file {f}: {e}")
                    
    except Exception as e:
        logger.warning(f"⚠️ Audio cleanup process failed: {e}")

def prepare_text_for_speech(text: str) -> str:
    """
    Prepares text for TTS by removing HTML tags, Markdown artifacts,
    and extra whitespace using the project's standard text utilities.
    """
    if not text:
        return ""

    # 1. Remove HTML tags (e.g., <br>, <div dir='rtl'>)
    clean = re.sub(r'<[^>]+>', '', text)
    
    # 2. Remove Markdown (Bold, Headers, etc.) using shared utility
    clean = soft_clean(clean)
    
    return clean.strip()

def generate_speech(text: str) -> str:
    """
    Generates TTS audio using Google Text-to-Speech (gTTS).
    
    Features:
    - Auto-detects language (Arabic/English).
    - Generates unique filenames for concurrency.
    - Handles network errors gracefully (returns None instead of crashing).
    """
    if not text or not text.strip():
        return None

    try:
        # 1. Maintenance: Cleanup old files
        cleanup_old_audio(Config.AUDIO_DIR)
        
        # 2. Pre-processing: Clean text artifacts
        speech_text = prepare_text_for_speech(text)
        if not speech_text:
            logger.warning("TTS Warning: Text became empty after cleaning.")
            return None

        # 3. Language Detection
        try:
            # Default to 'ar' for safety in this specific domain
            lang = detect(speech_text)
            tts_lang = 'en' if lang == 'en' else 'ar'
        except Exception:
            tts_lang = 'ar'
        
        # 4. Generate Audio (Network Call)
        # slow=False -> Normal speed
        tts = gTTS(text=speech_text, lang=tts_lang, slow=False)
        
        # 5. Save File
        # Using UUID to avoid conflicts between users/sessions
        filename = f"response_{uuid.uuid4().hex[:8]}.mp3"
        output_path = os.path.join(Config.AUDIO_DIR, filename)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        tts.save(output_path)
        logger.info(f"🎙️ TTS Generated ({tts_lang}): {filename}")
        
        return output_path

    except Exception as e:
        logger.error(f"❌ TTS Generation Failed: {e}")
        # Return None so the UI knows not to display an audio player
        return None