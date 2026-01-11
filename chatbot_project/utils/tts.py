import os
import uuid
import time
import glob
import re  # Added for HTML cleanup
from gtts import gTTS
from langdetect import detect
from config import Config
from utils.logger import setup_logger

# Initialize module logger
logger = setup_logger(__name__)

def cleanup_old_audio(directory: str, max_age_seconds: int = 600):
    """
    Removes audio files older than 'max_age_seconds' to prevent disk bloat.
    Runs silently in the background.
    """
    try:
        if not os.path.exists(directory):
            return

        now = time.time()
        files = glob.glob(os.path.join(directory, "*.mp3"))
        for f in files:
            if os.stat(f).st_mtime < now - max_age_seconds:
                os.remove(f)
                logger.debug(f"🧹 Deleted old audio file: {os.path.basename(f)}")
    except Exception as e:
        logger.warning(f"⚠️ Audio cleanup failed: {e}")

def clean_html_tags(text: str) -> str:
    """
    Removes HTML tags (like <div dir='rtl'>) from text to prevent TTS reading them.
    """
    clean = re.sub(r'<[^>]+>', '', text)
    return clean.strip()

def generate_speech(text: str) -> str:
    """
    Generates TTS audio with unique filenames to support concurrent users.
    Includes auto-cleanup mechanism and HTML tag stripping.
    """
    if not text or not text.strip():
        return None

    try:
        # 1. Cleanup old files first (Self-Maintenance)
        cleanup_old_audio(Config.AUDIO_DIR)
        
        # 2. Clean Text (Remove HTML tags from RAG output)
        clean_text = clean_html_tags(text)
        if not clean_text:
            return None

        # 3. Detect Language
        try:
            lang = detect(clean_text)
        except:
            lang = 'ar'
            
        tts_lang = 'en' if lang == 'en' else 'ar'
        
        # 4. Generate Audio
        # slow=False for faster, more natural response
        tts = gTTS(text=clean_text, lang=tts_lang, slow=False)
        
        # 5. Save with Unique ID (Critical for Multi-user)
        filename = f"response_{uuid.uuid4().hex[:8]}.mp3"
        output_path = os.path.join(Config.AUDIO_DIR, filename)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        tts.save(output_path)
        logger.info(f"✅ TTS Audio generated: {filename}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"❌ TTS Generation failed: {e}")
        return None