import os
from typing import Optional
from gtts import gTTS
from langdetect import detect
from config import Config
from utils.logger import setup_logger

# Initialize module logger
logger = setup_logger(__name__)

def generate_speech(text: str) -> Optional[str]:
    """
    Converts the provided text into an audio file using Google Text-to-Speech API.
    
    Features:
    - Automatic Language Detection (Switches accent between Arabic and English).
    - Error Handling (Falls back gracefully if API fails).
    - Directory Management (Ensures output path exists).

    Args:
        text (str): The string to be spoken.
        
    Returns:
        Optional[str]: Absolute path to the generated MP3 file, or None if failed.
    """
    # 1. Validation: Don't process empty strings
    if not text or not text.strip():
        logger.warning("⚠️ TTS called with empty text. Skipping.")
        return None
        
    try:
        # 2. Language Detection Logic
        # We try to detect the language to match the voice accent (English vs Arabic).
        try:
            lang = detect(text)
        except Exception:
            # Fallback to Arabic if detection fails (e.g., text is just numbers)
            lang = 'ar'
            
        # gTTS mapping: Strictly use 'ar' or 'en'. 
        # Any other language (like 'ur' or 'fr') defaults to 'ar' for consistency in this context.
        tts_lang = 'en' if lang == 'en' else 'ar'
        
        logger.info(f"🗣️ Generating TTS audio (Language: {tts_lang})...")
        
        # 3. Generate Audio Object
        # slow=False makes the speech speed normal conversational pace
        tts = gTTS(text=text, lang=tts_lang, slow=False)
        
        # 4. Save File
        # We overwrite 'response.mp3' every time to save disk space.
        # In a multi-user web app, this should use unique UUIDs, but for a local demo, this is fine.
        output_path = os.path.join(Config.AUDIO_DIR, "response.mp3")
        
        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        tts.save(output_path)
        logger.info(f"✅ Audio saved to: {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"❌ TTS Generation Error (Check Internet Connection): {e}")
        return None