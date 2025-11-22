import os
from gtts import gTTS
from langdetect import detect
from config import Config

def generate_speech(text: str) -> str:
    """
    Converts text to speech using Google TTS.
    Auto-detects language (ar/en) to choose the correct accent.
    
    Args:
        text (str): The text to read.
        
    Returns:
        str: Path to the generated audio file.
    """
    if not text or not text.strip():
        return None
        
    try:
        # 1. Detect language of the response
        try:
            lang = detect(text)
        except:
            lang = 'ar'
            
        # Map to gTTS supported languages
        # gTTS supports 'ar' and 'en' natively
        tts_lang = 'ar' if lang == 'ar' else 'en'
        
        # 2. Generate Audio
        tts = gTTS(text=text, lang=tts_lang, slow=False)
        
        # 3. Save to a temp file
        # We use a fixed name so it overwrites the old one (saves space)
        output_path = os.path.join(Config.AUDIO_DIR, "response.mp3")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        tts.save(output_path)
        return output_path
        
    except Exception as e:
        print(f"❌ TTS Error: {e}")
        return None