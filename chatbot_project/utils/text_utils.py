import re
import unicodedata

# Regex for Arabic Diacritics (Tashkeel)
ARABIC_DIAC = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]")

def normalize_arabic(text: str) -> str:
    """
    Normalizes Arabic text by removing diacritics and unifying characters.
    Example: 'إستخراج إقامة' -> 'استخراج اقامه'
    """
    if not isinstance(text, str):
        return ""
    
    # Normalize unicode (NFC)
    text = unicodedata.normalize("NFC", text)
    
    # Remove diacritics (Tashkeel)
    text = ARABIC_DIAC.sub("", text)
    
    # Unify Alef forms (أ إ آ -> ا)
    text = re.sub(r"[أإآٱ]", "ا", text)
    
    # Unify Ya/Alif Maqsura (ى -> ي)
    text = text.replace("ى", "ي")
    
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()

def soft_clean(text: str) -> str:
    """
    Removes markdown markers, special chars, and noisy formatting.
    Used for cleaning raw data before embedding.
    """
    if not isinstance(text, str):
        return ""
    
    # Remove markdown bold/italic (* or **)
    text = re.sub(r"\*{1,2}(.*?)\*{1,2}", r"\1", text)
    
    # Remove underscores
    text = re.sub(r"_+(.*?)_+", r"\1", text)
    
    # Remove hash headers (#)
    text = re.sub(r"#+\s*", "", text)
    
    # Remove citations like [cite]
    text = re.sub(r"\[cite[^\]]*\]", "", text)
    
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()

def is_arabic(text: str) -> bool:
    """Check if the text contains Arabic characters."""
    return bool(re.search(r'[\u0600-\u06FF]', text))

def looks_english(text: str) -> bool:
    """Check if the text contains English characters."""
    return bool(re.search(r"[A-Za-z]", text))