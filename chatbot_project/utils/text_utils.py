import re
import unicodedata
from typing import Optional

# --- 1. Pre-compiled Regex Patterns (Performance Optimization) ---
# Compiling patterns globally prevents re-compilation on every function call.

# Arabic Diacritics (Tashkeel)
ARABIC_DIAC = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]")
# Tatweel (Kashida) - Critical for normalization (e.g., "الـــسلام" -> "السلام")
TATWEEL = re.compile(r"\u0640")
# Zero-Width Characters (Invisible artifacts from PDFs/Web)
ZERO_WIDTH = re.compile(r"[\u200B\u200C\u200D\u200E\u200F\uFEFF]")

# Normalization Helpers
ALEF_PAT = re.compile(r"[أإآٱ]")
WHITESPACE = re.compile(r"\s+")

# Soft Clean Patterns
MD_BOLD_ITALIC = re.compile(r"\*{1,2}(.*?)\*{1,2}")
MD_UNDERSCORE = re.compile(r"_+(.*?)_+")
MD_HEADER = re.compile(r"#+\s*")
CITATION = re.compile(r"\[cite[^\]]*\]")

def normalize_arabic(text: Optional[str]) -> str:
    """
    Advanced Arabic Normalization Pipeline.
    Optimized for Search Recall & Vector Embedding alignment.
    
    Operations:
    1. Unicode Normalization (NFC).
    2. Remove Zero-Width chars.
    3. Remove Diacritics (Tashkeel).
    4. Remove Tatweel (Kashida).
    5. Unify Alef (أ, إ, آ -> ا).
    6. Unify Ya/Alef Maqsura (ى -> ي).
    7. Unify Taa Marbuta (ة -> ه).
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Unicode NFC
    text = unicodedata.normalize("NFC", text)
    
    # 2. Remove Invisible Characters (ZWNJ, etc.)
    text = ZERO_WIDTH.sub("", text)
    
    # 3. Remove Diacritics
    text = ARABIC_DIAC.sub("", text)
    
    # 4. Remove Tatweel (Kashida) -- NEW & CRITICAL
    text = TATWEEL.sub("", text)
    
    # 5. Unify Alef
    text = ALEF_PAT.sub("ا", text)
    
    # 6. Unify Ya (ى -> ي)
    text = text.replace("ى", "ي")
    
    # 7. Unify Taa Marbuta (ة -> ه)
    text = text.replace("ة", "ه")
    
    # 8. Collapse Whitespace
    text = WHITESPACE.sub(" ", text)
    
    return text.strip()

def soft_clean(text: Optional[str]) -> str:
    """
    Cleans text from formatting artifacts (Markdown, Citations, etc.).
    Uses pre-compiled patterns for speed.
    """
    if not isinstance(text, str):
        return ""
    
    # Markdown Cleanup
    text = MD_BOLD_ITALIC.sub(r"\1", text)
    text = MD_UNDERSCORE.sub(r"\1", text)
    text = MD_HEADER.sub("", text)
    
    # Remove Citations
    text = CITATION.sub("", text)
    
    # Collapse Whitespace
    text = WHITESPACE.sub(" ", text)
    
    return text.strip()

def is_arabic(text: Optional[str]) -> bool:
    """
    Heuristic check: Returns True if text contains Arabic characters.
    """
    if not isinstance(text, str):
        return False
    return bool(re.search(r'[\u0600-\u06FF]', text))

def looks_english(text: Optional[str]) -> bool:
    """
    Heuristic check: Returns True if text contains basic Latin characters.
    """
    if not isinstance(text, str):
        return False
    return bool(re.search(r"[A-Za-z]", text))