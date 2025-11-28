import re
import unicodedata
from typing import Optional

# --- Constants & Regex Patterns ---

# Regex for Arabic Diacritics (Tashkeel)
# Matches: Fatha, Damma, Kasra, Sukun, Shadda, Tanween, etc.
ARABIC_DIAC = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]")

def normalize_arabic(text: Optional[str]) -> str:
    """
    Normalizes Arabic text to standard forms to improve retrieval recall (Search).
    
    Operations:
    1. Unicode Normalization (NFC).
    2. Remove Diacritics (Tashkeel).
    3. Unify Alef forms (أ, إ, آ -> ا).
    4. Unify Ya/Alef Maqsura (ى -> ي).
    5. Unify Taa Marbuta (ة -> ه).
    
    Args:
        text (str): Input text containing Arabic characters.
        
    Returns:
        str: Normalized text string.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Normalize Unicode characters to NFC form (standard representation)
    text = unicodedata.normalize("NFC", text)
    
    # 2. Remove Diacritics (Tashkeel) - Critical for embedding consistency
    text = ARABIC_DIAC.sub("", text)
    
    # 3. Unify Alef forms (Hamza removal)
    # Example: "الإجراءات" -> "الاجراءات"
    text = re.sub(r"[أإآٱ]", "ا", text)
    
    # 4. Unify Ya forms
    # Example: "مستشفى" -> "مستشفي" (Common typing mismatch handling)
    text = text.replace("ى", "ي")
    
    # 5. Unify Taa Marbuta
    # Example: "خدمة" -> "خدمه" (Users often omit the dots)
    text = text.replace("ة", "ه")
    
    # 6. Collapse multiple whitespaces into one
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()

def soft_clean(text: Optional[str]) -> str:
    """
    Cleans raw text from formatting artifacts before embedding.
    Removes Markdown syntax, citations, and excessive whitespace.
    
    Args:
        text (str): Raw text possibly containing Markdown or citations.
        
    Returns:
        str: Clean plain text.
    """
    if not isinstance(text, str):
        return ""
    
    # Remove Markdown Bold/Italic (*word* or **word**)
    text = re.sub(r"\*{1,2}(.*?)\*{1,2}", r"\1", text)
    
    # Remove underscores (often used for formatting)
    text = re.sub(r"_+(.*?)_+", r"\1", text)
    
    # Remove Markdown headers (# Header)
    text = re.sub(r"#+\s*", "", text)
    
    # Remove citations patterns like [1], [cite], etc.
    text = re.sub(r"\[cite[^\]]*\]", "", text)
    
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()

def is_arabic(text: Optional[str]) -> bool:
    """
    Heuristic check to determine if text contains Arabic characters.
    Used for language detection logic.
    """
    if not isinstance(text, str):
        return False
    # Check unicode range for Arabic
    return bool(re.search(r'[\u0600-\u06FF]', text))

def looks_english(text: Optional[str]) -> bool:
    """
    Heuristic check to determine if text contains English (Latin) characters.
    """
    if not isinstance(text, str):
        return False
    # Check regex for basic Latin alphabet
    return bool(re.search(r"[A-Za-z]", text))