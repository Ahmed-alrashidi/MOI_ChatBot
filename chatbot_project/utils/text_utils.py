# =========================================================================
# File Name: utils/text_utils.py
# Purpose: Advanced Arabic Text Normalization & Cleaning.
# Project: Absher Smart Assistant (MOI ChatBot)
# Features:
# - Performance: Pre-compiled Regex patterns for high-speed batch processing.
# - Search Optimization: Normalizes Alef, Yaa, and Ta Marbuta to unify search hits.
# - Noise Reduction: Strips Diacritics (Tashkeel) and Kashida (Tatweel).
# - Robustness: Handles mixed Arabic/English punctuation and invisible chars.
# =========================================================================

import re
import string
import unicodedata

# --- Pre-compile Regex Patterns (Optimized for High Performance) ---
# Pre-compiling prevents the regex engine from re-parsing the patterns 
# during every function call, which is critical for indexing large datasets.

# 1. Diacritics (Tashkeel): Matches marks like Fatha, Damma, Kasra, etc.
_DIACRITICS_PATTERN = re.compile(r"[\u064B-\u065F\u0670]")

# 2. Tatweel (Kashida): Matches the horizontal stretching character (ـ).
_TATWEEL_PATTERN = re.compile(r"\u0640")

# 3. Zero-Width Characters: Matches invisible layout controls that often 
# sneak into text copied from PDFs or web pages.
_ZERO_WIDTH_PATTERN = re.compile(r"[\u200B\u200C\u200D\u200E\u200F\uFEFF]")

# 4. Alef variants: Standardizes all forms (أ، إ، آ، ٱ) to a plain Alif (ا).
_ALEF_PATTERN = re.compile(r"[أإآٱ]")

# 5. Ta Marbuta: Normalizes 'ة' to 'ه' to handle common spelling variations.
_TA_MARBUTA_PATTERN = re.compile(r"ة")

# 6. Alif Maqsura: Normalizes 'ى' to 'ي'.
# This is critical for search; users often search for "مستشفي" instead of "مستشفى".
_ALIF_MAQSURA_PATTERN = re.compile(r"ى")

# 7. Arabic Punctuation: Standard string.punctuation misses regional marks.
# This pattern includes Arabic commas, semicolons, and quotes.
_ARABIC_PUNCTUATION_PATTERN = re.compile(r"[،؟؛«»ـ–—…“”]")

def remove_diacritics(text: str) -> str:
    """
    Removes all Arabic diacritics (Tashkeel) from the given text.
    
    Args:
        text (str): The raw Arabic string.
        
    Returns:
        str: Cleaned string without diacritics.
    """
    if not text: return ""
    return _DIACRITICS_PATTERN.sub('', text)

def normalize_arabic(text: str) -> str:
    """
    The master normalization engine used by the RAG Pipeline for both 
    indexing documents and processing user queries.
    
    It transforms varied Arabic writing styles into a standardized 'canonical' 
    form to ensure a match is found regardless of user typing habits.

    Args:
        text (str): Input text (can be mixed Arabic/English).

    Returns:
        str: Fully normalized and cleaned text.
    """
    if not isinstance(text, str):
        return str(text)

    # 1. Lowercase English terms to ensure case-insensitive matching.
    text = text.lower()

    # 2. Unicode Normalization (NFKC): Ensures consistent character 
    # representation across different operating systems.
    text = unicodedata.normalize('NFKC', text)

    # 3. Noise Removal: Strip invisible control characters and Tatweel.
    text = _ZERO_WIDTH_PATTERN.sub('', text)
    text = _TATWEEL_PATTERN.sub('', text)

    # 4. Remove Tashkeel: Marks are usually irrelevant for semantic search.
    text = _DIACRITICS_PATTERN.sub('', text)

    # 5. Character Unification (Standardization):
    # This ensures "أحمد" and "احمد" are treated as the same word by the system.
    text = _ALEF_PATTERN.sub('ا', text)
    text = _TA_MARBUTA_PATTERN.sub('ه', text)
    text = _ALIF_MAQSURA_PATTERN.sub('ي', text)

    # 6. Punctuation Removal:
    # First, handle Arabic-specific marks.
    text = _ARABIC_PUNCTUATION_PATTERN.sub(' ', text)
    # Second, handle standard ASCII punctuation using a fast translation table.
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    text = text.translate(translator)

    # 7. White-space Collapsing:
    # Converts multiple spaces/newlines into a single space for clean indexing.
    text = " ".join(text.split())

    return text.strip()

# --- STANDALONE TESTING BLOCK ---
if __name__ == "__main__":
    # Sample cases representing common user input variations
    test_cases = [
        "كيف أجدد رخصة القيادة؟ (تجديد رخصه)",
        "الـــســلام عــلــيــكــم",          # Kashida test
        "أبشر أعمال vs Absher Individual", # Mixed language test
        "مستشفى الملك فيصل",                # Alif Maqsura test
        "يا هلا، ومرحبا!",                  # Arabic punctuation test
        "إصدار إقامة جديدة"                  # Alef variant test
    ]
    
    print("--- Normalization Results for RAG Ingestion ---")
    for t in test_cases:
        print(f"Original: '{t}'")
        print(f"Cleaned : '{normalize_arabic(t)}'")
        print("-" * 30)