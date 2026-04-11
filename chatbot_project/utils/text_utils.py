# =========================================================================
# File Name: utils/text_utils.py
# Purpose: High-Performance Arabic Text Normalization & Cleaning.
# Project: Absher Smart Assistant (MOI ChatBot)
# Features:
# - Performance: Optimized character mapping via str.maketrans (C-level execution).
# - Search Accuracy: Enforces symmetric normalization for query/index exact matching.
# - Polyglot Support: Unifies Urdu/Farsi characters (Kaf/Yaa) with Standard Arabic.
# =========================================================================

import re
import string
import unicodedata

# --- 1. Pre-compiled Regular Expressions (Optimized for Speed) ---

# Diacritics (Tashkeel): Matches marks like Fatha, Damma, Kasra, Shadda, etc.
_DIACRITICS_PATTERN = re.compile(r"[\u064B-\u065F\u0670]")

# Invisible Control Characters: Zero-Width spaces that sneak into text from PDFs.
_ZERO_WIDTH_PATTERN = re.compile(r"[\u200B\u200C\u200D\u200E\u200F\uFEFF]")

# Kashida/Tatweel: The Arabic stretching character (e.g., ســــلام).
_TATWEEL_PATTERN = re.compile(r"\u0640")

# Non-Standard Arabic Punctuation marks
_ARABIC_PUNCTUATION = "،؟؛«»–—…“”"

# --- 2. Character Unification Translation Tables ---

# This mapping aligns different forms of Alef, Ta Marbuta, Alif Maqsura, 
# and Non-Arabic script variants to a single standard character. 
# This is vital for accurate BM25 sparse retrieval.
_NORM_MAP = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", # Normalize all Alef variants to plain Alef
    "ة": "ه",                               # Normalize Ta Marbuta to Haa
    "ى": "ي",                               # Normalize Alif Maqsura to Yaa
    "ی": "ي",                               # Convert Farsi/Urdu Yaa to Arabic Yaa (Expats support)
    "ک": "ك",                               # Convert Farsi/Urdu Kaf to Arabic Kaf
})

# Maps standard ASCII punctuation and Arabic punctuation to a SPACE.
# Using a space rather than an empty string prevents words from merging (e.g., "Word1,Word2" -> "Word1 Word2").
_PUNCT_MAP = str.maketrans(
    string.punctuation + _ARABIC_PUNCTUATION, 
    ' ' * (len(string.punctuation) + len(_ARABIC_PUNCTUATION))
)

def remove_diacritics(text: str) -> str:
    """
    Strips all Arabic diacritics (Tashkeel) from the provided string.
    
    Args:
        text (str): The raw Arabic text.
    Returns:
        str: Text without diacritics.
    """
    if not text: 
        return ""
    return _DIACRITICS_PATTERN.sub('', text)

def normalize_arabic(text: str) -> str:
    """
    The Core Normalization Engine.
    Transforms raw text into a canonical, unified form to ensure symmetric 
    matching between user queries and stored documents.
    
    Pipeline: Lowercase -> NFKC Normalize -> Strip Noise -> Unify Characters -> Strip Punctuation.
    
    Args:
        text (str): The raw text (can be mixed Arabic/English).
    Returns:
        str: Cleaned, normalized, and unified text ready for BM25/FAISS.
    """
    if not isinstance(text, str):
        return str(text)

    # 1. Standardize English terms
    text = text.lower()

    # 2. Unicode Normalization (NFKC handles ligatures and special char representations)
    text = unicodedata.normalize('NFKC', text)

    # 3. Strip Noise (Invisible chars, kashida, and diacritics)
    text = _ZERO_WIDTH_PATTERN.sub('', text)
    text = _TATWEEL_PATTERN.sub('', text)
    text = _DIACRITICS_PATTERN.sub('', text)

    # 4. Character Unification (Uses the high-speed translation table)
    text = text.translate(_NORM_MAP)

    # 5. Fast Punctuation Removal
    text = text.translate(_PUNCT_MAP)

    # 6. Final Cleanup (Collapse multiple spaces into one and strip edges)
    text = " ".join(text.split())

    return text