# =========================================================================
# File Name: utils/text_utils.py
# Project: Absher Smart Assistant (MOI ChatBot)
# Architecture: Cross-Lingual Hybrid RAG (BGE-M3 + BM25 + ALLaM-7B)
#
# Affiliation: King Abdullah University of Science and Technology (KAUST)
# Team: Ahmed AlRashidi, Sultan Alshaibani, Fahad Alqahtani, 
#       Rakan Alharbi, Sultan Alotaibi, Abdulaziz Almutairi.
# Advisors: Prof. Naeemullah Khan & Dr. Salman Khan
# =========================================================================

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
    1. Unicode Normalization (NFKC).
    2. Zero-width character removal.
    3. Diacritics & Tatweel removal.
    4. Letter Standardization (Alef, Ya, Taa Marbuta).
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Unicode Normalization
    text = unicodedata.normalize('NFKC', text)
    
    # 2. Remove Zero-Width Chars
    text = ZERO_WIDTH.sub("", text)
    
    # 3. Remove Diacritics
    text = ARABIC_DIAC.sub("", text)
    
    # 4. Remove Tatweel (Kashida) -- NEW & CRITICAL
    text = TATWEEL.sub("", text)
    
    # 5. Unify Alef
    text = ALEF_PAT.sub("ا", text)
    
    # 6. Unify Ya (ى -> ي)
    # Note: Search engines often treat them identically. 
    # We standardize to 'ي' to match user input habits.
    text = text.replace("ى", "ي")
    
    # 7. Unify Taa Marbuta (ة -> ه)
    # Many users type "مدرسة" as "مدرسه". We standardize to 'ه'.
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
    Heuristic check if text contains Arabic characters.
    Useful for routing logic if handling mixed languages.
    """
    if not isinstance(text, str):
        return False
    # Check for presence of Arabic unicode block
    return bool(re.search(r'[\u0600-\u06FF]', text))