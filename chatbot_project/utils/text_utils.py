# =========================================================================
# File Name: utils/text_utils.py
# Purpose: High-Performance Arabic Text Normalization & Cleaning.
# Project: Absher Smart Assistant (MOI ChatBot)
# Version: 5.3.0 (Dual-Stream Normalization + Markdown Sanitizer)
#
# Changelog v5.2.0 → v5.3.0:
#   - [FIX] Added sanitize_markdown() for fair ROUGE-L scoring
#   - [FIX] Added normalize_for_dense() — light normalization for BGE-M3
#   - [FIX] Added extract_arabic_tokens() — pulls Arabic words from
#           mixed-language (polyglot) text for cross-lingual retrieval
#   - [FIX] Expanded _ARABIC_PUNCTUATION with missing Unicode marks
#   - normalize_arabic() unchanged — remains the BM25/sparse standard
#
# Features:
# - Performance: Optimized character mapping via str.maketrans (C-level).
# - Dual-Stream: normalize_arabic() for BM25, normalize_for_dense() for FAISS.
# - Polyglot: extract_arabic_tokens() handles Urdu/Chinese mixed queries.
# - Benchmark: sanitize_markdown() strips formatting for fair ROUGE scoring.
# =========================================================================

import re
import string
import unicodedata
from typing import List

# --- 1. Pre-compiled Regular Expressions (Optimized for Speed) ---

# Diacritics (Tashkeel): Fatha, Damma, Kasra, Shadda, Sukun, etc.
_DIACRITICS_PATTERN = re.compile(r"[\u064B-\u065F\u0670]")

# Invisible Control Characters: Zero-Width spaces from PDFs/web scraping.
_ZERO_WIDTH_PATTERN = re.compile(r"[\u200B\u200C\u200D\u200E\u200F\uFEFF]")

# Kashida/Tatweel: Arabic stretching character (e.g., ســــلام).
_TATWEEL_PATTERN = re.compile(r"\u0640")

# Markdown formatting: bold, headers, bullet points, numbered lists.
_MARKDOWN_PATTERN = re.compile(r"\*{1,3}|#{1,6}\s?|^[-•]\s|^\d+\.\s", re.MULTILINE)

# Arabic script range detector (covers Arabic, Urdu, Farsi characters).
_ARABIC_SCRIPT_PATTERN = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+")

# Non-Standard Arabic Punctuation marks (expanded set).
_ARABIC_PUNCTUATION = "،؟؛«»–—…""﴿﴾〈〉《》"

# --- 2. Character Unification Translation Tables ---

# BM25 / Sparse Retrieval: Aggressive unification for exact keyword matching.
# Maps variant forms to a single canonical character.
_NORM_MAP = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",  # Alef variants → plain Alef
    "ة": "ه",                                    # Ta Marbuta → Haa
    "ى": "ي",                                    # Alif Maqsura → Yaa
    "ی": "ي",                                    # Farsi/Urdu Yaa → Arabic Yaa
    "ک": "ك",                                    # Farsi/Urdu Kaf → Arabic Kaf
})

# Dense Retrieval / BGE-M3: Light unification only.
# Preserves semantic spelling differences that BGE-M3 was trained on.
# Only normalizes cross-script variants (Farsi/Urdu → Arabic), NOT Arabic→Arabic.
_DENSE_NORM_MAP = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",  # Alef variants (safe, all same letter)
    "ی": "ي",                                    # Farsi/Urdu Yaa → Arabic Yaa
    "ک": "ك",                                    # Farsi/Urdu Kaf → Arabic Kaf
    # NOTE: ة→ه and ى→ي deliberately EXCLUDED for dense retrieval.
    # BGE-M3 encodes these as semantically distinct (خدمة ≠ خدمه).
})

# Punctuation → SPACE mapping (prevents word concatenation).
_PUNCT_MAP = str.maketrans(
    string.punctuation + _ARABIC_PUNCTUATION,
    ' ' * (len(string.punctuation) + len(_ARABIC_PUNCTUATION))
)


# =========================================================================
# PUBLIC API
# =========================================================================

def remove_diacritics(text: str) -> str:
    """Strips all Arabic diacritics (Tashkeel) from text."""
    if not text:
        return ""
    return _DIACRITICS_PATTERN.sub('', text)


def normalize_arabic(text: str) -> str:
    """
    Aggressive normalization for BM25 / Sparse Retrieval.

    Transforms raw text into a canonical form for exact keyword matching.
    Pipeline: Lowercase → NFKC → Strip Noise → Unify Characters → Strip Punctuation.

    USE FOR: BM25 indexing, BM25 queries, KG service name matching.
    DO NOT USE FOR: FAISS/BGE-M3 embeddings (use normalize_for_dense instead).
    """
    if not isinstance(text, str):
        return str(text)

    text = text.lower()
    text = unicodedata.normalize('NFKC', text)
    text = _ZERO_WIDTH_PATTERN.sub('', text)
    text = _TATWEEL_PATTERN.sub('', text)
    text = _DIACRITICS_PATTERN.sub('', text)
    text = text.translate(_NORM_MAP)
    text = text.translate(_PUNCT_MAP)
    text = " ".join(text.split())

    return text


def normalize_for_dense(text: str) -> str:
    """
    Light normalization for Dense Retrieval (FAISS / BGE-M3).

    Preserves semantic spelling that BGE-M3 was trained on.
    Only normalizes cross-script variants and removes noise.

    KEY DIFFERENCE from normalize_arabic():
    - Keeps ة (Ta Marbuta) as-is — BGE-M3 distinguishes خدمة from خدمه
    - Keeps ى (Alif Maqsura) as-is — BGE-M3 distinguishes على from علي
    - Keeps punctuation — embedding models use sentence structure
    - Does NOT lowercase — BGE-M3 handles casing internally

    USE FOR: FAISS indexing, FAISS queries, embedding generation.
    """
    if not isinstance(text, str):
        return str(text)

    text = unicodedata.normalize('NFKC', text)
    text = _ZERO_WIDTH_PATTERN.sub('', text)
    text = _TATWEEL_PATTERN.sub('', text)
    text = _DIACRITICS_PATTERN.sub('', text)
    text = text.translate(_DENSE_NORM_MAP)
    text = " ".join(text.split())

    return text


def sanitize_markdown(text: str) -> str:
    """
    Strips Markdown formatting for fair metric computation (ROUGE-L, BLEU).

    Removes: **bold**, ##headers, - bullets, 1. numbered lists, \\n line breaks.
    Converts the result to a flat text string comparable to Ground Truth format.

    USE FOR: ROUGE-L scoring in benchmark (arena.py).
    DO NOT USE FOR: User-facing responses (keep formatting for readability).
    """
    if not isinstance(text, str):
        return str(text)

    # Strip Markdown bold/italic markers: *, **, ***
    text = re.sub(r'\*{1,3}', '', text)

    # Strip Markdown headers: # ## ### etc.
    text = re.sub(r'#{1,6}\s*', '', text)

    # Strip bullet points: - item, • item, * item (at line start)
    text = re.sub(r'(?m)^[\-•\*]\s+', '', text)

    # Strip numbered lists: 1. item, 2. item (at line start)
    text = re.sub(r'(?m)^\d+\.\s+', '', text)

    # Strip Markdown links: [text](url) → text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    # Strip HTML-like tags: <https://...>
    text = re.sub(r'<[^>]+>', '', text)

    # Collapse all whitespace (newlines, tabs, multiple spaces) to single space
    text = " ".join(text.split())

    return text


def extract_arabic_tokens(text: str) -> List[str]:
    """
    Extracts Arabic-script tokens from mixed-language (polyglot) text.

    Handles code-switching queries like:
    "تمديد تأشيرة خروج وعودة کے طریقہ کار اور فیس کیا ہیں؟"
    → ["تمديد", "تاشيره", "خروج", "وعوده"]

    Each extracted token is normalized with normalize_arabic() for matching.

    USE FOR: Cross-lingual service name matching in find_service(),
             KG lookup for polyglot queries, BM25 fallback for T-S-T.
    """
    if not isinstance(text, str):
        return []

    # Find all Arabic-script sequences (includes Urdu/Farsi characters)
    raw_tokens = _ARABIC_SCRIPT_PATTERN.findall(text)

    # Normalize each token and filter short ones (< 2 chars = noise)
    normalized = []
    for token in raw_tokens:
        clean = normalize_arabic(token)
        if len(clean) >= 2:
            normalized.append(clean)

    return normalized


def normalize_for_rouge(text: str) -> str:
    """
    Combined pipeline for ROUGE-L computation.

    Strips Markdown formatting FIRST, then applies standard normalization.
    This ensures that well-formatted model outputs are fairly compared
    against flat Ground Truth text.

    USE FOR: ROUGE-L computation in FairMetrics class (arena.py).
    """
    if not isinstance(text, str):
        return str(text)

    text = sanitize_markdown(text)
    text = normalize_arabic(text)

    return text