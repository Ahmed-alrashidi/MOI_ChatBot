# =========================================================================
# File Name: data/preprocessor.py
# Project: Absher Smart Assistant (MOI ChatBot)
# Architecture: Cross-Lingual Hybrid RAG (BGE-M3 + BM25 + ALLaM-7B)
#
# Affiliation: King Abdullah University of Science and Technology (KAUST)
# Team: Ahmed AlRashidi, Sultan Alshaibani, Fahad Alqahtani, 
#       Rakan Alharbi, Sultan Alotaibi, Abdulaziz Almutairi.
# Advisors: Prof. Naeemullah Khan & Dr. Salman Khan
# =========================================================================

import re
import string
from typing import str

class TextPreprocessor:
    """
    Handles Arabic text normalization and cleaning.
    Critical for ensuring query-document matching in the Retrieval phase.
    """

    @staticmethod
    def remove_diacritics(text: str) -> str:
        """
        Removes Arabic diacritics (Tashkeel).
        Example: 'أَبْشِرْ' -> 'أبشر'
        """
        arabic_diacritics = re.compile("""
                             ّ    | # Shadda
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatweel/Kashida
                         """, re.VERBOSE)
        return re.sub(arabic_diacritics, '', text)

    @staticmethod
    def normalize_arabic(text: str) -> str:
        """
        Standardizes Arabic characters to unify search space.
        
        Transformations:
        1. Normalize Alef variants (أ, إ, آ) -> ا
        2. Normalize Ta Marbuta (ة) -> ه (or vice versa, strictly consistent)
        3. Normalize Ya (ي) -> ى (Alif Maqsura) handling
        """
        if not isinstance(text, str):
            return str(text)

        # Remove diacritics first
        text = TextPreprocessor.remove_diacritics(text)

        # Normalize Alef
        text = re.sub("[إأآا]", "ا", text)
        
        # Normalize Ta Marbuta to Ha (common in search normalization)
        text = re.sub("ة", "ه", text)
        
        # Normalize Ya (Optional: depends on strictness, usually kept distinct in modern NLP)
        # text = re.sub("ى", "ي", text)

        return text

    @staticmethod
    def clean_text(text: str) -> str:
        """
        General cleaning pipeline for User Queries.
        Removes punctuation, extra spaces, and non-alphanumeric noise.
        """
        if not text:
            return ""

        # Normalize Arabic chars
        text = TextPreprocessor.normalize_arabic(text)
        
        # Remove punctuation
        # We keep alphanumeric + Arabic chars + spaces
        # Python's string.punctuation includes: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)
        
        # Remove extra whitespace (tabs, newlines, double spaces)
        text = " ".join(text.split())
        
        return text

# Helper block for testing
if __name__ == "__main__":
    sample_query = "كيف أجدد رخصة القيادة؟ (تجديد رخصه)"
    cleaned = TextPreprocessor.clean_text(sample_query)
    print(f"Original: {sample_query}")
    print(f"Cleaned : {cleaned}")