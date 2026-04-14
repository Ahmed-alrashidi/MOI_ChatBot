# =========================================================================
# File Name: core/translator.py
# Purpose: NLLB-200 Translation Engine & T-S-T Pipeline.
# Project: Absher Smart Assistant (MOI ChatBot)
# Version: 5.3.0 (Entity Protection + Dynamic Max Length)
#
# Changelog v1.0 → v5.3.0:
#   - [FIX] max_length now uses Config.NLLB_MAX_LENGTH (1024) instead of
#           hardcoded 512, preventing truncation of long responses.
#           (Engineer Report §6B)
#   - [FIX] Arabic entity protection in translate_to_arabic(): preserves
#           Arabic-script government terms (like تمديد تأشيرة) that already
#           appear in polyglot queries, preventing NLLB from garbling them.
#           (Engineer Report §6, §2 re: polyglot code-switching)
#   - [NEW] extract_and_augment_query(): for polyglot queries, extracts
#           Arabic tokens directly and appends them to the translated query,
#           ensuring BM25/FAISS can match against Arabic chunks.
#
# Architecture:
#   - Uses facebook/nllb-200-1.3B (~2.5GB VRAM in fp16)
#   - Singleton pattern: loads model once, reuses across all requests
#   - T-S-T: Non-Arabic query → Arabic (for RAG) → Answer → User's language
#   - Arabic/English queries bypass translation entirely (zero overhead)
#   - Graceful fallback: if NLLB fails, returns original text (no crash)
#
# VRAM Budget on A100-80GB:
#   ALLaM-7B (bf16):     ~14 GB
#   BGE-M3 embeddings:    ~2 GB
#   NLLB-200-1.3B (fp16): ~2.5 GB
#   Total:                ~18.5 GB — plenty of headroom
# =========================================================================
import re
import torch
import gc
from typing import Optional, Tuple, List
from config import Config
from utils.logger import setup_logger
from utils.text_utils import extract_arabic_tokens

logger = setup_logger("Translator")

# =========================================================================
# NLLB-200 LANGUAGE CODE MAPPING
# =========================================================================
NLLB_LANG_MAP = {
    'ar': 'arb_Arab',      # Arabic (Modern Standard)
    'en': 'eng_Latn',      # English
    'ur': 'urd_Arab',      # Urdu
    'fr': 'fra_Latn',      # French
    'es': 'spa_Latn',      # Spanish
    'de': 'deu_Latn',      # German
    'ru': 'rus_Cyrl',      # Russian
    'zh': 'zho_Hans',      # Chinese (Simplified)
    'zh-cn': 'zho_Hans',   # Chinese (langdetect variant)
    'hi': 'hin_Deva',      # Hindi (sometimes detected instead of Urdu)
    'tr': 'tur_Latn',      # Turkish (common in Saudi context)
    'pt': 'por_Latn',      # Portuguese
    'id': 'ind_Latn',      # Indonesian
    'ms': 'zsm_Latn',      # Malay
    'tl': 'tgl_Latn',      # Tagalog/Filipino (large expat community)
    'bn': 'ben_Beng',      # Bengali
    'ta': 'tam_Taml',      # Tamil
    'ko': 'kor_Hang',      # Korean
    'ja': 'jpn_Jpan',      # Japanese
}

PRIMARY_LANGS = {'ar', 'en'}
RAG_LANG = 'ar'
RAG_NLLB_CODE = 'arb_Arab'

# Arabic script regex for entity detection in polyglot queries
_ARABIC_SCRIPT_RE = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+')


class NLLBTranslator:
    """
    Singleton NLLB-200 translation engine with Arabic entity protection.
    """

    _instance: Optional['NLLBTranslator'] = None
    _model = None
    _tokenizer = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is not None:
            return

        model_name = getattr(Config, 'NLLB_MODEL_NAME', 'facebook/nllb-200-1.3B')
        logger.info(f"🌐 Loading NLLB Translation Engine: {model_name}")

        try:
            import os
            _hf_transfer_was = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER")
            try:
                import importlib.util
                if importlib.util.find_spec("hf_transfer") is None:
                    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
            except Exception:
                os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=Config.MODELS_CACHE_DIR,
                token=Config.HF_TOKEN,
            )

            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                cache_dir=Config.MODELS_CACHE_DIR,
                token=Config.HF_TOKEN,
                dtype=torch.float16,
                low_cpu_mem_usage=True,
            ).to(Config.DEVICE)

            self._model.eval()

            if _hf_transfer_was is not None:
                os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = _hf_transfer_was

            if torch.cuda.is_available():
                vram_used = torch.cuda.memory_allocated() / 1e9
                logger.info(f"✅ NLLB loaded | VRAM: {vram_used:.1f} GB | Device: {Config.DEVICE}")
            else:
                logger.info(f"✅ NLLB loaded (CPU mode)")

        except Exception as e:
            logger.error(f"❌ NLLB load failed: {e}. Translation will fall back to ALLaM.")
            self._model = None
            self._tokenizer = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    @staticmethod
    def is_primary_lang(lang_code: str) -> bool:
        return lang_code in PRIMARY_LANGS

    @staticmethod
    def is_supported(lang_code: str) -> bool:
        return lang_code in NLLB_LANG_MAP

    @staticmethod
    def get_nllb_code(lang_code: str) -> Optional[str]:
        return NLLB_LANG_MAP.get(lang_code)

    def translate(self, text: str, src_lang: str, tgt_lang: str,
                  max_length: Optional[int] = None) -> str:
        """
        Translate text from src_lang to tgt_lang using NLLB-200.

        Args:
            text: Text to translate
            src_lang: Source language (2-letter code)
            tgt_lang: Target language (2-letter code)
            max_length: Max tokens (defaults to Config.NLLB_MAX_LENGTH=1024)

        Returns:
            Translated text, or original text if translation fails
        """
        if src_lang == tgt_lang:
            return text

        if not self.is_loaded:
            logger.warning(f"⚠️ NLLB not loaded, returning original text")
            return text

        src_nllb = NLLB_LANG_MAP.get(src_lang)
        tgt_nllb = NLLB_LANG_MAP.get(tgt_lang)

        if not src_nllb or not tgt_nllb:
            logger.warning(f"⚠️ Unsupported lang pair: {src_lang}→{tgt_lang}")
            return text

        # [FIX v5.3.0] Use Config.NLLB_MAX_LENGTH (1024) instead of hardcoded 512
        if max_length is None:
            max_length = getattr(Config, 'NLLB_MAX_LENGTH', 1024)

        try:
            self._tokenizer.src_lang = src_nllb

            inputs = self._tokenizer(
                text, return_tensors="pt", truncation=True, max_length=max_length
            ).to(self._model.device)

            tgt_token_id = self._tokenizer.convert_tokens_to_ids(tgt_nllb)

            with torch.inference_mode():
                outputs = self._model.generate(
                    **inputs,
                    forced_bos_token_id=tgt_token_id,
                    max_new_tokens=max_length,
                    num_beams=4,
                    length_penalty=1.0,
                    early_stopping=True,
                )

            translated = self._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

            if not translated or translated == text:
                return text

            logger.info(f"🌐 NLLB: {src_lang}→{tgt_lang} | '{text[:50]}...' → '{translated[:50]}...'")
            return translated

        except Exception as e:
            logger.warning(f"⚠️ NLLB translation failed ({src_lang}→{tgt_lang}): {e}")
            return text

    def translate_to_arabic(self, text: str, src_lang: str) -> str:
        """
        T-S-T Step 1: Translate user query to Arabic for RAG retrieval.

        [FIX v5.3.0] For polyglot queries (e.g., Urdu mixed with Arabic service names),
        extracts Arabic tokens BEFORE translation and augments the result.
        This ensures government entity names survive translation intact.

        Example:
            Input:  "تمديد تأشيرة خروج وعودة کے طریقہ کار اور فیس کیا ہیں؟"
            Step 1: Extract Arabic tokens → ["تمديد", "تاشيره", "خروج", "وعوده"]
            Step 2: NLLB translates full query → "كيف يمكنني تمديد تأشيرة الخروج..."
            Step 3: Augment with preserved tokens if missing from translation
        """
        if src_lang in PRIMARY_LANGS:
            return text

        # [FIX v5.3.0] Extract Arabic entities from polyglot text BEFORE translation
        # These are government terms that NLLB might garble during translation
        arabic_entities = extract_arabic_tokens(text)

        # Standard NLLB translation
        translated = self.translate(text, src_lang, RAG_LANG)

        # [FIX v5.3.0] Augment: if key Arabic entities were lost in translation,
        # append them to ensure BM25/FAISS retrieval can still match
        if arabic_entities and translated != text:
            from utils.text_utils import normalize_arabic
            translated_normalized = normalize_arabic(translated)

            # Check which entities survived translation
            missing_entities = []
            for entity in arabic_entities:
                # Skip very short tokens (noise like "كے", "اور")
                if len(entity) < 3:
                    continue
                # Check if entity (or a close variant) appears in translation
                if entity not in translated_normalized:
                    missing_entities.append(entity)

            # Append missing entities as context hints for retrieval
            if missing_entities:
                hint = " ".join(missing_entities)
                translated = f"{translated} ({hint})"
                logger.info(f"🔗 Entity augment: +{len(missing_entities)} rescued: [{hint}]")

        return translated

    def translate_from_arabic(self, text: str, tgt_lang: str) -> str:
        """
        T-S-T Step 3: Translate Arabic RAG answer to user's language.
        Arabic/English targets pass through unchanged.
        """
        if tgt_lang in PRIMARY_LANGS:
            return text
        return self.translate(text, RAG_LANG, tgt_lang)

    def tst_pipeline(self, query: str, src_lang: str) -> Tuple[str, str]:
        """
        Full T-S-T orchestrator.

        Returns:
            (arabic_query, src_lang) — arabic_query for RAG processing
        """
        if src_lang in PRIMARY_LANGS:
            return query, src_lang

        arabic_query = self.translate_to_arabic(query, src_lang)
        logger.info(f"🔄 T-S-T Step 1: {src_lang}→AR | '{query[:60]}' → '{arabic_query[:60]}'")
        return arabic_query, src_lang

    @classmethod
    def unload(cls):
        """Release NLLB model VRAM with explicit deletion."""
        if cls._instance and cls._instance._model:
            logger.info("🧹 Unloading NLLB translation model...")
            vram_before = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0

            if cls._instance._model is not None:
                del cls._instance._model
            if cls._instance._tokenizer is not None:
                del cls._instance._tokenizer

            cls._instance._model = None
            cls._instance._tokenizer = None
            cls._instance = None

            gc.collect()
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                vram_after = torch.cuda.memory_allocated() / 1e9
                logger.info(f"✨ NLLB unloaded | VRAM freed: {vram_before - vram_after:.1f} GB")