import os
import torch
import gc
from typing import Optional, Tuple, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from config import Config
from utils.logger import setup_logger

# Initialize project logger
logger = setup_logger(__name__)

class ModelManager:
    """
    A Singleton class optimized for NVIDIA A100 GPU infrastructure.
    Manages loading/unloading of LLM, Embeddings, and ASR models with 
    mixed-precision (bfloat16) and Flash Attention 2 support.
    """
    
    # Singleton instances
    _embed_model: Optional[HuggingFaceEmbeddings] = None
    _llm_model: Optional[AutoModelForCausalLM] = None
    _llm_tokenizer: Optional[AutoTokenizer] = None
    _asr_pipeline: Optional[Any] = None

    @classmethod
    def get_embedding_model(cls) -> HuggingFaceEmbeddings:
        """
        Initializes and retrieves the Embedding Model (BGE-M3).
        Optimized to use CUDA directly.
        """
        if cls._embed_model is None:
            # ✅ FIX: Use EMBEDDING_MODEL_NAME (matches config.py)
            logger.info(f"🔹 Loading Embedding Model: {Config.EMBEDDING_MODEL_NAME}")
            try:
                # Force CUDA if available
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model_kwargs = {"device": device}
                encode_kwargs = {"normalize_embeddings": True}
                
                cls._embed_model = HuggingFaceEmbeddings(
                    model_name=Config.EMBEDDING_MODEL_NAME, # ✅ Corrected
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs
                )
                logger.info("✅ Embedding Model Loaded.")
            except Exception as e:
                logger.error(f"❌ Failed to load Embedding Model: {e}")
                raise e
                
        return cls._embed_model

    @classmethod
    def get_llm(cls) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Loads the LLM (ALLaM-7B) optimized for A100.
        Features:
        - bfloat16 precision (Native Ampere support)
        - Flash Attention 2 (Massive speedup for inference)
        - Device Map Auto (Smart VRAM allocation)
        """
        if cls._llm_model is None:
            # ✅ FIX: Use LLM_MODEL_NAME
            logger.info(f"🚀 Loading LLM: {Config.LLM_MODEL_NAME} on A100 (bfloat16)...")
            try:
                cls._llm_tokenizer = AutoTokenizer.from_pretrained(
                    Config.LLM_MODEL_NAME, # ✅ Corrected
                    trust_remote_code=True,
                    cache_dir=Config.MODELS_DIR
                )
                
                # Smart Check for Flash Attention 2
                attn_impl = "eager" # Default fallback
                if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                    try:
                        import flash_attn
                        attn_impl = "flash_attention_2"
                        logger.info("⚡ Flash Attention 2 detected and enabled.")
                    except ImportError:
                        logger.warning("⚠️ Flash Attention 2 library not found. Using standard attention.")

                # A100 Specific Optimization
                cls._llm_model = AutoModelForCausalLM.from_pretrained(
                    Config.LLM_MODEL_NAME, # ✅ Corrected
                    device_map="auto",
                    torch_dtype=torch.bfloat16,       # Best for A100
                    attn_implementation=attn_impl,    # Dynamic selection
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    cache_dir=Config.MODELS_DIR
                )
                
                # Set evaluation mode to disable dropout
                cls._llm_model.eval()
                
                logger.info("✅ LLM Loaded Successfully on GPU.")
            except Exception as e:
                logger.error(f"❌ Failed to load LLM: {e}")
                raise e
                
        return cls._llm_model, cls._llm_tokenizer

    @classmethod
    def get_asr_pipeline(cls) -> Optional[Any]:
        """
        Initializes the Whisper ASR pipeline with half-precision (float16)
        to save memory while maintaining accuracy.
        """
        if cls._asr_pipeline is None:
            # ✅ FIX: Use ASR_MODEL_NAME
            logger.info(f"🔹 Loading Whisper ASR: {Config.ASR_MODEL_NAME}")
            try:
                device_id = 0 if torch.cuda.is_available() else -1
                
                # ✅ FIX: Removed 'use_flash_attention_2' kwarg to prevent TypeError
                # Standard loading is safe and very fast on A100 with float16
                cls._asr_pipeline = pipeline(
                    "automatic-speech-recognition",
                    model=Config.ASR_MODEL_NAME, # ✅ Corrected
                    device=device_id,
                    torch_dtype=torch.float16, # Optimized for A100/T4
                    chunk_length_s=30,
                )
                logger.info("✅ ASR Pipeline Ready.")
            except Exception as e:
                logger.error(f"❌ Failed to load Whisper: {e}")
                return None
                
        return cls._asr_pipeline

    @classmethod
    def unload_all(cls):
        """
        Aggressive memory cleanup.
        Unloads all models and forces Garbage Collection & CUDA Cache clearing.
        Useful when switching tasks or restarting the pipeline.
        """
        logger.warning("🧹 Unloading all models from memory...")
        
        if cls._llm_model is not None:
            del cls._llm_model
            cls._llm_model = None
            
        if cls._llm_tokenizer is not None:
            del cls._llm_tokenizer
            cls._llm_tokenizer = None
            
        if cls._embed_model is not None:
            del cls._embed_model
            cls._embed_model = None
            
        if cls._asr_pipeline is not None:
            del cls._asr_pipeline
            cls._asr_pipeline = None

        # Force cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("✅ Memory Cleaned.")