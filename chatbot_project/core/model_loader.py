# =========================================================================
# File Name: core/model_loader.py
# Purpose: Efficient Model Loading & Memory Management (A100 Optimized)
# Features:
# - Singleton Pattern: Prevents redundant model loading and Out-of-Memory (OOM).
# - Smart Fallback: Dynamically detects Flash Attention 2 to optimize A100/H100.
# - Mixed Precision: Supports bfloat16 for Ampere GPUs and float16 for fallbacks.
# - Clean Cleanup: Force-flushes VRAM during system shutdown.
# =========================================================================

import torch
import gc
import importlib.util
from typing import Optional, Tuple, Any
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline
)
from langchain_huggingface import HuggingFaceEmbeddings
from config import Config
from utils.logger import setup_logger

# Initialize project logger linked to Config paths
logger = setup_logger("Model_Manager")

class ModelManager:
    """
    Static manager class responsible for loading and managing LLMs, Embeddings, 
    and ASR models. It uses the Singleton pattern to ensure that each model 
    is loaded into GPU memory only once.
    """
    
    # Singleton instances to keep models persistent in RAM/VRAM
    _embed_model_instance: Optional[HuggingFaceEmbeddings] = None
    _llm_instance: Optional[AutoModelForCausalLM] = None
    _tokenizer_instance: Optional[AutoTokenizer] = None
    _asr_pipeline_instance: Optional[Any] = None

    @staticmethod
    def _is_flash_attn_available():
        """
        Checks if the 'flash_attn' library is installed in the environment.
        This prevents the system from crashing if Flash Attention is requested 
        but not supported by the hardware/software stack.
        
        Returns:
            bool: True if flash_attn is available.
        """
        return importlib.util.find_spec("flash_attn") is not None

    @classmethod
    def get_embedding_model(cls) -> HuggingFaceEmbeddings:
        """
        Loads the Embedding Model (BGE-M3) into memory.
        Used primarily for generating vectors for the RAG pipeline.
        
        Returns:
            HuggingFaceEmbeddings: The persistent embedding model instance.
        """
        if cls._embed_model_instance is not None:
            return cls._embed_model_instance

        logger.info(f"🔹 Loading Embedding Model: {Config.EMBEDDING_MODEL_NAME}")
        
        try:
            # Device configuration: Move embeddings to GPU if available
            model_kwargs = {'device': Config.DEVICE, 'trust_remote_code': True}
            encode_kwargs = {'normalize_embeddings': True} 

            # Initialize with strict cache path to avoid re-downloading on Slurm/Ibex clusters
            cls._embed_model_instance = HuggingFaceEmbeddings(
                model_name=Config.EMBEDDING_MODEL_NAME,
                cache_folder=Config.MODELS_CACHE_DIR,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
                multi_process=False
            )

            logger.info("✅ Embedding Model Loaded Successfully.")
            return cls._embed_model_instance

        except Exception as e:
            logger.critical(f"❌ Failed to load Embedding Model: {e}")
            raise e

    @classmethod
    def get_llm(cls) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Loads the Large Language Model (ALLaM-7B) and its Tokenizer.
        Features automatic optimization for NVIDIA Ampere (A100) architecture.
        
        Returns:
            Tuple: (Model instance, Tokenizer instance).
        """
        # Singleton check: return existing instances if already loaded in VRAM
        if cls._llm_instance is not None and cls._tokenizer_instance is not None:
            return cls._llm_instance, cls._tokenizer_instance

        logger.info(f"🔹 Loading LLM: {Config.LLM_MODEL_NAME}...")
        logger.info(f"⚙️ Precision: {Config.TORCH_DTYPE} | Device: {Config.DEVICE}")

        try:
            # 1. Load Tokenizer with official Hugging Face Token
            tokenizer = AutoTokenizer.from_pretrained(
                Config.LLM_MODEL_NAME,
                token=Config.HF_TOKEN,
                cache_dir=Config.MODELS_CACHE_DIR,
                trust_remote_code=True,
                clean_up_tokenization_spaces=True
            )
            
            # Ensure the pad_token is set for stable generation and batching
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("🔧 Set pad_token to eos_token for stability.")

            # 2. Determine Attention Implementation (Smart Fallback)
            # Flash Attention 2 significantly speeds up inference on A100/H100 GPUs
            use_flash_attn = cls._is_flash_attn_available() and Config.DEVICE == "cuda"
            attn_impl = "flash_attention_2" if use_flash_attn else "eager"
            
            if use_flash_attn:
                logger.info("⚡ Flash Attention 2 is ENABLED (A100 Optimization Active).")
            else:
                logger.warning("⚠️ Flash Attention 2 library missing. Falling back to standard attention (Stable Mode).")

            # 3. Load Model with optimized data types (bfloat16 for A100)
            model = AutoModelForCausalLM.from_pretrained(
                Config.LLM_MODEL_NAME,
                token=Config.HF_TOKEN,
                cache_dir=Config.MODELS_CACHE_DIR,
                torch_dtype=Config.TORCH_DTYPE,
                device_map="auto", # Automatically balances layers across available GPUs
                trust_remote_code=True,
                attn_implementation=attn_impl 
            )

            cls._llm_instance = model
            cls._tokenizer_instance = tokenizer
            logger.info(f"✅ ALLaM-7B Model Loaded Successfully (Mode: {attn_impl}).")
            
            return model, tokenizer

        except Exception as e:
            logger.critical(f"❌ Failed to load LLM: {e}")
            raise e

    @classmethod
    def get_asr_pipeline(cls):
        """
        Loads the OpenAI Whisper Large-v3 model for Speech-to-Text conversion.
        Configured with long-form transcription support (chunk_length_s).
        
        Returns:
            pipeline: HuggingFace ASR Pipeline instance.
        """
        if cls._asr_pipeline_instance is not None:
            return cls._asr_pipeline_instance

        logger.info(f"🔹 Loading ASR Model: {Config.ASR_MODEL_NAME}")
        
        try:
            # Map device for HuggingFace Pipeline (0 for GPU, -1 for CPU)
            device_id = 0 if Config.DEVICE == "cuda" else -1
            
            # Flash Attention optimization for Whisper inference
            use_flash_attn = cls._is_flash_attn_available() and Config.DEVICE == "cuda"
            attn_impl = "flash_attention_2" if use_flash_attn else "eager"

            cls._asr_pipeline_instance = pipeline(
                "automatic-speech-recognition",
                model=Config.ASR_MODEL_NAME,
                device=device_id,
                # Dynamic dtype based on hardware support
                torch_dtype=torch.float16 if Config.DEVICE == "cuda" else torch.float32,
                chunk_length_s=30,
                model_kwargs={
                    "cache_dir": Config.MODELS_CACHE_DIR,
                    "attn_implementation": attn_impl
                }
            )
            logger.info(f"✅ ASR Pipeline Ready (Mode: {attn_impl}).")
            return cls._asr_pipeline_instance

        except Exception as e:
            logger.error(f"❌ Failed to load Whisper: {e}")
            return None

    @classmethod
    def unload_all(cls):
        """
        Clears the Singleton instances and force-flushes GPU VRAM.
        Critical for avoiding memory leaks during system reloads or shutdowns.
        """
        logger.warning("🧹 Unloading all models from memory...")
        
        # Explicitly delete Python references to the models
        if cls._llm_instance:
            del cls._llm_instance
            cls._llm_instance = None
            
        if cls._tokenizer_instance:
            del cls._tokenizer_instance
            cls._tokenizer_instance = None
            
        if cls._embed_model_instance:
            del cls._embed_model_instance
            cls._embed_model_instance = None
            
        if cls._asr_pipeline_instance:
            del cls._asr_pipeline_instance
            cls._asr_pipeline_instance = None
            
        # Trigger Garbage Collection and CUDA cache clearing
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("✅ Memory Cleaned.")