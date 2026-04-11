# =========================================================================
# File Name: core/model_loader.py
# Purpose: Expert Model Management & VRAM Optimization (A100/H100 Ready).
# Project: Absher Smart Assistant (MOI ChatBot)
# Features:
# - Unified Cache Path: Strictly uses /models directory for all AI assets.
# - SDPA Acceleration: Native PyTorch optimization for high-speed inference.
# - Singleton Persistence: Loads each model once, handles dynamic swapping.
# - Memory Flush: Robust cleanup to prevent CUDA Out-of-Memory (OOM).
# Fixed: Resolved Config.LLM_MODEL_NAME attribute error (was LL_MODEL_NAME).
# =========================================================================

import torch
import gc
import os
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

# Initialize module-specific logger
logger = setup_logger("Model_Manager")

class ModelManager:
    """
    Centralized manager for Large Language Models (LLMs), Embeddings, and ASR models.
    Implements a robust Singleton design pattern with path-awareness to prevent 
    redundant memory allocation and manage VRAM efficiently on the A100 cluster.
    """
    
    # Class-level variables holding the active singleton instances
    _embed_model_instance: Optional[HuggingFaceEmbeddings] = None
    _llm_instance: Optional[AutoModelForCausalLM] = None
    _tokenizer_instance: Optional[AutoTokenizer] = None
    _asr_pipeline_instance: Optional[Any] = None
    
    # Tracks the currently active LLM string to handle dynamic model swapping safely
    _current_llm_name: Optional[str] = None

    @staticmethod
    def _get_best_attn_impl() -> str:
        """
        Detects the optimal attention implementation for the current hardware architecture.
        Prioritizes Flash Attention 2 for Ampere/Hopper GPUs, falling back to native SDPA.
        
        Returns:
            str: The optimal attention backend ('flash_attention_2', 'sdpa', or 'eager').
        """
        if Config.DEVICE != "cuda":
            return "eager"
        
        # Check if the environment has Flash Attention 2 installed natively
        if importlib.util.find_spec("flash_attn") is not None:
            return "flash_attention_2"
        
        # Default to Scaled Dot Product Attention (SDPA), which is highly optimized in PyTorch 2.0+
        return "sdpa"

    @classmethod
    def get_embedding_model(cls) -> HuggingFaceEmbeddings:
        """
        Loads and returns the Embedding Engine (e.g., BGE-M3) using the unified local cache.
        Retrieves the singleton instance if it has already been loaded.
        
        Returns:
            HuggingFaceEmbeddings: The instantiated embedding model.
        """
        if cls._embed_model_instance is not None:
            return cls._embed_model_instance

        logger.info(f"🔹 Loading Embedding Engine: {Config.EMBEDDING_MODEL_NAME}")
        
        try:
            cls._embed_model_instance = HuggingFaceEmbeddings(
                model_name=Config.EMBEDDING_MODEL_NAME,
                cache_folder=Config.MODELS_CACHE_DIR, # Force usage of the unified /models directory
                model_kwargs={'device': Config.DEVICE, 'trust_remote_code': True},
                encode_kwargs={'normalize_embeddings': True} # Critical for Cosine Similarity
            )
            logger.info("✅ Embedding Model Synchronized and Ready.")
            return cls._embed_model_instance
        except Exception as e:
            logger.critical(f"❌ Failed to load Embedding Engine: {e}")
            raise e

    @classmethod
    def get_llm(cls, model_name: str = None) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Dynamically loads or swaps LLMs while maintaining VRAM integrity.
        Ensures the previously loaded model is completely purged before loading a new one 
        to prevent Out-of-Memory (OOM) exceptions during benchmarking.
        
        Args:
            model_name (str, optional): The specific HuggingFace repo ID. Defaults to Config.LLM_MODEL_NAME.
            
        Returns:
            Tuple[AutoModelForCausalLM, AutoTokenizer]: The loaded model and its corresponding tokenizer.
        """
        # [FIXED]: Changed LL_MODEL_NAME to LLM_MODEL_NAME to match config.py
        target_model = model_name or Config.LLM_MODEL_NAME

        # Return existing instance if the exact requested model is already active in VRAM
        if cls._llm_instance is not None and cls._current_llm_name == target_model:
            return cls._llm_instance, cls._tokenizer_instance

        # Force a targeted VRAM purge if a different LLM is currently loaded
        if cls._llm_instance is not None:
            logger.warning(f"🔄 Swapping Models: Unloading '{cls._current_llm_name}' -> Loading '{target_model}'")
            cls.unload_llm_only() # Only unload LLM to preserve Embeddings/ASR in memory

        logger.info(f"🔹 Initializing LLM: {target_model}")
        attn_impl = cls._get_best_attn_impl()
        logger.info(f"🚀 Acceleration Backend: {attn_impl} | Compute Precision: {Config.TORCH_DTYPE}")

        try:
            # 1. Initialize the Tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                target_model,
                token=Config.HF_TOKEN,
                cache_dir=Config.MODELS_CACHE_DIR,
                trust_remote_code=True,
                clean_up_tokenization_spaces=True
            )
            
            # Ensure a padding token exists to prevent generation warnings
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # 2. Load Model weights with the hardware-optimized precision (e.g., bfloat16)
            model = AutoModelForCausalLM.from_pretrained(
                target_model,
                token=Config.HF_TOKEN,
                cache_dir=Config.MODELS_CACHE_DIR,
                torch_dtype=Config.TORCH_DTYPE,  # Keep for backward compat with older transformers
                device_map="auto", 
                trust_remote_code=True,
                attn_implementation=attn_impl,
                low_cpu_mem_usage=True  # Additional memory optimization
            )

            # Update Class Singletons
            cls._llm_instance = model
            cls._tokenizer_instance = tokenizer
            cls._current_llm_name = target_model
            
            # Log VRAM usage for monitoring
            if torch.cuda.is_available():
                vram_used = torch.cuda.memory_allocated() / 1e9
                vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(f"✅ LLM '{target_model}' loaded | VRAM: {vram_used:.1f}/{vram_total:.1f} GB")
            else:
                logger.info(f"✅ LLM '{target_model}' loaded (CPU mode).")
            return model, tokenizer

        except Exception as e:
            logger.critical(f"❌ LLM Initialization Fatal Error: {e}")
            raise e

    @classmethod
    def get_asr_pipeline(cls) -> Optional[Any]:
        """
        Initializes the Whisper Automatic Speech Recognition (ASR) pipeline.
        
        Returns:
            pipeline: A HuggingFace pipeline object configured for speech-to-text.
        """
        if cls._asr_pipeline_instance is not None:
            return cls._asr_pipeline_instance

        logger.info(f"🔹 Loading ASR Engine: {Config.ASR_MODEL_NAME}")
        
        try:
            device_id = 0 if Config.DEVICE == "cuda" else -1

            cls._asr_pipeline_instance = pipeline(
                "automatic-speech-recognition",
                model=Config.ASR_MODEL_NAME,
                device=device_id,
                # Use torch_dtype to comply with modern Transformers pipeline API
                torch_dtype=torch.float16 if Config.DEVICE == "cuda" else torch.float32,
                chunk_length_s=30,
                model_kwargs={
                    "cache_dir": Config.MODELS_CACHE_DIR
                }
            )
            logger.info(f"✅ ASR Pipeline Synchronized and Ready.")
            return cls._asr_pipeline_instance
        except Exception as e:
            logger.error(f"❌ ASR Initialization Failure: {e}")
            return None

    @classmethod
    def unload_llm_only(cls):
        """
        Selectively unloads the LLM and Tokenizer to free up bulk VRAM.
        Leaves the Embedding model and ASR pipeline intact for continued retrieval tasks.
        """
        logger.warning("🧹 Purging LLM from VRAM...")
        
        # Snapshot VRAM before purge
        vram_before = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        
        cls._llm_instance = None
        cls._tokenizer_instance = None
        cls._current_llm_name = None
        
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            vram_after = torch.cuda.memory_allocated() / 1e9
            logger.info(f"✨ LLM purged | VRAM freed: {vram_before - vram_after:.1f} GB | Remaining: {vram_after:.1f} GB")
        else:
            logger.info("✨ LLM VRAM targeted reset complete.")

    @classmethod
    def unload_all(cls):
        """
        Hard reset for GPU memory. Purges all Singletons (LLM, Embeddings, ASR) 
        and fully flushes the CUDA cache.
        Essential for safely switching between massive models during Benchmarking (e.g., 7B to 32B).
        """
        logger.warning("🧹 Purging VRAM: Flushing all model instances...")
        
        vram_before = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        
        cls._llm_instance = None
        cls._tokenizer_instance = None
        cls._embed_model_instance = None
        cls._asr_pipeline_instance = None
        cls._current_llm_name = None
            
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            vram_after = torch.cuda.memory_allocated() / 1e9
            logger.info(f"✨ Full VRAM reset | Freed: {vram_before - vram_after:.1f} GB | Remaining: {vram_after:.1f} GB")
        else:
            logger.info("✨ Full VRAM hard reset complete.")