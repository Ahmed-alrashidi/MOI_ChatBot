# =========================================================================
# File Name: core/model_loader.py
# Project: Absher Smart Assistant (MOI ChatBot)
# Architecture: Cross-Lingual Hybrid RAG (BGE-M3 + BM25 + ALLaM-7B)
#
# Affiliation: King Abdullah University of Science and Technology (KAUST)
# Team: Ahmed AlRashidi, Sultan Alshaibani, Fahad Alqahtani, 
#       Rakan Alharbi, Sultan Alotaibi, Abdulaziz Almutairi.
# Advisors: Prof. Naeemullah Khan & Dr. Salman Khan
# =========================================================================

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
    Manages loading/unloading of LLM, Embeddings, and ASR models.
    """
    
    # Singleton instances to prevent reloading
    _embed_model: Optional[HuggingFaceEmbeddings] = None
    _llm_model: Optional[AutoModelForCausalLM] = None
    _llm_tokenizer: Optional[AutoTokenizer] = None
    _asr_pipeline: Optional[Any] = None

    @classmethod
    def get_embedding_model(cls) -> HuggingFaceEmbeddings:
        """
        Initializes and retrieves the Embedding Model (BGE-M3).
        Optimized to use CUDA directly for fast retrieval.
        """
        if cls._embed_model is None:
            logger.info(f"🔹 Loading Embedding Model: {Config.EMBEDDING_MODEL_NAME}")
            try:
                # Use GPU if available
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                model_kwargs = {"device": device, "trust_remote_code": True}
                # Normalize embeddings is crucial for Cosine Similarity
                encode_kwargs = {"normalize_embeddings": True} 
                
                cls._embed_model = HuggingFaceEmbeddings(
                    model_name=Config.EMBEDDING_MODEL_NAME,
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs,
                    multi_process=False
                )
                logger.info("✅ Embedding Model Loaded Successfully.")
            except Exception as e:
                logger.error(f"❌ Failed to load Embedding Model: {e}")
                raise e
                
        return cls._embed_model

    @classmethod
    def get_llm(cls) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Loads the Saudi-native ALLaM-7B model using bfloat16 for A100 optimization.
        """
        if cls._llm_model is None or cls._llm_tokenizer is None:
            logger.info(f"🔹 Loading LLM: {Config.LLM_MODEL_NAME}")
            try:
                # Load Tokenizer
                cls._llm_tokenizer = AutoTokenizer.from_pretrained(
                    Config.LLM_MODEL_NAME,
                    trust_remote_code=True
                )
                
                # Load Model with bfloat16 precision (Recommended for Ampere GPUs)
                cls._llm_model = AutoModelForCausalLM.from_pretrained(
                    Config.LLM_MODEL_NAME,
                    device_map="auto",
                    torch_dtype=torch.bfloat16, 
                    trust_remote_code=True
                )
                
                logger.info("✅ ALLaM-7B Model Loaded Successfully.")
            except Exception as e:
                logger.error(f"❌ Failed to load LLM: {e}")
                raise e

        return cls._llm_model, cls._llm_tokenizer

    @classmethod
    def get_asr_pipeline(cls):
        """
        Loads Whisper Large-v3 for Arabic Speech Recognition.
        """
        if cls._asr_pipeline is None:
            logger.info(f"🔹 Loading ASR Model: {Config.ASR_MODEL_NAME}")
            try:
                device_id = 0 if torch.cuda.is_available() else -1
                cls._asr_pipeline = pipeline(
                    "automatic-speech-recognition",
                    model=Config.ASR_MODEL_NAME,
                    device=device_id,
                    torch_dtype=torch.float16,
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
        Aggressive memory cleanup to free GPU VRAM.
        Useful when restarting the pipeline or switching tasks.
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

        # Force Garbage Collection and CUDA Cache clearing
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("✅ Memory Cleaned.")