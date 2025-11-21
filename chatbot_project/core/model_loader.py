import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from config import Config
from utils.logger import setup_logger

logger = setup_logger(__name__)

class ModelManager:
    """
    Singleton class to manage heavy AI models.
    Ensures models are loaded only once globally.
    """
    _embed_model = None
    _llm_model = None
    _llm_tokenizer = None
    _asr_pipeline = None

    @classmethod
    def get_embedding_model(cls):
        if cls._embed_model is None:
            logger.info(f"🔹 Loading Embedding Model: {Config.EMBEDDING_MODEL_NAME}")
            cls._embed_model = HuggingFaceEmbeddings(
                model_name=Config.EMBEDDING_MODEL_NAME,
                cache_folder=Config.MODELS_DIR,
                model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
        return cls._embed_model

    @classmethod
    def get_llm(cls):
        if cls._llm_model is None:
            logger.info(f"🔹 Loading LLM: {Config.LLM_MODEL_NAME} (bf16)")
            
            # Set cache dir for transformers
            os.environ["TRANSFORMERS_CACHE"] = Config.MODELS_DIR
            
            cls._llm_tokenizer = AutoTokenizer.from_pretrained(
                Config.LLM_MODEL_NAME,
                cache_dir=Config.MODELS_DIR,
                use_fast=False
            )
            
            cls._llm_model = AutoModelForCausalLM.from_pretrained(
                Config.LLM_MODEL_NAME,
                torch_dtype=torch.bfloat16, # Optimized for A100
                device_map="auto",
                cache_dir=Config.MODELS_DIR,
                low_cpu_mem_usage=True
            )
        return cls._llm_model, cls._llm_tokenizer

    @classmethod
    def get_asr_pipeline(cls):
        if cls._asr_pipeline is None:
            logger.info(f"🔹 Loading Whisper ASR: {Config.ASR_MODEL_NAME}")
            try:
                cls._asr_pipeline = pipeline(
                    "automatic-speech-recognition",
                    model=Config.ASR_MODEL_NAME,
                    device="cuda:0" if torch.cuda.is_available() else "cpu",
                    torch_dtype=torch.float32,
                    model_kwargs={
                        "cache_dir": Config.MODELS_DIR,
                        "attn_implementation": "sdpa"
                    }
                )
            except Exception as e:
                logger.error(f"❌ Failed to load Whisper: {e}")
                return None
        return cls._asr_pipeline