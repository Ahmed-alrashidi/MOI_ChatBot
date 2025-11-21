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
    Ensures models are loaded only once globally and handles device placement intelligently.
    """
    _embed_model = None
    _llm_model = None
    _llm_tokenizer = None
    _asr_pipeline = None

    @classmethod
    def get_embedding_model(cls):
        """
        Loads the Embedding Model (BGE-M3).
        Optimized for LangChain which prefers device strings (e.g., 'cuda').
        """
        if cls._embed_model is None:
            logger.info(f"🔹 Loading Embedding Model: {Config.EMBEDDING_MODEL_NAME}")
            
            # LangChain/SentenceTransformers prefer string identifiers for devices
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
            
            cls._embed_model = HuggingFaceEmbeddings(
                model_name=Config.EMBEDDING_MODEL_NAME,
                cache_folder=Config.MODELS_DIR,
                model_kwargs={"device": device_str},
                encode_kwargs={"normalize_embeddings": True}
            )
        return cls._embed_model

    @classmethod
    def get_llm(cls):
        """
        Loads the LLM (ALLaM-7B).
        Uses 'device_map="auto"' to handle VRAM distribution automatically across environments (Colab/IBEX/Local).
        """
        if cls._llm_model is None:
            logger.info(f"🔹 Loading LLM: {Config.LLM_MODEL_NAME} (bf16)")
            
            # Ensure transformers cache uses our project directory
            os.environ["TRANSFORMERS_CACHE"] = Config.MODELS_DIR
            
            try:
                cls._llm_tokenizer = AutoTokenizer.from_pretrained(
                    Config.LLM_MODEL_NAME,
                    cache_dir=Config.MODELS_DIR,
                    use_fast=False
                )
                
                cls._llm_model = AutoModelForCausalLM.from_pretrained(
                    Config.LLM_MODEL_NAME,
                    torch_dtype=torch.bfloat16, # Best for A100/RTX3000+, compatible with others
                    device_map="auto",          # Smart dispatch (CPU offload if VRAM is full)
                    cache_dir=Config.MODELS_DIR,
                    low_cpu_mem_usage=True
                )
            except Exception as e:
                logger.error(f"❌ Failed to load LLM: {e}")
                raise e
                
        return cls._llm_model, cls._llm_tokenizer

    @classmethod
    def get_asr_pipeline(cls):
        """
        Loads Whisper for Speech-to-Text.
        CRITICAL FIX: Uses integer device IDs (0 for GPU, -1 for CPU) to force GPU usage on pipelines.
        """
        if cls._asr_pipeline is None:
            logger.info(f"🔹 Loading Whisper ASR: {Config.ASR_MODEL_NAME}")
            try:
                # Pipelines behave better with integer device IDs
                # 0 = First GPU, -1 = CPU
                device_id = 0 if torch.cuda.is_available() else -1
                
                cls._asr_pipeline = pipeline(
                    "automatic-speech-recognition",
                    model=Config.ASR_MODEL_NAME,
                    device=device_id,  # Forces GPU usage if available
                    torch_dtype=torch.float32,
                    model_kwargs={
                        "cache_dir": Config.MODELS_DIR,
                        "attn_implementation": "sdpa" # Optimized attention for PyTorch 2.0+
                    }
                )
            except Exception as e:
                logger.error(f"❌ Failed to load Whisper: {e}")
                return None
        return cls._asr_pipeline