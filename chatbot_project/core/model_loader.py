import os
import torch
from typing import Optional, Tuple, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from config import Config
from utils.logger import setup_logger

# Initialize project logger
logger = setup_logger(__name__)

class ModelManager:
    """
    A Singleton class responsible for managing the lifecycle of heavy AI models.
    
    This class ensures that large models (LLM, Embeddings, ASR) are loaded into memory 
    only once to prevent resource exhaustion (OOM). It handles device allocation 
    (CPU vs GPU) and optimizes loading parameters for the specific hardware environment.
    """
    
    # Singleton instances
    _embed_model: Optional[HuggingFaceEmbeddings] = None
    _llm_model: Optional[AutoModelForCausalLM] = None
    _llm_tokenizer: Optional[AutoTokenizer] = None
    _asr_pipeline: Optional[Any] = None

    @classmethod
    def get_embedding_model(cls) -> HuggingFaceEmbeddings:
        """
        Initializes and retrieves the Embedding Model (specifically BGE-M3).
        
        This method checks if the model is already loaded. If not, it loads the model
        using LangChain's wrapper, ensuring the correct device string is passed
        since LangChain prefers explicit device strings over integer IDs.
        
        Returns:
            HuggingFaceEmbeddings: The loaded embedding model instance ready for vector operations.
        """
        if cls._embed_model is None:
            logger.info(f"🔹 Loading Embedding Model: {Config.EMBEDDING_MODEL_NAME}")
            
            # LangChain prefers 'cuda' string over integer device IDs
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
            
            cls._embed_model = HuggingFaceEmbeddings(
                model_name=Config.EMBEDDING_MODEL_NAME,
                cache_folder=Config.MODELS_DIR,
                model_kwargs={"device": device_str},
                encode_kwargs={"normalize_embeddings": True}
            )
            
        return cls._embed_model

    @classmethod
    def get_llm(cls) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Loads the ALLaM-7B Large Language Model and its tokenizer.
        
        This method utilizes 'device_map="auto"' to intelligently distribute the model 
        layers across available GPUs and CPU offload if necessary. It loads the model 
        in bfloat16 precision to optimize performance on modern GPUs (Ampere+).
        
        Returns:
            Tuple[AutoModelForCausalLM, AutoTokenizer]: A tuple containing the model and tokenizer.
        
        Raises:
            Exception: If the model fails to load due to memory or path issues.
        """
        if cls._llm_model is None:
            logger.info(f"🔹 Loading LLM: {Config.LLM_MODEL_NAME} (bf16)")
            
            # Force transformers to use the project's model directory
            os.environ["TRANSFORMERS_CACHE"] = Config.MODELS_DIR
            
            try:
                # Load Tokenizer
                cls._llm_tokenizer = AutoTokenizer.from_pretrained(
                    Config.LLM_MODEL_NAME,
                    cache_dir=Config.MODELS_DIR,
                    use_fast=False # Keeping slow tokenizer as per requirement
                )
                
                # Load Model with smart dispatch
                cls._llm_model = AutoModelForCausalLM.from_pretrained(
                    Config.LLM_MODEL_NAME,
                    torch_dtype=torch.bfloat16,     # Optimized for modern GPUs
                    device_map="auto",              # Handles VRAM offloading automatically
                    cache_dir=Config.MODELS_DIR,
                    low_cpu_mem_usage=True
                )
            except Exception as e:
                logger.error(f"❌ Failed to load LLM: {e}")
                raise e
                
        return cls._llm_model, cls._llm_tokenizer

    @classmethod
    def get_asr_pipeline(cls) -> Optional[Any]:
        """
        Initializes the Whisper ASR (Automatic Speech Recognition) pipeline.
        
        This method creates a pipeline for converting speech to text. It includes a 
        critical fix for device allocation where integer IDs are strictly required 
        by the pipeline constructor to utilize the GPU correctly.
        
        Returns:
            Pipeline: The Hugging Face ASR pipeline, or None if loading fails.
        """
        if cls._asr_pipeline is None:
            logger.info(f"🔹 Loading Whisper ASR: {Config.ASR_MODEL_NAME}")
            try:
                # Pipelines require integer device IDs (0=GPU, -1=CPU)
                device_id = 0 if torch.cuda.is_available() else -1
                
                cls._asr_pipeline = pipeline(
                    "automatic-speech-recognition",
                    model=Config.ASR_MODEL_NAME,
                    device=device_id,
                    torch_dtype=torch.float32,
                    model_kwargs={
                        "cache_dir": Config.MODELS_DIR,
                        "attn_implementation": "sdpa" # PyTorch 2.0+ optimization
                    }
                )
            except Exception as e:
                logger.error(f"❌ Failed to load Whisper: {e}")
                return None
                
        return cls._asr_pipeline