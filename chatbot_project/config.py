import os
import sys
from typing import Optional, List
from dotenv import load_dotenv

# Load environment variables from a .env file if present (Crucial for Local/IBEX development)
load_dotenv()

class Config:
    """
    Central Configuration Class for the MOI Chatbot Project.
    
    This class acts as a single source of truth for:
    1. File Paths (Dynamic & Absolute).
    2. Model Identifiers (Hugging Face Hub IDs).
    3. RAG Hyperparameters (Chunking, Retrieval, Generation).
    4. Environment & Security Settings (Tokens).
    """

    # =====================================================
    # 1. Project Paths (Platform Agnostic & Dynamic)
    # =====================================================
    # Get the absolute path of the directory containing this file (project root)
    PROJECT_ROOT: str = os.path.dirname(os.path.abspath(__file__))
    
    # Data Directories
    # We use os.getenv to allow overriding paths on the Cluster (IBEX) if needed
    DATA_ROOT: str = os.getenv("MOI_DATA_ROOT", os.path.join(PROJECT_ROOT, "data"))
    DATA_MASTER_DIR: str = os.path.join(DATA_ROOT, "Data_Master")
    DATA_CHUNKS_DIR: str = os.path.join(DATA_ROOT, "Data_chunks")

    # Processed Data & Vector Database
    PROCESSED_DIR: str = os.path.join(DATA_ROOT, "data_processed")
    VECTOR_DB_DIR: str = os.path.join(DATA_ROOT, "vector_db")
    
    # Models Cache Directory (Important for keeping IBEX home directory clean)
    MODELS_DIR: str = os.getenv("MOI_MODELS_DIR", os.path.join(PROJECT_ROOT, "models"))

    # Outputs (Logs & Audio)
    OUTPUTS_DIR: str = os.getenv("MOI_OUTPUTS_DIR", os.path.join(PROJECT_ROOT, "outputs"))
    LOGS_DIR: str = os.path.join(OUTPUTS_DIR, "logs")
    AUDIO_DIR: str = os.path.join(OUTPUTS_DIR, "audio")
    LOG_FILE: str = os.path.join(LOGS_DIR, "app.log")

    # =====================================================
    # 2. Authentication (Hugging Face Token Strategy)
    # =====================================================
    @staticmethod
    def get_hf_token() -> Optional[str]:
        """
        Retrieves the Hugging Face Token intelligently based on the environment.
        
        Priority:
        1. Google Colab Secrets (Secure storage in Colab).
        2. Environment Variable 'HF_TOKEN' (For Local/Docker/IBEX).
        
        Returns:
            Optional[str]: The token string if found, else None.
        """
        # 1. Try Colab Secrets (Specific to Google Colab environment)
        try:
            from google.colab import userdata
            token = userdata.get('HF_TOKEN')
            if token: return token
        except ImportError:
            pass # Not running on Colab, proceed to next method

        # 2. Try Environment Variables (Standard practice)
        token = os.getenv("HF_TOKEN")
        if token: return token
        
        return None

    # =====================================================
    # 3. Model Configurations
    # =====================================================
    # Embedding: BGE-M3 is excellent for Multilingual/Arabic retrieval
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
    
    # LLM: ALLaM-7B (Saudi LLM) - The core intelligence
    LLM_MODEL_NAME: str = "ALLaM-AI/ALLaM-7B-Instruct-preview"
    
    # ASR: Whisper Large v3 - State of the art for Speech-to-Text
    ASR_MODEL_NAME: str = "openai/whisper-large-v3"

    # =====================================================
    # 4. RAG Engine Settings (Hyperparameters)
    # =====================================================
    # Chunking: Larger chunks (1500) provide more context for the LLM to understand regulations
    CHUNK_SIZE: int = 1500
    CHUNK_OVERLAP: int = 250
    
    # Retrieval: Fetch more documents (12) initially, then filter down
    RETRIEVAL_K: int = 12
    RERANK_TOP_K: int = 6 # The final number of chunks passed to the LLM
    
    # Generation parameters (Balanced for Factual Accuracy vs. Fluency)
    MAX_NEW_TOKENS: int = 700
    TEMPERATURE: float = 0.55       # Lower temperature = More deterministic/factual
    TOP_P: float = 0.92             # Nucleus sampling
    REPETITION_PENALTY: float = 1.10 # Prevents the model from looping phrases

    # Feature Flags
    ENABLE_ARABIC_NORMALIZATION: bool = True
    ENABLE_QUERY_REWRITING: bool = True

    @classmethod
    def setup_directories(cls) -> None:
        """
        Automatically creates the necessary directory structure on startup.
        This prevents 'FileNotFoundError' when saving logs or models.
        """
        dirs_to_create: List[str] = [
            cls.DATA_ROOT, cls.PROCESSED_DIR, cls.VECTOR_DB_DIR,
            cls.OUTPUTS_DIR, cls.LOGS_DIR, cls.AUDIO_DIR, cls.MODELS_DIR
        ]
        
        for directory in dirs_to_create:
            try:
                os.makedirs(directory, exist_ok=True)
            except OSError as e:
                # We use print here because logger might not be set up yet
                print(f"⚠️ Warning: Could not create directory {directory}: {e}")

# Automatically initialize directories when config is imported
Config.setup_directories()