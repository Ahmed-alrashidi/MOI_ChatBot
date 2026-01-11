import os
import sys
from typing import Optional, List
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()

class Config:
    """
    Central Configuration Class - Optimized for NVIDIA A100 Infrastructure.
    """

    # =====================================================
    # 1. Project Paths
    # =====================================================
    PROJECT_ROOT: str = os.path.dirname(os.path.abspath(__file__))
    
    # Data Directories
    DATA_ROOT: str = os.getenv("MOI_DATA_ROOT", os.path.join(PROJECT_ROOT, "data"))
    DATA_MASTER_DIR: str = os.path.join(DATA_ROOT, "Data_Master")
    DATA_CHUNKS_DIR: str = os.path.join(DATA_ROOT, "Data_chunks")

    # Processed Data & Vector Database
    PROCESSED_DIR: str = os.path.join(DATA_ROOT, "data_processed")
    VECTOR_DB_DIR: str = os.path.join(DATA_ROOT, "vector_db")
    
    # Models Cache Directory
    MODELS_DIR: str = os.getenv("MOI_MODELS_DIR", os.path.join(PROJECT_ROOT, "models"))

    # Outputs (Logs & Audio)
    OUTPUTS_DIR: str = os.getenv("MOI_OUTPUTS_DIR", os.path.join(PROJECT_ROOT, "outputs"))
    LOGS_DIR: str = os.path.join(OUTPUTS_DIR, "logs")
    AUDIO_DIR: str = os.path.join(OUTPUTS_DIR, "audio")
    LOG_FILE: str = os.path.join(LOGS_DIR, "app.log")

    # =====================================================
    # 2. Authentication
    # =====================================================
    @staticmethod
    def get_hf_token() -> Optional[str]:
        """
        Retrieves the Hugging Face Token intelligently.
        """
        # 1. Try Colab Secrets
        try:
            from google.colab import userdata
            token = userdata.get('HF_TOKEN')
            if token: return token
        except ImportError:
            pass

        # 2. Try Environment Variables
        token = os.getenv("HF_TOKEN")
        if token: return token
        
        return None

    # Accessor for consistency
    HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")

    # =====================================================
    # 3. Model Configuration (A100 Tier)
    # =====================================================
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
    LLM_MODEL_NAME: str = "ALLaM-AI/ALLaM-7B-Instruct-preview"
    ASR_MODEL_NAME: str = "openai/whisper-large-v3"

    # =====================================================
    # 4. RAG Pipeline Hyperparameters
    # =====================================================
    CHUNK_SIZE: int = 1200
    CHUNK_OVERLAP: int = 300
    
    RETRIEVAL_K: int = 15
    RERANK_TOP_K: int = 5 
    
    MAX_NEW_TOKENS: int = 1024
    TEMPERATURE: float = 0.3
    TOP_P: float = 0.9
    REPETITION_PENALTY: float = 1.1

    # App Logic
    HISTORY_SUMMARY_THRESHOLD: int = 3
    AUDIO_RETENTION_SECONDS: int = 600

    @classmethod
    def setup_directories(cls) -> None:
        dirs_to_create: List[str] = [
            cls.DATA_ROOT, cls.PROCESSED_DIR, cls.VECTOR_DB_DIR,
            cls.OUTPUTS_DIR, cls.LOGS_DIR, cls.AUDIO_DIR, cls.MODELS_DIR
        ]
        
        for directory in dirs_to_create:
            try:
                os.makedirs(directory, exist_ok=True)
            except OSError as e:
                print(f"⚠️ Warning: Could not create directory {directory}: {e}")

# Setup on import
Config.setup_directories()