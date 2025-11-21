import os
import sys
from dotenv import load_dotenv

# Load environment variables from a .env file if present (For Local/IBEX)
load_dotenv()

class Config:
    # =====================================================
    # 1. Project Paths (Platform Agnostic & Dynamic)
    # =====================================================
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    
    # Data Directories
    DATA_ROOT = os.getenv("MOI_DATA_ROOT", os.path.join(PROJECT_ROOT, "data"))
    DATA_MASTER_DIR = os.path.join(DATA_ROOT, "Data_Master")
    DATA_CHUNKS_DIR = os.path.join(DATA_ROOT, "Data_chunks")

    # Processed & Vector DB
    PROCESSED_DIR = os.path.join(DATA_ROOT, "data_processed")
    VECTOR_DB_DIR = os.path.join(DATA_ROOT, "vector_db")
    
    # Models Directory
    MODELS_DIR = os.getenv("MOI_MODELS_DIR", os.path.join(PROJECT_ROOT, "models"))

    # Outputs Directory
    OUTPUTS_DIR = os.getenv("MOI_OUTPUTS_DIR", os.path.join(PROJECT_ROOT, "outputs"))
    LOGS_DIR = os.path.join(OUTPUTS_DIR, "logs")
    AUDIO_DIR = os.path.join(OUTPUTS_DIR, "audio")
    LOG_FILE = os.path.join(LOGS_DIR, "app.log")

    # =====================================================
    # 2. Authentication (Hugging Face Token Strategy)
    # =====================================================
    @staticmethod
    def get_hf_token():
        """
        Smartly retrieves the HF_TOKEN from various sources:
        1. Google Colab Secrets (Best for Colab)
        2. Environment Variable / .env file (Best for Local/IBEX)
        """
        # 1. Try Colab Secrets
        try:
            from google.colab import userdata
            token = userdata.get('HF_TOKEN')
            if token: return token
        except ImportError:
            pass # Not running on Colab

        # 2. Try Environment Variables
        token = os.getenv("HF_TOKEN")
        if token: return token
        
        return None

    # =====================================================
    # 3. Model Configurations
    # =====================================================
    EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
    LLM_MODEL_NAME = "ALLaM-AI/ALLaM-7B-Instruct-preview"
    ASR_MODEL_NAME = "openai/whisper-large-v3"

    # =====================================================
    # 4. RAG Engine Settings
    # =====================================================
    CHUNK_SIZE = 1500
    CHUNK_OVERLAP = 250
    RETRIEVAL_K = 12
    RERANK_TOP_K = 6
    
    MAX_NEW_TOKENS = 700
    TEMPERATURE = 0.55
    TOP_P = 0.92
    REPETITION_PENALTY = 1.10

    ENABLE_ARABIC_NORMALIZATION = True
    ENABLE_QUERY_REWRITING = True

    @classmethod
    def setup_directories(cls):
        dirs_to_create = [
            cls.DATA_ROOT, cls.PROCESSED_DIR, cls.VECTOR_DB_DIR,
            cls.OUTPUTS_DIR, cls.LOGS_DIR, cls.AUDIO_DIR, cls.MODELS_DIR
        ]
        for directory in dirs_to_create:
            try:
                os.makedirs(directory, exist_ok=True)
            except OSError as e:
                print(f"⚠️ Warning: Could not create directory {directory}: {e}")

# Initialize directories
Config.setup_directories()