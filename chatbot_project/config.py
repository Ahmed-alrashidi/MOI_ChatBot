import os

class Config:
    # =====================================================
    # 1. Project Paths (Dynamic & Relative)
    # =====================================================
    # Base Directory: /ibex/.../chatbot_project
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    
    # -----------------------------------------------------
    # Data Directory Structure (New Organization)
    # -----------------------------------------------------
    # Main Data Folder: chatbot_project/data
    DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

    # Raw Data Inputs
    # Path: chatbot_project/data/Data_Master
    DATA_MASTER_DIR = os.path.join(DATA_ROOT, "Data_Master")
    # Path: chatbot_project/data/Data_chunks
    DATA_CHUNKS_DIR = os.path.join(DATA_ROOT, "Data_chunks")

    # Processed Data & Vector DB
    # Path: chatbot_project/data/data_processed
    PROCESSED_DIR = os.path.join(DATA_ROOT, "data_processed")
    # Path: chatbot_project/data/vector_db
    VECTOR_DB_DIR = os.path.join(DATA_ROOT, "vector_db")
    
    # -----------------------------------------------------
    # Models Directory
    # -----------------------------------------------------
    # Path: chatbot_project/models
    # (Assuming you renamed '3_models' to 'models' in the root)
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

    # -----------------------------------------------------
    # Outputs Directory
    # -----------------------------------------------------
    # Path: chatbot_project/outputs (Renamed from 4_outputs)
    OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
    
    # Sub-directories for logs and audio
    LOGS_DIR = os.path.join(OUTPUTS_DIR, "logs")
    AUDIO_DIR = os.path.join(OUTPUTS_DIR, "audio")
    
    # Log File Path
    LOG_FILE = os.path.join(LOGS_DIR, "app.log")

    # =====================================================
    # 2. Model Configurations
    # =====================================================
    # HuggingFace Model IDs
    EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
    LLM_MODEL_NAME = "ALLaM-AI/ALLaM-7B-Instruct-preview"
    ASR_MODEL_NAME = "openai/whisper-large-v3"

    # =====================================================
    # 3. RAG Engine Settings
    # =====================================================
    # Text Splitting
    CHUNK_SIZE = 1500
    CHUNK_OVERLAP = 250
    
    # Retrieval Strategies
    RETRIEVAL_K = 12       # Initial candidates retrieved
    RERANK_TOP_K = 6       # Final candidates passed to LLM
    
    # Generation Hyperparameters
    MAX_NEW_TOKENS = 700
    TEMPERATURE = 0.55
    TOP_P = 0.92
    REPETITION_PENALTY = 1.10

    # Logic Flags
    ENABLE_ARABIC_NORMALIZATION = True
    ENABLE_QUERY_REWRITING = True

    @classmethod
    def setup_directories(cls):
        """Ensure all required output directories exist."""
        dirs_to_create = [
            cls.PROCESSED_DIR,
            cls.VECTOR_DB_DIR,
            cls.OUTPUTS_DIR,
            cls.LOGS_DIR,
            cls.AUDIO_DIR,
            # We ensure model dir exists too, just in case
            cls.MODELS_DIR
        ]
        for directory in dirs_to_create:
            os.makedirs(directory, exist_ok=True)
            
# Automatically create directories when config is imported
Config.setup_directories()