import os
import sys
import logging
import warnings
from typing import NoReturn

# 1. Suppress annoying tokenizer warnings & FutureWarnings
# This must be done before importing transformers or heavy libs
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)

from huggingface_hub import login

# Import Config first to setup paths
from config import Config
from utils.logger import setup_logger

# Initialize Main Application Logger
logger = setup_logger("MainApp")

def validate_environment() -> None:
    """
    Performs pre-flight checks to ensure the environment is ready.
    Specifically checks for the Hugging Face Token (HF_TOKEN) required for gated models.
    
    If the token is missing, it prints a colorful, instructive guide and exits.
    """
    token = Config.get_hf_token()
    
    if not token:
        # ANSI Color codes for terminal output
        RED = "\033[91m"
        RESET = "\033[0m"
        BOLD = "\033[1m"
        
        error_msg = f"""
        {RED}{BOLD}❌ CRITICAL ERROR: Hugging Face Token (HF_TOKEN) is missing!{RESET}
        
        To run this project, you need access to gated models (like ALLaM).
        
        {BOLD}👉 OPTION 1: If using Google Colab:{RESET}
           1. Click on the 'Keys' icon (🔑) on the left sidebar.
           2. Add a new secret named 'HF_TOKEN' with your key.
           3. Toggle 'Notebook access' to ON.
           
        {BOLD}👉 OPTION 2: If running Locally or on IBEX:{RESET}
           1. Create a file named '.env' in the project root.
           2. Add this line: HF_TOKEN=hf_your_token_here
           3. OR export it in terminal: export HF_TOKEN=hf_your_token_here
           
        🔗 Get your token here: https://huggingface.co/settings/tokens
        """
        print(error_msg)
        logger.critical("HF_TOKEN missing. Execution stopped.")
        sys.exit(1)
    
    # If token exists, log in to Hub to ensure access to gated repos
    try:
        login(token=token)
        logger.info("✅ Successfully logged in to Hugging Face Hub.")
    except Exception as e:
        logger.error(f"❌ HF Login failed: {e}")
        sys.exit(1)

def main() -> None:
    """
    Main Execution Flow:
    1. Authenticate environment.
    2. Load Data (ETL).
    3. Initialize/Load Vector Database.
    4. Warm up AI Models (LLM + ASR).
    5. Launch Gradio Interface.
    """
    
    # 1. Pre-flight Check (Authentication)
    validate_environment()

    # 2. Lazy imports (Import only AFTER validation to avoid errors)
    # This prevents loading heavy libraries if the token is missing, saving time.
    from data.ingestion import DataIngestor
    from core.model_loader import ModelManager
    from core.vector_store import VectorStoreManager
    from core.rag_pipeline import ProRAGChain
    from ui.app import create_app

    logger.info("🚀 Starting MOI Chatbot Application...")
    
    # =====================================================
    # 3. Data & Vector Store Setup
    # =====================================================
    embedding_model = ModelManager.get_embedding_model()
    
    # We always need raw documents for BM25 (Keyword Search), 
    # even if the Vector Store (FAISS) already exists on disk.
    logger.info("🔹 Loading documents from CSVs (Required for Hybrid Search)...")
    ingestor = DataIngestor()
    all_documents = ingestor.load_and_process()
    
    if not all_documents:
        logger.error("❌ No documents found! Please check 'data/Data_Master' and 'data/Data_chunks'. Exiting.")
        return

    # Handle Vector Store (FAISS)
    # Strategy: Check if index exists on disk. If yes, load it. If no, build it.
    index_path = os.path.join(Config.VECTOR_DB_DIR, "index.faiss")
    
    if os.path.exists(index_path):
        logger.info("🔹 Existing Vector DB found. Loading index...")
        # Load existing index without rebuilding (Faster restart)
        vector_store = VectorStoreManager.load_or_build(embedding_model)
    else:
        logger.info("🔸 No Vector DB found. Building fresh index...")
        # Build new index from documents (First run)
        vector_store = VectorStoreManager.load_or_build(embedding_model, documents=all_documents)

    # =====================================================
    # 4. Initialize RAG Engine
    # =====================================================
    logger.info("🔹 Initializing RAG Chain (Models & Logic)...")
    
    # Pre-load heavy models to GPU to avoid latency on the first user request
    # This effectively "warms up" the system.
    logger.info("   - Warming up LLM...")
    ModelManager.get_llm()
    
    logger.info("   - Warming up Whisper ASR...")
    ModelManager.get_asr_pipeline()
    
    # Initialize the Brain (The Chain)
    rag_chain = ProRAGChain(vector_store, all_documents)
    
    # =====================================================
    # 5. Launch UI
    # =====================================================
    logger.info("✅ System Ready. Launching UI...")
    
    app = create_app(rag_chain)
    
    # Launch Configuration
    # - server_name="0.0.0.0": Essential for exposing the port on IBEX/Docker/Colab
    # - share=True: Creates a temporary public link for easy sharing/testing
    app.queue().launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        share=True,
        inline=False,
        favicon_path="ui/moi_logo.png" if os.path.exists("ui/moi_logo.png") else None
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("🛑 Application stopped by user.")
    except Exception as e:
        logger.exception(f"❌ Critical Error: {e}")