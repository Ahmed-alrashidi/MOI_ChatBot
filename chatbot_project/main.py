import os
import sys
import torch
import warnings
from huggingface_hub import login

# 1. Suppress warnings for cleaner logs
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

from config import Config
from utils.logger import setup_logger
from core.model_loader import ModelManager
from core.vector_store import VectorStoreManager
from core.rag_pipeline import ProRAGChain
from data.ingestion import DataIngestor

# ✅ FIX: Correct import path for the UI app
from ui.app import create_app

# Initialize Main Application Logger
logger = setup_logger("Main_Launcher")

def check_gpu_status():
    """
    Verifies that we are actually running on the A100 GPU.
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"🚀 Hardware Check: Detected GPU: {gpu_name} ({gpu_mem:.2f} GB VRAM)")
        
        if "A100" in gpu_name:
            logger.info("✅ Excellent! System is optimized for this hardware.")
        else:
            logger.warning(f"⚠️ Performance Warning: Running on {gpu_name}, not A100.")
    else:
        logger.error("❌ CRITICAL: No GPU detected! The system will be extremely slow.")

def main():
    logger.info("==================================================")
    logger.info("   🏛️  MOI SMART ASSISTANT - SYSTEM STARTUP  🏛️   ")
    logger.info("==================================================")

    # 1. Environment & Auth Check
    token = Config.get_hf_token() # Updated to use the getter method safely
    if not token:
        logger.error("❌ HF_TOKEN is missing! Cannot load gated models (ALLaM).")
        return
    
    try:
        logger.info("🔐 Authenticating with Hugging Face...")
        login(token=token)
        logger.info("✅ Authentication successful.")
    except Exception as e:
        logger.error(f"❌ Authentication failed: {e}")
        return

    # 2. Hardware Check
    check_gpu_status()

    # 3. Model Warmup (Pre-load to GPU)
    logger.info("🔹 Warming up AI Models on A100...")
    ModelManager.get_embedding_model() # BGE-M3
    ModelManager.get_llm()             # ALLaM-7B (bfloat16)
    ModelManager.get_asr_pipeline()    # Whisper-Large-v3
    
    # 4. Data Loading (Crucial for Hybrid Search)
    # Even if Vector DB exists, we MUST load raw docs to initialize BM25 (Keyword Search)
    logger.info("🔹 Loading documents for Hybrid Search (BM25)...")
    ingestor = DataIngestor()
    all_documents = ingestor.load_and_process()
    
    if not all_documents:
        logger.error("❌ No documents found! Please check 'data/Data_Master' and 'data/Data_chunks'. Exiting.")
        return

    # 5. Vector Store Management
    index_path = os.path.join(Config.VECTOR_DB_DIR, "index.faiss")
    embed_model = ModelManager.get_embedding_model()

    if os.path.exists(index_path):
        logger.info("📂 Found existing Vector DB. Loading...")
        # Load existing index (Fast)
        vector_store = VectorStoreManager.load_or_build(embed_model, None)
    else:
        logger.info("⚡ Building fresh Vector Index...")
        # Build new index from documents (First run)
        vector_store = VectorStoreManager.load_or_build(embed_model, documents=all_documents)

    # 6. Initialize The Brain (RAG Chain)
    logger.info("🧠 Initializing RAG Logic (Memory & Context)...")
    # Now we pass 'all_documents' so BM25 Retriever can be initialized correctly
    rag_chain = ProRAGChain(vector_store, all_documents=all_documents) 

    # 7. Launch UI
    logger.info("🎨 Launching MOI Assistant Interface...")
    app = create_app(rag_chain)
    
    # Check for logo existence to avoid UI warnings
    logo_path = "ui/assets/moi_logo.png"
    if not os.path.exists(logo_path):
        logger.warning(f"⚠️ Logo not found at {logo_path}, UI will load without it.")
        logo_path = None

    # Launch Settings
    app.queue().launch(
        server_name="0.0.0.0",  # Expose to network
        server_port=7860, 
        share=True,             # Create public link
        favicon_path=logo_path
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("🛑 System shutting down by user.")
    except Exception as e:
        logger.critical(f"❌ Fatal Error: {e}", exc_info=True)