# =========================================================================
# File Name: main.py
# Project: Absher Smart Assistant (MOI ChatBot)
# Architecture: Cross-Lingual Hybrid RAG (BGE-M3 + BM25 + ALLaM-7B)
#
# Affiliation: King Abdullah University of Science and Technology (KAUST)
# Team: Ahmed AlRashidi, Sultan Alshaibani, Fahad Alqahtani, 
#       Rakan Alharbi, Sultan Alotaibi, Abdulaziz Almutairi.
# Advisors: Prof. Naeemullah Khan & Dr. Salman Khan
# =========================================================================

import os
import sys
import torch
import warnings

# Suppress minor warnings for cleaner logs
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from utils.logger import setup_logger
from core.model_loader import ModelManager
from core.vector_store import VectorStoreManager
from core.rag_pipeline import RAGPipeline
from data.ingestion import DataIngestor
from ui.app import create_app

# Initialize Main Logger
logger = setup_logger("Main_Launcher")

def check_hardware_status():
    """
    Diagnostics: Verifies GPU availability (A100 Preference).
    """
    print("\n" + "="*50)
    print("   🚀 ABSHER SMART ASSISTANT - SYSTEM DIAGNOSTICS")
    print("="*50)
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"✅ Hardware Detected: {gpu_name} ({gpu_mem:.2f} GB VRAM)")
        
        if "A100" in gpu_name:
            logger.info("🔥 OPTIMIZED: Running on NVIDIA A100 Architecture.")
        else:
            logger.warning(f"⚠️ Performance Warning: Running on {gpu_name}. A100 is recommended.")
    else:
        logger.critical("❌ CRITICAL: No GPU detected! Inference will be extremely slow.")

def ensure_vector_db_ready():
    """
    Checks if the Vector Database exists. If not, runs the Ingestion pipeline
    to build it from scratch using 'Data_Master'.
    """
    index_path = os.path.join(Config.VECTOR_DB_DIR, "index.faiss")
    
    if not os.path.exists(index_path):
        logger.warning("⚠️ Vector DB not found. Starting Automatic Build Process...")
        
        # 1. Load Data
        logger.info("📂 Ingesting Master Data...")
        ingestor = DataIngestor()
        documents = ingestor.load_master_documents()
        
        if not documents:
            logger.error("❌ No documents found in 'Data_Master'. Please run 'prepare_data.py' first.")
            sys.exit(1)
            
        # 2. Load Embedding Model
        logger.info("🧠 Loading Embedding Model...")
        embed_model = ModelManager.get_embedding_model()
        
        # 3. Build Index
        logger.info("⚡ Building FAISS Index...")
        VectorStoreManager.load_or_build(embed_model, documents)
        logger.info("✅ Vector DB Built Successfully.")
    else:
        logger.info("✅ Vector DB found. Skipping ingestion.")

def main():
    """
    Main Execution Flow:
    1. Hardware Check -> 2. Data Check -> 3. Load Models -> 4. Launch UI
    """
    # 1. System Checks
    check_hardware_status()
    Config.ensure_directories()
    
    # 2. Ensure Data Readiness
    ensure_vector_db_ready()
    
    # 3. Initialize The Brain (RAG Pipeline)
    try:
        logger.info("🤖 Initializing RAG Pipeline (This may take a moment)...")
        rag_system = RAGPipeline()
    except Exception as e:
        logger.critical(f"🔥 Failed to initialize RAG Pipeline: {e}")
        sys.exit(1)

    # 4. Launch UI
    logger.info("🎨 Launching Gradio Interface...")
    try:
        app = create_app(rag_system)
        
        # Define logo path for favicon
        logo_path = os.path.join(Config.PROJECT_ROOT, "ui", "assets", "moi_logo.png")
        if not os.path.exists(logo_path):
            logo_path = None # Fallback if logo missing
            
        # Launch Options
        app.queue().launch(
            server_name="0.0.0.0",  # Allow external connections
            server_port=7860,       # Standard Gradio port
            share=True,             # Generate public link
            favicon_path=logo_path,
            allowed_paths=[Config.AUDIO_DIR] # Allow serving generated audio
        )
    except KeyboardInterrupt:
        logger.info("👋 User interrupted. Shutting down...")
    except Exception as e:
        logger.critical(f"❌ UI Launch Error: {e}")
    finally:
        # Cleanup on exit
        ModelManager.unload_all()
        logger.info("✅ System Shutdown Complete.")

if __name__ == "__main__":
    main()