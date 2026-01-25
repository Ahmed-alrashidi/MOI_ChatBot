# =========================================================================
# File Name: main.py
# Purpose: Main Entry Point and System Orchestrator.
# Project: Absher Smart Assistant (MOI ChatBot)
# Version: 1.0 (Stable Release)
# Features:
# - Hardware Diagnostics: Verifies GPU readiness and A100/H100 optimizations.
# - Automated ETL: Triggered indexing if the Vector Database is missing.
# - Fault Tolerance: Robust error handling to ensure system stability.
# - Resource Management: Strict VRAM cleanup and garbage collection on exit.
# =========================================================================

import os
import sys
import torch
import warnings
import gc

# Environment Optimization: Prevent tokenizer parallelism issues in multi-threaded environments
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

# Path Configuration: Ensure the project root is accessible for relative imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from utils.logger import setup_logger
from core.model_loader import ModelManager
from core.vector_store import VectorStoreManager
from core.rag_pipeline import RAGPipeline
from data.ingestion import DataIngestor
from ui.app import create_app

# Initialize the main system logger
logger = setup_logger("Main_Launcher")

def check_hardware_status():
    """
    Performs critical hardware diagnostics to ensure the environment 
    supports high-speed inference. It specifically checks for NVIDIA 
    Ampere/Hopper (A100/H100) GPUs to enable bfloat16 optimizations.
    """
    print("\n" + "="*60)
    print("    🚀 ABSHER SMART ASSISTANT - SOVEREIGN AI SYSTEM")
    print("="*60)
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"✅ Hardware Detected: {gpu_name} ({gpu_mem:.2f} GB VRAM)")
        
        # Optimization check for Enterprise-grade GPUs
        if "A100" in gpu_name or "H100" in gpu_name:
            logger.info("🔥 OPTIMIZED: Running on NVIDIA Ampere/Hopper Architecture.")
        else:
            logger.warning(f"⚠️ Performance Warning: Running on {gpu_name}. A100 is recommended for production.")
    else:
        logger.critical("❌ CRITICAL: No GPU detected! Inference will be extremely slow (CPU Mode).")

def ensure_vector_db_ready():
    """
    Implements a self-healing data check. If the FAISS index is missing, 
    the system automatically triggers the Data Ingestion and Indexing 
    pipeline to rebuild the 'Long-Term Memory' of the AI.
    """
    index_path = os.path.join(Config.VECTOR_DB_DIR, "index.faiss")
    
    if not os.path.exists(index_path):
        logger.warning("⚠️ Vector DB not found. Starting Automatic Build Process...")
        
        # 1. Trigger Data Ingestion: Extract text from Master CSVs
        logger.info("📂 Ingesting Master Data...")
        ingestor = DataIngestor()
        documents = ingestor.load_and_process() 
        
        if not documents:
            logger.error("❌ No documents found in 'Data_Master'. Ingestion failed.")
            sys.exit(1)
            
        # 2. Load the Embedding Model for Vectorization
        logger.info("🧠 Loading Embedding Model...")
        embed_model = ModelManager.get_embedding_model()
        
        # 3. Build and Persist the FAISS Index
        logger.info("⚡ Building FAISS Index (this may take a few minutes)...")
        VectorStoreManager.load_or_build(embed_model, documents)
        logger.info("✅ Vector DB Built and Persisted Successfully.")
    else:
        logger.info("✅ Vector DB found. Skipping automated ingestion.")

def main():
    """
    Main Application Lifecycle Controller.
    Flow: Hardware Checks -> Env Setup -> Data Prep -> Pipeline Init -> UI Launch.
    """
    # 1. Preliminary Diagnostics & Environment Setup
    check_hardware_status()
    Config.setup_environment()
    
    # 2. Verify Data Readiness (Self-Healing Check)
    ensure_vector_db_ready()
    
    # Initialize component variables for safe resource cleanup in 'finally' block
    rag_system = None
    app = None

    try:
        # 3. Pipeline Initialization (The Reasoning Engine)
        logger.info("🤖 Initializing RAG Pipeline (Loading ALLaM-7B and logic modules)...")
        rag_system = RAGPipeline()
        
        # 4. Interface Construction (Gradio UI)
        logger.info("🎨 Launching Gradio Interface (Production Release v3.50.2)...")
        app = create_app(rag_system)
        
        # Branding: Load official favicon/logo
        logo_path = os.path.join(Config.PROJECT_ROOT, "ui", "assets", "moi_logo.png")
        if not os.path.exists(logo_path):
            logger.warning(f"⚠️ Favicon not found at: {logo_path}")
            logo_path = None 
            
        # Ensure audio directories are accessible to the web server
        if not os.path.exists(Config.AUDIO_DIR):
            os.makedirs(Config.AUDIO_DIR, exist_ok=True)

        # 5. Launch the Web Server
        # - server_name="0.0.0.0" allows remote network access.
        # - share=True enables a Gradio proxy for external testing.
        app.queue().launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,
            favicon_path=logo_path,
            # Critical: Allow Gradio to serve files from the audio and assets folders
            allowed_paths=[Config.AUDIO_DIR, os.path.join(Config.PROJECT_ROOT, "ui", "assets")],
            inbrowser=True
        )
        
    except KeyboardInterrupt:
        logger.info("👋 User interrupted the process (Ctrl+C). Starting safe shutdown...")
    except Exception as e:
        logger.critical(f"❌ Critical System Error: {e}", exc_info=True)
    finally:
        # --- ROBUST RESOURCE CLEANUP ---
        # This section is vital for preventing memory leaks on shared clusters (KAUST Ibex)
        logger.info("🧹 Starting Memory Cleanup and VRAM Release...")
        
        # Unload heavy models from GPU memory
        if hasattr(ModelManager, 'unload_all'):
            ModelManager.unload_all()
        
        # Clear large Python objects and trigger Garbage Collection
        if rag_system:
            del rag_system
        if app:
            del app
            
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache() # Clear CUDA cache to free up VRAM for other users
        
        logger.info("✅ System Cleanup and Shutdown Complete.")

if __name__ == "__main__":
    main()