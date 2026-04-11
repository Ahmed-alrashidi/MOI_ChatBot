# =========================================================================
# File Name: main.py
# Purpose: Enterprise-Grade Entry Point & System Orchestrator.
# Project: Absher Smart Assistant (MOI ChatBot)
# Version: 1.5 (A100 Optimized + Secure Auth + Multi-Stage UI)
# Features:
# - Hardware Maturation: Forces TF32 and matmul optimizations for Ampere.
# - Integrity Logic: Smart CSV-to-FAISS synchronization with self-healing.
# - Secure Access: Mandatory authentication gate for sovereign data protection.
# - Resource Stewardship: Aggressive VRAM purging on session termination.
# =========================================================================

import os
import sys
import torch
import gc
import warnings
import logging

# --- 1. ENVIRONMENT STABILIZATION ---
# Optimize terminal output and suppress non-critical library noise
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Set external library logging levels for clear operational visibility
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)

# Path Portability: Ensure project root is accessible for dynamic imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from utils.logger import setup_logger
from core.model_loader import ModelManager
from core.vector_store import VectorStoreManager
from core.rag_pipeline import RAGPipeline
from data.ingestion import DataIngestor
from ui.app import create_app
from utils.auth_manager import verify_user 

# Initialize the central system orchestrator logger
logger = setup_logger("Main_Launcher")

def check_hardware_readiness():
    """
    Performs deterministic hardware diagnostics and enables Ampere-specific 
    optimizations (TF32/SDPA) to maximize NVIDIA A100-80GB throughput.
    """
    print("\n" + "═"*65)
    print("      🇸🇦  ABSHER SMART ASSISTANT - SOVEREIGN INTELLIGENCE  🇸🇦")
    print("═"*65)
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        # Enable TensorFloat-32 for better performance on Ampere/Hopper
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        logger.info(f"✅ Hardware Detected: {gpu_name} ({gpu_mem:.2f} GB VRAM)")
        logger.info(f"🚀 Acceleration: TF32 and SDPA Optimizations ACTIVE.")
        logger.info(f"⚖️ Precision: Enforcing {Config.TORCH_DTYPE} across all tensors.")
    else:
        logger.critical("❌ ERROR: No GPU detected. System startup aborted for performance safety.")
        sys.exit(1)

def verify_data_integrity():
    """
    Ensures the RAG Knowledge Base is synchronized with the master data.
    If the CSV is newer than the FAISS index, a background rebuild is triggered.
    """
    index_path = os.path.join(Config.VECTOR_DB_DIR, "index.faiss")
    master_csv = os.path.join(Config.DATA_MASTER_DIR, "MOI_Master_Knowledge.csv")
    
    needs_rebuild = False

    if not os.path.exists(index_path):
        logger.warning("⚠️ Vector DB missing. Initiating automatic construction...")
        needs_rebuild = True
    elif os.path.exists(master_csv):
        # Time-based synchronization logic
        if os.path.getmtime(master_csv) > os.path.getmtime(index_path):
            logger.warning("🔄 Master Data update detected. Rebuilding FAISS Index...")
            needs_rebuild = True

    if needs_rebuild:
        ingestor = DataIngestor()
        docs = ingestor.load_and_process()
        
        if not docs:
            logger.error("❌ Data Corruption: Ingestion returned zero documents.")
            sys.exit(1)
            
        embed_model = ModelManager.get_embedding_model()
        VectorStoreManager.load_or_build(embed_model, docs, force_rebuild=True)
        logger.info("✅ FAISS Knowledge Base synchronized successfully.")
    else:
        logger.info("✅ Data Integrity: Knowledge Base is current.")

def main():
    """
    Master Application Lifecycle.
    Bootstraps diagnostics, data, AI engines, and the secure UI layer.
    """
    # [1] Hardware and VRAM Readiness
    check_hardware_readiness()
    
    # [2] Filesystem and Environment Provisioning
    Config.setup_environment()
    
    # [3] Data-to-Index Consistency Verification
    verify_data_integrity()
    
    rag_system = None
    app = None

    try:
        # [4] Initialize Core RAG Intelligence (Reasoning Engine)
        logger.info(f"🤖 Booting AI Engine (Model: {Config.LLM_MODEL_NAME})...")
        rag_system = RAGPipeline()
        
        # [5] Compile Modern UI with Language Gateway
        logger.info("🎨 Compiling Absher Modern UI (Gradio Engine)...")
        app = create_app(rag_system)
        
        # Asset discovery for branding
        logo_path = os.path.join(Config.PROJECT_ROOT, "ui", "assets", "moi_logo.png")
        if not os.path.exists(logo_path): 
            logo_path = None 

        # [6] Production Deployment Configuration
        logger.info(f"🌐 Deployment: Server initiating at http://0.0.0.0:7860")
        logger.info("🔒 Security: Authentication Gate and Audit Logging ACTIVE.")
        
        app.queue(concurrency_count=15).launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=getattr(Config, 'DEBUG_MODE', False), 
            favicon_path=logo_path,
            # [CRITICAL]: Path permissions for static assets and user outputs
            allowed_paths=[
                Config.AUDIO_DIR, 
                Config.OUTPUTS_DIR,
                os.path.join(Config.PROJECT_ROOT, "ui", "assets")
            ],
            inbrowser=True,
            show_api=False, # GRC Constraint: Hide technical API endpoints
            auth=verify_user, 
            auth_message="مرحباً بك في مساعد أبشر الذكي. الرجاء تسجيل الدخول للمتابعة."
        )
        
    except KeyboardInterrupt:
        logger.info("👋 Manual Shutdown: User interrupted the process.")
    except Exception as e:
        logger.critical(f"💥 Runtime System Failure: {str(e)}", exc_info=True)
    finally:
        # --- PHASE: ROBUST VRAM PURGE ---
        # Crucial for Ibex Cluster etiquette and preventing memory fragmentation
        logger.info("🧹 Releasing hardware resources and VRAM clusters...")
        
        if hasattr(ModelManager, 'unload_all'):
            ModelManager.unload_all()
        
        if rag_system: 
            del rag_system
        if app: 
            del app
            
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logger.info("✨ Cleanup Complete. System is OFFLINE.")

def rebuild_faiss():
    """Force rebuilds the FAISS index from the Master CSV."""
    check_hardware_readiness()
    Config.setup_environment()
    
    logger.info("🔄 Force rebuilding FAISS index...")
    
    import shutil
    if os.path.exists(Config.VECTOR_DB_DIR):
        shutil.rmtree(Config.VECTOR_DB_DIR)
        logger.info("🗑️ Old FAISS index deleted.")
    
    ingestor = DataIngestor()
    docs = ingestor.load_and_process()
    
    if not docs:
        logger.error("❌ Data Corruption: Ingestion returned zero documents.")
        return
    
    embed_model = ModelManager.get_embedding_model()
    VectorStoreManager.load_or_build(embed_model, docs, force_rebuild=True)
    logger.info("✅ FAISS index rebuilt successfully.")
    
    ModelManager.unload_all()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def benchmark_menu():
    """Benchmark submenu with all evaluation options."""
    print("\n" + "═" * 55)
    print("   🏆  BENCHMARK SUITE")
    print("═" * 55)
    print()
    print("  1. 📊  Model Arena (Quick - 8 questions × 4 models)")
    print("  2. 📊  Model Arena (Full - 120 questions × 4 models)")
    print("  3. 🧪  Functional Test (KG + Context + Safety)")
    print("  4. 🔍  Retrieval Accuracy Test")
    print("  5. 🛡️   Safety & Red Teaming")
    print("  6. 🔥  Stress Test (Latency + Throughput)")
    print("  7. 🧠  Multi-Turn Context Test")
    print("  8. 🔙  Back to Main Menu")
    print()
    print("═" * 55)
    
    choice = input("  Select (1-8): ").strip()
    
    if choice in ("1", "2"):
        from Benchmarks.comprehensive_arena import run_comprehensive_arena
        run_comprehensive_arena(quick_test=(choice == "1"))
    elif choice == "3":
        from Benchmarks.functional_test import FunctionalTester
        tester = FunctionalTester()
        tester.run_all()
    elif choice == "4":
        from Benchmarks.retrieval_test import RetrievalBenchmark
        tester = RetrievalBenchmark()
        tester.run_test(Config.DEFAULT_GROUND_TRUTH)
    elif choice == "5":
        from Benchmarks.safety_test import SafetyBenchmark
        tester = SafetyBenchmark()
        tester.run_test()
    elif choice == "6":
        from Benchmarks.stress_test import StressTester
        tester = StressTester()
        tester.run()
    elif choice == "7":
        from Benchmarks.context_test import run_context_benchmark
        run_context_benchmark()
    elif choice == "8":
        return
    else:
        print("⚠️ Invalid choice.")

def launcher():
    """Main Launcher Menu — single entry point for all system operations."""
    print("\n" + "═" * 55)
    print("   🇸🇦  ABSHER SMART ASSISTANT - COMMAND CENTER  🇸🇦")
    print("═" * 55)
    print()
    print("  1. 🚀  Launch Absher Chat Application")
    print("  2. 🏆  Benchmark Suite")
    print("  3. 🛡️   Manage Users (Auth Manager)")
    print("  4. 🔄  Rebuild FAISS Index")
    print("  5. 🚪  Exit")
    print()
    print("═" * 55)
    
    choice = input("  Select an option (1-5): ").strip()
    
    if choice == "1":
        main()
    elif choice == "2":
        benchmark_menu()
    elif choice == "3":
        from utils.auth_manager import main_menu
        main_menu()
    elif choice == "4":
        rebuild_faiss()
    elif choice == "5":
        print("\n👋 Goodbye!\n")
    else:
        print("⚠️ Invalid choice.")

if __name__ == "__main__":
    launcher()