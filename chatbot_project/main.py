# =========================================================================
# File Name: main.py
# Purpose: Enterprise-Grade Entry Point & System Orchestrator.
# Project: Absher Smart Assistant (MOI ChatBot)
# Version: 5.3.0 (Concurrency Safety + VRAM Leak Fix)
#
# Changelog v5.1.2 → v5.3.0:
#   - [FIX] concurrency_count=15 → 2 to prevent VRAM OOM with 7B LLM.
#           15 simultaneous inference threads would exceed A100 capacity.
#           (Engineer Report §5B)
#   - [FIX] verify_data_integrity() now cleans up embedding model after
#           FAISS rebuild to prevent VRAM leak before main app loads.
#           (Engineer Report §5A)
# =========================================================================
import os
import sys
import torch
import gc
import warnings
import logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config
from utils.logger import setup_logger
from core.model_loader import ModelManager
from core.vector_store import VectorStoreManager
from core.rag_pipeline import RAGPipeline
from data.ingestion import DataIngestor
from ui.app import create_app
from utils.auth_manager import verify_user
logger = setup_logger("Main_Launcher")
# Bilingual login screen (Gradio 3.x renders auth_message as Markdown)
AUTH_MESSAGE = "مرحباً بك في مساعد أبشر الذكي — الرجاء تسجيل الدخول للمتابعة"

def check_hardware_readiness():
    print("\n" + "=" * 65)
    print("      \U0001f1f8\U0001f1e6  ABSHER SMART ASSISTANT - SOVEREIGN INTELLIGENCE  \U0001f1f8\U0001f1e6")
    print("=" * 65)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info(f"\u2705 Hardware Detected: {gpu_name} ({gpu_mem:.2f} GB VRAM)")
        logger.info(f"\U0001f680 Acceleration: TF32 and SDPA Optimizations ACTIVE.")
        logger.info(f"\u2696\ufe0f Precision: Enforcing {Config.TORCH_DTYPE} across all tensors.")
    else:
        logger.critical("\u274c ERROR: No GPU detected. System startup aborted.")
        sys.exit(1)
def verify_data_integrity():
    index_path = os.path.join(Config.VECTOR_DB_DIR, "index.faiss")
    master_csv = os.path.join(Config.DATA_MASTER_DIR, "MOI_Master_Knowledge.csv")
    needs_rebuild = False
    if not os.path.exists(index_path):
        logger.warning("\u26a0\ufe0f Vector DB missing. Initiating automatic construction...")
        needs_rebuild = True
    elif os.path.exists(master_csv):
        if os.path.getmtime(master_csv) > os.path.getmtime(index_path):
            logger.warning("\U0001f504 Master Data update detected. Rebuilding FAISS Index...")
            needs_rebuild = True
    if needs_rebuild:
        ingestor = DataIngestor()
        docs = ingestor.load_and_process()
        if not docs:
            logger.error("\u274c Data Corruption: Ingestion returned zero documents.")
            sys.exit(1)
        embed_model = ModelManager.get_embedding_model()
        VectorStoreManager.load_or_build(embed_model, docs, force_rebuild=True)
        logger.info("\u2705 FAISS Knowledge Base synchronized successfully.")
        # [FIX v5.3.0] Clean up embedding model used for rebuild to prevent VRAM leak.
        # Without this, the rebuild's embed_model persists alongside the main app's copy.
        del docs, ingestor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        logger.info("\u2705 Data Integrity: Knowledge Base is current.")
def main():
    check_hardware_readiness()
    Config.setup_environment()
    verify_data_integrity()
    rag_system = None
    app = None
    try:
        logger.info(f"\U0001f916 Booting AI Engine (Model: {Config.LLM_MODEL_NAME})...")
        rag_system = RAGPipeline()
        logger.info("\U0001f3a8 Compiling Absher Modern UI (Gradio Engine)...")
        app = create_app(rag_system)
        logo_path = os.path.join(Config.PROJECT_ROOT, "ui", "assets", "moi_logo.png")
        if not os.path.exists(logo_path):
            logo_path = None
        logger.info(f"\U0001f310 Deployment: Server initiating at http://0.0.0.0:7860")
        logger.info("\U0001f512 Security: Authentication Gate and Audit Logging ACTIVE.")
        # [FIX v5.3.0] Reduced from 15 to 2. Native PyTorch inference with a 7B LLM
        # cannot handle 15 concurrent threads — each needs ~16GB VRAM for KV-cache.
        # For higher concurrency, use vLLM or TGI as inference backend.
        app.queue(concurrency_count=2).launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=getattr(Config, 'DEBUG_MODE', False),
            favicon_path=logo_path,
            allowed_paths=[
                Config.AUDIO_DIR,
                Config.OUTPUTS_DIR,
                os.path.join(Config.PROJECT_ROOT, "ui", "assets")
            ],
            inbrowser=True,
            show_api=False,
            auth=verify_user,
            auth_message=AUTH_MESSAGE
        )
    except KeyboardInterrupt:
        logger.info("\U0001f44b Manual Shutdown: User interrupted the process.")
    except Exception as e:
        logger.critical(f"\U0001f4a5 Runtime System Failure: {str(e)}", exc_info=True)
    finally:
        logger.info("\U0001f9f9 Releasing hardware resources and VRAM clusters...")
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
        logger.info("\u2728 Cleanup Complete. System is OFFLINE.")
def rebuild_faiss():
    check_hardware_readiness()
    Config.setup_environment()
    logger.info("\U0001f504 Force rebuilding FAISS index...")
    import shutil
    if os.path.exists(Config.VECTOR_DB_DIR):
        shutil.rmtree(Config.VECTOR_DB_DIR)
        logger.info("\U0001f5d1\ufe0f Old FAISS index deleted.")
    ingestor = DataIngestor()
    docs = ingestor.load_and_process()
    if not docs:
        logger.error("\u274c Data Corruption: Ingestion returned zero documents.")
        return
    embed_model = ModelManager.get_embedding_model()
    VectorStoreManager.load_or_build(embed_model, docs, force_rebuild=True)
    logger.info("\u2705 FAISS index rebuilt successfully.")
    ModelManager.unload_all()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
def benchmark_menu():
    while True:
        print("\n" + "=" * 55)
        print("   \U0001f3c6  BENCHMARK SUITE")
        print("=" * 55)
        print()
        print("  1. \U0001f4ca  Model Arena (Quick - 8 questions x 4 models)")
        print("  2. \U0001f4ca  Model Arena (Full - 120 questions x 4 models)")
        print("  3. \U0001f9ea  Functional Test (KG + Context + Safety)")
        print("  4. \U0001f50d  Retrieval Accuracy Test")
        print("  5. \U0001f6e1\ufe0f   Safety & Red Teaming")
        print("  6. \U0001f525  Stress Test (Latency + Throughput)")
        print("  7. \U0001f9e0  Multi-Turn Context Test")
        print("  8. \U0001f3c6  Run ALL Tests (3-7)")
        print("  9. \U0001f519  Back to Main Menu")
        print()
        print("=" * 55)
        choice = input("  Select (1-9): ").strip()
        if choice in ("1", "2"):
            from Benchmarks.comprehensive_arena import run_comprehensive_arena
            run_comprehensive_arena(quick_test=(choice == "1"))
        elif choice == "3":
            from Benchmarks.unified_benchmark import FunctionalTester
            FunctionalTester().run_all()
        elif choice == "4":
            from Benchmarks.unified_benchmark import RetrievalBenchmark
            RetrievalBenchmark().run_test()
        elif choice == "5":
            from Benchmarks.unified_benchmark import SafetyBenchmark
            SafetyBenchmark().run_test()
        elif choice == "6":
            from Benchmarks.unified_benchmark import StressTester
            StressTester().run()
        elif choice == "7":
            from Benchmarks.unified_benchmark import run_context_benchmark
            run_context_benchmark()
        elif choice == "8":
            from Benchmarks.unified_benchmark import UnifiedBenchmark
            engine = UnifiedBenchmark()
            engine.run_all()
        elif choice == "9":
            return
        else:
            print("\u26a0\ufe0f Invalid choice.")
def launcher():
    while True:
        print("\n" + "=" * 55)
        print("   \U0001f1f8\U0001f1e6  ABSHER SMART ASSISTANT - COMMAND CENTER  \U0001f1f8\U0001f1e6")
        print("=" * 55)
        print()
        print("  1. \U0001f680  Launch Absher Chat Application")
        print("  2. \U0001f3c6  Benchmark Suite")
        print("  3. \U0001f6e1\ufe0f   Manage Users (Auth Manager)")
        print("  4. \U0001f504  Rebuild FAISS Index")
        print("  5. \U0001f6aa  Exit")
        print()
        print("=" * 55)
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
            print("\n\U0001f44b Goodbye!\n")
            break
        else:
            print("\u26a0\ufe0f Invalid choice.")
if __name__ == "__main__":
    launcher()