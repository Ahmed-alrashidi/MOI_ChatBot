# =========================================================================
# File Name: config.py
# Project: Absher Smart Assistant (MOI ChatBot)
# Purpose: Enterprise Configuration & Single Source of Truth.
# Optimized for: NVIDIA A100-80GB (Ibex Cluster) & ALLaM Sovereign AI.
# Version: 2.1 (Telemetry & Gateway Optimized)
# =========================================================================

import os
import torch
import warnings
import logging

from dotenv import load_dotenv

# --- 1. GLOBAL NOISE REDUCTION ---
# Maintains professional terminal clarity during high-performance inference.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)

# Load environment variables (Essential for HF_TOKEN)
load_dotenv()

class Config:
    """
    Centralized configuration orchestrator.
    Manages filesystem trees, hardware acceleration, and RAG logic.
    """

    # =====================================================
    # A. PATH ARCHITECTURE (Absolute & Dynamic)
    # =====================================================
    PROJECT_ROOT: str = os.path.dirname(os.path.abspath(__file__))

    # Data & Indexing Layers
    DATA_DIR: str = os.path.join(PROJECT_ROOT, "data")
    DATA_MASTER_DIR: str = os.path.join(DATA_DIR, "Data_Master")
    DATA_CHUNK_DIR: str = os.path.join(DATA_DIR, "Data_Chunk")
    DATA_PROCESSED_DIR: str = os.path.join(DATA_DIR, "data_processed") 
    VECTOR_DB_DIR: str = os.path.join(DATA_DIR, "faiss_index")
    
    # Logs, Assets & Telemetry
    LOG_DIR: str = os.path.join(PROJECT_ROOT, "logs")
    OUTPUTS_DIR: str = os.path.join(PROJECT_ROOT, "outputs")
    AUDIO_DIR: str = os.path.join(OUTPUTS_DIR, "audio")
    TELEMETRY_DIR: str = os.path.join(OUTPUTS_DIR, "user_analytics") # [NEW]: Dedicated path for logs
    
    # Benchmarking & Ground Truth
    BENCHMARK_DIR: str = os.path.join(PROJECT_ROOT, "Benchmarks")
    BENCHMARK_RESULTS_DIR: str = os.path.join(BENCHMARK_DIR, "results")
    DEFAULT_GROUND_TRUTH: str = "ground_truth_polyglot_V2.csv"
    BENCHMARK_DATA_DIR: str = DATA_PROCESSED_DIR

    # Unified Local Model Cache (Crucial for Ibex Cluster nodes)
    MODELS_CACHE_DIR: str = os.path.join(PROJECT_ROOT, "models")
    os.environ["HF_HOME"] = MODELS_CACHE_DIR
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1" 

    # =====================================================
    # B. HARDWARE ACCELERATION STRATEGY
    # =====================================================
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Precision: Forcing bfloat16 on A100 to maximize throughput vs precision balance.
    TORCH_DTYPE = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    
    # Authentication Security
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")

    # =====================================================
    # C. SOVEREIGN MODEL STACK
    # =====================================================
    LLM_MODEL_NAME: str = "ALLaM-AI/ALLaM-7B-Instruct-preview"
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
    ASR_MODEL_NAME: str = "openai/whisper-large-v3"
    JUDGE_MODEL_NAME: str = "Qwen/Qwen2.5-32B-Instruct"

    # =====================================================
    # D. INFERENCE & RAG HYPERPARAMETERS
    # =====================================================
    RETRIEVAL_K: int = 5         # Increased slightly to provide more context for LLM reasoning.
    RRF_K: int = 60              # Hybrid smoothing factor for Reciprocal Rank Fusion.
    MAX_NEW_TOKENS: int = 1024   # Supports long-form official steps/requirements.
    TEMPERATURE: float = 0.2         # Used by rag_pipeline for generation.    # Near-zero to enforce factual consistency.
    REPETITION_PENALTY: float = 1.1
    DEBUG_MODE: bool = True      # Enabled for evaluation and public tunnel testing.

    # =====================================================
    # E. ABSHER PERSONA (Advanced System Prompt)
    # =====================================================
    SYSTEM_PROMPT_CONTENT: str = """You are the official "Absher Smart Assistant" (مساعد أبشر الذكي), a highly formal and helpful AI for the Saudi Ministry of Interior.

**OPERATIONAL PROTOCOL:**
1. **AUTHENTICITY:** Use ONLY the provided [CONTEXT] to answer. If unsure, admit lack of information politely.
2. **ATTRIBUTION:** Every response MUST begin with: "Based on the official records of Absher platform..." (or equivalent in the target language).
3. **TONE:** Professional, concise, and helpful.
4. **STYLE:** Use bullet points for steps and **bold** for fees/prices.
5. **TARGET LANGUAGE:** Always respond in {target_lang} as per the user's interface selection. """

    # =====================================================
    # F. INFRASTRUCTURE PROVISIONING
    # =====================================================
    @classmethod
    def setup_environment(cls):
        """
        Automates local filesystem readiness and enables hardware-level optimizations.
        """
        try:
            # 1. Directory Tree Provisioning
            dirs = [
                cls.LOG_DIR, cls.DATA_MASTER_DIR, cls.DATA_CHUNK_DIR, 
                cls.DATA_PROCESSED_DIR, cls.VECTOR_DB_DIR, cls.AUDIO_DIR, 
                cls.TELEMETRY_DIR, cls.BENCHMARK_RESULTS_DIR, cls.MODELS_CACHE_DIR
            ]
            for d in dirs:
                os.makedirs(d, exist_ok=True)

            # 2. Hardware-Level Optimizations (Targeting NVIDIA Ampere/A100)
            if cls.DEVICE == "cuda":
                # TF32 is critical for A100 math performance
                torch.set_float32_matmul_precision('high')
                
            # 3. Security Check
            if not cls.HF_TOKEN:
                print("⚠️ SECURITY WARNING: HF_TOKEN missing. Gated models (ALLaM) may fail to load.")
                
        except Exception as e:
            print(f"⚠️ ENVIRONMENT ERROR: Infrastructure setup failed: {e}")

# Self-initialize environment on import
Config.setup_environment()