# =========================================================================
# File Name: config.py
# Project: Absher Smart Assistant (MOI ChatBot)
# Purpose: Centralized Configuration & Single Source of Truth
#
# Technical Stack: 
# - LLM: ALLaM-7B (Saudi-Sovereign Model)
# - Hardware Optimization: NVIDIA A100 (Ampere Architecture)
# - Environment Management: Python-Dotenv
# =========================================================================

import os
import torch
import sys
import warnings
from dotenv import load_dotenv

# Initialize environment variables from a .env file for security (e.g., HF_TOKEN)
load_dotenv()

class Config:
    """
    Central control class for application-wide settings.
    Organized into functional blocks: Paths, Compute, Models, RAG, and Persona.
    """

    # =====================================================
    # A. Dynamic Path Configuration
    # =====================================================
    # Uses absolute path calculation to ensure the project runs 
    # seamlessly regardless of the deployment directory.
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

    # Data Storage Layers
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    DATA_MASTER_DIR = os.path.join(DATA_DIR, "Data_Master") # Original CSV files
    DATA_CHUNK_DIR = os.path.join(DATA_DIR, "Data_Chunk")   # Processed text fragments
    VECTOR_DB_DIR = os.path.join(DATA_DIR, "faiss_index")  # FAISS database storage
    
    # Logs & Generated Assets
    LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
    OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
    AUDIO_DIR = os.path.join(OUTPUTS_DIR, "audio") # Cached TTS files
    
    # Benchmarking & Analytics
    BENCHMARK_DIR = os.path.join(PROJECT_ROOT, "Benchmarks")
    BENCHMARK_DATA_DIR = os.path.join(BENCHMARK_DIR, "data")
    BENCHMARK_RESULTS_DIR = os.path.join(BENCHMARK_DIR, "results")

    # Local storage for model weights to avoid redundant downloads
    MODELS_CACHE_DIR = os.path.join(PROJECT_ROOT, "models")

    # =====================================================
    # B. Compute & Hardware Optimization
    # =====================================================
    # Automatic hardware detection
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Memory/Speed Optimization: 
    # Use bfloat16 for modern GPUs (A100/H100) to maintain precision while doubling speed.
    # Fallback to float16 for older GPUs.
    TORCH_DTYPE = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    
    # Retrieve API Token for Hugging Face gated models (like ALLaM)
    HF_TOKEN = os.getenv("HF_TOKEN")
    
    # Speed up model fetching using 'hf_transfer' logic
    os.environ["HF_HOME"] = MODELS_CACHE_DIR
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1" 

    # =====================================================
    # C. Model Architecture (Selection)
    # =====================================================
    LLM_MODEL_NAME = "ALLaM-AI/ALLaM-7B-Instruct-preview" # Main reasoning engine
    EMBEDDING_MODEL_NAME = "BAAI/bge-m3"                # Multilingual search engine
    ASR_MODEL_NAME = "openai/whisper-large-v3"          # Speech recognition
    JUDGE_MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"      # Benchmarking Evaluator

    # =====================================================
    # D. RAG Pipeline Hyperparameters
    # =====================================================
    RETRIEVAL_K = 4       # Number of documents to retrieve per query
    RRF_K = 60            # Reciprocal Rank Fusion constant for hybrid search
    MAX_NEW_TOKENS = 1024 # Maximum length of AI response
    TEMPERATURE = 0.01    # Extremely low value to ensure factual/consistent answers
    TOP_P = 0.9           # Nucleus sampling for creative diversity (within facts)
    REPETITION_PENALTY = 1.15
    
    # Toggle detailed logs for developer troubleshooting
    DEBUG_MODE = True 

    # =====================================================
    # E. System Persona & Guardrails
    # =====================================================
    # This template forces the AI to stay within KSA context and avoid hallucinations.
    SYSTEM_PROMPT_TEMPLATE = """<s>[INST] <<SYS>>
You are "Absher Smart Assistant" (مساعد أبشر الذكي), the official AI for the **Kingdom of Saudi Arabia (KSA)** Ministry of Interior.
Your goal is to answer the user's question based **STRICTLY** on the provided Arabic context.

**CRITICAL RULES:**
1. **SCOPE (SAUDI ARABIA ONLY):**
   - You act ONLY within the laws and procedures of **Saudi Arabia**.
   - NEVER mention other countries. If the context doesn't apply to KSA, say "I don't have this info."
   - If the context is irrelevant, strictly say: "Sorry, I do not have sufficient information about this topic in the official Saudi documents."

2. **LANGUAGE ADAPTATION:**
   - The user is speaking in **{target_lang}**. You MUST reply in **{target_lang}**.
   - Logic: Arabic Context -> {target_lang} Answer.

3. **FORMAT:**
   - Use bullet points for steps.
   - Mention fees/prices explicitly if found.

<</SYS>>

[CONTEXT (Official KSA Documents)]:
{context}

[HISTORY]:
{chat_history}

[USER QUESTION ({target_lang})]:
{question} [/INST]"""

    # =====================================================
    # F. System Integrity & Setup
    # =====================================================
    @classmethod
    def setup_environment(cls):
        """
        Initializes the filesystem and validates critical credentials.
        This ensures the application is 'Self-Healing' on a fresh deployment.
        """
        
        # 1. Directory Initialization
        dirs_to_create = [
            cls.LOG_DIR, cls.DATA_MASTER_DIR, cls.DATA_CHUNK_DIR,
            cls.VECTOR_DB_DIR, cls.AUDIO_DIR, cls.BENCHMARK_RESULTS_DIR,
            cls.BENCHMARK_DATA_DIR, cls.MODELS_CACHE_DIR
        ]
        for d in dirs_to_create:
            os.makedirs(d, exist_ok=True)

        # 2. Credential Check
        if not cls.HF_TOKEN:
            warnings.warn("⚠️ HF_TOKEN is missing! Downloads for gated models will fail.")

        # 3. High-Performance Math (A100 Specific)
        if cls.DEVICE == "cuda":
            # Enable TensorFloat-32 for faster matrix multiplication on Ampere GPUs
            torch.set_float32_matmul_precision('high') 
            
# Execute environment setup immediately upon module import
Config.setup_environment()