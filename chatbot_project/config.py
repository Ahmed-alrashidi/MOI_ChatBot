# =========================================================================
# File Name: config.py
# Project: Absher Smart Assistant (MOI ChatBot)
# Architecture: Cross-Lingual Hybrid RAG (BGE-M3 + BM25 + ALLaM-7B)
#
# Affiliation: King Abdullah University of Science and Technology (KAUST)
# Team: Ahmed AlRashidi, Sultan Alshaibani, Fahad Alqahtani, 
#       Rakan Alharbi, Sultan Alotaibi, Abdulaziz Almutairi.
# Advisors: Prof. Naeemullah Khan & Dr. Salman Khan
# =========================================================================

import os
import torch
from dotenv import load_dotenv

# Load environment variables from .env file (if exists)
load_dotenv()

class Config:
    """
    Central Configuration Class.
    Acts as the 'Control Center' for Paths, Models, Hyperparameters, and Prompts.
    Optimized for NVIDIA A100 Infrastructure.
    """

    # =====================================================
    # 1. System & Compute Settings
    # =====================================================
    # Auto-detect best device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Use bfloat16 for A100 (Ampere) to maximize throughput, else float16
    TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Hugging Face Token (Critical for Gate-kept models like ALLaM)
    HF_TOKEN = os.getenv("HF_TOKEN")

    # =====================================================
    # 2. Project Paths & Directory Structure
    # =====================================================
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    
    # Data Layer
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    DATA_MASTER_DIR = os.path.join(DATA_DIR, "Data_Master")
    DATA_CHUNK_DIR = os.path.join(DATA_DIR, "Data_Chunk") # Directory for sparse retrieval files
    
    # Storage Layer
    VECTOR_DB_DIR = os.path.join(DATA_DIR, "faiss_index")
    
    # Outputs Layer
    LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
    AUDIO_DIR = os.path.join(PROJECT_ROOT, "outputs", "audio")

    # =====================================================
    # 3. Model Architecture (SOTA Selection)
    # =====================================================
    # LLM: Saudi-Native ALLaM-7B (Optimized for Arabic context)
    LLM_MODEL_NAME = "ALLaM-AI/ALLaM-7B-Instruct-preview"
    
    # Embedding: BGE-M3 (Best for Multilingual Dense Retrieval)
    EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
    
    # ASR: Whisper Large V3 (Best for Arabic Speech-to-Text)
    ASR_MODEL_NAME = "openai/whisper-large-v3"

    # =====================================================
    # 4. RAG Pipeline Hyperparameters
    # =====================================================
    # Retrieval Settings
    RETRIEVAL_K = 5        # Documents to retrieve from EACH source (Dense/Sparse)
    RRF_K = 60             # Reciprocal Rank Fusion constant
    
    # Generation Settings
    MAX_NEW_TOKENS = 512
    TEMPERATURE = 0.1      # Keep it low (0.1) for factual, hallucination-free responses
    TOP_P = 0.9
    REPETITION_PENALTY = 1.1

    # Text Splitting (if needed for raw text)
    CHUNK_SIZE = 700
    CHUNK_OVERLAP = 100

    # =====================================================
    # 5. System Persona (The "Soul" of the Agent)
    # =====================================================
    # Centralizing the prompt here makes it easier to tune without touching logic code.
    SYSTEM_PROMPT_TEMPLATE = """<s>[INST] <<SYS>>
أنت "مساعد أبشر الذكي"، نظام ذكاء اصطناعي سيادي تابع لوزارة الداخلية السعودية.
مهمتك هي الإجابة على استفسارات المواطنين والمقيمين بدقة متناهية بناءً على المعلومات الرسمية المقدمة فقط.

القواعد الصارمة:
1. **المصدر:** استمد إجابتك **حصرياً** من "السياق المسترجع" أدناه. لا تخترع معلومات.
2. **اللغة:** استخدم اللغة العربية الفصحى بأسلوب رسمي ومهذب.
3. **التنسيق:** إذا كانت الإجابة تتضمن خطوات أو شروط، استخدم النقاط (Bullet points).
4. **الرفض:** إذا لم تجد الإجابة في السياق، قل: "عذراً، لا تتوفر لدي معلومات كافية حول هذا الموضوع في الوثائق الرسمية الحالية."
<</SYS>>

السياق الرسمي:
{context}

سجل المحادثة:
{chat_history}

سؤال المستخدم:
{question} [/INST]"""

    # =====================================================
    # 6. Helper Methods
    # =====================================================
    @classmethod
    def ensure_directories(cls):
        """Creates necessary directories if they don't exist."""
        dirs = [cls.DATA_DIR, cls.LOG_DIR, cls.AUDIO_DIR, cls.VECTOR_DB_DIR]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

# Run setup on import to guarantee folder existence
Config.ensure_directories()