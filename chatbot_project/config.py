# =========================================================================
# File Name: config.py
# Project: Absher Smart Assistant (MOI ChatBot)
# Purpose: Enterprise Configuration & Single Source of Truth.
# Optimized for: NVIDIA A100-80GB (Ibex Cluster) & ALLaM Sovereign AI.
# Version: 5.3.0 (RRF Recall Fix + T-S-T Prompt Collision Fix)
#
# Changelog v5.1.2 → v5.3.0:
#   - [FIX] Added FETCH_K=20 for pre-RRF retrieval. Individual retrievers
#           now fetch 20 candidates each before RRF merges and truncates to
#           RETRIEVAL_K=5. Previously both used K=5, defeating RRF's
#           statistical advantage. (Engineer Report §7B)
#   - [FIX] Split system prompt into direct vs T-S-T versions.
#           When T-S-T is active, prompt forces "أجب بالعربية دائماً"
#           to prevent ALLaM from generating broken Urdu/Chinese that
#           NLLB then double-translates into nonsense. (Engineer Report §7A)
#   - [FIX] Added NLLB_MAX_LENGTH=1024 for translation output (was 512,
#           truncating long responses). (Engineer Report §6B)
# =========================================================================
import os
import torch
import warnings
import logging
from dotenv import load_dotenv

# --- 1. GLOBAL NOISE REDUCTION ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)
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

    DATA_DIR: str = os.path.join(PROJECT_ROOT, "data")
    DATA_MASTER_DIR: str = os.path.join(DATA_DIR, "Data_Master")
    DATA_CHUNK_DIR: str = os.path.join(DATA_DIR, "Data_Chunk")
    DATA_PROCESSED_DIR: str = os.path.join(DATA_DIR, "data_processed")
    VECTOR_DB_DIR: str = os.path.join(DATA_DIR, "faiss_index")

    LOG_DIR: str = os.path.join(PROJECT_ROOT, "logs")
    OUTPUTS_DIR: str = os.path.join(PROJECT_ROOT, "outputs")
    AUDIO_DIR: str = os.path.join(OUTPUTS_DIR, "audio")
    TELEMETRY_DIR: str = os.path.join(OUTPUTS_DIR, "user_analytics")

    BENCHMARK_DIR: str = os.path.join(PROJECT_ROOT, "Benchmarks")
    BENCHMARK_RESULTS_DIR: str = os.path.join(BENCHMARK_DIR, "results")
    DEFAULT_GROUND_TRUTH: str = "ground_truth_polyglot_V2.csv"
    BENCHMARK_DATA_DIR: str = DATA_PROCESSED_DIR

    MODELS_CACHE_DIR: str = os.path.join(PROJECT_ROOT, "models")
    os.environ["HF_HOME"] = MODELS_CACHE_DIR
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    # =====================================================
    # B. HARDWARE ACCELERATION STRATEGY
    # =====================================================
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    TORCH_DTYPE = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")

    # =====================================================
    # C. SOVEREIGN MODEL STACK
    # =====================================================
    LLM_MODEL_NAME: str = "ALLaM-AI/ALLaM-7B-Instruct-preview"
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
    ASR_MODEL_NAME: str = "openai/whisper-large-v3"
    JUDGE_MODEL_NAME: str = "Qwen/Qwen2.5-32B-Instruct"

    # T-S-T Translation Engine (Translate → Search → Translate)
    NLLB_MODEL_NAME: str = "facebook/nllb-200-1.3B"
    TST_ENABLED: bool = True

    # [FIX v5.3.0] NLLB output max length — was 512, truncating long responses.
    NLLB_MAX_LENGTH: int = 1024

    # =====================================================
    # D. INFERENCE & RAG HYPERPARAMETERS
    # =====================================================

    # [FIX v5.3.0] Dual-K Retrieval Strategy for RRF.
    # OLD: RETRIEVAL_K=5 used for BOTH individual retrievers AND final output.
    #      This meant RRF merged at most 5+5=10 candidates with heavy overlap,
    #      often producing <5 unique results — defeating RRF's purpose.
    # NEW: FETCH_K=20 per retriever (40 total candidates), RRF merges all,
    #      then truncates to RETRIEVAL_K=5 best. ~4x more recall surface.
    FETCH_K: int = 20          # Per-retriever fetch count (before RRF merge)
    RETRIEVAL_K: int = 5       # Final top-K after RRF fusion
    RRF_K: int = 60            # Reciprocal Rank Fusion constant

    MAX_NEW_TOKENS: int = 1024

    # Temperature strategy (two-tier):
    TEMPERATURE: float = 0.2
    REWRITE_TEMPERATURE: float = 0.1
    REPETITION_PENALTY: float = 1.1

    DEBUG_MODE: bool = True

    # =====================================================
    # E. ABSHER PERSONA (Arabic-First System Prompt v5.3)
    # =====================================================

    # --- DIRECT PROMPT (for Arabic & English queries) ---
    # Used when user speaks AR or EN — ALLaM generates in target language directly.
    # {target_lang} is replaced at runtime with "العربية" or "الإنجليزية".
    SYSTEM_PROMPT_CONTENT: str = """أنت "مساعد أبشر الذكي"، مساعد رسمي وموثوق تابع لوزارة الداخلية في المملكة العربية السعودية.

**بروتوكول التشغيل:**
1. **المصداقية:** استخدم فقط المعلومات الموجودة في [CONTEXT] للإجابة. إذا لم تجد المعلومة، اعتذر بأدب وقل "لا تتوفر لدي معلومات رسمية حول هذا الاستفسار".
2. **المصدر:** نوّع في بداية إجاباتك. استخدم عبارات مختلفة مثل:
   - "وفقاً لمنصة أبشر..."
   - "حسب السجلات الرسمية..."
   - "تتيح منصة أبشر..."
   - "من خلال منصة أبشر..."
   - "بحسب المعلومات الرسمية..."
   لا تكرر نفس العبارة الافتتاحية في كل إجابة.
3. **الأسلوب:** رسمي، مختصر، ومفيد. استخدم النقاط المرقمة للخطوات و**الخط العريض** للرسوم والأسعار.
4. **الرسوم:** لا تخمن أي سعر أبداً. استخدم فقط الأسعار الموجودة في [المصدر الرسمي] من قاعدة المعرفة. إذا لم يكن السعر موجوداً، قل "يرجى مراجعة منصة أبشر للاطلاع على الرسوم المحدثة".
5. **لغة الرد:** أجب دائماً بلغة {target_lang}. لا تخلط بين اللغات أبداً في نفس الإجابة. لا تستخدم كلمات إنجليزية في إجابة عربية والعكس.
6. **نطاق التخصص:** تخصصك فقط خدمات وزارة الداخلية عبر منصة أبشر (الجوازات، الأحوال المدنية، المرور، الأمن العام، المديرية العامة للسجون، خدمات الوزارة). لا تذكر أبداً خدمات التعليم أو الصحة أو التجارة أو أي وزارة أخرى.
7. **الإيجاز:** لا تكرر المعلومات. أجب بأقل عدد كلمات يوصل المعلومة بوضوح."""

    # --- T-S-T PROMPT (for translated queries: FR, ES, DE, RU, ZH, UR) ---
    # [FIX v5.3.0] When T-S-T is active, NLLB handles the translation.
    # ALLaM MUST generate in Arabic ONLY. If told to generate in Urdu/Chinese,
    # it produces broken output that NLLB then double-translates into nonsense.
    #
    # CRITICAL: rag_pipeline.py must use this prompt (not SYSTEM_PROMPT_CONTENT)
    #           when the query was translated by NLLB.
    SYSTEM_PROMPT_TST: str = """أنت "مساعد أبشر الذكي"، مساعد رسمي وموثوق تابع لوزارة الداخلية في المملكة العربية السعودية.

**بروتوكول التشغيل:**
1. **المصداقية:** استخدم فقط المعلومات الموجودة في [CONTEXT] للإجابة. إذا لم تجد المعلومة، اعتذر بأدب وقل "لا تتوفر لدي معلومات رسمية حول هذا الاستفسار".
2. **المصدر:** نوّع في بداية إجاباتك. استخدم عبارات مختلفة مثل:
   - "وفقاً لمنصة أبشر..."
   - "حسب السجلات الرسمية..."
   - "تتيح منصة أبشر..."
   - "من خلال منصة أبشر..."
   - "بحسب المعلومات الرسمية..."
   لا تكرر نفس العبارة الافتتاحية في كل إجابة.
3. **الأسلوب:** رسمي، مختصر، ومفيد. استخدم النقاط المرقمة للخطوات و**الخط العريض** للرسوم والأسعار.
4. **الرسوم:** لا تخمن أي سعر أبداً. استخدم فقط الأسعار الموجودة في [المصدر الرسمي] من قاعدة المعرفة. إذا لم يكن السعر موجوداً، قل "يرجى مراجعة منصة أبشر للاطلاع على الرسوم المحدثة".
5. **لغة الرد:** أجب دائماً باللغة العربية الفصحى فقط. هذا الاستفسار مترجم تلقائياً وسيتم ترجمة إجابتك آلياً إلى لغة المستخدم. لا تحاول الرد بأي لغة غير العربية.
6. **نطاق التخصص:** تخصصك فقط خدمات وزارة الداخلية عبر منصة أبشر (الجوازات، الأحوال المدنية، المرور، الأمن العام، المديرية العامة للسجون، خدمات الوزارة). لا تذكر أبداً خدمات التعليم أو الصحة أو التجارة أو أي وزارة أخرى.
7. **الإيجاز:** لا تكرر المعلومات. أجب بأقل عدد كلمات يوصل المعلومة بوضوح.
8. **تنبيه:** لا تستخدم أبداً أي لغة أجنبية في ردك. العربية فقط. الترجمة ستتم تلقائياً."""

    # =====================================================
    # F. INFRASTRUCTURE PROVISIONING
    # =====================================================
    @classmethod
    def setup_environment(cls):
        """Automates local filesystem readiness and enables hardware-level optimizations."""
        try:
            dirs = [
                cls.LOG_DIR, cls.DATA_MASTER_DIR, cls.DATA_CHUNK_DIR,
                cls.DATA_PROCESSED_DIR, cls.VECTOR_DB_DIR, cls.AUDIO_DIR,
                cls.TELEMETRY_DIR, cls.BENCHMARK_RESULTS_DIR, cls.MODELS_CACHE_DIR
            ]
            for d in dirs:
                os.makedirs(d, exist_ok=True)
            if cls.DEVICE == "cuda":
                torch.set_float32_matmul_precision('high')
            if not cls.HF_TOKEN:
                print("⚠️ SECURITY WARNING: HF_TOKEN missing. Gated models (ALLaM) may fail to load.")
        except Exception as e:
            print(f"⚠️ ENVIRONMENT ERROR: Infrastructure setup failed: {e}")


Config.setup_environment()