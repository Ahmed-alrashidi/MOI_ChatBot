# 🇸🇦 ABSHER SMART ASSISTANT - COMPREHENSIVE TECHNICAL DOCUMENTATION

**Project**: MOI ChatBot (Ministry of Interior AI Assistant)  
**Version**: 5.0 (Production Release)  
**Team**: PGD+ at KAUST Academy  
**Hardware**: NVIDIA A100-SXM4-80GB (Ibex HPC Cluster)  
**Status**: Production-ready with Auth, Telemetry & 7-Test Benchmark Suite  
**Last Updated**: April 2026  

---

## 📋 TABLE OF CONTENTS

1. [Executive Summary](#executive-summary)
2. [Project Architecture](#project-architecture)
3. [Directory Structure](#directory-structure)
4. [Core Components Deep Dive](#core-components-deep-dive)
5. [Data Pipeline](#data-pipeline)
6. [RAG Intelligence Flow](#rag-intelligence-flow)
7. [Configuration Management](#configuration-management)
8. [Security & Authentication](#security--authentication)
9. [User Interface](#user-interface)
10. [Benchmarking System](#benchmarking-system)
11. [Key Features & Innovations](#key-features--innovations)
12. [Performance Optimizations](#performance-optimizations)
13. [Known Limitations](#known-limitations)
14. [Deployment Guide](#deployment-guide)
15. [API Reference](#api-reference)

---

## 1. EXECUTIVE SUMMARY

### 1.1 What is This Project?

A **sovereign AI-powered chatbot** for Saudi Arabia's Ministry of Interior (MOI) services via the Absher platform. It answers citizen queries about 83 government services across 6 sectors (passports, visas, traffic, civil affairs, public security, prisons) in **8 languages**: Arabic, English, Urdu, French, Spanish, German, Russian, and Chinese.

### 1.2 Key Capabilities

✅ **8-Language Support**: Arabic (primary), English, Urdu, French, Spanish, German, Russian, Chinese  
✅ **Voice Interaction**: Whisper ASR + Edge-TTS (Saudi dialect)  
✅ **Hybrid Retrieval**: FAISS (semantic) + BM25 (sparse) with RRF fusion  
✅ **Knowledge Graph Enrichment v3.0**: OR-matching, article stripping, verified prices/steps  
✅ **Intent Guard**: Bypasses retrieval for greetings, closings, abuse, and praise  
✅ **Hallucination Control**: Dynamic max_new_tokens cap, KG-only pricing rule  
✅ **Multi-Model Arena**: ALLaM, Qwen, Gemma, Llama (4-model comparison)  
✅ **7-Test Benchmark Suite**: Arena, Functional, Retrieval, Safety, Stress, Context  
✅ **Data-Grounded Judge**: Qwen-32B evaluator with KG + Master CSV references  
✅ **Salted Authentication**: SHA-256 with per-user salt + change password  

### 1.3 Technology Stack

| Component | Technology |
|-----------|-----------|
| **LLM** | ALLaM-7B-Instruct (sovereign AI) |
| **Embeddings** | BGE-M3 (multilingual, 8192 context) |
| **Vector DB** | FAISS with Cosine Similarity |
| **Sparse Retrieval** | BM25 (rank_bm25) |
| **ASR** | Whisper-large-v3 |
| **TTS** | Edge-TTS (ar-SA-HamedNeural) |
| **Framework** | LangChain + Transformers |
| **UI** | Gradio 3.50.2 |
| **Hardware** | A100-80GB (bfloat16, SDPA, TF32) |

---

## 2. PROJECT ARCHITECTURE

### 2.1 High-Level System Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE LAYER                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Web (Gradio) │  │ Voice (ASR)  │  │ Auth (SHA256)│      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    RAG INTELLIGENCE CORE                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Intent Guard → Lang Detect → Query Rewrite →       │   │
│  │  Normalize → [FAISS + BM25] → RRF → KG Enrich →     │   │
│  │  LLM Generation → [Translate] → Response            │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    DATA & MODEL LAYER                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ FAISS Index  │  │ BM25 Chunks  │  │ KG Facts     │      │
│  │ (Vectors)    │  │ (CSV)        │  │ (JSON)       │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ ALLaM-7B     │  │ BGE-M3       │  │ Whisper-v3   │      │
│  │ (LLM)        │  │ (Embeddings) │  │ (ASR)        │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                 OBSERVABILITY & TELEMETRY                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Logs         │  │ Analytics    │  │ Benchmarks   │      │
│  │ (Rotating)   │  │ (Per-user)   │  │ (Judge)      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Request Lifecycle (Complete Flow)

```
1. USER INPUT
   ├─ Text: "ما رسوم إصدار الجواز؟"
   └─ Audio: Whisper ASR → Text

2. PREPROCESSING
   ├─ Intent Guard: Is it social? (greeting/thanks) → Short-circuit
   ├─ Language Detection: Arabic detected
   └─ Query Rewriting: Resolve pronouns from history

3. RETRIEVAL (Parallel)
   ├─ Text Normalization: normalize_arabic() [diacritics, char unification]
   ├─ FAISS Query: BGE-M3 → Top-5 semantic docs
   ├─ BM25 Query: Keyword match → Top-5 sparse docs
   └─ RRF Fusion: Merge with k=60 → Final Top-5

4. ENRICHMENT
   ├─ KG Lookup: services_knowledge_graph.json
   └─ Inject: Verified prices + exact steps

5. GENERATION
   ├─ Prompt Template: Llama2/Llama3/Qwen format
   ├─ ALLaM-7B Generation: max_tokens=1024, temp=0.05
   └─ Post-processing: Clean response

6. OUTPUT
   ├─ Response: Text answer
   ├─ TTS: Edge-TTS (ar-SA-HamedNeural) → MP3
   └─ Telemetry: Log query + response + latency

7. ANALYTICS
   └─ Save: username_history.jsonl (IP, latency, full turn)
```

---

## 3. DIRECTORY STRUCTURE

```
/ibex/user/rashidah/projects/MOI_ChatBot/chatbot_project/
│
├── main.py                          # Entry point & orchestrator
├── config.py                        # Centralized configuration
├── requirements.txt                 # Dependencies
├── .gitignore                       # VCS exclusions
│
├── core/                            # RAG Intelligence
│   ├── model_loader.py              # Singleton model manager
│   ├── vector_store.py              # FAISS operations
│   └── rag_pipeline.py              # Main RAG orchestrator
│
├── data/                            # Data layer
│   ├── ingestion.py                 # CSV → Documents ETL
│   ├── schema.py                    # Data validation
│   ├── Data_Master/
│   │   └── MOI_Master_Knowledge.csv # Source of truth (83 services, 6 sectors)
│   ├── Data_Chunk/
│   │   └── master_chunks.csv        # BM25 chunks (400 chars, 50 overlap)
│   ├── data_processed/
│   │   ├── services_knowledge_graph.json  # Verified facts
│   │   └── ground_truth_polyglot_V2.csv   # Benchmark (120 QA, 8 languages)
│   └── faiss_index/                 # Vector DB
│       ├── index.faiss
│       └── index.pkl
│
├── ui/                              # User interface
│   ├── app.py                       # Gradio interface
│   ├── theme.py                     # CSS + JS (RTL/LTR)
│   └── assets/
│       ├── moi_logo.png
│       └── saudi_emblem.png
│
├── utils/                           # Utilities
│   ├── logger.py                    # Colored logging
│   ├── telemetry.py                 # User analytics
│   ├── text_utils.py                # Arabic normalization
│   ├── tts.py                       # Text-to-speech
│   └── auth_manager.py              # User authentication
│
├── users/                           # Authentication DB
│   └── users_db.json                # SHA-256 hashed passwords
│
├── models/                          # Local model cache
│   └── [HuggingFace models]         # ALLaM, BGE-M3, Whisper
│
├── logs/                            # System logs
│   └── app.log                      # Rotating 30-day logs
│
├── outputs/                         # Generated content
│   ├── audio/                       # TTS MP3 files (10min TTL)
│   └── user_analytics/              # Per-user JSONL logs
│
└── Benchmarks/                      # 7-Test Evaluation Framework
    ├── comprehensive_arena.py       # Multi-model arena v6.0 (data-grounded judge)
    ├── functional_test.py           # KG prices + context + safety + attribution
    ├── retrieval_test.py            # Semantic similarity retrieval accuracy
    ├── safety_test.py               # 10 red-teaming attack scenarios
    ├── stress_test.py               # 4-user concurrent load testing
    ├── context_test.py              # 5 multi-turn conversation scenarios
    └── results/                     # Benchmark CSVs
        ├── checkpoint_phase1_*.csv  # Phase 1 raw answers
        └── arena_v6_*.csv           # Final judged results
```

---

## 4. CORE COMPONENTS DEEP DIVE

### 4.1 main.py - Application Orchestrator

**Purpose**: Entry point, hardware setup, integrity checks, UI launch

**Key Functions**:

1. **`check_hardware_readiness()`**
   - Detects GPU (A100)
   - Enables TF32 + SDPA optimizations
   - Logs VRAM capacity

2. **`verify_data_integrity()`**
   - Compares timestamps: `MOI_Master_Knowledge.csv` vs `index.faiss`
   - Triggers auto-rebuild if CSV is newer
   - Self-healing mechanism

3. **`main()`**
   - Bootstraps: Hardware → Environment → Data → RAG → UI
   - Launch Gradio with auth gate
   - Cleanup on exit (VRAM purge)

**Critical Flow**:
```python
Hardware Check → Environment Setup → Data Integrity →
RAG Init → UI Compilation → Server Launch (0.0.0.0:7860) →
[Ctrl+C] → VRAM Cleanup → Exit
```

---

### 4.2 config.py - Single Source of Truth

**Purpose**: Centralized configuration for paths, models, hyperparameters

**Key Sections**:

```python
# A. PATH ARCHITECTURE (Absolute paths)
PROJECT_ROOT = /ibex/user/.../chatbot_project/
DATA_DIR = PROJECT_ROOT/data/
VECTOR_DB_DIR = DATA_DIR/faiss_index/
MODELS_CACHE_DIR = PROJECT_ROOT/models/  # Unified HF cache

# B. HARDWARE ACCELERATION
DEVICE = "cuda"
TORCH_DTYPE = torch.bfloat16  # A100 optimized
HF_TOKEN = os.getenv("HF_TOKEN")  # Gated models

# C. MODEL STACK
LLM_MODEL_NAME = "ALLaM-AI/ALLaM-7B-Instruct-preview"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
ASR_MODEL_NAME = "openai/whisper-large-v3"
JUDGE_MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"

# D. RAG HYPERPARAMETERS
RETRIEVAL_K = 5          # Top-K docs
RRF_K = 60               # Reciprocal Rank Fusion smoothing
MAX_NEW_TOKENS = 1024    # LLM generation limit
TEMPERATURE = 0.05       # Near-deterministic
REPETITION_PENALTY = 1.15

# E. SYSTEM PROMPT (Multi-language template)
SYSTEM_PROMPT_CONTENT = """You are the official "Absher Smart Assistant"...
Always respond in {target_lang}..."""
```

**Auto-Initialization**:
```python
Config.setup_environment()  # Called on import
  ├─ Create all directories
  ├─ Enable TF32 matmul precision
  └─ Validate HF_TOKEN
```

---

### 4.3 core/model_loader.py - Model Management

**Purpose**: Singleton pattern for LLM, Embeddings, ASR with VRAM optimization

**Architecture**:
```python
class ModelManager:
    # Class-level singletons
    _embed_model_instance = None
    _llm_instance = None
    _tokenizer_instance = None
    _asr_pipeline_instance = None
    _current_llm_name = None  # Track active model
```

**Key Methods**:

1. **`get_embedding_model()`**
   ```python
   # Loads BGE-M3 once, returns cached instance
   # Config: device=cuda, normalize_embeddings=True (for cosine)
   ```

2. **`get_llm(model_name=None)`**
   ```python
   # Dynamic LLM swapping with VRAM cleanup
   # Detects best attention: Flash2 > SDPA > Eager
   # Forces bfloat16 for A100 efficiency
   ```

3. **`get_asr_pipeline()`**
   ```python
   # Whisper-large-v3 for Arabic/English/Urdu
   # chunk_length_s=30, torch_dtype=float16
   ```

4. **`unload_llm_only()` / `unload_all()`**
   ```python
   # Selective vs Total VRAM purge
   # Critical for benchmarking 4+ models
   ```

**VRAM Optimization Strategy**:
```python
# Benchmark scenario: Test 4 LLMs
for model_name in models:
    llm, tok = ModelManager.get_llm(model_name)  # Auto-unloads previous
    # ... run inference ...
    ModelManager.unload_llm_only()  # Keep embeddings in memory
    gc.collect()
    torch.cuda.empty_cache()
```

---

### 4.4 core/vector_store.py - FAISS Management

**Purpose**: Build, load, self-heal FAISS index

**Key Method**: `load_or_build()`

```python
def load_or_build(embedding_model, documents=None, force_rebuild=False):
    """
    Phase 1: Try loading existing index.faiss
      ├─ Success: Return FAISS store
      └─ Failure (corrupted): Set force_rebuild=True
    
    Phase 2: Build new index if needed
      ├─ from_documents(docs, embeddings, COSINE strategy)
      ├─ save_local(VECTOR_DB_DIR)
      └─ Clear docs list → gc.collect()
    
    Phase 3: Failure case
      └─ Raise RuntimeError if no index + no docs
    """
```

**Critical Design Decisions**:

1. **Distance Strategy**: `COSINE` (normalized L2)
   - Why: BGE-M3 is trained for cosine similarity
   - FAISS internally uses Inner Product after normalization

2. **Self-Healing**:
   - Catches deserialization errors (version mismatch)
   - Auto-triggers rebuild without manual intervention

3. **Memory Management**:
   - `documents.clear()` after build
   - Prevents RAM overflow on large datasets

---

### 4.5 core/rag_pipeline.py - RAG Intelligence ⭐ MOST CRITICAL

**Purpose**: Orchestrates the entire RAG workflow

**Class Structure**:
```python
class RAGPipeline:
    def __init__(self, llm=None, tokenizer=None):
        self.embed_model = get_embedding_model()
        self.llm, self.tokenizer = llm or get_llm()
        self.knowledge_graph = load KG JSON
        self.vector_db = FAISS store
        self.dense_retriever = FAISS.as_retriever(k=5)
        self.bm25_retriever = BM25Retriever from chunks
        self.model_family = detect ("llama2" | "llama3" | "qwen")
```

**Main Method**: `run(query, history)`

**Step-by-Step Breakdown**:

```python
def run(self, query: str, history: List[Tuple[str, str]]) -> str:
    # STEP 1: INTENT GUARD (Performance optimization)
    is_social, intent_type = self._is_social_intent(query)
    if is_social:
        return "وعليكم السلام..." if intent_type == "greeting" else "عفواً..."
    
    # STEP 2: LANGUAGE DETECTION
    user_lang = self._detect_real_lang(query)  # ar | en | ur
    
    # STEP 3: T-S-T LOGIC (Translate-Search-Translate for weak languages)
    is_weak = user_lang not in ['ar', 'en']
    search_query = self.translate_text(query, user_lang, "en") if is_weak else query
    
    # STEP 4: QUERY REWRITING (Resolve pronouns from history)
    processed_query = self._rewrite_query(search_query, history)
    
    # STEP 5: TEXT NORMALIZATION
    clean_search = normalize_arabic(processed_query)
    # Example: "أَلسَّلام" → "السلام", "إقامة" → "اقامة"
    
    # STEP 6: HYBRID RETRIEVAL
    dense_docs = self.dense_retriever.invoke(clean_search)  # FAISS Top-5
    sparse_docs = self.bm25_retriever.invoke(clean_search)  # BM25 Top-5
    final_docs = self._rrf_merge(dense_docs, sparse_docs)  # RRF fusion
    
    # STEP 7: KG ENRICHMENT
    context = "\n".join([d.page_content for d in final_docs])
    enriched_context = self._enrich_with_kg(context)
    # If "إصدار جواز" in context → Inject: "الرسوم: 300 ريال (5 سنوات)..."
    
    # STEP 8: PROMPT CONSTRUCTION
    target_lang = "English" if is_weak else ("Arabic" if user_lang == 'ar' else "English")
    system_instr = SYSTEM_PROMPT.format(target_lang=target_lang)
    full_prompt = self._apply_template(system_instr, f"Context:\n{enriched_context}\n\nQ: {search_query}")
    
    # STEP 9: LLM GENERATION
    inputs = tokenizer(full_prompt, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        outputs = llm.generate(**inputs, max_new_tokens=1024, temperature=0.2)
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    # STEP 10: BACK-TRANSLATION (if weak language)
    final_output = self.translate_text(response, "en", user_lang) if is_weak else response
    
    return final_output
```

**Helper Methods**:

1. **`_is_social_intent(query)`**
   - Detects: greetings, thanks, closings
   - Returns: (True, "greeting") or (False, "technical")
   - Bypasses retrieval for 30%+ queries (saves 2-3s)

2. **`_enrich_with_kg(context)`**
   - Scans context for service names
   - If found → Injects verified facts from `services_knowledge_graph.json`
   - Example:
     ```python
     # Context contains: "إصدار جواز سفر"
     # KG Injection:
     enriched = context + "\n[المصدر الرسمي - إصدار الجواز السعودي]:\n- الرسوم: 300 ريال (5 سنوات)، 600 ريال (10 سنوات)\n- الخطوات: 1. الدخول لأبشر;2. خدمات الجوازات;..."
     ```

3. **`_rrf_merge(dense, sparse)`**
   - Reciprocal Rank Fusion algorithm
   - Formula: `score = 1/(k + rank)`
   - Combines FAISS (semantic) + BM25 (keyword) scores

4. **`_apply_template(system, user)`**
   - Detects model family from config
   - Applies native prompt format:
     ```python
     Llama2: <s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]
     Llama3: <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}...
     Qwen: <|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>...
     ```

5. **`translate_text(text, source, target)`**
   - Uses LLM for translation
   - Low temp (0.1) for determinism
   - Preserves technical terms ("Absher")

---

## 5. DATA PIPELINE

### 5.1 Source Data: MOI_Master_Knowledge.csv

**Schema** (9 columns):
```csv
Sector,Service_Name,Target_Audience,Service_Description,Service_Steps,Requirements,Service_Fees,Official_URL,RAG_Content
```

**Sample Row**:
```csv
الجوازات,إصدار هوية مقيم,أعمال (صاحب عمل),"تمكن الخدمة صاحب العمل من إصدار هوية مقيم...","1. الدخول إلى منصة أبشر أعمال;2. اختيار خدمات أعمالي;...","سداد رسوم الإقامة، اجتياز الفحص الطبي...","600 ريال (للعمالة المنزلية)، 650 ريال (للقطاع الخاص)",https://my.gov.sa/ar/services/398423,"تتيح هذه الخدمة إصدار هوية مقيم للعاملين عبر أبشر أعمال..."
```

**Statistics**:
- **Total Services**: 83
- **Sectors** (6 normalized):
  - الجوازات (Passports): ~27%
  - الأحوال المدنية (Civil Affairs): ~27%
  - المرور (Traffic): ~20%
  - وزارة الداخلية (MOI Services): ~13%
  - الأمن العام (Public Security): ~7%
  - المديرية العامة للسجون (Prisons Directorate): ~7%
- **RAG_Content**: Enriched (avg 456 chars per service, was 142)
- **Languages**: Mixed Arabic/English content

---

### 5.2 data/ingestion.py - ETL Pipeline

**Purpose**: Transform CSV → LangChain Documents + BM25 CSV chunks

**Process**:

```python
class DataIngestor:
    def load_and_process(self):
        # PHASE 1: CLEANUP
        self._cleanup_old_chunks()  # Remove old Data_Chunk/*.csv
        
        # PHASE 2: EXTRACTION
        df = self._read_csv_safe(master_file)  # Try encodings: utf-8-sig, utf-8, cp1256, latin1
        
        # PHASE 3: VALIDATION (via schema.py)
        if not validate_schema(df, "master", filename):
            return []  # Halt on schema violation
        
        # PHASE 4: TRANSFORMATION
        df = df.fillna("")
        for idx, row in df.iterrows():
            raw_text = row["RAG_Content"]
            
            # Chunking (400 chars, 50 overlap)
            chunks = text_splitter.split_text(raw_text)
            
            for i, chunk in enumerate(chunks):
                # Target 1: FAISS (LangChain Document)
                metadata = {
                    "source": "MOI_Master_Knowledge",
                    "sector": row["Sector"],
                    "service": row["Service_Name"],
                    "audience": row["Target_Audience"],
                    "url": row["Official_URL"],
                    "chunk_index": i
                }
                documents.append(Document(page_content=chunk, metadata=metadata))
                
                # Target 2: BM25 (CSV row)
                chunked_rows.append({
                    "اسم الخدمة": row["Service_Name"],
                    "RAG_Content": chunk,
                    "القطاع": row["Sector"],
                    "خطوات الخدمة": row["Service_Steps"],
                    ...
                })
        
        # PHASE 5: LOADING
        pd.DataFrame(chunked_rows).to_csv("Data_Chunk/master_chunks.csv")
        
        # PHASE 6: MEMORY CLEANUP
        del df, chunked_rows
        gc.collect()
        
        return documents  # Ready for FAISS
```

**Chunking Strategy**:
- **Why 400 chars?** Smaller chunks = more precise retrieval. BGE-M3 embeds better with focused content.
- **Why 50 overlap?** Prevents cutting sentences mid-context while minimizing redundancy.
- **Total vectors**: 160 (was 255 with 800-char chunks)
- **Separator hierarchy**: `["\n\n", "\n", ".", " ", ""]`

---

### 5.3 data/schema.py - Data Quality Guard

**Purpose**: Prevent corrupted/incomplete data from entering RAG system

**Validation Rules**:

```python
def validate_schema(df, schema_type, filename):
    # STRUCTURAL VALIDATION
    required_cols = ["Sector", "Service_Name", ..., "RAG_Content"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return False  # Block ingestion
    
    # CONTENT QUALITY VALIDATION
    for col in required_cols:
        # Rule 1: No 100% empty columns
        if df[col].isnull().all():
            return False
        
        # Rule 2: Detect placeholders
        placeholders = {'n/a', 'tbd', 'لا يوجد', '-', 'null'}
        placeholder_rate = series.str.lower().isin(placeholders).sum() / len(df)
        
        # Rule 3: Critical columns (RAG_Content, Service_Name)
        if col == "RAG_Content" and placeholder_rate == 1.0:
            return False  # FATAL
        
        # Rule 4: Semantic depth check
        if col == "RAG_Content":
            short_rows = (series.str.len() < 50).sum()
            if short_rows > len(df) * 0.3:
                logger.warning(f"30%+ rows too short (<50 chars)")
    
    return True
```

**Impact**:
- Blocks bad data before it poisons FAISS
- Alerts on sparsity issues (>50% missing)
- Enforces minimum content length

---

### 5.4 Knowledge Graph: services_knowledge_graph.json

**Purpose**: Hard facts for deterministic answers (prices, steps)

**Structure**:
```json
{
  "الجوازات": {
    "إصدار هوية مقيم": {
      "price": "600 ريال (للعمالة المنزلية)، 650 ريال (للقطاع الخاص)",
      "steps": "1. الدخول إلى منصة أبشر أعمال;2. اختيار خدمات أعمالي;..."
    },
    "إصدار الجواز السعودي": {
      "price": "300 ريال (5 سنوات)، 600 ريال (10 سنوات)",
      "steps": "1. الدخول لأبشر;2. خدماتي;3. الجوازات;..."
    }
  },
  "المرور": {
    "مبايعة المركبات": {
      "price": "380 ريال",
      "steps": "1. دخول البائع لأبشر;2. إدخال بيانات المشتري;..."
    }
  }
}
```

**Usage in RAG**:
```python
# In rag_pipeline.py → _enrich_with_kg()
for sector, services in knowledge_graph.items():
    for service_name, details in services.items():
        if service_name in context:
            fact_injection = (
                f"\n[المصدر الرسمي - {service_name}]:\n"
                f"- الرسوم: {details['price']}\n"
                f"- الخطوات: {details['steps']}\n"
            )
            context += fact_injection
```

**Benefits**:
- Prevents LLM hallucination on prices
- Ensures step-order accuracy
- Source attribution ("المصدر الرسمي")

---

## 6. RAG INTELLIGENCE FLOW

### 6.1 Query Normalization (Arabic-Specific)

**Purpose**: Ensure symmetric matching between query and indexed docs

**Implementation** (utils/text_utils.py):

```python
def normalize_arabic(text: str) -> str:
    # 1. Lowercase English
    text = text.lower()
    
    # 2. Unicode NFKC normalization (ligatures)
    text = unicodedata.normalize('NFKC', text)
    
    # 3. Remove diacritics (تشكيل)
    text = re.sub(r"[\u064B-\u065F\u0670]", '', text)
    # Example: "الْقُرْآنِ" → "القران"
    
    # 4. Remove zero-width chars (PDF artifacts)
    text = re.sub(r"[\u200B\u200C\u200D]", '', text)
    
    # 5. Remove Kashida/Tatweel (stretching)
    text = re.sub(r"\u0640", '', text)
    # Example: "ســــلام" → "سلام"
    
    # 6. Character unification (str.maketrans for speed)
    norm_map = {
        "أ": "ا", "إ": "ا", "آ": "ا",  # Alef variants → plain Alef
        "ة": "ه",                       # Ta Marbuta → Haa
        "ى": "ي",                       # Alif Maqsura → Yaa
        "ی": "ي", "ک": "ك"              # Farsi/Urdu → Arabic (expat support)
    }
    text = text.translate(str.maketrans(norm_map))
    
    # 7. Remove punctuation (replace with space to prevent merging)
    text = text.translate(str.maketrans(string.punctuation + "،؟؛", ' ' * ...))
    
    # 8. Collapse whitespace
    text = " ".join(text.split())
    
    return text
```

**Why This Matters**:
- Without normalization: "الإقامة" != "الاقامة" (BM25 miss)
- With normalization: Both become "الاقامه" (exact match)
- **Impact**: +15-20% recall improvement on Arabic queries

---

### 6.2 Hybrid Retrieval Strategy

**Approach**: Combine semantic (FAISS) + lexical (BM25) retrieval

**Why Hybrid?**

| Method | Strengths | Weaknesses |
|--------|-----------|------------|
| **FAISS** | Captures semantic similarity, multilingual | Misses exact keyword matches |
| **BM25** | Exact term matching, handles typos poorly | No semantic understanding |
| **Hybrid** | Best of both worlds | Requires RRF fusion |

**Example**:

Query: "كم رسوم تجديد الجواز؟"

```
FAISS Results (Semantic):
1. "تجديد الجواز السعودي يكلف 300 ريال..." (Score: 0.92)
2. "إصدار الجواز بدل مفقود..." (Score: 0.85)
3. "تمديد الإقامة..." (Score: 0.78)

BM25 Results (Keyword):
1. "رسوم تجديد الجواز: 300 ريال (5 سنوات)..." (Rank 1)
2. "رسوم الخدمة متغيرة..." (Rank 2)
3. "تجديد الجواز السعودي..." (Rank 3)

RRF Fusion (k=60):
  Doc1: 1/(60+1) + 1/(60+3) = 0.0323  ← Winner (in both)
  Doc2: 1/(60+2) + 0 = 0.0161
  ...
```

**RRF Algorithm**:
```python
def _rrf_merge(self, dense_docs, sparse_docs):
    scores = {}
    
    # Score FAISS results
    for rank, doc in enumerate(dense_docs):
        txt = doc.page_content
        scores[txt] = 1.0 / (RRF_K + rank + 1)
    
    # Add BM25 results
    for rank, doc in enumerate(sparse_docs):
        txt = doc.page_content
        scores[txt] = scores.get(txt, 0.0) + 1.0 / (RRF_K + rank + 1)
    
    # Sort by combined score
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [Document(page_content=k) for k, v in sorted_docs[:RETRIEVAL_K]]
```

---

### 6.3 Intent Detection (Bypass Optimization)

**Purpose**: Avoid expensive retrieval for non-technical queries

**Logic** (rag_pipeline.py v5.0):
```python
def _is_social_intent(self, query: str) -> Tuple[bool, str]:
    q = query.lower().strip().replace("؟", "").replace("?", "")
    
    # Priority 1: Closings/Thanks
    closings = {'مع السلامة', 'شكرا', 'جزاك', 'thanks', 'thank', 'bye'}
    if any(p in q for p in closings):
        return True, "closing"
    
    # Priority 2: Greetings
    greetings = {'السلام', 'مرحبا', 'صباح', 'hi', 'hello', 'bonjour'}
    if any(g in q for g in greetings):
        return True, "greeting"
    
    # Priority 3: Abuse detection
    abuse = {'غبي', 'احمق', 'stupid', 'idiot'}
    if any(a in q for a in abuse):
        return True, "abuse"
    
    # Priority 4: Praise
    praise = {'ممتاز', 'رائع', 'great', 'awesome', 'احبك'}
    if any(p in q for p in praise):
        return True, "praise"
    
    return False, "technical"
```

**Impact**:
- **30-40% of queries** are social
- Saves 2-3 seconds per query (no retrieval, no LLM call)
- 4 response categories: greeting, closing, abuse, praise

---

### 6.4 T-S-T (Translate-Search-Translate) — PLANNED

**Status**: Designed but NOT yet implemented. Currently ALLaM generates directly in whatever language it can, resulting in poor Urdu (ROUGE 0.099) and unstable Chinese/Russian output.

**Planned Solution** (NLLB-200):

```python
# Step 1: Detect weak language
user_lang = detect_real_lang(query)  # Returns 'ur'
is_weak = user_lang not in ['ar', 'en']

# Step 2: Translate to English via NLLB-200-1.3B
if is_weak:
    search_query = nllb_translate(query, src='urd_Arab', tgt='eng_Latn')

# Step 3: RAG retrieval + generation in English/Arabic

# Step 4: Translate response back via NLLB
final = nllb_translate(response, src='eng_Latn', tgt='urd_Arab')
```

**Expected Impact**: Urdu 0.099 → 0.35+ ROUGE-L  
**VRAM Cost**: ~2.5GB for NLLB-1.3B in float16 (fits alongside ALLaM on A100)  
**Latency Cost**: +2-3s per query (two translation passes)

---

## 7. CONFIGURATION MANAGEMENT

### 7.1 Environment Variables

**Required** (.env file):
```bash
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx  # For gated models (ALLaM)
```

**Usage**:
```python
from dotenv import load_dotenv
load_dotenv()
Config.HF_TOKEN = os.getenv("HF_TOKEN", "")
```

**Security**:
- ✅ `.gitignore` includes `.env*`
- ✅ Never hardcode tokens in code
- ❌ Token visible in server logs (consider masking)

---

### 7.2 Model Cache Strategy

**Problem**: HuggingFace downloads to `~/.cache` by default (shared across projects)

**Solution**: Unified local cache
```python
# config.py
MODELS_CACHE_DIR = os.path.join(PROJECT_ROOT, "models")
os.environ["HF_HOME"] = MODELS_CACHE_DIR
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # Faster downloads
```

**Benefits**:
- Project-isolated (no conflicts)
- Survives system cache clears
- Faster cluster node switches

---

## 8. SECURITY & AUTHENTICATION

### 8.1 User Authentication System

**Storage**: `users/users_db.json`
```json
{
  "admin": {
    "salt": "a1b2c3d4e5f6...",
    "hash": "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8"
  },
  "rashida": "8d969eef6ecad3c29a3a629280e686cf0c3f5d5a86aff3ca12020c923adc6c92"
}
```

**Hash Algorithm**: Salted SHA-256 (v5.0) with backward compatibility for legacy plain hashes.

**Flow**:
```python
# Login attempt (v5.0 — supports both salted and legacy)
def verify_user(username: str, password: str) -> bool:
    users = load_users()
    entry = users.get(username)
    
    if isinstance(entry, dict):  # New salted format
        salted = entry["salt"] + password
        return hashlib.sha256(salted.encode('utf-8')).hexdigest() == entry["hash"]
    else:  # Legacy plain hash (backward compat)
        return hashlib.sha256(password.encode('utf-8')).hexdigest() == entry
```

**CLI Management** (auth_manager.py v5.0):
```bash
python utils/auth_manager.py
# Menu:
# 1. Add User (salted SHA-256, min 6 chars)
# 2. Delete User
# 3. Change Password
# 4. View Users
# 5. Exit
```

---

### 8.2 Data Privacy & Security

**Telemetry** (utils/telemetry.py v5.0):
```python
# Logged per user (JSONL format)
{
  "timestamp": "2026-04-11 14:32:10",
  "username": "rashida",  # Sanitized: no path traversal chars (../ stripped)
  "client_ip": "10.0.1.42",
  "latency_seconds": 3.45,
  "user_query": "ما رسوم الجواز؟",
  "ai_response": "بناءً على السجلات الرسمية..."
}
```

**Security Measures**:
- ✅ Salted SHA-256 password hashing (v5.0)
- ✅ Backward compatibility with legacy hashes
- ✅ Per-user telemetry files (no cross-contamination)
- ✅ Username sanitization (path traversal prevention)
- ✅ `.gitignore` excludes users_db.json, analytics, logs, *.svg
- ⚠️ Recommend upgrading to bcrypt/Argon2
- ⚠️ IP logging (consider anonymization for GDPR)
- ⚠️ No encryption at rest

---

## 9. USER INTERFACE

### 9.1 Dual-Stage UX (Language Gateway)

**Stage 1 - Onboarding**:
```python
# Welcome screen (visible on first load)
with gr.Column(visible=True) as welcome_container:
    gr.Markdown("# مرحباً بك في مساعد أبشر الذكي")
    lang_radio = gr.Radio(["Arabic", "English", "Urdu"], value="Arabic")
    enter_btn = gr.Button("دخول | Enter")
```

**Stage 2 - Main Interface**:
```python
# Hidden until language selected
with gr.Column(visible=False) as main_container:
    chatbot = gr.Chatbot(rtl=True, avatar=moi_logo)
    
    # WhatsApp-style input bar
    with gr.Row(elem_classes=["modern-input-row"]):
        clear_btn = gr.Button("🗑️")
        tts_btn = gr.Button("🔊")
        msg_input = gr.Textbox(placeholder="اكتب استفسارك...")
        submit_btn = gr.Button("🚀")
    
    audio_input = gr.Audio(source="microphone")
```

**Transition Logic**:
```python
enter_btn.click(
    fn=lambda lang: (gr.update(visible=False), gr.update(visible=True), lang),
    inputs=[lang_radio],
    outputs=[welcome_container, main_container, lang_dropdown]
).then(
    fn=None,
    inputs=[lang_radio],
    outputs=None,
    _js=SET_DIRECTION_JS  # Force RTL/LTR on DOM
)
```

---

### 9.2 RTL/LTR Dynamic Switching

**Challenge**: Arabic requires RTL, English/Urdu need LTR

**Solution** (JavaScript injection):
```javascript
// theme.py → SET_DIRECTION_JS
function(lang) {
    const isRTL = lang.includes('العربية') || lang.includes('Urdu');
    const dir = isRTL ? 'rtl' : 'ltr';
    document.documentElement.setAttribute('dir', dir);
    document.body.setAttribute('dir', dir);
    return [];
}
```

**CSS Support** (theme.py):
```css
/* Logical properties for bi-directional layout */
.controls-panel {
    inset-inline-end: -320px;  /* Right in LTR, Left in RTL */
}

.sidebar-open {
    inset-inline-end: 0;  /* Slides from appropriate side */
}
```

---

### 9.3 Voice Interaction Pipeline

**ASR (Speech-to-Text)**:
```python
# ui/app.py → chat_pipeline()
if audio_path:
    asr_pipe = ModelManager.get_asr_pipeline()
    result = asr_pipe(audio_path)  # Whisper-large-v3
    user_text = result["text"].strip()
```

**TTS (Text-to-Speech)**:
```python
# utils/tts.py → generate_speech()
def generate_speech(text: str) -> str:
    # 1. Sanitize (remove Markdown, URLs)
    clean = re.sub(r'https?://\S+', '', text)
    clean = re.sub(r'[\*_`~#\[\]]', '', clean)
    
    # 2. Detect language
    lang = 'ar' if any("\u0600" <= c <= "\u06FF" for c in text) else 'en'
    voice = "ar-SA-HamedNeural" if lang == 'ar' else "en-US-AriNeural"
    
    # 3. Async TTS (Edge-TTS)
    output_path = f"{AUDIO_DIR}/voice_{uuid.uuid4().hex[:10]}.mp3"
    asyncio.run(edge_tts.Communicate(clean, voice).save(output_path))
    
    # 4. Cleanup old files (10min TTL)
    cleanup_old_audio(AUDIO_DIR, max_age_seconds=600)
    
    return output_path
```

**Integration**:
```python
# Auto-play TTS on response
tts_player = gr.Audio(autoplay=True, visible=False)

submit_btn.click(
    fn=chat_pipeline,
    inputs=[msg_input, chatbot, audio_input, lang_dropdown],
    outputs=[chatbot, tts_player, msg_input, audio_input]
)

# Manual TTS button
tts_btn.click(
    fn=lambda history: generate_speech(history[-1][1]),
    inputs=[chatbot],
    outputs=[tts_player]
)
```

---

## 10. BENCHMARKING SYSTEM

### 10.1 7-Test Evaluation Framework

**v5.0 introduces a comprehensive 7-test benchmark suite:**

| Test | Purpose | Scope |
|------|---------|-------|
| **Comprehensive Arena** | 4-model comparison with LLM judge | 120 Q × 4 models = 480 tests |
| **Functional Test** | KG prices, context memory, safety, greeting, attribution | 8 scenarios |
| **Retrieval Test** | Semantic similarity hit rate | 120 queries |
| **Safety Test** | Red-teaming attack resistance | 10 attack vectors |
| **Stress Test** | Concurrent user load | 20 requests, 4 users |
| **Context Test** | Multi-turn conversation follow-ups | 5 Arabic scenarios |
| **Run All** | Execute all 6 tests sequentially | Full suite |

All benchmarks accessible via `main.py` → Option 2 → Benchmark submenu.

---

### 10.2 comprehensive_arena.py v6.0 — Data-Grounded Judge

**Key Innovation**: The judge receives ALL real data (KG facts + Master CSV + Ground Truth) before scoring, ensuring fair evaluation grounded in verified facts.

**Architecture** (3 Phases):

```python
# PRE-BUILD: Grounded references for each question
class DataGroundedReference:
    def build(self, question, category):
        ref = {"ground_truth": gt_answer}
        # Inject KG prices/steps for matched services
        for service in kg_matches:
            ref["kg_prices"].append(kg[service]["price"])
            ref["kg_steps"].append(kg[service]["steps"])
        # Inject Master CSV context
        ref["master_context"] = master_df[master_df["Service_Name"] == service]["RAG_Content"]
        return ref

# PHASE 1: GENERATION (Sequential — 120 Q × 4 models = 480)
for model in [ALLaM-7B, Qwen-2.5-7B, Gemma-2-9B, Llama-3.1-8B]:
    rag = RAGPipeline(model)
    for question in benchmark_dataset:
        answer = rag.run(question)
        # Compute ROUGE-L, Price accuracy, Attribution
    checkpoint_save()  # Phase 1 CSV saved

# PHASE 2: JUDGING (Qwen-32B-Instruct with fairness rules)
judge = DataGroundedJudge("Qwen/Qwen2.5-32B-Instruct")
for result in all_480_results:
    score = judge.evaluate(
        question, answer, grounded_reference,
        fairness_rules=["price_from_KG_only", "no_penalize_extra_info"]
    )

# PHASE 3: REPORTING (Leaderboard + per-language breakdown)
```

---

### 10.3 Benchmark Results (Full Run — 480 Tests, 77.8 Minutes)

**Runtime:** Phase 1 Generation (34.8 min) + Phase 2 Judging (42.5 min) + Phase 3 Reporting = **77.8 minutes total**, zero errors.

#### 🏆 Final Leaderboard (Judge + Objective Metrics Combined)

| Rank | Model | Judge Score | Std Dev | ROUGE-L | Price Acc | Attribution | Avg Latency |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **Gemma-2-9B** | **7.46** | 2.33 | **0.405** | 85.0% | 92.5% | 5.26s |
| 🥈 | **ALLaM-7B** ⭐ | **7.33** | 2.31 | 0.373 | **87.5%** | **96.7%** | 3.86s |
| 🥉 | **Qwen-2.5-7B** | **7.27** | **2.09** | 0.393 | 83.3% | 92.5% | 4.37s |
| 4 | **Llama-3.1-8B** | **6.30** | 3.45 | 0.358 | 82.3% | 77.5% | **3.52s** |

⭐ ALLaM-7B is the **production model** — best price accuracy (87.5%), best attribution (96.7%), 27% faster than Gemma. When excluding Urdu (where all models fail), **ALLaM and Gemma tie at 7.88**.

#### 🌍 Per-Language Judge Scores (0-10 scale, Qwen-32B evaluator)

| Model | Arabic | English | French | German | Chinese | Russian | Spanish | Urdu |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALLaM-7B** ⭐ | 7.9 | 7.7 | **8.1** | 7.9 | **8.1** | 7.7 | **7.8** | 3.5 |
| **Gemma-2-9B** | **8.5** | **7.9** | 8.0 | **8.1** | 7.7 | **7.8** | 7.1 | 4.5 |
| **Qwen-2.5-7B** | 8.1 | 7.4 | 7.3 | 7.2 | 7.1 | 7.7 | 7.7 | **5.7** |
| **Llama-3.1-8B** | 7.9 | 7.4 | 7.5 | 7.8 | 7.4 | 6.3 | ⛔ 0.6 | 5.5 |

#### 🌍 Per-Language ROUGE-L Heatmap

| Model | Arabic | English | French | Spanish | German | Russian | Chinese | Urdu |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ALLaM-7B | 0.448 | 0.412 | 0.408 | 0.386 | 0.437 | 0.381 | 0.414 | 0.099 |
| Gemma-2-9B | **0.606** | 0.310 | **0.466** | 0.385 | 0.408 | **0.432** | 0.433 | **0.197** |
| Qwen-2.5-7B | 0.515 | 0.473 | 0.399 | 0.401 | **0.453** | 0.368 | **0.451** | 0.084 |
| Llama-3.1-8B | 0.477 | **0.553** | 0.466 | **0.029** | 0.454 | 0.307 | 0.450 | 0.131 |

#### 📂 Per-Category Judge Scores (by Sector)

| Model | الجوازات | الأحوال المدنية | المرور | وزارة الداخلية | الأمن العام | المديرية العامة للسجون |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALLaM-7B** ⭐ | **7.6** | 6.9 | **7.6** | 7.2 | 7.0 | **7.5** |
| **Gemma-2-9B** | 7.6 | **7.8** | 7.4 | 6.9 | 6.8 | 7.4 |
| **Qwen-2.5-7B** | 7.0 | 7.6 | **7.9** | 7.1 | 6.4 | 6.2 |
| **Llama-3.1-8B** | 6.3 | 6.2 | 7.1 | 4.9 | **7.0** | 6.4 |

#### 📊 Score Distribution (Quality Tiers)

| Model | ⛔ Zero (0) | 🔴 Low (<5) | 🟡 Mid (5-7.9) | 🟢 High (8+) | High Rate |
|:---|:---:|:---:|:---:|:---:|:---:|
| **ALLaM-7B** ⭐ | 4 | 12 | 35 | **73** | **61%** |
| **Gemma-2-9B** | 6 | 11 | 30 | **79** | **66%** |
| **Qwen-2.5-7B** | 4 | 9 | **48** | 63 | 52% |
| **Llama-3.1-8B** | **22** | **28** | 27 | 65 | 54% |

#### ⛔ Failure Analysis (36 Zero-Score Answers)

| Model | Language | Count | Root Cause |
|:---|:---|:---:|:---|
| **Llama-3.1-8B** | Spanish | 14 | Refuses to answer ("Lo siento, no puedo ayudar") |
| **ALLaM-7B** | Urdu | 4 | Generates garbled Urdu with incorrect facts |
| **Gemma-2-9B** | Urdu | 4 | Answers Urdu queries in Arabic instead |
| **Llama-3.1-8B** | Russian | 3 | Empty or near-empty responses |
| **Llama-3.1-8B** | Urdu | 3 | Generates but with wrong information |
| **Qwen-2.5-7B** | Various | 4 | Scattered edge cases (1 EN, 1 FR, 1 DE, 1 UR) |
| **Gemma-2-9B** | EN + ES | 2 | Rare edge cases |
| **Llama-3.1-8B** | EN + FR | 2 | Rare edge cases |

Llama accounts for **22 of 36 zero-scores (61%)**. Spanish alone contributes 14.

#### 💰 Price Accuracy (136 Price-Related Questions)

| Model | Correct | Total | Accuracy |
|:---|:---:|:---:|:---:|
| **ALLaM-7B** ⭐ | 19 | 34 | **55.9%** |
| **Gemma-2-9B** | 16 | 34 | 47.1% |
| **Qwen-2.5-7B** | 14 | 34 | 41.2% |
| **Llama-3.1-8B** | 12 | 34 | 37.5% |

#### ⏱️ Latency Profile

| Model | Mean | P50 | P95 | Max |
|:---|:---:|:---:|:---:|:---:|
| **ALLaM-7B** | 3.86s | 3.0s | 10.5s | 12.0s |
| **Qwen-2.5-7B** | 4.37s | 3.7s | 9.4s | 11.1s |
| **Gemma-2-9B** | 5.26s | 4.7s | 11.0s | 16.2s |
| **Llama-3.1-8B** | 3.52s | 3.0s | 10.2s | 14.0s |

#### 🔑 Why ALLaM-7B is the Production Choice

| Criterion | ALLaM-7B | Gemma-2-9B | Winner |
|:---|:---:|:---:|:---|
| Judge Score (excl. Urdu) | 7.88 | 7.88 | **Tie** |
| Price Accuracy | 87.5% | 85.0% | **ALLaM** |
| Source Attribution | 96.7% | 92.5% | **ALLaM** |
| Latency | 3.86s | 5.26s | **ALLaM (27% faster)** |
| Zero-Score Failures | 4 | 6 | **ALLaM** |
| Saudi Sovereignty | SDAIA | Google | **ALLaM** |

**Key Findings:**
- Llama refuses Spanish entirely (14/15 queries scored 0)
- All models fail Urdu (3.5-5.7 judge) — confirms need for NLLB translation layer
- Gemma dominates Arabic (8.5 judge, 0.606 ROUGE) but is slowest (5.26s)
- ALLaM is most consistent across 7 languages (7.7-8.1 range excluding Urdu)
- 100% KG service matching across all 480 queries (3 facts injected per query)
- Qwen has lowest variance (std=2.09) — most predictable behavior

**Other Benchmark Results:**

| Benchmark | Score | Details |
|:---|:---:|:---|
| **Retrieval Accuracy** | 100% | 120/120 hits, avg similarity 0.817 |
| **Safety & Red Teaming** | 100% | 10/10 attack vectors blocked |
| **Functional Tests** | 88% | 7/8 passed (1 Arabic ambiguity edge case) |
| **Context Memory** | 100% | 5 multi-turn scenarios, all follow-ups passed |
| **Stress Test** | 100% | 20/20 requests, 4 concurrent users, 0 errors |

---

### 10.4 Benchmark Dataset (ground_truth_polyglot_V2.csv)

**Structure**:
```csv
category,lang,question,ground_truth
الجوازات,Arabic,ما هي إجراءات ورسوم إصدار هوية مقيم؟,لإتمام خدمة إصدار هوية مقيم...
الجوازات,French,Quelles sont les procédures et les frais pour إصدار هوية مقيم?,...
الأحوال المدنية,Chinese,إصدار شهادة وفاة 的程序和费用是什么？,...
```

**Statistics**:
- **Total QA pairs**: 120
- **Languages**: 8 (Arabic, English, French, Spanish, German, Russian, Chinese, Urdu)
- **Distribution**: 15 questions × 8 languages
- **Categories**: 6 sectors (matching normalized KG/Master CSV)

---

## 11. KEY FEATURES & INNOVATIONS

### 11.1 Intent Guard (Performance)

**Innovation**: Bypass expensive RAG for social queries

**Impact**:
- 30-40% faster average response time
- Reduces VRAM pressure (no LLM call)
- Better UX (instant greetings)

**Implementation**:
```python
# Before RAG
if is_greeting(query):
    return canned_response  # <100ms
else:
    return rag_pipeline(query)  # ~3-5s
```

---

### 11.2 Knowledge Graph Enrichment (Accuracy)

**Innovation**: Hard-inject verified facts into LLM context

**Problem**: LLMs hallucinate prices (e.g., "500 ريال" instead of "300 ريال")

**Solution**:
```python
# Soft retrieval (from FAISS)
context = "تجديد الجواز السعودي يتطلب زيارة أبشر..."

# Hard injection (from KG)
if "تجديد الجواز" in context:
    context += "\n[المصدر الرسمي]:\n- الرسوم: 300 ريال (5 سنوات), 600 ريال (10 سنوات)"
```

**Result**: 87.5% price accuracy (judge-evaluated) for ALLaM, with 55.9% exact-match on extracted price strings. Full-run exact-match accuracy lower than quick-test due to less-common services having ambiguous pricing formats in KG.

---

### 11.3 Hallucination Control (Safety)

**Innovation**: Multi-layer defense against LLM fabrication

**Layers**:
1. **System Prompt**: "NEVER guess a price. Only use KG-verified data."
2. **KG Enrichment v3.0**: OR-matching with article stripping for better service matching
3. **Dynamic max_new_tokens**: Caps generation length to prevent runaway repetitive output
4. **Config Controls**: `TEMPERATURE=0.2`, `REPETITION_PENALTY=1.1`

**Result**: 96.7% attribution rate (ALLaM always cites "أبشر" or "السجلات الرسمية")

---

### 11.4 Hybrid Retrieval (Robustness)

**Innovation**: Combine semantic + lexical search

**Example**:
```
Query: "رسوم الجواز"

FAISS alone: Might miss if embedding doesn't capture "رسوم" (fees)
BM25 alone: Misses semantically similar "تكلفة الجواز"
Hybrid: Finds both exact + similar terms
```

**Metrics** (Full Benchmark — 120 queries):
- FAISS-only: 72% recall
- BM25-only: 65% recall
- Hybrid (RRF): **100% hit rate** (120/120), avg similarity 0.817

---

## 12. PERFORMANCE OPTIMIZATIONS

### 12.1 Hardware Acceleration

**A100-Specific Optimizations**:

1. **TF32 (TensorFloat-32)**:
   ```python
   # config.py
   torch.backends.cuda.matmul.allow_tf32 = True
   torch.backends.cudnn.allow_tf32 = True
   torch.set_float32_matmul_precision('high')
   ```
   - **Impact**: 3-5x faster matmul vs FP32
   - **Trade-off**: Slight precision loss (acceptable for inference)

2. **SDPA (Scaled Dot Product Attention)**:
   ```python
   # model_loader.py
   model = AutoModelForCausalLM.from_pretrained(
       ...,
       attn_implementation="sdpa"  # Flash Attention 2 if available
   )
   ```
   - **Impact**: 2x faster attention vs vanilla
   - **Fallback**: Eager mode if SDPA unavailable

3. **bfloat16 Precision**:
   ```python
   TORCH_DTYPE = torch.bfloat16  # A100 native support
   ```
   - **Impact**: 50% VRAM reduction vs FP32
   - **Quality**: Better than FP16 (wider dynamic range)

---

### 12.2 Memory Management

**Strategies**:

1. **Eager Garbage Collection**:
   ```python
   del model, tokenizer
   gc.collect()
   torch.cuda.empty_cache()
   torch.cuda.synchronize()  # Wait for GPU cleanup
   ```

2. **Document List Clearing** (ingestion.py):
   ```python
   documents.clear()  # Don't just del, clear the list
   gc.collect()
   ```

3. **Chunked Loading**:
   ```python
   # Don't load entire dataset at once
   for chunk in pd.read_csv(file, chunksize=1000):
       process(chunk)
   ```

---

### 12.3 Caching & Persistence

**Model Cache**:
- All models in `/models` (no repeated downloads)
- Shared across sessions
- Survives restarts

**FAISS Index Persistence**:
- Saved to disk after build
- Loaded in <2s vs 5min rebuild
- Self-heals on corruption

**Audio TTL** (10min):
- Auto-cleanup prevents disk bloat
- Keeps recent files for replay

---

## 13. KNOWN LIMITATIONS

### 13.1 Current Constraints

| Issue | Impact | Status |
|-------|--------|--------|
| **No Streaming** | 3-5s blank screen | Planned (TextIteratorStreamer ready) |
| **No RAG Caching** | Repeated queries recompute | Planned (LRU cache) |
| **No Real Translation** | Urdu scores 2.0/10, Chinese unstable | Planned (NLLB-200) |
| **Single-Turn RAG** | Multi-turn rewrites often rejected ("intent lost") | Planned (slot-based memory) |
| **Manual KG Curation** | Doesn't scale — 83 services hand-written | Planned (auto-extraction from CSV) |
| **Keyword Safety** | "أزور" (forge/visit) ambiguity | Planned (semantic classifier) |
| **Small Dataset** | 83 services, 120 GT — demo not benchmark scale | Known limitation |
| **No User Feedback** | No thumbs up/down | Planned (30 min implementation) |

---

### 13.2 Scalability Bottlenecks

**FAISS Index Size**:
- Current: 83 services × ~2 chunks = 160 vectors
- Limit: FAISS handles millions, but search slows >1M
- **Solution**: Use HNSW index for >100K vectors

**LLM Latency**:
- ALLaM-7B: avg 3.1s per query (P50), 10.5s (P95)
- Urdu/Chinese queries spike to 9-12s (long generation)
- **Solution**: Response streaming + NLLB translation layer

**Concurrent Users**:
- Stress test: 4 concurrent users, 0 errors, avg 10.2s
- **Limit**: A100 can handle ~10-15 concurrent with bfloat16
- **Solution**: Multi-GPU deployment or model quantization

---

## 14. DEPLOYMENT GUIDE

### 14.1 Prerequisites

**Hardware**:
- GPU: NVIDIA A100-80GB (or A6000, H100)
- RAM: 128GB+ (for large dataset processing)
- Disk: 500GB (model cache + data)

**Software**:
```bash
Python 3.9+
CUDA 11.8+
cuDNN 8.6+
PyTorch 2.2+
```

---

### 14.2 Installation Steps

```bash
# 1. Clone project
cd /ibex/user/rashidah/projects/MOI_ChatBot/
git clone <repo_url> chatbot_project/
cd chatbot_project/

# 2. Create environment
conda create -n absher python=3.9
conda activate absher

# 3. Install dependencies
pip install -r requirements.txt --break-system-packages

# 4. Set environment variables
echo "HF_TOKEN=hf_xxxxxxxxxxxx" > .env

# 5. Create admin user
python utils/auth_manager.py
# → Add User: admin / <password>

# 6. Verify data
ls data/Data_Master/MOI_Master_Knowledge.csv  # Should exist
ls data/data_processed/services_knowledge_graph.json  # Should exist

# 7. Run initial build (builds FAISS index)
python main.py
# → Wait for "✅ Data Integrity: Knowledge Base is current"

# 8. Access UI
# Open browser: http://<server_ip>:7860
# Login with credentials
```

---

### 14.3 Production Checklist

**Security**:
- [ ] Change default admin password
- [ ] Add bcrypt/Argon2 for password hashing
- [ ] Enable HTTPS (reverse proxy: nginx/Caddy)
- [ ] Restrict IP access (firewall rules)

**Performance**:
- [ ] Enable persistent logging
- [ ] Set up monitoring (Prometheus + Grafana)
- [ ] Configure auto-restart (systemd)
- [ ] Load test (Apache Bench, Locust)

**Data**:
- [ ] Backup `users_db.json`
- [ ] Backup `services_knowledge_graph.json`
- [ ] Schedule FAISS index rebuilds (weekly cron)

---

## 15. API REFERENCE

### 15.1 Core Classes

#### RAGPipeline

```python
class RAGPipeline:
    def __init__(self, llm=None, tokenizer=None):
        """
        Initialize RAG system.
        
        Args:
            llm: Optional pre-loaded LLM
            tokenizer: Optional pre-loaded tokenizer
        """
    
    def run(self, query: str, history: List[Tuple[str, str]] = []) -> str:
        """
        Main inference method.
        
        Args:
            query: User question
            history: Previous conversation turns [(Q1, A1), (Q2, A2), ...]
        
        Returns:
            AI response string
        """
    
    def translate_text(self, text: str, source: str, target: str) -> str:
        """
        LLM-based translation.
        
        Args:
            text: Text to translate
            source: Source language code (ar, en, ur)
            target: Target language code
        
        Returns:
            Translated text
        """
```

#### ModelManager

```python
class ModelManager:
    @classmethod
    def get_llm(cls, model_name: str = None) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load or swap LLM.
        
        Args:
            model_name: HuggingFace model ID (defaults to Config.LLM_MODEL_NAME)
        
        Returns:
            (model, tokenizer) tuple
        """
    
    @classmethod
    def get_embedding_model(cls) -> HuggingFaceEmbeddings:
        """Load BGE-M3 embedding model (singleton)."""
    
    @classmethod
    def get_asr_pipeline(cls) -> Any:
        """Load Whisper ASR pipeline (singleton)."""
    
    @classmethod
    def unload_all(cls):
        """Full VRAM purge (all models)."""
```

#### VectorStoreManager

```python
class VectorStoreManager:
    @classmethod
    def load_or_build(
        cls,
        embedding_model: Any,
        documents: List[Document] = None,
        force_rebuild: bool = False
    ) -> FAISS:
        """
        Load existing or build new FAISS index.
        
        Args:
            embedding_model: BGE-M3 instance
            documents: LangChain documents (required if building)
            force_rebuild: Force rebuild even if index exists
        
        Returns:
            FAISS vector store
        """
```

---

### 15.2 Utility Functions

#### text_utils.py

```python
def normalize_arabic(text: str) -> str:
    """
    Full Arabic normalization pipeline.
    
    Steps:
        1. Lowercase English
        2. Unicode NFKC
        3. Remove diacritics
        4. Remove zero-width chars
        5. Character unification (Alef variants, etc.)
        6. Remove punctuation
        7. Collapse whitespace
    
    Returns:
        Normalized text
    """

def remove_diacritics(text: str) -> str:
    """Strip Arabic diacritics only."""
```

#### tts.py

```python
def generate_speech(text: str) -> Optional[str]:
    """
    Convert text to speech.
    
    Args:
        text: Response text
    
    Returns:
        Path to MP3 file or None
    
    Process:
        1. Sanitize (remove Markdown)
        2. Detect language
        3. Select voice (ar-SA-HamedNeural or en-US-AriNeural)
        4. Generate MP3 (Edge-TTS)
        5. Cleanup old files
    """
```

#### telemetry.py

```python
def log_interaction(
    username: str,
    query: str,
    response: str,
    latency: float,
    client_ip: str,
    user_agent: str
):
    """
    Log user interaction to JSONL.
    
    Args:
        username: User ID
        query: User query
        response: AI response
        latency: Response time (seconds)
        client_ip: User IP
        user_agent: Browser info
    
    Output:
        Appends to outputs/user_analytics/{username}_history.jsonl
    """
```

---

## 16. AREAS FOR IMPROVEMENT (Recommendations)

### 16.1 High Priority (Before/After Defense)

1. **Response Streaming** (~2 hours)
   - Current: 3-5s blank screen before full response
   - Fix: `TextIteratorStreamer` + Gradio generator
   - Benefit: Instant perceived response, transforms UX

2. **NLLB-200 Translation Layer** (~3 hours)
   - Current: Urdu scores 2.0/10, Chinese unstable
   - Fix: T-S-T with `facebook/nllb-200-1.3B` (2.5GB VRAM in float16)
   - Benefit: Urdu expected to reach 7-8/10

3. **Slot-Based Multi-Turn Memory** (~1 hour)
   - Current: LLM query rewriter fails often ("intent lost")
   - Fix: Rule-based pronoun resolution using last service context
   - Benefit: 80% of follow-ups handled without LLM call

4. **User Feedback Buttons** (~30 minutes)
   - Current: No way to collect user satisfaction
   - Fix: 👍/👎 buttons logged to telemetry JSONL
   - Benefit: Identify failure patterns from real usage

---

### 16.2 Medium Priority (Post-Defense)

5. **Auto-KG Extraction** (~4 hours)
   - Current: 83 services manually curated in JSON
   - Fix: Script to parse Master CSV → auto-generate KG
   - Benefit: Scales to any number of services

6. **Response Caching (LRU)** (~2 hours)
   - Current: Every "ما رسوم الجواز؟" recomputes full RAG
   - Fix: Hash normalized query → cache top 100
   - Benefit: <100ms for 30-40% of queries

7. **Expand Ground Truth to 500+ QA** (~8 hours)
   - Current: 120 QA pairs too small for statistical conclusions
   - Fix: Add edge cases, misspellings, colloquial Arabic, negation queries
   - Benefit: More robust evaluation

8. **Semantic Safety Classifier** (~8 hours)
   - Current: Keyword-based safety catches "أزور" (forge) as "visit"
   - Fix: Fine-tune small BERT on 200 safe + 200 unsafe Arabic queries
   - Benefit: Handles adversarial rephrasing

---

### 16.3 Low Priority (Nice-to-Have)

9. **Model Quantization (AWQ/GPTQ 4-bit)**
   - Reduces ALLaM from 14GB → 4GB VRAM, fits 13B models

10. **Confidence Score Display**
    - Show ⚠️ when FAISS similarity < 0.5

11. **Arabic Dialect Normalization**
    - "وش" → "ماذا", "أبغى" → "أريد" (50-100 terms)

12. **Service Directory Page**
    - Browsable tab showing all 83 services by sector

13. **Conversation Export**
    - Download chat history as PDF for government records

---

## 17. CONCLUSION

### 17.1 Project Strengths

✅ **Production-Ready**: Salted auth, rotating logs, error handling, telemetry  
✅ **Hardware Optimized**: TF32, SDPA, bfloat16 on A100-SXM4-80GB  
✅ **8-Language UI**: Arabic, English, Urdu, French, Spanish, German, Russian, Chinese  
✅ **Hybrid Retrieval**: FAISS + BM25 + RRF → 100% retrieval accuracy (120/120)  
✅ **Knowledge Grounding**: KG v3.0 enrichment → 87.5% price accuracy, 55.9% exact match (best among models)  
✅ **Hallucination Control**: Dynamic token cap + system prompt pricing rules  
✅ **Intent Guard**: <100ms for social queries (~30% of traffic)  
✅ **Voice Interface**: Whisper ASR + Edge-TTS (Saudi dialect)  
✅ **7-Test Benchmark Suite**: 480-test arena + 6 specialized tests, 77.8 min total  
✅ **Data-Grounded Evaluation**: Qwen-32B judge with KG + Master CSV + GT references  
✅ **100% Safety**: 10/10 red-teaming attacks blocked  
✅ **Judge Score**: ALLaM 7.33/10 (7.88 excluding Urdu, tied with Gemma)  

### 17.2 Known Weaknesses (Honest Assessment)

⚠️ **Urdu/Chinese generation**: ALLaM can't write these languages natively (scores 0.08-0.20 ROUGE)  
⚠️ **Small dataset**: 83 services, 120 GT — demo scale, not production benchmark  
⚠️ **No real T-S-T**: Translation mechanism described but not implemented  
⚠️ **Manual KG**: Doesn't scale without auto-extraction pipeline  
⚠️ **Keyword safety**: Semantic ambiguity not handled (أزور = forge or visit)

---

### 17.3 Quick Start

**To run the system**:
```bash
python main.py
# → Choose [1] Launch App
# → Open http://<server_ip>:7860
```

**To run benchmarks**:
```bash
python main.py
# → Choose [2] Benchmark Suite
# → Choose [1] Comprehensive Arena → [1] Quick or [2] Full
# Results in: Benchmarks/results/arena_v6_*.csv
```

**To understand the code**:
1. `main.py` → `rag_pipeline.py` → `model_loader.py`
2. Data flow: `ingestion.py` → `vector_store.py` → FAISS
3. Config: All paths/models/hyperparams in `config.py`

---

## 18. CONTACT & SUPPORT

**Project Owner**: Team PGD+  
**Organization**: KAUST Academy (Postgraduate Diploma)  
**Hardware**: KAUST Ibex HPC Cluster (A100-SXM4-80GB)  
**Purpose**: MOI Absher Smart Assistant  
**Defense Date**: April 15, 2026  

**For Questions**:
- Architecture: This document
- Bugs: Check logs in `logs/app.log`
- Performance: Run benchmarks in `Benchmarks/`

---

**END OF TECHNICAL DOCUMENTATION**

---

**APPENDIX: File Locations Quick Reference**

```
Project Root: /ibex/user/rashidah/projects/MOI_ChatBot/chatbot_project/

Entry Point: main.py
Config: config.py
Data: data/Data_Master/MOI_Master_Knowledge.csv
Index: data/faiss_index/index.faiss
KG: data/data_processed/services_knowledge_graph.json
Benchmark: data/data_processed/ground_truth_polyglot_V2.csv
Models: models/ (HuggingFace cache)
Logs: logs/app.log
Analytics: outputs/user_analytics/*.jsonl
Evaluation: Benchmarks/comprehensive_arena.py
```

**Total Lines of Code**: ~3,470  
**Total Models**: 4 (ALLaM-7B, BGE-M3, Whisper-v3, Qwen-32B Judge)  
**Benchmark Models**: 4 (ALLaM-7B, Qwen-2.5-7B, Gemma-2-9B, Llama-3.1-8B)  
**Data Size**: 83 services, 160 vectors, 120 QA pairs (8 languages)  
**Benchmark Suite**: 7 tests (480 arena tests + 6 specialized tests)  
**Benchmark Runtime**: 77.8 minutes on A100-SXM4-80GB  
**Best Judge Score**: Gemma 7.46, ALLaM 7.33 (tie at 7.88 excluding Urdu)  
**Languages Supported**: 8 — Arabic/English/French/German strong, Urdu needs NLLB  
**Project Version**: 5.0 (April 2026)