# 📂 ABSHER CHATBOT v5.0 — COMPLETE FILE TREE

**Visual Map of Every File in the Project**

---

## 🌳 DIRECTORY TREE

```
/ibex/user/rashidah/projects/MOI_ChatBot/chatbot_project/
│
├── 📄 main.py                          [Command Center — App + Benchmarks + Auth + FAISS]
├── 📄 config.py                        [Central configuration (TEMP=0.2, REP_PENALTY=1.1)]
├── 📄 README.md                        [Project README v5.0]
├── 📄 requirements.txt                 [Python dependencies (~25 packages)]
├── 📄 LICENSE                          [Academic & Non-Commercial License]
├── 📄 .gitignore                       [VCS exclusions (*.svg, users_db.json)]
├── 📄 .env                             [Environment variables — NOT IN GIT]
│
├── 📁 core/                            [RAG Intelligence Layer]
│   ├── 📄 __init__.py
│   ├── 📄 model_loader.py              [Singleton model manager + VRAM logging]
│   ├── 📄 vector_store.py              [FAISS operations + vector count logging]
│   └── 📄 rag_pipeline.py              [Main RAG orchestrator v5.0 ⭐ CRITICAL]
│
├── 📁 data/                            [Data Layer]
│   ├── 📄 __init__.py
│   ├── 📄 ingestion.py                 [CSV → 160 chunks ETL (400 chars, 50 overlap)]
│   ├── 📄 schema.py                    [Data validation + duplicate detection]
│   │
│   ├── 📁 Data_Master/                 [Source of Truth]
│   │   └── 📄 MOI_Master_Knowledge.csv [83 services, 9 columns, 6 sectors]
│   │
│   ├── 📁 Data_Chunk/                  [Processed Chunks for BM25]
│   │   └── 📄 master_chunks.csv        [~160 chunks, 400 chars each]
│   │
│   ├── 📁 data_processed/              [Curated Assets]
│   │   ├── 📄 services_knowledge_graph.json  [Verified prices/steps, 6 sectors]
│   │   └── 📄 ground_truth_polyglot_V2.csv   [120 QA pairs, 8 languages]
│   │
│   └── 📁 faiss_index/                 [Vector Database — AUTO-GENERATED]
│       ├── 📄 index.faiss              [FAISS binary index]
│       └── 📄 index.pkl                [Docstore pickle]
│
├── 📁 ui/                              [User Interface Layer]
│   ├── 📄 __init__.py
│   ├── 📄 app.py                       [Gradio interface v5.0 (8 languages)]
│   ├── 📄 theme.py                     [CSS/JS/RTL-LTR v4.0 (6 breakpoints)]
│   │
│   └── 📁 assets/                      [Static Files]
│       ├── 🖼️ saudi_emblem.svg         [Saudi emblem (welcome screen)]
│       ├── 🖼️ moi_logo.png            [Ministry logo]
│       └── 🖼️ KAUST.png               [KAUST Academy logo]
│
├── 📁 utils/                           [Utility Functions]
│   ├── 📄 __init__.py
│   ├── 📄 logger.py                    [Colored rotating logger (30-day)]
│   ├── 📄 telemetry.py                 [Per-user JSONL analytics + sanitization]
│   ├── 📄 text_utils.py                [Arabic normalization (C-level str.maketrans)]
│   ├── 📄 tts.py                       [Text-to-speech (Edge-TTS, Saudi + EN voices)]
│   └── 📄 auth_manager.py              [Salted SHA-256 auth + change password]
│
├── 📁 users/                           [Authentication Database — NOT IN GIT]
│   └── 📄 users_db.json                [Salted SHA-256 hashed passwords]
│
├── 📁 models/                          [HuggingFace Model Cache — LARGE]
│   ├── 📁 ALLaM-AI--ALLaM-7B-Instruct-preview/
│   ├── 📁 BAAI--bge-m3/
│   ├── 📁 openai--whisper-large-v3/
│   └── 📁 Qwen--Qwen2.5-32B-Instruct/
│
├── 📁 logs/                            [System Logs — AUTO-GENERATED]
│   └── 📄 app.log                      [Rotating 30-day logs]
│
├── 📁 outputs/                         [Generated Content]
│   ├── 📁 audio/                       [TTS MP3 files — 10min TTL]
│   └── 📁 user_analytics/              [Per-user JSONL telemetry]
│
└── 📁 Benchmarks/                      [7-Test Evaluation Framework]
    ├── 📄 comprehensive_arena.py       [Multi-model arena v6.0 (data-grounded judge)]
    ├── 📄 functional_test.py           [KG prices + context + safety + attribution]
    ├── 📄 retrieval_test.py            [Semantic similarity retrieval accuracy]
    ├── 📄 safety_test.py               [10 red-teaming attack scenarios]
    ├── 📄 stress_test.py               [4-user concurrent load testing]
    ├── 📄 context_test.py              [5 multi-turn conversation scenarios]
    │
    ├── 📁 data/                        [Benchmark Datasets]
    │   └── 📄 ground_truth_polyglot_V2.csv  [Symlink to data_processed/]
    │
    └── 📁 results/                     [Benchmark Outputs — AUTO-GENERATED]
        ├── 📄 checkpoint_phase1_*.csv  [Phase 1 raw answers (480 rows)]
        ├── 📄 arena_v6_*.csv           [Final judged results (480 rows)]
        └── 📄 summary_*.txt            [Leaderboard + per-language tables]
```

---

## 📊 FILE COUNT SUMMARY

```
Total Directories: 16
Total Code Files: 28
Total Data Files: 5
Total Config Files: 5
Total Assets: 3

Breakdown:
├─ Python Code: 23 files (core: 3, data: 3, ui: 2, utils: 5, benchmarks: 7, main+config: 2, __init__: 4)
├─ CSV Data: 3 files
├─ JSON Data: 2 files
├─ Config: 5 files (.env, .gitignore, requirements.txt, config.py, LICENSE)
├─ Images: 3 files (svg, png, png)
├─ Markdown: 1 file (README.md)
└─ Generated: ~10-20 files (logs, audio, benchmark results)
```

---

## 🎯 FILE IMPORTANCE MATRIX

### 🔴 CRITICAL (Must Read)

| File | LoC | Purpose | Read Priority |
|------|-----|---------|---------------|
| `core/rag_pipeline.py` | ~450 | Main RAG logic — intent guard, KG enrichment, hallucination control | 1️⃣ |
| `core/model_loader.py` | ~250 | Singleton model manager with VRAM tracking | 2️⃣ |
| `config.py` | ~130 | All settings, paths, hyperparameters, system prompt | 3️⃣ |
| `main.py` | ~220 | Command Center — 5-option menu + 7-benchmark submenu | 4️⃣ |

### 🟡 IMPORTANT (Understand)

| File | LoC | Purpose |
|------|-----|---------|
| `data/ingestion.py` | ~180 | CSV → 160 chunks ETL (400 chars, 50 overlap) |
| `core/vector_store.py` | ~120 | FAISS build/load with vector count logging |
| `ui/app.py` | ~250 | Gradio v5.0 — 8 languages, suggestion chips, telemetry safety |
| `utils/text_utils.py` | ~100 | Arabic normalization (Alef/Taa unification) |
| `Benchmarks/comprehensive_arena.py` | ~500 | Data-grounded arena v6.0 with Qwen-32B judge |

### 🟢 REFERENCE (As Needed)

| File | LoC | Purpose |
|------|-----|---------|
| `data/schema.py` | ~90 | Data validation + duplicate Service_Name detection |
| `utils/tts.py` | ~100 | Edge-TTS (ar-SA-HamedNeural + en-US) |
| `utils/telemetry.py` | ~60 | JSONL analytics + path traversal prevention |
| `utils/logger.py` | ~80 | Color-coded rotating logger |
| `utils/auth_manager.py` | ~170 | Salted SHA-256, backward compat, change password |
| `ui/theme.py` | ~200 | CSS/JS — RTL/LTR, 6 breakpoints, iOS safe-area |
| `Benchmarks/functional_test.py` | ~150 | KG prices, context memory, safety, greeting |
| `Benchmarks/retrieval_test.py` | ~100 | 120-query semantic similarity test |
| `Benchmarks/safety_test.py` | ~120 | 10 red-teaming attack vectors |
| `Benchmarks/stress_test.py` | ~100 | Concurrent user load testing |
| `Benchmarks/context_test.py` | ~100 | Multi-turn Arabic conversation scenarios |

---

## 📦 DATA FILE DETAILS

### Source Data
```
data/Data_Master/MOI_Master_Knowledge.csv
├─ Size: ~180 KB
├─ Rows: 83
├─ Columns: 9
├─ Sectors: 6 (normalized: الجوازات، الأحوال المدنية، المرور، 
│              وزارة الداخلية، الأمن العام، المديرية العامة للسجون)
├─ Format: UTF-8 CSV
├─ RAG_Content: Enriched (avg 456 chars per service)
└─ Content: Government services (Arabic/English mix)
```

### Processed Chunks
```
data/Data_Chunk/master_chunks.csv
├─ Size: ~100 KB
├─ Rows: ~160
├─ Chunk Size: 400 chars (was 800)
├─ Overlap: 50 chars (was 100)
├─ Format: UTF-8 CSV
└─ Content: Chunked text for BM25 keyword search
```

### Knowledge Graph
```
data/data_processed/services_knowledge_graph.json
├─ Size: ~50 KB
├─ Format: JSON (3-level hierarchy)
├─ Structure: {sector: {service: {price, steps}}}
├─ Sectors: 6 (normalized, matching Master CSV)
└─ Content: Verified hard facts — injected into LLM context
```

### Benchmark Dataset
```
data/data_processed/ground_truth_polyglot_V2.csv
├─ Size: ~100 KB
├─ Rows: 120
├─ Columns: 4 (category, lang, question, ground_truth)
├─ Languages: 8 (Arabic, English, French, Spanish, German, Russian, Chinese, Urdu)
├─ Distribution: 15 questions × 8 languages
└─ Content: QA pairs for evaluation across 6 service categories
```

---

## 🗂️ GENERATED FILES (Not in Git)

### FAISS Index
```
data/faiss_index/
├─ index.faiss     [~5 MB, 160 vectors @ 1024 dims]
└─ index.pkl       [~200 KB, docstore metadata]
```

### Model Cache (Large!)
```
models/
├─ ALLaM-7B:       ~14 GB (bfloat16)        [Production LLM]
├─ BGE-M3:         ~2 GB (float32)           [Embeddings]
├─ Whisper-v3:     ~3 GB (float16)           [ASR — voice input]
└─ Qwen-32B:       ~65 GB (bfloat16)         [Judge only — benchmarks]
```

### Benchmark Results
```
Benchmarks/results/
├─ checkpoint_phase1_*.csv    [480 rows — raw model answers + ROUGE-L + price accuracy]
├─ arena_v6_*.csv             [480 rows — final judged scores (0-10) + reasons]
├─ summary_*.txt              [Leaderboard + per-language + per-category tables]
├─ Columns: Model, Question, GT, Answer, Lang, Category,
│           Latency, ROUGE_L, Price_Score, Attribution,
│           Judge_Score, Reason
├─ Size: ~2 MB per full run
└─ Latest run: 480 tests, 77.8 min, zero errors
    ├─ Gemma-2-9B:   Judge 7.46 | ROUGE 0.405 | Price 85.0%
    ├─ ALLaM-7B ⭐:  Judge 7.33 | ROUGE 0.373 | Price 87.5%
    ├─ Qwen-2.5-7B:  Judge 7.27 | ROUGE 0.393 | Price 83.3%
    └─ Llama-3.1-8B: Judge 6.30 | ROUGE 0.358 | Price 82.3%
```

---

## 🔍 FILE DEPENDENCIES GRAPH

```
main.py
  ├─→ config.py
  ├─→ utils/logger.py
  ├─→ core/model_loader.py
  │    └─→ config.py
  ├─→ core/vector_store.py
  │    ├─→ config.py
  │    └─→ utils/logger.py
  ├─→ core/rag_pipeline.py
  │    ├─→ config.py
  │    ├─→ core/model_loader.py
  │    ├─→ core/vector_store.py
  │    ├─→ utils/logger.py
  │    └─→ utils/text_utils.py
  ├─→ data/ingestion.py
  │    ├─→ config.py
  │    ├─→ utils/logger.py
  │    └─→ data/schema.py
  ├─→ utils/auth_manager.py
  ├─→ Benchmarks/*.py           [7 benchmark scripts]
  └─→ ui/app.py
       ├─→ config.py
       ├─→ core/model_loader.py
       ├─→ core/rag_pipeline.py
       ├─→ ui/theme.py
       ├─→ utils/logger.py
       ├─→ utils/tts.py
       └─→ utils/telemetry.py
```

---

## 🚀 EXECUTION FLOW

### Command Center Menu
```
main.py → display_menu()
  ├─→ [1] Launch App      → ui/app.py
  ├─→ [2] Benchmark Suite  → 7 benchmark options
  │    ├─→ [1] Comprehensive Arena (Quick/Full)
  │    ├─→ [2] Functional Test
  │    ├─→ [3] Retrieval Test
  │    ├─→ [4] Safety Test
  │    ├─→ [5] Stress Test
  │    ├─→ [6] Context Test
  │    └─→ [7] Run All
  ├─→ [3] Auth Manager    → utils/auth_manager.py
  ├─→ [4] Rebuild FAISS   → core/vector_store.py
  └─→ [5] Exit
```

### Query Processing Flow
```
ui/app.py → chat_pipeline()
  ↓
core/rag_pipeline.py → run()
  ├─→ Intent Guard (greeting/closing/abuse/praise → canned response)
  ├─→ Language Detection
  ├─→ Query Rewrite (multi-turn pronoun resolution)
  ├─→ utils/text_utils.py → normalize_arabic()
  ├─→ core/vector_store.py → FAISS retrieve (Top-5)
  ├─→ BM25 retrieve (Top-5)
  ├─→ RRF Fusion → Final Top-5
  ├─→ KG Enrichment v3.0 (OR matching, article stripping)
  ├─→ Dynamic max_new_tokens cap
  ├─→ core/model_loader.py → LLM generate (ALLaM-7B, bfloat16)
  └─→ utils/tts.py → generate_speech() [optional]
  ↓
utils/telemetry.py → log_interaction()
```

---

## 📏 CODE METRICS

### Lines of Code
```
Core Logic:
├─ rag_pipeline.py:     ~450 LoC  (v5.0 — intent guard, KG v3.0, hallucination ctrl)
├─ model_loader.py:     ~250 LoC  (VRAM logging)
├─ vector_store.py:     ~120 LoC  (vector count logging)
└─ Total Core:          ~820 LoC

Data Pipeline:
├─ ingestion.py:        ~180 LoC  (chunk_size=400, overlap=50)
├─ schema.py:           ~90 LoC   (duplicate detection)
└─ Total Data:          ~270 LoC

UI Layer:
├─ app.py:              ~250 LoC  (v5.0 — 8 langs, SUPPORTED_LANGUAGES const)
├─ theme.py:            ~200 LoC  (v4.0 — RTL/LTR, 6 breakpoints, iOS safe-area)
└─ Total UI:            ~450 LoC

Utilities:
├─ text_utils.py:       ~100 LoC
├─ tts.py:              ~100 LoC
├─ logger.py:           ~80 LoC
├─ telemetry.py:        ~60 LoC   (path traversal prevention)
├─ auth_manager.py:     ~170 LoC  (salted SHA-256, change password)
└─ Total Utils:         ~510 LoC

Main & Config:
├─ main.py:             ~220 LoC  (command center + benchmark submenu)
├─ config.py:           ~130 LoC  (TEMPERATURE, REPETITION_PENALTY, BENCHMARK_DATA_DIR)
└─ Total:               ~350 LoC

Benchmarks:
├─ comprehensive_arena.py: ~500 LoC  (v6.0 — data-grounded, pre-built refs, checkpoint)
├─ functional_test.py:     ~150 LoC  (v2.0)
├─ retrieval_test.py:      ~100 LoC  (v2.0)
├─ safety_test.py:         ~120 LoC  (v2.0)
├─ stress_test.py:         ~100 LoC  (v2.0)
├─ context_test.py:        ~100 LoC  (v2.0)
└─ Total Benchmarks:       ~1,070 LoC

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL PROJECT:          ~3,470 LoC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 🔐 SENSITIVE FILES (Never Commit)

```
⛔ .env                       [HF_TOKEN]
⛔ users/users_db.json        [Salted password hashes]
⛔ outputs/user_analytics/*   [User queries/IPs — JSONL]
⛔ logs/app.log               [May contain PII]
⛔ models/*                   [~85 GB, use HF cache]
⛔ *.svg                      [Generated SVG assets]
```

---

## 🎯 WHERE TO START (By Goal)

### Goal: Understand RAG Logic
```
1. core/rag_pipeline.py      [v5.0 — intent guard, KG enrichment, hallucination ctrl]
2. core/model_loader.py      [Model management + VRAM tracking]
3. core/vector_store.py      [FAISS retrieval]
4. utils/text_utils.py       [Arabic normalization]
```

### Goal: Improve Data Quality
```
1. data/ingestion.py         [ETL — chunk_size=400, overlap=50]
2. data/schema.py            [Validation + duplicate detection]
3. data/Data_Master/*.csv    [83 services, enriched RAG_Content]
4. data/data_processed/*.json [KG — 6 normalized sectors]
```

### Goal: Enhance UI/UX
```
1. ui/app.py                 [v5.0 — 8 languages, suggestion chips]
2. ui/theme.py               [v4.0 — RTL/LTR, responsive, dark/light]
3. utils/tts.py              [Saudi + English voice output]
4. utils/auth_manager.py     [Salted auth + change password]
```

### Goal: Run Benchmarks
```
1. main.py                   [Command Center → option 2 → 7 benchmarks]
2. Benchmarks/comprehensive_arena.py  [v6.0 — data-grounded judge, 77.8 min full run]
3. data/data_processed/ground_truth_polyglot_V2.csv [120 QA × 8 langs]
4. Benchmarks/results/*.csv  [Checkpoint + judged results + summary]
   Latest: ALLaM 7.33/10, Gemma 7.46/10, Qwen 7.27/10, Llama 6.30/10
```

### Goal: Deploy to Production
```
1. main.py                   [Entry point]
2. config.py                 [Settings — TEMPERATURE=0.2, REP_PENALTY=1.1]
3. requirements.txt          [Dependencies]
4. .env                      [HF_TOKEN]
5. users/users_db.json       [Auth DB — salted SHA-256]
```

---

## 📋 CHANGELOG (v4.0 → v5.0)

| What Changed | Old (v4.0) | New (v5.0) |
|---|---|---|
| Services | 85 | 83 (normalized) |
| Sectors | 8 | 6 (normalized) |
| Chunk size | 800 chars | 400 chars |
| Chunk overlap | 100 | 50 |
| Vectors | ~255 | 160 |
| RAG_Content avg | 142 chars | 456 chars (enriched) |
| Languages (UI) | 3 | 8 |
| Ground Truth | 122 QA (3 langs) | 120 QA (8 langs) |
| Benchmark scripts | 1 | 7 |
| Auth | Plain SHA-256 | Salted SHA-256 + change password |
| Config | Hardcoded temp | TEMPERATURE=0.2, REPETITION_PENALTY=1.1 |
| Total LoC | ~2,400 | ~3,470 |
| rag_pipeline.py | v3.0 | v5.0 (intent guard, KG v3.0, hallucination ctrl) |
| theme.py | v2.0 | v4.0 (RTL/LTR, 6 breakpoints, iOS) |
| app.py | v3.0 | v5.0 (8 langs, telemetry safety) |

---

**Last Updated**: April 2026
**Project Version**: 5.0
**Total Files Documented**: 28 code files + 5 data files + 3 assets
**Total Size (without models)**: ~500 MB
**Total Size (with models)**: ~85 GB