# Absher Smart Assistant

### 🏛️ Sovereign AI for Saudi MOI Services | Version 5.0

### 🏛️ Sovereign AI • 🔎 Hybrid RAG • 🤖 ALLaM Engine • 🌍 8-Language Support • 🎤 Speech-to-Speech

![Status](https://img.shields.io/badge/Status-Production_Ready-success?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
[![Model - ALLaM](https://img.shields.io/badge/Model-ALLaM--7B--Instruct-green?style=flat-square&logo=huggingface)](https://huggingface.co/ALLaM-AI/ALLaM-7B-Instruct-preview)
![Architecture](https://img.shields.io/badge/Architecture-Hybrid_RAG-purple?style=flat-square)
![Hardware](https://img.shields.io/badge/GPU-A100_Optimized-orange?style=flat-square&logo=nvidia)
[![ASR - Whisper](https://img.shields.io/badge/ASR-Whisper_Large--v3-blueviolet?style=flat-square)](https://huggingface.co/openai/whisper-large-v3)
![TTS](https://img.shields.io/badge/TTS-ar--SA--HamedNeural-red?style=flat-square)
![Languages](https://img.shields.io/badge/Languages-8-yellow?style=flat-square)
[![License](https://img.shields.io/badge/License-Academic_Non_Commercial-lightgrey?style=flat-square)](LICENSE)

---

## 📖 Overview

**Absher Smart Assistant** (مساعد أبشر الذكي) is a sovereign AI conversational system designed to democratize access to Saudi Ministry of Interior (MOI) services. The system employs a **Cross-Lingual Hybrid Retrieval-Augmented Generation (RAG)** architecture with Knowledge Graph enrichment to anchor generative capabilities to a curated, verified knowledge base of **83 MOI services** across **6 sectors**.

**Key Achievements (480-Test Full Benchmark):**

- **7.33/10** Judge Score (ALLaM-7B, evaluated by Qwen-32B)
- **100%** Retrieval Accuracy (120/120 queries)
- **100%** Safety Compliance (10/10 red-teaming attacks blocked)
- **96.7%** Source Attribution (highest among all tested models)
- **87.5%** Price Accuracy (KG-grounded, best among all models)
- **3.86s** Average Latency on A100-80GB
- **8 Languages** with strong Arabic/English/French/German performance
- **Zero errors** across all 480 benchmark tests

---

## ✨ Technical Features

### 🧠 Sovereign Saudi Intelligence (ALLaM-7B)

Powered by [ALLaM-7B-Instruct-preview](https://huggingface.co/ALLaM-AI/ALLaM-7B-Instruct-preview), developed by **SDAIA (Saudi Data & AI Authority)**.

- **Training:** Pretrained on **5.2 Trillion tokens** (4T English + 1.2T Arabic/English)
- **Optimization:** bfloat16 precision with TF32 matmul and SDPA attention on A100
- **VRAM Usage:** 14GB (bfloat16) — leaves 66GB free for embeddings, ASR, and TTS

### 🔍 Hybrid Retrieval with RRF Fusion

Synergizes dense vector retrieval (**BGE-M3**, 160 vectors @ 1024 dims) with sparse keyword matching (**BM25**) using **Reciprocal Rank Fusion**:

$$RRF(d) = \sum_{j \in \{Dense, Sparse\}} \frac{1}{k + r_j(d)}, \quad k=60$$

**Benchmark result:** 100% hit rate on 120 queries, average similarity 0.817.

### 📊 Knowledge Graph Enrichment v3.0

Verified prices and steps injected directly into LLM context from a curated JSON knowledge graph (83 services × 6 sectors):

- **OR-matching** — matches any word in a service name (not all words required)
- **Article stripping** — removes Arabic articles (ال) for better matching
- **3 facts injected per query** — consistently across all 480 benchmark tests
- **Result:** 87.5% price accuracy (ALLaM, best among all 4 models)

### 🎭 Intent Guard System

Bypasses RAG for social interactions with pattern-matched responses in 4 categories:

| Category | Examples | Response Time |
|:---|:---|:---:|
| **Greeting** | السلام عليكم, Hello, Bonjour | <100ms |
| **Closing** | شكرا, Thanks, مع السلامة | <100ms |
| **Abuse** | غبي, Stupid, احمق | <100ms |
| **Praise** | ممتاز, Great, رائع | <100ms |

Reduces average system latency by bypassing expensive retrieval + LLM for ~30% of queries.

### 🌍 8-Language UI Support

Arabic, English, Urdu, French, Spanish, German, Russian, Chinese — with:

- RTL/LTR auto-switching based on detected language
- WhatsApp-style input area with suggestion chips per language
- 6 responsive breakpoints (mobile → desktop)
- Dark/light theme toggle with iOS safe-area support

### 🛡️ Safety & Hallucination Control

- **10/10 safety score** against red-teaming attacks (politics, jailbreaks, harmful content)
- **Dynamic `max_new_tokens`** cap prevents runaway generation
- **System prompt** enforces KG-only pricing ("NEVER guess a price")
- **`TEMPERATURE=0.2`**, **`REPETITION_PENALTY=1.1`**
- **Keyword safety guard** catches harmful queries before retrieval

### 🎤 Speech-to-Speech Pipeline

- **ASR:** OpenAI Whisper-large-v3 (Arabic + multilingual)
- **TTS:** Edge-TTS with Saudi dialect voice (ar-SA-HamedNeural)
- **Auto-cleanup:** 10-minute TTL on generated audio files

### 🔐 Authentication & Telemetry

- **Salted SHA-256** password hashing with backward compatibility for legacy hashes
- **Per-user JSONL** telemetry logs (query, response, latency, IP)
- **Username sanitization** prevents path traversal attacks
- **Change password** + minimum 6-char enforcement via CLI auth manager

---

## 📊 Benchmark Results (v6.0 — Data-Grounded)

### Methodology

**480 tests** = 120 questions × 4 models, run on **NVIDIA A100-SXM4-80GB** (KAUST Ibex HPC).

**3-Phase Pipeline:**

1. **Phase 1 — Generation** (34.8 min): Each model answers 120 questions sequentially. Models are loaded/unloaded one at a time to manage VRAM. Objective metrics computed inline: ROUGE-L (LCS), per-service price extraction, attribution detection.

2. **Phase 2 — Judging** (42.5 min): Qwen-2.5-32B-Instruct scores all 480 answers on a 0-10 scale. The judge receives **data-grounded references**: KG prices/steps + Master CSV context + Ground Truth answers. Fairness rules: don't penalize extra information, verify prices only from KG data.

3. **Phase 3 — Reporting**: Automated leaderboard, per-language heatmap, per-category breakdown, failure analysis, CSV export.

**Total runtime:** 77.8 minutes, zero errors, zero retries.

---

### 🏆 Final Leaderboard

| Rank | Model | Judge Score | Std Dev | ROUGE-L | Price Acc | Attribution | Avg Latency |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **Gemma-2-9B** | **7.46** | 2.33 | **0.405** | 85.0% | 92.5% | 5.26s |
| 🥈 | **ALLaM-7B** ⭐ | **7.33** | 2.31 | 0.373 | **87.5%** | **96.7%** | 3.86s |
| 🥉 | **Qwen-2.5-7B** | **7.27** | **2.09** | 0.393 | 83.3% | 92.5% | 4.37s |
| 4 | **Llama-3.1-8B** | **6.30** | 3.45 | 0.358 | 82.3% | 77.5% | **3.52s** |

⭐ **ALLaM-7B is the production model** — best price accuracy (87.5%), best attribution (96.7%), and 27% faster than Gemma. When excluding Urdu (where all models fail), ALLaM and Gemma tie at **7.88**.

---

### 🌍 Per-Language Judge Scores (0-10)

| Model | Arabic | English | French | German | Chinese | Russian | Spanish | Urdu |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALLaM-7B** ⭐ | 7.9 | 7.7 | **8.1** | 7.9 | **8.1** | 7.7 | **7.8** | 3.5 |
| **Gemma-2-9B** | **8.5** | **7.9** | 8.0 | **8.1** | 7.7 | **7.8** | 7.1 | 4.5 |
| **Qwen-2.5-7B** | 8.1 | 7.4 | 7.3 | 7.2 | 7.1 | 7.7 | 7.7 | **5.7** |
| **Llama-3.1-8B** | 7.9 | 7.4 | 7.5 | 7.8 | 7.4 | 6.3 | ⛔ 0.6 | 5.5 |

**Key observations:**

- **Gemma dominates Arabic** (8.5) — best single-language score
- **ALLaM is the most consistent** across 7 languages (7.7–8.1 range, excluding Urdu)
- **Llama refuses Spanish entirely** — 14/15 Spanish queries scored 0 ("Lo siento, no puedo ayudar")
- **All models fail Urdu** (3.5–5.7) — confirms need for NLLB-200 translation layer
- **Qwen has lowest variance** (std=2.09) — most predictable behavior

---

### 🌍 Per-Language ROUGE-L Scores

| Model | Arabic | English | French | German | Chinese | Russian | Spanish | Urdu |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALLaM-7B** | 0.448 | 0.412 | 0.408 | 0.437 | 0.414 | 0.381 | 0.386 | 0.099 |
| **Gemma-2-9B** | **0.606** | 0.310 | **0.466** | 0.408 | 0.433 | **0.432** | 0.385 | **0.197** |
| **Qwen-2.5-7B** | 0.515 | 0.473 | 0.399 | **0.453** | **0.451** | 0.368 | 0.401 | 0.084 |
| **Llama-3.1-8B** | 0.477 | **0.553** | 0.466 | 0.454 | 0.450 | 0.307 | 0.029 | 0.131 |

---

### 📂 Per-Category Judge Scores (by Sector)

| Model | الجوازات | الأحوال المدنية | المرور | وزارة الداخلية | الأمن العام | المديرية العامة للسجون |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALLaM-7B** ⭐ | **7.6** | 6.9 | **7.6** | 7.2 | 7.0 | **7.5** |
| **Gemma-2-9B** | 7.6 | **7.8** | 7.4 | 6.9 | 6.8 | 7.4 |
| **Qwen-2.5-7B** | 7.0 | 7.6 | **7.9** | 7.1 | 6.4 | 6.2 |
| **Llama-3.1-8B** | 6.3 | 6.2 | 7.1 | 4.9 | **7.0** | 6.4 |

ALLaM is the most balanced across all 6 sectors (6.9–7.6 range).

---

### 📊 Score Distribution (Quality Tiers)

| Model | ⛔ Zero (0) | 🔴 Low (<5) | 🟡 Mid (5–7.9) | 🟢 High (8+) | High Rate |
|:---|:---:|:---:|:---:|:---:|:---:|
| **ALLaM-7B** ⭐ | 4 | 12 | 35 | **73** | **61%** |
| **Gemma-2-9B** | 6 | 11 | 30 | **79** | **66%** |
| **Qwen-2.5-7B** | 4 | 9 | **48** | 63 | 52% |
| **Llama-3.1-8B** | **22** | **28** | 27 | 65 | 54% |

- **Gemma has the most 8+ scores** (79/120) but also 6 zero-scores
- **ALLaM has fewest failures** — only 4 zero-scores (all Urdu)
- **Llama has 22 zero-scores** — catastrophic for production use

---

### ⛔ Failure Analysis (36 Zero-Score Answers)

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

**Llama accounts for 22 of 36 zero-scores (61%).** Spanish alone contributes 14.

---

### 💰 Price Accuracy (136 Price-Related Questions)

| Model | Correct | Total | Accuracy |
|:---|:---:|:---:|:---:|
| **ALLaM-7B** ⭐ | 19 | 34 | **55.9%** |
| **Gemma-2-9B** | 16 | 34 | 47.1% |
| **Qwen-2.5-7B** | 14 | 34 | 41.2% |
| **Llama-3.1-8B** | 12 | 34 | 37.5% |

ALLaM's KG enrichment advantage is clear — 50% better than Llama on price extraction.

---

### ⏱️ Latency Profile

| Model | Mean | P50 | P95 | Max |
|:---|:---:|:---:|:---:|:---:|
| **ALLaM-7B** | 3.86s | 3.0s | 10.5s | 12.0s |
| **Qwen-2.5-7B** | 4.37s | 3.7s | 9.4s | 11.1s |
| **Gemma-2-9B** | 5.26s | 4.7s | 11.0s | 16.2s |
| **Llama-3.1-8B** | 3.52s | 3.0s | 10.2s | 14.0s |

Urdu/Chinese queries spike to 9–12s due to longer generation in non-native scripts. Intent Guard catches return in 0.00s (no LLM call).

---

### 📋 Complete Benchmark Suite

| Benchmark | Score | Details |
|:---|:---:|:---|
| **Comprehensive Arena** | 7.33/10 | 480 tests, 4 models × 120 Q, Qwen-32B judge |
| **Retrieval Accuracy** | **100%** | 120/120 queries, avg similarity 0.817 |
| **Safety & Red Teaming** | **100%** | 10/10 attacks blocked |
| **Functional Tests** | **88%** | 7/8 passed (1 Arabic ambiguity edge case) |
| **Context Memory** | **100%** | 5 multi-turn scenarios, all follow-ups correct |
| **Stress Test** | **100%** | 20/20 requests, 4 concurrent users, 0 errors |

---

### 🔑 Why ALLaM-7B is the Production Choice

| Criterion | ALLaM-7B | Gemma-2-9B | Winner |
|:---|:---:|:---:|:---|
| **Judge Score (excl. Urdu)** | 7.88 | 7.88 | **Tie** |
| **Price Accuracy** | 87.5% | 85.0% | **ALLaM** |
| **Source Attribution** | 96.7% | 92.5% | **ALLaM** |
| **Latency** | 3.86s | 5.26s | **ALLaM (27% faster)** |
| **Zero-Score Failures** | 4 | 6 | **ALLaM** |
| **Consistency** | std 2.31 | std 2.33 | **Tie** |
| **Saudi Sovereignty** | SDAIA | Google | **ALLaM** |

---

## 🧠 RAG Intelligence Flow (v5.0 — 11 Steps)

```
 1. Intent Guard → Social (greeting/closing/abuse/praise)? Return canned response (<100ms)
 2. Lang Detect → ar | en | fr | es | de | ru | zh | ur
 3. Query Rewrite → Resolve pronouns from conversation history
 4. Normalize → Arabic char unification (Alef/Taa/Kashida), diacritics removal
 5. FAISS Retrieve → Top-5 semantic matches (BGE-M3, 160 vectors, cosine similarity)
 6. BM25 Retrieve → Top-5 keyword matches (master_chunks.csv, TF-IDF scoring)
 7. RRF Fusion → Merge + rerank with k=60 → Final Top-5 documents
 8. KG Enrich v3.0 → OR-match services, strip articles, inject prices/steps
 9. Dynamic Token Cap → Adjust max_new_tokens based on context length
10. Generate → ALLaM-7B (bfloat16, temp=0.2, rep_penalty=1.1, SDPA attention)
11. TTS → Optional voice output (ar-SA-HamedNeural via Edge-TTS)
```

---

## 📂 Project Structure

```
chatbot_project/
├── main.py                    # Command Center (App + 7 Benchmarks + Auth + FAISS)
├── config.py                  # Central configuration & system prompt
├── README.md                  # This document
├── requirements.txt           # ~25 Python dependencies
├── LICENSE                    # Academic & Non-Commercial License
├── .gitignore                 # Excludes: models/, users_db.json, *.svg, analytics/
├── .env                       # HF_TOKEN — NOT IN GIT
│
├── core/                      # RAG Intelligence Layer
│   ├── rag_pipeline.py        # Main RAG orchestrator v5.0 ⭐ (~450 LoC)
│   ├── model_loader.py        # Singleton model manager + VRAM tracking (~250 LoC)
│   └── vector_store.py        # FAISS build/load + vector count logging (~120 LoC)
│
├── data/                      # Data Layer
│   ├── ingestion.py           # CSV → 160 chunks ETL (400 chars, 50 overlap) (~180 LoC)
│   ├── schema.py              # Data validation + duplicate detection (~90 LoC)
│   ├── Data_Master/           # MOI_Master_Knowledge.csv (83 services, 6 sectors)
│   ├── Data_Chunk/            # master_chunks.csv (~160 BM25 chunks)
│   ├── data_processed/        # KG JSON + Ground Truth V2 (120 QA, 8 langs)
│   └── faiss_index/           # Auto-generated: index.faiss + index.pkl
│
├── ui/                        # User Interface Layer
│   ├── app.py                 # Gradio v5.0 — 8 languages, suggestion chips (~250 LoC)
│   ├── theme.py               # CSS/JS v4.0 — RTL/LTR, 6 breakpoints (~200 LoC)
│   └── assets/                # saudi_emblem.svg, KAUST.png, moi_logo.png
│
├── utils/                     # Utilities
│   ├── auth_manager.py        # Salted SHA-256 auth + change password (~170 LoC)
│   ├── logger.py              # Color-coded rotating logger (~80 LoC)
│   ├── telemetry.py           # Per-user JSONL analytics + sanitization (~60 LoC)
│   ├── text_utils.py          # Arabic normalization (C-level str.maketrans) (~100 LoC)
│   └── tts.py                 # Edge-TTS (Saudi + English voices) (~100 LoC)
│
├── models/                    # HuggingFace Cache (~85 GB total)
│   ├── ALLaM-7B (~14 GB)     # Production LLM
│   ├── BGE-M3 (~2 GB)        # Embeddings
│   ├── Whisper-v3 (~3 GB)    # ASR
│   └── Qwen-32B (~65 GB)     # Benchmark judge only
│
└── Benchmarks/                # 7-Test Evaluation Framework
    ├── comprehensive_arena.py # Multi-model arena v6.0 (~500 LoC)
    ├── functional_test.py     # KG prices + context + safety (~150 LoC)
    ├── retrieval_test.py      # Semantic similarity accuracy (~100 LoC)
    ├── safety_test.py         # 10 red-teaming attacks (~120 LoC)
    ├── stress_test.py         # 4-user concurrent load (~100 LoC)
    ├── context_test.py        # 5 multi-turn scenarios (~100 LoC)
    └── results/               # checkpoint_phase1_*.csv + arena_v6_*.csv
```

**Total: ~3,470 Lines of Code** across 28 Python files.

---

## 🛠️ Installation & Execution

### Prerequisites

- **Hardware:** NVIDIA GPU with 20GB+ VRAM (A100/H100 recommended)
- **Software:** Python 3.9+, CUDA 12.x, PyTorch 2.2+

### Setup

```bash
git clone https://github.com/Ahmed-alrashidi/MOI_ChatBot.git
cd MOI_ChatBot/chatbot_project
conda create -n absher python=3.9 && conda activate absher
pip install -r requirements.txt --break-system-packages
echo "HF_TOKEN=hf_your_token" > .env
python utils/auth_manager.py   # Add admin user
python main.py                 # Launch Command Center
```

---

## 📈 Known Limitations & Roadmap

### Current Limitations

| Issue | Severity | Details |
|:---|:---:|:---|
| **Urdu generation** | 🔴 | All models score 3.5–5.7/10 |
| **No streaming** | 🔴 | 3–5s blank screen before response |
| **Small dataset** | 🟡 | 83 services, 120 GT — demo scale |
| **Manual KG** | 🟡 | JSON hand-curated, doesn't auto-update |
| **Single-turn RAG** | 🟡 | Multi-turn rewriter often fails |

### Priority Improvements

| # | Improvement | Effort | Impact |
|:---:|:---|:---:|:---|
| 1 | Response Streaming | 2h | Instant UX |
| 2 | Feedback Buttons 👍👎 | 30min | User satisfaction data |
| 3 | Slot-Based Memory | 1h | Fix multi-turn |
| 4 | NLLB-200 Translation | 3h | Urdu 3.5 → 7+ |
| 5 | Response Caching | 2h | <100ms for 30% queries |

---

## 📜 Credits & Citations

### Model Acknowledgment

This project utilizes the **ALLaM** model series by **SDAIA (Saudi Data & AI Authority)**.

```bibtex
@inproceedings{
    bari2025allam,
    title={{ALL}aM: Large Language Models for Arabic and English},
    author={M Saiful Bari and Yazeed Alnumay and others},
    booktitle={ICLR 2025},
    year={2025},
    url={https://openreview.net/forum?id=MscdsFVZrN}
}
```

---

## 📄 Academic Context

Developed as a capstone project for the **AI Master (PGD+ Team)** program at
**King Abdullah University of Science and Technology (KAUST) — 2026**

**Team:** PGD+
**Version:** 5.0 (Production Release)
**Defense Date:** April 15, 2026
**Last Updated:** April 2026
