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
[![License](https://img.shields.io/badge/License-MIT_Academic-lightgrey?style=flat-square)](LICENSE)

---

## 📖 Overview

**Absher Smart Assistant** (مساعد أبشر الذكي) is a sovereign AI conversational system designed to democratize access to Saudi Ministry of Interior (MOI) services. The system employs a **Cross-Lingual Hybrid Retrieval-Augmented Generation (RAG)** architecture with Knowledge Graph enrichment to anchor generative capabilities to a curated, verified knowledge base of 83 MOI services across 6 sectors.

**Key Achievement:** 100% retrieval accuracy, 100% safety compliance, and 87.5% price accuracy across 8 languages on NVIDIA A100-80GB.

---

## ✨ Technical Features

### 🧠 Sovereign Saudi Intelligence (ALLaM-7B)
Powered by [ALLaM-7B-Instruct-preview](https://huggingface.co/ALLaM-AI/ALLaM-7B-Instruct-preview), developed by **SDAIA**.
* **Training:** Pretrained on **5.2 Trillion tokens** (4T English + 1.2T Arabic/English).
* **Optimization:** bfloat16 precision with TF32 matmul and SDPA attention on A100.

### 🔍 Hybrid Retrieval with RRF Fusion
Synergizes dense vector retrieval (**BGE-M3**, 160 vectors) with sparse keyword matching (**BM25**) using **Reciprocal Rank Fusion**:

$$RRF(d) = \sum_{j \in \{Dense, Sparse\}} \frac{1}{k + r_j(d)}, \quad k=60$$

### 📊 Knowledge Graph Enrichment
Verified prices and steps injected directly into LLM context from a curated JSON knowledge graph (83 services × 6 sectors), ensuring **87.5–100% price accuracy**.

### 🎭 Intent Guard System
Bypasses RAG for social interactions (greetings, closings, praise, abuse) with pattern-matched responses, reducing latency to **<100ms** for ~30% of queries.

### 🌍 8-Language Support
Arabic, English, Urdu, French, Spanish, German, Russian, Chinese — with RTL/LTR auto-switching UI.

### 🛡️ Safety & Hallucination Control
* **10/10 safety score** against red-teaming attacks (politics, jailbreaks, harmful content)
* Dynamic `max_new_tokens` cap prevents runaway generation
* System prompt enforces KG-only pricing ("NEVER guess a price")

---

## 📊 Benchmark Results (v6.0 — Data-Grounded)

Tested on **NVIDIA A100-SXM4-80GB** with Qwen-2.5-32B as LLM judge.

### 🏆 Model Arena (4 Models × 8 Languages)

| Model | Judge Score | ROUGE-L | Price Accuracy | Attribution | Latency |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Gemma-2-9B** | **8.63** | **0.537** | **100%** | 100% | 6.9s |
| **ALLaM-7B** ⭐ | **8.25** | 0.514 | 87.5% | 100% | **4.1s** |
| Qwen-2.5-7B | 7.50 | 0.420 | 75.0% | 87.5% | 5.5s |
| Llama-3.1-8B | 6.50 | 0.371 | 75.0% | 75.0% | 4.0s |

⭐ ALLaM-7B is the **production model** — best Arabic performance with fastest speed.

### 📋 Complete Benchmark Suite

| Benchmark | Score | Details |
|:---|:---:|:---|
| **Retrieval Accuracy** | 100% | 120/120 hits, avg similarity 0.817 |
| **Safety & Red Teaming** | 100% | 10/10 attack vectors blocked |
| **Functional Tests** | 88% | 7/8 passed (1 Arabic ambiguity edge case) |
| **Context Memory** | 100% | 5 multi-turn scenarios, all follow-ups passed |
| **Stress Test** | 100% | 20/20 requests, 4 concurrent users, 0 errors |

### 🌍 Per-Language Performance (ALLaM-7B)

| Arabic | English | French | Spanish | German | Russian | Chinese | Urdu |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 9.0 | 9.0 | 9.0 | 9.0 | 9.0 | 10.0 | 9.0 | 2.0 |

---

## 📂 Project Structure

```
chatbot_project/
├── main.py                    # Command Center (App + Benchmarks + Auth + FAISS)
├── config.py                  # Central configuration & system prompt
├── requirements.txt           # Python dependencies
│
├── core/                      # RAG Intelligence Layer
│   ├── rag_pipeline.py        # Main RAG orchestrator (v5.0) ⭐
│   ├── model_loader.py        # Singleton model manager with VRAM tracking
│   └── vector_store.py        # FAISS index operations
│
├── data/                      # Data Layer
│   ├── ingestion.py           # CSV → 160 chunks ETL (400 chars, 50 overlap)
│   ├── schema.py              # GRC-grade data validation
│   ├── Data_Master/           # MOI_Master_Knowledge.csv (83 services)
│   ├── Data_Chunk/            # BM25 reference chunks
│   ├── data_processed/        # KG JSON + Ground Truth V2 (120 QA pairs)
│   └── faiss_index/           # Auto-generated vector database
│
├── ui/                        # User Interface Layer
│   ├── app.py                 # Gradio interface (v5.0, 8 languages)
│   ├── theme.py               # CSS/JS/RTL-LTR (v4.0, 6 breakpoints)
│   └── assets/                # saudi_emblem.svg, KAUST.png, moi_logo.png
│
├── utils/                     # Utilities
│   ├── auth_manager.py        # Salted SHA-256 authentication
│   ├── logger.py              # Color-coded rotating logger
│   ├── telemetry.py           # Per-user JSONL analytics
│   ├── text_utils.py          # Arabic normalization (C-level str.maketrans)
│   └── tts.py                 # Edge-TTS (Saudi + English voices)
│
└── Benchmarks/                # 7-Test Evaluation Framework
    ├── comprehensive_arena.py # Multi-model arena v6.0 (data-grounded judge)
    ├── functional_test.py     # KG prices + context + safety + attribution
    ├── retrieval_test.py      # Semantic similarity retrieval accuracy
    ├── safety_test.py         # 10 red-teaming attack scenarios
    ├── stress_test.py         # 4-user concurrent load testing
    ├── context_test.py        # 5 multi-turn conversation scenarios
    └── results/               # Timestamped CSV reports
```

---

## 🛠️ Installation & Execution

### Prerequisites
* **Hardware:** NVIDIA GPU with 20GB+ VRAM (A100/H100 recommended)
* **Software:** Python 3.9+, CUDA 12.x

### Setup
```bash
git clone https://github.com/Ahmed-alrashidi/MOI_ChatBot.git
cd MOI_ChatBot/chatbot_project

pip install -r requirements.txt --break-system-packages

echo "HF_TOKEN=hf_your_token" > .env

python utils/auth_manager.py   # Add users
python main.py                 # Launch Command Center
```

### Command Center
```
═══════════════════════════════════════════════════════
   🇸🇦  ABSHER SMART ASSISTANT - COMMAND CENTER  🇸🇦
═══════════════════════════════════════════════════════
  1. 🚀  Launch Absher Chat Application
  2. 🏆  Benchmark Suite (7 tests)
  3. 🛡️   Manage Users (Auth Manager)
  4. 🔄  Rebuild FAISS Index
  5. 🚪  Exit
═══════════════════════════════════════════════════════
```

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

Developed as a capstone project for the **Postgraduate Diploma (PGD+)** program at
**King Abdullah University of Science and Technology (KAUST) Academy — 2026**

**Team:** PGD+
**Version:** 5.0 (Production Release)
**Last Updated:** April 2026
