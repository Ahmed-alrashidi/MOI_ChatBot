# MOI Universal Assistant

> **An enterprise-grade AI conversational agent designed for the Ministry of Interior (MOI).**

This system utilizes the **Sovereign Saudi LLM (ALLaM-7B)** and a **Hybrid RAG architecture** to provide accurate, real-time assistance for Passport, Traffic, and Security services. It is strictly optimized for **NVIDIA A100** infrastructure.

---

## ✨ Key Features

### 🧠 Saudi-Native Intelligence
Powered by **ALLaM-7B-Instruct** to deeply understand local dialects, regulations, and cultural context.

### ⚡ A100 Optimized Architecture
Built with `bfloat16` precision and **Flash Attention 2** support for lightning-fast inference on High-Performance Computing (HPC/IBEX) clusters.

### 🔍 Hybrid RAG Engine
Implements **Reciprocal Rank Fusion (RRF)** combining:
* **Semantic Search:** Dense retrieval via `BAAI/bge-m3` (Cosine Similarity).
* **Keyword Search:** Sparse retrieval via `BM25` for precise terminology matching.

### 🗣️ Multimodal Interface
* **Voice-to-Text:** `Whisper Large v3` for high-accuracy Arabic speech recognition.
* **Text-to-Speech:** Integrated `gTTS` with auto-cleanup logic for seamless audio responses.

### 🛡️ Robust Data Pipeline
Advanced ETL with strict schema validation, **"Smart Chunking,"** and automatic Arabic text normalization (removing Tatweel/Diacritics).

### 🧠 Smart Memory
Features an **"Infinite Context"** mechanism that summarizes conversation history dynamically to maintain long-term context without exhausting tokens.

---

## 📊 Benchmark Results (v3.0)

Tested against a ground-truth dataset for *Jawazat* and *Muroor* regulations on NVIDIA A100.

| Metric | Score | Status |
| :--- | :--- | :--- |
| **Semantic Accuracy** | **91.50%** | ✅ Excellent |
| **Avg. Latency** | **2.08 sec** | ⚡ Real-time |
| **Dialect Understanding** | **High** | 🇸🇦 Native |

---

## 🛠️ Tech Stack

### Infrastructure
* **Language:** Python 3.9
* **Hardware:** NVIDIA A100 (80GB/40GB), CUDA 12.x

### Models
* **LLM:** `ALLaM-AI/ALLaM-7B-Instruct-preview`
* **Embedding:** `BAAI/bge-m3`
* **ASR:** `openai/whisper-large-v3`

### Tools
* **Orchestration:** LangChain (v0.3), Transformers (v4.38+)
* **Database:** FAISS (GPU-Accelerated Vector Store)
* **UI:** Gradio 3.50.2 (Custom MOI Theme & RTL Support)

---

## 📂 Project Structure

```text
MOI_Universal_Assistant/
├── core/
│   ├── model_loader.py   # Singleton Model Manager (LLM/ASR/Embeddings) on A100
│   ├── rag_pipeline.py   # RAG Logic, RRF Merge, Memory Summarization
│   └── vector_store.py   # FAISS Index Management & Recovery
├── data/
│   ├── ingestion.py      # ETL Pipeline (CSV -> Documents)
│   ├── preprocessor.py   # Text Cleaning & Sector Mapping
│   ├── schema.py         # Strict Validation Rules
│   ├── Data_Master/      # High-level Service CSVs
│   └── Data_chunks/      # Detailed Procedure CSVs
├── ui/
│   ├── app.py            # Gradio Application Logic
│   └── theme.py          # CSS Styling & HTML Headers
├── utils/
│   ├── logger.py         # Rotational Logging System
│   ├── tts.py            # Text-to-Speech with File Management
│   └── text_utils.py     # Advanced Arabic Normalization (NLP)
├── config.py             # Central Configuration (Hyperparameters)
└── main.py               # Application Entry Point
```
## ⚡ Quick Start

### 1. Prerequisites
* **Hardware:** NVIDIA GPU (A100 Recommended).
* **Auth:** Hugging Face Token (required for ALLaM model access).

### 2. Installation
Install dependencies (skips `flash-attn` build if needed):
```bash
pip install -r requirements.txt
```
### 3. Setup Environment
Export your Hugging Face token:
```bash
export HF_TOKEN=your_hf_token_here
```
### 4. Run System
The system handles data ingestion and model warmup automatically.

```Bash

python main.py
```
Access the UI at:
``` URL
http://localhost:7860
```

### ⚠️ Troubleshooting
* **HF_TOKEN Error:** If the app crashes on startup, ensure your Hugging Face token has specific permissions to access `ALLaM-AI/ALLaM-7B-Instruct-preview`.
* **OOM (Out of Memory):** If running on a smaller GPU, try reducing `CHUNK_SIZE` in `config.py` or enabling `load_in_8bit` (requires `bitsandbytes`).
* **Flash Attention:** For maximum speed on A100, ensure `flash-attn` is installed. The system will fallback to standard attention if missing.

---

## 📄 License
Developed for KAUST course - 2026

**Version:** 3.0
**Last Updated:** 2026
