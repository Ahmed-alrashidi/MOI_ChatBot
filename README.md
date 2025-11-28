# 🇸🇦 MOI Universal Assistant (المساعد الذكي الشامل)

![MOI Logo](ui/moi_logo.png)

> An advanced AI-powered conversational agent designed for the Ministry of Interior (MOI), utilizing the **ALLaM-7B** Saudi LLM and a Hybrid RAG architecture to provide accurate, real-time assistance for Passport, Traffic, and Security services.

---

## 🚀 Key Features

* **Hybrid RAG Engine:** Combines Semantic Search (Dense Vectors via `BAAI/bge-m3`) and Keyword Search (Sparse via `BM25`) using Reciprocal Rank Fusion (RRF).
* **Saudi Identity:** Built on **ALLaM-7B-Instruct** to understand local dialect and context, with a custom UI reflecting MOI branding.
* **Multimodal Interface:** Supports **Voice-to-Text** (Whisper v3) and **Text-to-Speech** (gTTS) for accessibility.
* **High Performance:** Optimized for HPC (IBEX) with Lazy Loading & GPU Warmup logic.
* **Robust Data Pipeline:** Advanced ETL pipeline with schema validation, de-duplication, and smart chunking.

---

## 📊 Benchmark Results (v3.0)

We tested the system against a verified ground-truth dataset covering Jawazat, Traffic, and Security regulations.

| Metric | Score | Status |
| :--- | :--- | :--- |
| **Semantic Accuracy** | **91.50%** | ✅ Excellent |
| **Avg. Latency** | **2.08 sec** | ⚡ Real-time |

---

## 🛠️ Tech Stack

* **Core:** Python 3.9, LangChain, Transformers.
* **Models:** * LLM: `ALLaM-AI/ALLaM-7B-Instruct-preview`
    * Embedding: `BAAI/bge-m3`
    * ASR: `openai/whisper-large-v3`
* **Database:** FAISS (Vector Store).
* **UI/Web:** Gradio (Custom CSS/Theme).

---

## 👥 The Team

| Member | Role | Key Contributions |
| :--- | :--- | :--- |
| **Sultan Alshaibani** | Project Lead | Strategy, Model Selection (ALLaM), Resource Mgmt. |
| **Ahmed Alrashidi** | System Architect | RAG Pipeline Logic (RRF), Orchestration, Gradio UI Logic. |
| **Sultan Alotaibi** | Search Engine | Vector Store (FAISS), Embeddings Optimization. |
| **Fahad Alqahtani** | Data Engineer | ETL Pipeline, Schema Validation, Data Cleaning. |
| **Abdulaziz Almutairi**| UI/UX & Translation | Frontend Design (MOI Theme), RTL Support, TTS Integration. |
| **Rakan Alharbi** | QA & Testing | Benchmarking, Stress Testing, System Logging. |

---

## 📂 Project Structure

```bash
├── core/             # RAG Logic, Model Loaders, Vector Store
├── data/             # Ingestion Pipeline, Schema, Raw Data
├── ui/               # Gradio App, CSS Themes
├── utils/            # Logger, Text Cleaning, TTS
├── config.py         # Central Configuration
└── main.py           # Application Entry Point
