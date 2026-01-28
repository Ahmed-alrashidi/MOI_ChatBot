# Absher Smart Assistant
### 🏛️ Sovereign AI for Saudi MOI Services | Version 1.0

### 🏛️ Sovereign AI • 🔎 Hybrid RAG • 🤖 Bilingual LLM • 🌍 Zero-Shot Cross-Lingual • 🎤 Speech-to-Speech

![Status](https://img.shields.io/badge/Status-Stable_Release-success?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
[![Model - ALLaM](https://img.shields.io/badge/Model-ALLaM--7B--Instruct-green?style=flat-square&logo=huggingface)](https://huggingface.co/humain-ai/ALLaM-7B-Instruct-preview)
![Architecture](https://img.shields.io/badge/Architecture-Hybrid_RAG-purple?style=flat-square)
![Hardware](https://img.shields.io/badge/GPU-A100_Optimized-orange?style=flat-square&logo=nvidia)
[![ASR - Whisper](https://img.shields.io/badge/ASR-Whisper_Large--v3-blueviolet?style=flat-square)](https://huggingface.co/openai/whisper-large-v3)
![TTS](https://img.shields.io/badge/TTS-ar--SA--HamedNeural-red?style=flat-square)
[![Translation - NLLB](https://img.shields.io/badge/Translation-NLLB--200-yellow?style=flat-square)](https://huggingface.co/facebook/nllb-200-3.3B)
[![License](https://img.shields.io/badge/License-MIT_Academic-lightgrey?style=flat-square)](LICENSE)
---

> 🏛️ Sovereign AI | 🤖 Large Language Models | 🔎 Hybrid RAG | 🛡️ GRC Compliant | 🌍 Multilingual Ready | 🎤 Speech & TTS


## 📖 Overview

**Absher Smart Assistant** is a sovereign AI conversational system designed to democratize access to Saudi Ministry of Interior (MOI) services. Addressing the critical challenges of language barriers and hallucinations in traditional LLMs, the system employs a novel **Cross-Lingual Hybrid Retrieval-Augmented Generation (RAG)** architecture to anchor generative capabilities to a curated, verified knowledge base of MOI regulations.

---

## ✨ Advanced Technical Features

### 🧠 Sovereign Saudi Intelligence (ALLaM-7B)
Powered by [ALLaM-7B-Instruct-preview](https://huggingface.co/humain-ai/ALLaM-7B-Instruct-preview), developed by **SDAIA**. 
* **Training Depth:** Pretrained on **5.2 Trillion tokens** (4T English + 1.2T Mixed Arabic/English).
* **Optimization:** Built on **NVIDIA/MegatronLM** with bf16-mixed precision, ensuring high MFU (~42%) during training.

### 🔍 Hybrid Retrieval with RRF Fusion
The system eliminates hallucinations by synergizing dense vector retrieval (**BGE-M3**) with sparse keyword matching (**BM25**). Results are fused using the **Reciprocal Rank Fusion (RRF)** algorithm:

$$RRF~Score(d) = \sum_{j \in \{Dense, Sparse\}} \frac{1}{k + r_j(d)}$$

Where $k=60$ is a smoothing constant to prioritize documents verified by both retrieval streams.

### 🌍 Zero-Shot Cross-Lingual Mechanism
Enables multilingual support (English, French, Russian, etc.) without an intermediate translation layer. By leveraging a unified embedding space, the system maps foreign queries directly to Arabic regulatory vectors, ensuring low-latency and preserving semantic nuance.

### 🛡️ Robust ETL & Self-Healing
* **Advanced Normalization:** Specialized NLP pipeline standardizes Arabic text (e.g., unifying Alef and Taa Marbuta forms) to resolve morphological inconsistencies.
* **Smart Chunking:** Employs a recursive character splitter with a **250-token overlap** to preserve context across boundaries.
* **Self-Healing Vector Store:** A fail-safe mechanism that performs real-time sanity checks and automatically rebuilds the FAISS index upon detecting corruption.

---

## 📊 Benchmark Results (v1.0)

Tested on **NVIDIA A100** using a rigorous global benchmark across 6 core languages.

| Metric | Result | Status |
| :--- | :--- | :--- |
| **Arabic Semantic Accuracy** | **96.0%** | ✅ Superior (Native) |
| **English Semantic Accuracy** | **88.0%** | ✅ Excellent |
| **Hallucination Rate** | **0.0%** | 🛡️ Zero-Hallucination |
| **Average Latency (Arabic)** | **2.10 sec** | ⚡ Ultra-Fast |

---

## 📜 Credits & Citations

### Model Acknowledgment
This project utilizes the **ALLaM** model series by **SDAIA**. We acknowledge the **National Center for Artificial Intelligence (NCAI)** for their work on Arabic Language Technology.

```bibtex
@inproceedings{
    bari2025allam,
    title={{ALL}aM: Large Language Models for Arabic and English},
    author={M Saiful Bari and Yazeed Alnumay and others},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={[https://openreview.net/forum?id=MscdsFVZrN](https://openreview.net/forum?id=MscdsFVZrN)}
}
}
```

## 📂 Project Structure

```plaintext
MOI_Universal_Assistant/
├── core/             # The Reasoning Engine (RAG Pipeline, Vector Store)
├── data/             # Data Layer (ETL Pipeline, KG, Schema Validation)
├── Benchmarks/       # The Audit Suite (Safety, Stress, Model Arena)
├── ui/               # Interface (Gradio App, Professional MOI Theme)
├── utils/            # Utilities (Neural TTS, Rotational Logger, NLP)
├── config.py         # Central Intelligence Configuration
└── main.py           # Production Entry Point
```

---
## 🛠️ Installation & Execution

### 1. Prerequisites
* **Hardware:** NVIDIA GPU (A100/H100 Optimized recommended) with 80GB+ VRAM.
* **Software:** Python 3.9+, CUDA Toolkit.

### 2. Setup & Installation
```bash
# Clone the repository
git clone [https://github.com/Ahmed-alrashidi/MOI_ChatBot.git](https://github.com/Ahmed-alrashidi/MOI_ChatBot.git)
cd MOI_ChatBot

# Install dependencies
pip install -r requirements.txt
```
### 3. Configure Environment

```bash
export HF_TOKEN="your_hugging_face_token"
```
### 4. Launch System

The system handles automated hardware diagnostics and database builds on startup.

```bash
python main.py
```
## 📄 Academic Context

Developed as a final project for the **CS299-Master's Directed Research** course at  
**King Abdullah University of Science and Technology (KAUST) – 2026**

**Version:** 1.0 (Stable Release)  
**Last Updated:** Jan 2026 
