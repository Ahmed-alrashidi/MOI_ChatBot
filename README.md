🇸🇦 MOI Smart Assistant

An Advanced RAG-based Chatbot for Saudi Ministry of Interior Services

MOI Smart Assistant is a specialized AI chatbot designed to answer queries regarding Saudi Ministry of Interior (MOI) services, such as Passports (Jawazat), Civil Affairs (Ahwal), and Traffic (Muroor). It leverages ALLaM-7B (a premier Arabic LLM), Whisper for voice recognition, and a Hybrid RAG engine to provide accurate, context-aware responses.

✨ Key Features

Hybrid RAG Engine: Combines Dense Vector Search (Embedding) with Keyword Search (BM25) for maximum retrieval accuracy.

Native Arabic Support: Powered by ALLaM-7B, optimized for Saudi dialect and formal Arabic.

Voice Interaction: Supports voice input using OpenAI's Whisper model.

Platform Agnostic: Runs seamlessly on Google Colab, HPC Clusters (IBEX), or local machines.

Smart Query Rewriting: Automatically translates and expands queries to find better matches in the database.

🚀 Quick Start

Follow these steps to set up the project on any environment.

1. Clone the Repository

git clone [https://github.com/Ahmed-alrashidi/MOI_ChatBot.git](https://github.com/Ahmed-alrashidi/MOI_ChatBot.git)
cd MOI_ChatBot


2. One-Click Setup

We provide a setup script that automatically handles Python dependencies, system libraries (ffmpeg), and environment configuration.

bash setup.sh


🔑 Authentication (Important)

To access the ALLaM-7B model, you need a valid Hugging Face Token.

Step 1: Get your Token

Go to your Hugging Face Settings.

Create a new token with Read permissions.

Copy the token (starts with hf_...).

Step 2: Add Token to the Project

🅰️ Option A: Google Colab (Recommended)

On the left sidebar, click the Secrets (Key icon 🔑).

Add a new secret:

Name: HF_TOKEN

Value: Paste your token.

Toggle Notebook access to ON.

🅱️ Option B: Local Machine / Terminal / IBEX

Open the .env file created by the setup script.

Paste your token inside:

HF_TOKEN=hf_your_token_here


Save the file.

▶️ Usage

Once setup is complete and the token is added, launch the application:

python main.py


What happens next?

The system ingests the CSV data and builds the Vector Database (if not already built).

It loads the AI models (ALLaM & Whisper) onto the GPU.

It launches a Gradio Web Interface.

A Public URL will be displayed in the terminal (e.g., https://xxxx.gradio.live) which you can share or open on any device.

📂 Project Structure

The project follows a modular architecture for easy maintenance:

MOI_ChatBot/
├── core/               # The AI Brain
│   ├── model_loader.py # Handles loading LLMs & Embeddings (Singleton)
│   ├── rag_pipeline.py # RAG Logic (Retrieval + Reranking + Generation)
│   └── vector_store.py # FAISS Database Management
│
├── data/               # Data Layer
│   ├── Data_Master/    # Raw CSVs (Service Descriptions)
│   ├── Data_chunks/    # Raw CSVs (Detailed chunks)
│   └── vector_db/      # Generated FAISS Index
│
├── ui/                 # Frontend
│   ├── app.py          # Gradio Interface Logic
│   └── theme.py        # Custom CSS & Branding
│
├── utils/              # Utilities
│   ├── logger.py       # Centralized Logging
│   └── text_utils.py   # Arabic Normalization & Cleaning
│
├── config.py           # Central Configuration (Paths & Hyperparameters)
├── main.py             # Entry Point
└── setup.sh            # Installation Script


🛠 Hardware Requirements

GPU: NVIDIA A100, V100, or T4 (Min 16GB VRAM recommended).

RAM: 32GB+ System RAM.

Storage: At least 20GB free space for models.

Developed by Ahmed Alrashidi for the MOI Chatbot Project.