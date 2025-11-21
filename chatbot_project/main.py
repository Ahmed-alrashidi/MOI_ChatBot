import os
import sys
from config import Config
from utils.logger import setup_logger
from data.ingestion import DataIngestor
from core.model_loader import ModelManager
from core.vector_store import VectorStoreManager
from core.rag_pipeline import ProRAGChain
from ui.app import create_app

# Initialize Logger
logger = setup_logger("MainApp")

def main():
    logger.info("🚀 Starting MOI Chatbot Application...")
    
    # =====================================================
    # 1. Data & Vector Store Setup
    # =====================================================
    embedding_model = ModelManager.get_embedding_model()
    
    # Check if Vector DB exists
    if os.path.exists(os.path.join(Config.VECTOR_DB_DIR, "index.faiss")):
        logger.info("🔹 Vector DB found. Loading existing index...")
        vector_store = VectorStoreManager.load_or_build(embedding_model)
        
        # We need documents for BM25 even if Vector DB exists
        # Ideally, we should save/load raw docs too, but for now we re-ingest for BM25
        # (In production, you'd pickle the 'all_documents' list to save time)
        logger.info("🔹 Loading documents for Keyword Search (BM25)...")
        ingestor = DataIngestor()
        all_documents = ingestor.load_and_process()
        
    else:
        logger.info("🔸 No Vector DB found. Starting fresh ingestion...")
        ingestor = DataIngestor()
        all_documents = ingestor.load_and_process()
        
        if not all_documents:
            logger.error("❌ No documents found! Exiting.")
            return

        logger.info("🔹 Building Vector Store...")
        vector_store = VectorStoreManager.load_or_build(embedding_model, documents=all_documents)

    # =====================================================
    # 2. Initialize RAG Engine
    # =====================================================
    logger.info("🔹 Initializing RAG Chain (Models & Logic)...")
    
    # Pre-load LLM and Whisper to avoid latency during first request
    ModelManager.get_llm()
    ModelManager.get_asr_pipeline()
    
    rag_chain = ProRAGChain(vector_store, all_documents)
    
    # =====================================================
    # 3. Launch UI
    # =====================================================
    logger.info("✅ System Ready. Launching UI...")
    
    app = create_app(rag_chain)
    
    # Launch with public link (share=True) for easy access
    app.queue().launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        share=True,
        inline=False
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("🛑 Application stopped by user.")
    except Exception as e:
        logger.exception(f"❌ Critical Error: {e}")