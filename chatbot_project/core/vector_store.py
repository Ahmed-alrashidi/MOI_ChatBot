# =========================================================================
# File Name: core/vector_store.py
# Purpose: Manages FAISS Vector Database (Storage & Retrieval)
# Project: Absher Smart Assistant (MOI ChatBot)
# Features:
# - Self-Healing: Automatically detects and rebuilds corrupted indices.
# - Optimized for BGE-M3: Enforces Cosine Similarity for semantic precision.
# - Memory Management: Implements explicit GC to free VRAM during build.
# - Singleton-friendly: Designed to serve as the long-term memory for RAG.
# =========================================================================

import os
import shutil
import gc
from typing import List, Optional, Any
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.schema import Document
from config import Config
from utils.logger import setup_logger

# Initialize project logger for the vector store module
logger = setup_logger(__name__)

class VectorStoreManager:
    """
    Handles the lifecycle of the FAISS Vector Database.
    Responsible for persisting, loading, and refreshing the semantic index 
    built from the 'Data_Master' source files.
    """
    
    @classmethod
    def load_or_build(cls, embedding_model: Any, documents: Optional[List[Document]] = None, force_rebuild: bool = False) -> FAISS:
        """
        Main entry point for accessing the Vector DB. 
        Decides whether to load an existing index from disk or create a fresh one.
        
        Args:
            embedding_model: The HuggingFace embedding engine (e.g., BGE-M3).
            documents: Optional list of documents to index if building from scratch.
            force_rebuild: If True, deletes any existing index and starts fresh.
            
        Returns:
            FAISS: An initialized and ready-to-use FAISS vector store instance.
        """
        db_path = Config.VECTOR_DB_DIR
        index_file = os.path.join(db_path, "index.faiss")
        
        # --- Logic Gate for Decision Making ---
        # 1. We load existing if: file exists AND not forced AND no new docs provided.
        should_load_existing = os.path.exists(index_file) and not force_rebuild and not documents

        # --- Attempt 1: Load Existing Index from Disk ---
        if should_load_existing:
            try:
                logger.info(f"📂 Found existing FAISS index at: {db_path}")
                
                # Load the local index enforcing COSINE similarity strategy.
                # This is critical for BGE-M3 models to maintain high retrieval accuracy.
                vector_store = FAISS.load_local(
                    folder_path=db_path, 
                    embeddings=embedding_model, 
                    distance_strategy=DistanceStrategy.COSINE,
                    allow_dangerous_deserialization=True # Safe as it is a local internal file
                )
                return vector_store
            except Exception as e:
                # SELF-HEALING: If loading fails (e.g., corrupted file), log warning and prepare for rebuild
                logger.warning(f"⚠️ Index corrupted: {e}. Preparing to rebuild...")
                if os.path.exists(db_path):
                    shutil.rmtree(db_path)

        # --- Attempt 2: Build New Index (ETL Process) ---
        if documents:
            logger.info(f"⚡ Building new FAISS index from {len(documents)} documents...")
            
            try:
                # Workspace preparation: Ensure the directory is clean to prevent merge conflicts
                if os.path.exists(db_path):
                    shutil.rmtree(db_path) 
                os.makedirs(db_path, exist_ok=True)
                
                # Embedding Process: Convert text documents into numerical vectors.
                # normalize_L2 is automatically handled via the COSINE DistanceStrategy.
                vector_store = FAISS.from_documents(
                    documents=documents, 
                    embedding=embedding_model,
                    distance_strategy=DistanceStrategy.COSINE 
                )
                
                # Persistence: Save the binary index to the local file system for future use.
                vector_store.save_local(db_path)
                logger.info(f"✅ FAISS index saved successfully to: {db_path}")
                
                # --- MEMORY CLEANUP ---
                # Explicitly delete the document list from RAM and trigger Garbage Collection.
                # This is vital when processing large datasets on a shared GPU server (Ibex).
                del documents
                gc.collect()
                
                return vector_store
                
            except Exception as e:
                # Log critical failure in case of indexing errors
                logger.error(f"❌ Failed to build FAISS index: {e}")
                raise RuntimeError(f"Critical Error: Could not create Vector DB. {e}")
        
        # --- Failure Case: No Index and No Source Data ---
        error_msg = f"❌ No Vector DB found at {db_path} and no documents provided for building."
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # Maintain alias for backward compatibility with earlier pipeline versions
    load_faiss_index = load_or_build