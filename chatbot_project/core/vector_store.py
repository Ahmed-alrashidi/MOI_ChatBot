# =========================================================================
# File Name: core/vector_store.py
# Purpose: Manages FAISS Vector Database (Storage, Retrieval & Self-Healing).
# Project: Absher Smart Assistant (MOI ChatBot)
# Features:
# - Self-Healing: Automatically detects and rebuilds corrupted binary indices.
# - Optimized for BGE-M3: Enforces Cosine Similarity for semantic precision.
# - Memory Management: Implements destructive GC to free VRAM during massive builds.
# - Singleton-friendly: Designed to serve as the long-term memory for the RAG Engine.
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
logger = setup_logger("Vector_Store")

class VectorStoreManager:
    """
    Handles the complete lifecycle of the FAISS Vector Database.
    Responsible for persisting, loading, and refreshing the semantic index 
    built from the chunked 'Data_Master' source files.
    """
    
    @classmethod
    def load_or_build(
        cls, 
        embedding_model: Any, 
        documents: Optional[List[Document]] = None, 
        force_rebuild: bool = False
    ) -> FAISS:
        """
        Main entry point for accessing the Vector DB. 
        Intelligently decides whether to load an existing local index or trigger an ETL rebuild.
        
        Note on Memory: If a rebuild is triggered, the `documents` list is destructively 
        cleared from memory at the end of the process to free up RAM/VRAM.
        
        Args:
            embedding_model (Any): The HuggingFace embedding engine (e.g., BGE-M3).
            documents (Optional[List[Document]]): List of LangChain documents to index (required if building).
            force_rebuild (bool): If True, aggressively deletes any existing index and starts fresh.
            
        Returns:
            FAISS: An initialized, ready-to-query FAISS vector store instance.
            
        Raises:
            ValueError: If a rebuild is needed but no documents are provided.
            RuntimeError: If FAISS fails to build or save the index.
        """
        db_path = Config.VECTOR_DB_DIR
        index_file = os.path.join(db_path, "index.faiss")
        
        # --- Phase 1: Attempt to Load Existing Index from Disk ---
        if os.path.exists(index_file) and not force_rebuild:
            try:
                logger.info(f"📂 Found existing FAISS index at: {db_path}")
                
                # Load the local index. 
                # Note: 'distance_strategy' is intentionally omitted here because the metric type 
                # (Inner Product for Cosine Similarity) is already compiled into the .faiss binary.
                # 'allow_dangerous_deserialization' is required by modern LangChain for local trusted files.
                vector_store = FAISS.load_local(
                    folder_path=db_path, 
                    embeddings=embedding_model, 
                    allow_dangerous_deserialization=True 
                )
                logger.info(f"✅ Loaded {vector_store.index.ntotal} vectors from existing index.")
                return vector_store
                
            except Exception as e:
                # SELF-HEALING: If deserialization fails (e.g., corrupted binary or version mismatch)
                logger.warning(f"⚠️ Index corrupted or incompatible: {e}. Initiating Self-Healing (Rebuild)...")
                force_rebuild = True

        # --- Phase 2: Build New Index (ETL Process) ---
        if documents or force_rebuild:
            if not documents:
                error_msg = "❌ Rebuild requested (or index missing) but no documents were provided."
                logger.error(error_msg)
                raise ValueError(error_msg)

            logger.info(f"⚡ Building new FAISS index from {len(documents)} dense documents...")
            
            try:
                # Workspace Preparation: Ensure the directory is clean to prevent overlapping chunks
                if os.path.exists(db_path):
                    shutil.rmtree(db_path) 
                os.makedirs(db_path, exist_ok=True)
                
                # Embedding Process: Convert text documents into high-dimensional numerical vectors.
                # The COSINE DistanceStrategy automatically normalizes the L2 vectors during ingestion.
                vector_store = FAISS.from_documents(
                    documents=documents, 
                    embedding=embedding_model,
                    distance_strategy=DistanceStrategy.COSINE 
                )
                
                # Persistence: Save the binary index and docstore to the local file system.
                vector_store.save_local(db_path)
                
                # Log build stats for debugging
                index_size_mb = os.path.getsize(os.path.join(db_path, "index.faiss")) / (1024 * 1024)
                logger.info(f"✅ FAISS index saved: {vector_store.index.ntotal} vectors | {index_size_mb:.1f} MB | {db_path}")
                
                # --- PROACTIVE MEMORY CLEANUP ---
                # Destructive consumption: Clear the list to free up Host RAM before moving to LLM inference.
                documents.clear()
                gc.collect()
                
                return vector_store
                
            except Exception as e:
                logger.error(f"❌ Failed to build FAISS index: {e}", exc_info=True)
                raise RuntimeError(f"Critical Error: Could not create Vector DB. {e}")
        
        # --- Phase 3: Total Failure Case (No Index + No Source Data) ---
        error_msg = f"❌ No Vector DB found at {db_path} and no documents provided to initiate a build."
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # Maintain alias for backward compatibility with earlier pipeline architectures
    load_faiss_index = load_or_build