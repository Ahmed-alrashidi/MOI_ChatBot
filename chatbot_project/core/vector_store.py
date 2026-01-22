# =========================================================================
# File Name: core/vector_store.py
# Project: Absher Smart Assistant (MOI ChatBot)
# Architecture: Cross-Lingual Hybrid RAG (BGE-M3 + BM25 + ALLaM-7B)
#
# Affiliation: King Abdullah University of Science and Technology (KAUST)
# Team: Ahmed AlRashidi, Sultan Alshaibani, Fahad Alqahtani, 
#       Rakan Alharbi, Sultan Alotaibi, Abdulaziz Almutairi.
# Advisors: Prof. Naeemullah Khan & Dr. Salman Khan
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

logger = setup_logger(__name__)

class VectorStoreManager:
    """
    Manages the FAISS Vector Database built from 'Data_Master'.
    Features:
    - Uses Cosine Similarity (DistanceStrategy.COSINE) specifically for BGE-M3.
    - Optimized for storing high-level semantic descriptions.
    """
    
    @staticmethod
    def load_or_build(embedding_model: Any, documents: Optional[List[Document]] = None) -> FAISS:
        """
        Retrieves the existing Master FAISS index or builds a new one.
        """
        db_path = Config.VECTOR_DB_DIR
        index_file = os.path.join(db_path, "index.faiss")

        # --- Attempt 1: Load Existing Index ---
        if os.path.exists(index_file):
            try:
                logger.info(f"📂 Found existing FAISS index (Master DB) at: {db_path}")
                
                # Load the index enforcing Cosine Strategy
                vector_store = FAISS.load_local(
                    folder_path=db_path, 
                    embeddings=embedding_model, 
                    distance_strategy=DistanceStrategy.COSINE,
                    allow_dangerous_deserialization=True # Trusted local source
                )
                return vector_store
            except Exception as e:
                logger.warning(f"⚠️ Failed to load index: {e}. Rebuilding...")
                shutil.rmtree(db_path, ignore_errors=True)

        # --- Attempt 2: Build New Index ---
        if documents:
            logger.info(f"⚡ Building new FAISS index from {len(documents)} Master documents...")
            
            try:
                os.makedirs(db_path, exist_ok=True)
                
                # Create Index
                vector_store = FAISS.from_documents(
                    documents=documents, 
                    embedding=embedding_model,
                    distance_strategy=DistanceStrategy.COSINE 
                )
                
                # Persist to disk
                vector_store.save_local(db_path)
                logger.info(f"✅ Master FAISS index saved to: {db_path}")
                
                # Memory Cleanup
                del documents
                gc.collect()
                
                return vector_store
                
            except Exception as e:
                logger.error(f"❌ Failed to build FAISS index: {e}")
                raise RuntimeError(f"Critical Error: Could not create Vector DB. {e}")
        
        # --- Failure Case ---
        error_msg = "❌ No Vector DB found and no documents provided."
        logger.error(error_msg)
        raise RuntimeError(error_msg)