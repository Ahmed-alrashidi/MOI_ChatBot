import os
import shutil
import gc
from typing import List, Optional, Any
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.schema import Document
from config import Config
from utils.logger import setup_logger

# Initialize module logger
logger = setup_logger(__name__)

class VectorStoreManager:
    """
    Manages the lifecycle of the FAISS Vector Database with A100 optimizations.
    
    Features:
    - Uses Cosine Similarity (DistanceStrategy.COSINE) for BGE-M3 compatibility.
    - Implements strict memory cleanup after indexing.
    - Handles corruption recovery automatically.
    """
    
    @staticmethod
    def load_or_build(embedding_model: Any, documents: Optional[List[Document]] = None) -> FAISS:
        """
        Retrieves an existing FAISS index or builds a new one.
        Optimized for semantic search accuracy and memory hygiene.
        """
        db_path = Config.VECTOR_DB_DIR
        index_file = os.path.join(db_path, "index.faiss")

        # --- Attempt 1: Load Existing Index ---
        if os.path.exists(index_file):
            try:
                logger.info(f"📂 Found existing FAISS index at: {db_path}")
                
                # Load the index with Cosine Strategy enforcement
                vector_store = FAISS.load_local(
                    folder_path=db_path, 
                    embeddings=embedding_model, 
                    distance_strategy=DistanceStrategy.COSINE, # Force Cosine for accuracy
                    allow_dangerous_deserialization=True
                )
                
                # SANITY CHECK: Perform a dummy search to ensure index is readable
                vector_store.similarity_search("test", k=1)
                
                logger.info("✅ FAISS index loaded and verified successfully.")
                return vector_store
                
            except Exception as e:
                logger.error(f"❌ FAISS index corrupted or incompatible: {e}")
                logger.warning("🗑️ Deleting corrupted index directory and rebuilding...")
                shutil.rmtree(db_path, ignore_errors=True)

        # --- Attempt 2: Build New Index ---
        if documents:
            logger.info(f"⚡ Building new FAISS index for {len(documents)} documents (Cosine Metric)...")
            
            try:
                os.makedirs(db_path, exist_ok=True)
                
                # Generate Embeddings and Index with COSINE Similarity
                vector_store = FAISS.from_documents(
                    documents=documents, 
                    embedding=embedding_model,
                    distance_strategy=DistanceStrategy.COSINE # CRITICAL for BGE-M3
                )
                
                # Persist to disk
                vector_store.save_local(db_path)
                logger.info(f"✅ New FAISS index saved successfully to: {db_path}")
                
                # Memory Cleanup: Free up RAM/VRAM used during embedding generation
                del documents
                gc.collect()
                
                return vector_store
                
            except Exception as e:
                logger.error(f"❌ Failed to build/save FAISS index: {e}")
                raise RuntimeError(f"Critical Error: Could not create Vector DB. Details: {e}")
        
        # --- Failure Case ---
        error_msg = "❌ No existing Vector DB found and no documents provided to build one."
        logger.error(error_msg)
        raise RuntimeError(error_msg)