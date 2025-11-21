import os
import shutil
from typing import List, Optional, Any
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from config import Config
from utils.logger import setup_logger

logger = setup_logger(__name__)

class VectorStoreManager:
    """
    Manages the FAISS Vector Database.
    Handles loading existing indices, verifying integrity, and building new ones.
    """
    
    @staticmethod
    def load_or_build(embedding_model: Any, documents: Optional[List[Document]] = None) -> FAISS:
        """
        Strategy:
        1. Try to load an existing index from disk.
        2. If loading fails (corrupted/missing), delete it.
        3. If documents are provided, build a new index and save it.
        
        Args:
            embedding_model: The HuggingFaceEmbeddings instance.
            documents: List of documents to index (required if building new).
            
        Returns:
            FAISS: The initialized vector store.
            
        Raises:
            RuntimeError: If neither loading nor building is possible.
        """
        db_path = Config.VECTOR_DB_DIR
        index_file = os.path.join(db_path, "index.faiss")

        # --- Attempt 1: Load Existing Index ---
        if os.path.exists(index_file):
            try:
                logger.info(f"📂 Found existing FAISS index at: {db_path}")
                
                # 'allow_dangerous_deserialization' is needed for local pickle files
                vector_store = FAISS.load_local(
                    folder_path=db_path, 
                    embeddings=embedding_model, 
                    allow_dangerous_deserialization=True
                )
                
                # Sanity Check: Run a dummy search to ensure index is readable
                vector_store.similarity_search("test", k=1)
                
                logger.info("✅ FAISS index loaded and verified successfully.")
                return vector_store
                
            except Exception as e:
                logger.error(f"❌ FAISS index corrupted or incompatible: {e}")
                logger.warning("🗑️ Deleting corrupted index directory and rebuilding...")
                shutil.rmtree(db_path, ignore_errors=True)

        # --- Attempt 2: Build New Index ---
        if documents:
            logger.info(f"⚡ Building new FAISS index for {len(documents)} documents...")
            
            try:
                # Ensure directory exists
                os.makedirs(db_path, exist_ok=True)
                
                # Build Index (Runs on CPU by default, which is safer for portability)
                # Note: FAISS automatically uses efficient kernels.
                vector_store = FAISS.from_documents(
                    documents=documents, 
                    embedding=embedding_model
                )
                
                # Save to Disk for future use
                vector_store.save_local(db_path)
                logger.info(f"✅ New FAISS index saved successfully to: {db_path}")
                return vector_store
                
            except Exception as e:
                logger.error(f"❌ Failed to build/save FAISS index: {e}")
                raise RuntimeError(f"Critical Error: Could not create Vector DB. {e}")
        
        # --- Failure Case ---
        error_msg = "❌ No existing Vector DB found and no documents provided to build one."
        logger.error(error_msg)
        raise RuntimeError(error_msg)