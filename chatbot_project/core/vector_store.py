import os
import shutil
from typing import List, Optional, Any
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from config import Config
from utils.logger import setup_logger

# Initialize module logger
logger = setup_logger(__name__)

class VectorStoreManager:
    """
    Manages the lifecycle of the FAISS Vector Database (Creation, Loading, and Validation).
    
    This class ensures that the vector index is always available and valid. It handles:
    1. Loading existing indices from the disk.
    2. Validating index integrity via dummy searches.
    3. rebuilding the index from scratch if corruption is detected or if it doesn't exist.
    """
    
    @staticmethod
    def load_or_build(embedding_model: Any, documents: Optional[List[Document]] = None) -> FAISS:
        """
        Retrieves an existing FAISS index or builds a new one from raw documents.
        
        The method follows a strict recovery strategy:
        - First, it attempts to load the index from the path defined in Config.
        - It runs a 'Sanity Check' (dummy search) to ensure the file isn't corrupted.
        - If loading fails or the file is corrupt, it deletes the old file and triggers a rebuild.
        - If no index exists, it builds a new one using the provided documents.

        Args:
            embedding_model: The embedding function (e.g., HuggingFaceEmbeddings) used to vectorise text.
            documents: A list of Document objects to be indexed. Required if building a new DB.
            
        Returns:
            FAISS: An initialized and verified FAISS vector store object.
            
        Raises:
            RuntimeError: If the index cannot be loaded AND no documents are provided to build a new one.
        """
        db_path = Config.VECTOR_DB_DIR
        index_file = os.path.join(db_path, "index.faiss")

        # --- Attempt 1: Load Existing Index ---
        if os.path.exists(index_file):
            try:
                logger.info(f"📂 Found existing FAISS index at: {db_path}")
                
                # Load the index
                # 'allow_dangerous_deserialization' is set to True because we trust our own local files.
                vector_store = FAISS.load_local(
                    folder_path=db_path, 
                    embeddings=embedding_model, 
                    allow_dangerous_deserialization=True
                )
                
                # SANITY CHECK: Perform a lightweight search to verify index integrity.
                # This catches corrupted files before they cause errors in the chat loop.
                vector_store.similarity_search("test", k=1)
                
                logger.info("✅ FAISS index loaded and verified successfully.")
                return vector_store
                
            except Exception as e:
                logger.error(f"❌ FAISS index corrupted or incompatible: {e}")
                logger.warning("🗑️ Deleting corrupted index directory and rebuilding...")
                
                # Remove the corrupted directory to ensure a clean rebuild
                shutil.rmtree(db_path, ignore_errors=True)

        # --- Attempt 2: Build New Index ---
        # This block runs if:
        # 1. No index existed previously.
        # 2. The previous index was corrupted and deleted in Attempt 1.
        if documents:
            logger.info(f"⚡ Building new FAISS index for {len(documents)} documents...")
            
            try:
                # Ensure the target directory exists
                os.makedirs(db_path, exist_ok=True)
                
                # Generate Embeddings and Index
                # FAISS.from_documents handles the batching and embedding generation automatically.
                vector_store = FAISS.from_documents(
                    documents=documents, 
                    embedding=embedding_model
                )
                
                # Persist the index to disk for future runs
                vector_store.save_local(db_path)
                logger.info(f"✅ New FAISS index saved successfully to: {db_path}")
                return vector_store
                
            except Exception as e:
                logger.error(f"❌ Failed to build/save FAISS index: {e}")
                raise RuntimeError(f"Critical Error: Could not create Vector DB. Details: {e}")
        
        # --- Failure Case ---
        # If we reach here, it means we have no index on disk AND no documents to build one.
        # This usually happens if the data ingestion pipeline hasn't run yet.
        error_msg = "❌ No existing Vector DB found and no documents provided to build one."
        logger.error(error_msg)
        raise RuntimeError(error_msg)