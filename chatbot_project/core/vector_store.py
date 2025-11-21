import os
import shutil
from langchain_community.vectorstores import FAISS
from config import Config
from utils.logger import setup_logger

logger = setup_logger(__name__)

class VectorStoreManager:
    @staticmethod
    def load_or_build(embedding_model, documents=None):
        """
        Loads existing FAISS index or builds a new one if documents are provided.
        """
        path = Config.VECTOR_DB_DIR
        index_file = os.path.join(path, "index.faiss")

        # 1. Try loading existing index
        if os.path.exists(index_file):
            try:
                logger.info(f"📂 Found existing FAISS index at {path}")
                vs = FAISS.load_local(
                    path, 
                    embedding_model, 
                    allow_dangerous_deserialization=True
                )
                # Sanity Check
                vs.similarity_search("test", k=1)
                logger.info("✅ FAISS index loaded and verified.")
                return vs
            except Exception as e:
                logger.error(f"❌ FAISS index corrupted: {e}. Rebuilding...")
                shutil.rmtree(path, ignore_errors=True)

        # 2. Build new index (if docs provided)
        if documents:
            logger.info(f"⚡ Building new FAISS index for {len(documents)} documents...")
            os.makedirs(path, exist_ok=True)
            vs = FAISS.from_documents(documents, embedding=embedding_model)
            vs.save_local(path)
            logger.info(f"✅ New FAISS index saved to {path}")
            return vs
        
        raise RuntimeError("❌ No Vector DB found and no documents provided to build one.")