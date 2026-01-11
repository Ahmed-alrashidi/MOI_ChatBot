import os
import glob
import gc
from typing import List
import pandas as pd
from tqdm import tqdm
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import Config
from utils.logger import setup_logger
from data.schema import validate_schema, MASTER_REQUIRED, CHUNK_REQUIRED
from data.preprocessor import infer_sector, clean_dataframe, build_master_text_representation

# Initialize module logger
logger = setup_logger(__name__)

class DataIngestor:
    """
    Orchestrates the ETL (Extract, Transform, Load) pipeline.
    Optimized for memory efficiency and robustness.
    """
    
    def __init__(self):
        self.master_dir = Config.DATA_MASTER_DIR
        self.chunks_dir = Config.DATA_CHUNKS_DIR
        
        # Text Splitter configuration
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_and_process(self) -> List[Document]:
        """
        Loads all CSV data, validates schemas, creates Documents, 
        and performs strict memory cleanup after each file.
        """
        docs = []
        
        # --- Phase 1: Master Files ---
        master_files = glob.glob(os.path.join(self.master_dir, "*.csv"))
        logger.info(f"📂 Found {len(master_files)} Master files.")
        
        for f in tqdm(master_files, desc="Ingesting Masters"):
            try:
                df = pd.read_csv(f)
                
                # Skip empty files
                if df.empty:
                    logger.warning(f"⚠️ Skipping empty file: {f}")
                    continue

                # Schema Validation
                # FIX: Pass filename 'f' and rely on Exception raising instead of return value
                validate_schema(df, MASTER_REQUIRED, f)
                
                # Cleaning & Processing
                # FIX: Specify columns to clean
                df = clean_dataframe(df, ["service_title_ar", "description_full"])
                sector = infer_sector(f)
                
                for _, row in df.iterrows():
                    text_representation = build_master_text_representation(row)
                    
                    meta = {
                        "sector": sector,
                        "service_id": str(row["service_id"]),
                        "service_title": row.get("service_title_ar", ""),
                        "doc_level": "service_master",
                        "source_file": os.path.basename(f)
                    }
                    
                    docs.append(Document(page_content=text_representation, metadata=meta))
                
                # --- MEMORY CLEANUP ---
                del df
                gc.collect()

            except Exception as e:
                logger.error(f"❌ Error processing Master file '{os.path.basename(f)}': {e}")

        # --- Phase 2: Chunk Files ---
        chunk_files = glob.glob(os.path.join(self.chunks_dir, "*.csv"))
        logger.info(f"📂 Found {len(chunk_files)} Chunk files.")
        
        for f in tqdm(chunk_files, desc="Ingesting Chunks"):
            try:
                df = pd.read_csv(f)
                
                if df.empty: continue

                # Schema Validation
                validate_schema(df, CHUNK_REQUIRED, f)
                
                # Cleaning & Processing
                # FIX: Specify columns to clean for chunks
                df = clean_dataframe(df, ["chunk_text", "chunk_title"])
                sector = infer_sector(f)
                
                for _, row in df.iterrows():
                    base_meta = {
                        "sector": sector,
                        "service_id": str(row["service_id"]),
                        "chunk_id": str(row["chunk_id"]),
                        "chunk_title": row.get("chunk_title", ""), # Safe access
                        "doc_level": "service_chunk",
                        "source_file": os.path.basename(f)
                    }
                    
                    text = row["chunk_text"]
                    
                    # Safety Split: Verify chunk fits embedding window
                    if len(text) > Config.CHUNK_SIZE:
                        parts = self.splitter.split_text(text)
                        for i, p in enumerate(parts):
                            meta = base_meta.copy()
                            meta["chunk_part"] = f"{i+1}/{len(parts)}"
                            docs.append(Document(page_content=p, metadata=meta))
                    else:
                        docs.append(Document(page_content=text, metadata=base_meta))
                
                # --- MEMORY CLEANUP ---
                del df
                gc.collect()
                        
            except Exception as e:
                logger.error(f"❌ Error processing Chunk file '{os.path.basename(f)}': {e}")
        
        logger.info(f"✅ Ingestion Complete. Total Documents Created: {len(docs)}")
        return docs

# Helper for standalone execution
if __name__ == "__main__":
    ingestor = DataIngestor()
    ingestor.load_and_process()