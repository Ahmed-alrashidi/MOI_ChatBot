# =========================================================================
# File Name: data/ingestion.py
# Purpose: ETL Pipeline for Master Data Ingestion (A100 Optimized).
# Project: Absher Smart Assistant (MOI ChatBot)
# Features:
# - Schema Enforcement: Integrates with schema.py to block corrupted data.
# - Hybrid Readiness: Prepares identical chunks for both FAISS and BM25.
# - Memory Optimization: Employs eager garbage collection during chunking.
# =========================================================================

import os
import glob
import gc
import pandas as pd
from typing import List
from tqdm import tqdm
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import Config
from utils.logger import setup_logger
from data.schema import validate_schema  # [ADDED]: Integrating the GRC Schema Validator

# Initialize module-specific logger
logger = setup_logger("Data_Ingestor")

class DataIngestor:
    """
    Orchestrates the transformation of raw CSV tabular data into 
    chunked LangChain Document objects (for Dense Retrieval) and 
    synchronized CSV chunks (for Sparse BM25 Retrieval).
    """
    
    def __init__(self):
        """
        Initializes directory paths and configures the text splitting logic.
        Optimized for the BGE-M3 embedding model context window.
        """
        # Direct path mapping to the Master Knowledge Base
        self.master_file = os.path.join(Config.DATA_MASTER_DIR, "MOI_Master_Knowledge.csv")
        self.chunk_dir = Config.DATA_CHUNK_DIR
        
        # BGE-M3 optimized splitter: 400 chars (~256 tokens) with a 50 char sliding window overlap.
        # Tuned for enriched RAG_Content (~456 chars avg) to produce 1-2 chunks per service,
        # yielding ~160 dense vectors instead of 83 (nearly 2x retrieval surface area).
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def _cleanup_old_chunks(self):
        """
        System Maintenance: Removes previously generated CSV chunks 
        to ensure fresh indexing and prevent duplicate sparse data.
        """
        logger.info("🧹 Cleaning up old chunks before executing ingestion pipeline...")
        os.makedirs(self.chunk_dir, exist_ok=True)
        for file in glob.glob(os.path.join(self.chunk_dir, "*.csv")):
            try:
                os.remove(file)
            except Exception as e:
                logger.warning(f"⚠️ Failed to remove old chunk {file}: {e}")

    def _read_csv_safe(self, file_path: str) -> pd.DataFrame:
        """
        Attempts to load CSV with multiple encoding fallbacks.
        Crucial for handling Arabic datasets originating from different OS environments.
        
        Args:
            file_path (str): Absolute path to the CSV file.
            
        Returns:
            pd.DataFrame: The loaded data, or an empty DataFrame if all encodings fail.
        """
        encodings = ['utf-8-sig', 'utf-8', 'cp1256', 'latin1']
        for enc in encodings:
            try:
                # The 'python' engine is more robust for mixed-encoding files and bad lines
                return pd.read_csv(file_path, encoding=enc, engine='python', on_bad_lines='skip')
            except (UnicodeDecodeError, Exception):
                continue
        
        logger.error(f"❌ Critical Error: Could not decode {file_path} with any supported encodings.")
        return pd.DataFrame()

    def load_and_process(self) -> List[Document]:
        """
        Main ETL execution logic.
        Flow: Read Master -> Validate Schema -> Chunk -> Export for BM25 -> Return for FAISS.
        
        Returns:
            List[Document]: LangChain documents ready for vectorization.
        """
        # 1. Purge legacy data to avoid index duplication
        self._cleanup_old_chunks()
        all_documents = []
        
        if not os.path.exists(self.master_file):
            logger.error(f"❌ Master knowledge file missing at: {self.master_file}")
            return []
            
        logger.info(f"📂 Starting Ingestion Pipeline for Master Knowledge Base...")

        try:
            # 2. Safe Extraction
            df = self._read_csv_safe(self.master_file)
            
            # 3. [NEW] Strict GRC Validation via schema.py
            # Protects the Vector DB from being poisoned by malformed data
            if not validate_schema(df, "master", os.path.basename(self.master_file)):
                logger.error("❌ Ingestion halted: Master Data failed schema validation.")
                return []

            # 4. Data Sanitization
            df = df.fillna("")
            chunked_rows_buffer = []

            # 5. Transformation (Chunking & Enrichment)
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing Services"):
                raw_text = str(row["RAG_Content"]).strip()
                
                # Filter out rows with insufficient semantic value
                if not raw_text or len(raw_text) < 20:
                    continue
                
                # Extract English columns (Aligned with V2 Schema)
                service_name = str(row.get("Service_Name", "Unknown")).strip()
                sector = str(row.get("Sector", "Unknown")).strip()
                audience = str(row.get("Target_Audience", "General")).strip()
                steps = str(row.get("Service_Steps", "")).strip()
                reqs = str(row.get("Requirements", "")).strip()
                fees = str(row.get("Service_Fees", "")).strip()
                url = str(row.get("Official_URL", "")).strip()
                
                # Transform large RAG_Content into manageable semantic chunks
                text_chunks = self.text_splitter.split_text(raw_text)
                
                for i, chunk_txt in enumerate(text_chunks):
                    # Construct standardized metadata for FAISS payload
                    metadata = {
                        "source": "MOI_Master_Knowledge",
                        "sector": sector,
                        "service": service_name,
                        "audience": audience,
                        "url": url,
                        "chunk_index": i
                    }
                    
                    # Target 1: Append to primary Document list for Vector DB (FAISS)
                    all_documents.append(Document(page_content=chunk_txt, metadata=metadata))
                    
                    # Target 2: Append to Buffer for BM25 CSV Export
                    # Note: Keys are mapped back to Arabic to match the BM25 pipeline logic
                    chunked_rows_buffer.append({
                        "اسم الخدمة": service_name,
                        "RAG_Content": chunk_txt,
                        "القطاع": sector,
                        "خطوات الخدمة": steps,
                        "المستندات المطلوبة": reqs,
                        "سعر الخدمة": fees,
                        "رابط الخدمة": url
                    })

            # 6. Load (Export processed CSV for BM25 Symmetric Retrieval)
            if chunked_rows_buffer:
                output_csv = os.path.join(self.chunk_dir, "master_chunks.csv")
                pd.DataFrame(chunked_rows_buffer).to_csv(output_csv, index=False, encoding='utf-8-sig')
                logger.info(f"✅ Saved BM25 reference chunks to: {output_csv}")
            
            # 7. Proactive Memory Management for A100 environment
            del df, chunked_rows_buffer
            gc.collect()

        except Exception as e:
            logger.error(f"❌ Critical ETL Failure during processing: {e}", exc_info=True)
            return all_documents  # Return whatever was processed before failure

        # 8. Log pipeline stats
        if all_documents:
            chunk_lengths = [len(d.page_content) for d in all_documents]
            services_count = len(set(d.metadata.get('service', '') for d in all_documents))
            logger.info(
                f"✅ ETL Pipeline Success: {len(all_documents)} chunks from {services_count} services | "
                f"Avg chunk: {sum(chunk_lengths)//len(chunk_lengths)} chars | "
                f"Min: {min(chunk_lengths)} | Max: {max(chunk_lengths)}"
            )
        else:
            logger.warning("⚠️ ETL Pipeline produced 0 documents.")
        
        return all_documents

if __name__ == "__main__":
    # Local execution for standalone testing
    ingestor = DataIngestor()
    results = ingestor.load_and_process()