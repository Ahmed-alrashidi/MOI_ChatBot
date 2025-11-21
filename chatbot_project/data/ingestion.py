import os
import glob
from typing import List
import pandas as pd
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import Config
from utils.logger import setup_logger
from data.schema import validate_schema, MASTER_REQUIRED, CHUNK_REQUIRED
from data.preprocessor import infer_sector, clean_dataframe, build_master_text_representation

logger = setup_logger(__name__)

class DataIngestor:
    """
    Handles the ETL pipeline: Extract (load CSVs), Transform (clean & split), Load (create Documents).
    """
    def __init__(self):
        self.master_dir = Config.DATA_MASTER_DIR
        self.chunks_dir = Config.DATA_CHUNKS_DIR
        
        # Initialize Splitter using values from Config to ensure consistency
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len
        )

    def load_and_process(self) -> List[Document]:
        """
        Main pipeline execution.
        1. Scans directories for CSV files.
        2. Validates schema and cleans data.
        3. Converts rows into LangChain Documents.
        
        Returns:
            List[Document]: A list of processed documents ready for embedding.
        """
        logger.info("🚀 Starting Data Ingestion Pipeline...")
        
        # Get list of files
        master_files = sorted(glob.glob(os.path.join(self.master_dir, "*.csv")))
        chunk_files = sorted(glob.glob(os.path.join(self.chunks_dir, "*.csv")))
        
        # Early exit if no data found
        if not master_files and not chunk_files:
            logger.error(f"❌ No CSV files found in:\n - {self.master_dir}\n - {self.chunks_dir}")
            return []

        logger.info(f"📦 Found {len(master_files)} Master files and {len(chunk_files)} Chunk files.")

        master_docs = self._process_master_files(master_files)
        chunk_docs = self._process_chunk_files(chunk_files)

        total_docs = master_docs + chunk_docs
        
        if not total_docs:
            logger.warning("⚠️ Ingestion finished but 0 documents were created. Check your CSV files.")
        else:
            logger.info(f"✅ Ingestion Complete. Created {len(total_docs)} documents (Masters: {len(master_docs)}, Chunks: {len(chunk_docs)})")
            
        return total_docs

    def _process_master_files(self, files: List[str]) -> List[Document]:
        """Helper to process Master-level CSVs."""
        docs = []
        for f in files:
            try:
                df = pd.read_csv(f)
                if df.empty:
                    logger.warning(f"⚠️ Skipped empty file: {f}")
                    continue

                validate_schema(df, MASTER_REQUIRED, f)
                
                # Clean text columns
                cols_to_clean = ["service_title_ar", "description_full", "beneficiaries", "fees", "conditions"]
                df = clean_dataframe(df, cols_to_clean)
                
                sector = infer_sector(f)
                
                # Group by service_id to de-duplicate and pick the longest description
                for service_id, group in df.groupby("service_id"):
                    # Pick the row with the longest description as the representative
                    best_row = group.loc[group["description_full"].str.len().idxmax()]
                    
                    text_content = build_master_text_representation(best_row)
                    
                    doc = Document(
                        page_content=text_content,
                        metadata={
                            "sector": sector,
                            "service_id": service_id,
                            "service_title": best_row["service_title_ar"],
                            "doc_level": "service_master",
                            "source_file": os.path.basename(f)
                        }
                    )
                    docs.append(doc)
                    
            except Exception as e:
                logger.warning(f"⚠️ Error processing Master file '{os.path.basename(f)}': {e}")
        return docs

    def _process_chunk_files(self, files: List[str]) -> List[Document]:
        """Helper to process Chunk-level CSVs."""
        docs = []
        for f in files:
            try:
                df = pd.read_csv(f)
                if df.empty:
                    logger.warning(f"⚠️ Skipped empty file: {f}")
                    continue

                validate_schema(df, CHUNK_REQUIRED, f)
                
                # Clean text columns
                df = clean_dataframe(df, ["chunk_title", "chunk_text"])
                
                # Remove rows with empty text
                df = df[df["chunk_text"].str.strip() != ""]
                
                sector = infer_sector(f)
                
                for _, row in df.iterrows():
                    base_meta = {
                        "sector": sector,
                        "service_id": row["service_id"],
                        "chunk_id": row["chunk_id"],
                        "chunk_title": row["chunk_title"],
                        "doc_level": "service_chunk",
                        "source_file": os.path.basename(f)
                    }
                    
                    text = row["chunk_text"]
                    
                    # Split logic: If text > Config.CHUNK_SIZE, split it. Else keep as is.
                    # We use a slightly lower threshold (1000) to trigger splitting logic if needed,
                    # but the splitter itself respects Config.CHUNK_SIZE
                    if len(text) > Config.CHUNK_SIZE:
                        parts = self.splitter.split_text(text)
                        for i, p in enumerate(parts):
                            meta = base_meta.copy()
                            meta["chunk_part"] = f"{i+1}/{len(parts)}"
                            docs.append(Document(page_content=p, metadata=meta))
                    else:
                        docs.append(Document(page_content=text, metadata=base_meta))
                        
            except Exception as e:
                logger.warning(f"⚠️ Error processing Chunk file '{os.path.basename(f)}': {e}")
        return docs

# Simple helper to run ingestion directly if needed
def load_all_documents() -> List[Document]:
    ingestor = DataIngestor()
    return ingestor.load_and_process()