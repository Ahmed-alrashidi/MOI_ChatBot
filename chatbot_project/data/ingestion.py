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

# Initialize module logger
logger = setup_logger(__name__)

class DataIngestor:
    """
    Orchestrates the ETL (Extract, Transform, Load) pipeline for the RAG system.
    
    Responsibilities:
    1. Scan directories for raw CSV data (Masters and Chunks).
    2. Validate schema compliance to ensure data integrity.
    3. Clean and normalize text data.
    4. Convert tabular data into LangChain 'Document' objects ready for embedding.
    """
    
    def __init__(self):
        """
        Initializes the Ingestor with paths and splitting configurations.
        
        The splitter is initialized here to ensure consistent chunking parameters 
        (size and overlap) across all data sources, pulled directly from Config.
        """
        self.master_dir = Config.DATA_MASTER_DIR
        self.chunks_dir = Config.DATA_CHUNKS_DIR
        
        # Text Splitter for safety (in case manual chunks are still too large)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len
        )

    def load_and_process(self) -> List[Document]:
        """
        Executes the main ingestion pipeline.
        
        Steps:
        1. Identify all CSV files in Master and Chunk directories.
        2. Process 'Master' files (high-level service descriptions).
        3. Process 'Chunk' files (detailed sub-information).
        4. Merge all documents into a single list.
        
        Returns:
            List[Document]: A combined list of processed documents ready for the Vector Store.
        """
        logger.info("🚀 Starting Data Ingestion Pipeline...")
        
        # 1. File Discovery
        master_files = sorted(glob.glob(os.path.join(self.master_dir, "*.csv")))
        chunk_files = sorted(glob.glob(os.path.join(self.chunks_dir, "*.csv")))
        
        # Graceful exit if no data is found
        if not master_files and not chunk_files:
            logger.error(f"❌ No CSV files found in:\n - {self.master_dir}\n - {self.chunks_dir}")
            return []

        logger.info(f"📦 Found {len(master_files)} Master files and {len(chunk_files)} Chunk files.")

        # 2. Process Files
        master_docs = self._process_master_files(master_files)
        chunk_docs = self._process_chunk_files(chunk_files)

        # 3. Aggregate Results
        total_docs = master_docs + chunk_docs
        
        if not total_docs:
            logger.warning("⚠️ Ingestion finished but 0 documents were created. Check your CSV files.")
        else:
            logger.info(f"✅ Ingestion Complete. Created {len(total_docs)} documents (Masters: {len(master_docs)}, Chunks: {len(chunk_docs)})")
            
        return total_docs

    def _process_master_files(self, files: List[str]) -> List[Document]:
        """
        Processes 'Master' CSV files containing general service information.
        
        Key Logic:
        - Validates columns against MASTER_REQUIRED schema.
        - Performs de-duplication: Groups by 'service_id' and selects the row with 
          the longest description to ensure maximum context.
        """
        docs = []
        for f in files:
            try:
                df = pd.read_csv(f)
                if df.empty:
                    logger.warning(f"⚠️ Skipped empty file: {f}")
                    continue

                # Schema Validation
                validate_schema(df, MASTER_REQUIRED, f)
                
                # Text Cleaning
                cols_to_clean = ["service_title_ar", "description_full", "beneficiaries", "fees", "conditions"]
                df = clean_dataframe(df, cols_to_clean)
                
                # Metadata Inference
                sector = infer_sector(f)
                
                # Deduplication Strategy:
                # Some services might appear multiple times. We group by ID and pick 
                # the one with the richest content (longest description).
                for service_id, group in df.groupby("service_id"):
                    best_row = group.loc[group["description_full"].str.len().idxmax()]
                    
                    # Convert row to formatted text string
                    text_content = build_master_text_representation(best_row)
                    
                    doc = Document(
                        page_content=text_content,
                        metadata={
                            "sector": sector,
                            "service_id": service_id,
                            "service_title": best_row["service_title_ar"],
                            "doc_level": "service_master", # Tagging for filtering later
                            "source_file": os.path.basename(f)
                        }
                    )
                    docs.append(doc)
                    
            except Exception as e:
                logger.warning(f"⚠️ Error processing Master file '{os.path.basename(f)}': {e}")
        return docs

    def _process_chunk_files(self, files: List[str]) -> List[Document]:
        """
        Processes 'Chunk' CSV files containing detailed procedural data.
        
        Key Logic:
        - Validates columns against CHUNK_REQUIRED schema.
        - Checks text length: If a pre-defined chunk is still larger than Config.CHUNK_SIZE,
          it uses the recursive splitter to break it down further (Safety Net).
        """
        docs = []
        for f in files:
            try:
                df = pd.read_csv(f)
                if df.empty:
                    logger.warning(f"⚠️ Skipped empty file: {f}")
                    continue

                # Schema Validation
                validate_schema(df, CHUNK_REQUIRED, f)
                
                # Text Cleaning
                df = clean_dataframe(df, ["chunk_title", "chunk_text"])
                
                # Filter invalid rows
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
                    
                    # Safety Split:
                    # Even though data comes from "Chunks" folder, we verify it fits our embedding window.
                    if len(text) > Config.CHUNK_SIZE:
                        parts = self.splitter.split_text(text)
                        for i, p in enumerate(parts):
                            meta = base_meta.copy()
                            meta["chunk_part"] = f"{i+1}/{len(parts)}" # Track split parts
                            docs.append(Document(page_content=p, metadata=meta))
                    else:
                        docs.append(Document(page_content=text, metadata=base_meta))
                        
            except Exception as e:
                logger.warning(f"⚠️ Error processing Chunk file '{os.path.basename(f)}': {e}")
        return docs

# Helper function to allow standalone execution of ingestion logic
def load_all_documents() -> List[Document]:
    ingestor = DataIngestor()
    return ingestor.load_and_process()