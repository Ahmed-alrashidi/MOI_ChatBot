import os
import glob
import pandas as pd
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import Config
from utils.logger import setup_logger
from data.schema import validate_schema, MASTER_REQUIRED, CHUNK_REQUIRED
from data.preprocessor import infer_sector, clean_dataframe, build_master_text_representation

logger = setup_logger(__name__)

class DataIngestor:
    def __init__(self):
        self.master_dir = Config.DATA_MASTER_DIR
        self.chunks_dir = Config.DATA_CHUNKS_DIR
        
        # Splitter configuration (Same as Cell 1)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=120,
            length_function=len
        )

    def load_and_process(self):
        """
        Main pipeline: Load CSVs -> Validate -> Clean -> Create Documents.
        Returns a list of LangChain Documents ready for FAISS.
        """
        logger.info("🚀 Starting Data Ingestion Pipeline...")
        
        master_files = sorted(glob.glob(os.path.join(self.master_dir, "*.csv")))
        chunk_files = sorted(glob.glob(os.path.join(self.chunks_dir, "*.csv")))
        
        if not master_files or not chunk_files:
            logger.error("❌ No CSV files found in data directories.")
            return []

        logger.info(f"📦 Found {len(master_files)} Master files and {len(chunk_files)} Chunk files.")

        # 1. Process Master Files (High-level service info)
        master_docs = []
        for f in master_files:
            try:
                df = pd.read_csv(f)
                validate_schema(df, MASTER_REQUIRED, f)
                
                df = clean_dataframe(df, ["service_title_ar", "description_full", "beneficiaries", "fees", "conditions"])
                sector = infer_sector(f)
                
                # Group by service to avoid duplicates, pick best description
                for service_id, group in df.groupby("service_id"):
                    best_row = group.loc[group["description_full"].str.len().idxmax()]
                    text_content = build_master_text_representation(best_row)
                    
                    doc = Document(
                        page_content=text_content,
                        metadata={
                            "sector": sector,
                            "service_id": service_id,
                            "service_title": best_row["service_title_ar"],
                            "doc_level": "service_master",
                            "source_file": f
                        }
                    )
                    master_docs.append(doc)
            except Exception as e:
                logger.warning(f"⚠️ Error processing master file {f}: {e}")

        # 2. Process Chunk Files (Detailed info)
        chunk_docs = []
        for f in chunk_files:
            try:
                df = pd.read_csv(f)
                validate_schema(df, CHUNK_REQUIRED, f)
                
                df = clean_dataframe(df, ["chunk_title", "chunk_text"])
                df = df[df["chunk_text"].str.strip() != ""] # Remove empty rows
                sector = infer_sector(f)
                
                for _, row in df.iterrows():
                    base_meta = {
                        "sector": sector,
                        "service_id": row["service_id"],
                        "chunk_id": row["chunk_id"],
                        "chunk_title": row["chunk_title"],
                        "doc_level": "service_chunk",
                        "source_file": f
                    }
                    
                    text = row["chunk_text"]
                    
                    # Split if too long, otherwise keep as is
                    if len(text) > 1000:
                        parts = self.splitter.split_text(text)
                        for i, p in enumerate(parts):
                            meta = base_meta.copy()
                            meta["chunk_part"] = f"{i+1}/{len(parts)}"
                            chunk_docs.append(Document(page_content=p, metadata=meta))
                    else:
                        chunk_docs.append(Document(page_content=text, metadata=base_meta))
                        
            except Exception as e:
                logger.warning(f"⚠️ Error processing chunk file {f}: {e}")

        total_docs = master_docs + chunk_docs
        logger.info(f"✅ Ingestion Complete. Created {len(total_docs)} documents (Masters: {len(master_docs)}, Chunks: {len(chunk_docs)})")
        return total_docs

# Simple helper to run ingestion
def load_all_documents():
    ingestor = DataIngestor()
    return ingestor.load_and_process()