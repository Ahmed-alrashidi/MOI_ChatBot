# =========================================================================
# File Name: data/ingestion.py
# Purpose: ETL (Extract, Transform, Load) Pipeline for Master Data Ingestion.
# Project: Absher Smart Assistant (MOI ChatBot)
# Features:
# - Encoding Autonomy: Robust handling of Arabic text (UTF-8, CP1256).
# - Data Validation: Filters empty rows and ensures content integrity.
# - Metadata Mapping: Enriches documents with Sector, Audience, and Service info.
# =========================================================================

import os
import glob
import pandas as pd
from typing import List
from tqdm import tqdm
from langchain.schema import Document
from config import Config
from utils.logger import setup_logger

# Initialize module logger linked to project-wide Config
logger = setup_logger("Data_Ingestor")

class DataIngestor:
    """
    Handles the ingestion of raw CSV data into the system.
    It transforms flat tabular data (Master CSVs) into LangChain Document 
    objects suitable for vector embedding and RAG retrieval.
    """
    
    def __init__(self):
        """Initializes the ingestor with the master data directory from Config."""
        self.master_dir = Config.DATA_MASTER_DIR
        
    def _read_csv_safe(self, file_path: str) -> pd.DataFrame:
        """
        Attempts to read CSV files using multiple encodings.
        This is critical for Arabic datasets which are often exported in 
        CP1256 (Windows Arabic) or UTF-8-SIG (Excel UTF-8).
        
        Args:
            file_path: Absolute path to the CSV file.
            
        Returns:
            pd.DataFrame: Loaded data or an empty DataFrame if all attempts fail.
        """
        encodings = ['utf-8-sig', 'utf-8', 'cp1256', 'latin1']
        for enc in encodings:
            try:
                # engine='python' is used for better handling of multi-line strings or complex delimiters
                df = pd.read_csv(file_path, encoding=enc, engine='python', on_bad_lines='skip')
                return df
            except UnicodeDecodeError:
                # Silently try the next encoding in the list
                continue
            except Exception as e:
                logger.error(f"❌ Read Error ({enc}) for {file_path}: {e}")
                break
        
        logger.error(f"❌ Failed to read {file_path} with all attempted encodings.")
        return pd.DataFrame()

    def load_and_process(self) -> List[Document]:
        """
        Main orchestration method for the ingestion pipeline.
        Steps: 1. Scan Dir -> 2. Safe Read -> 3. Clean/Fill -> 4. Map Metadata.
        
        Returns:
            List[Document]: A list of ready-to-index LangChain Document objects.
        """
        documents = []
        
        # 1. Directory Integrity Check
        if not os.path.exists(self.master_dir):
            logger.error(f"❌ Master Data directory missing: {self.master_dir}")
            return []

        # 2. Scanning for Master Data CSV files
        csv_files = glob.glob(os.path.join(self.master_dir, "*.csv"))
        
        if not csv_files:
            logger.warning(f"⚠️ No CSV files found in {self.master_dir}")
            return []
            
        logger.info(f"📂 Found {len(csv_files)} Master CSV files to ingest.")

        # 3. Iterative Processing with Progress Bar
        for file_path in tqdm(csv_files, desc="Ingesting Master Data"):
            try:
                # Read the CSV with robust encoding detection
                df = self._read_csv_safe(file_path)
                if df.empty:
                    logger.warning(f"⚠️ Skipping empty file: {os.path.basename(file_path)}")
                    continue
                
                # Validation: 'RAG_Content' is the mandatory column for indexing
                if "RAG_Content" not in df.columns:
                    logger.warning(f"⚠️ Skipping {os.path.basename(file_path)}: Missing 'RAG_Content' column.")
                    continue

                # Data Cleaning: Convert NaNs to empty strings to avoid crashes during concatenation
                df = df.fillna("")

                # Row-by-row transformation into Document objects
                for _, row in df.iterrows():
                    content = str(row["RAG_Content"]).strip()
                    
                    # Heuristic Filter: Ensure content has enough substance to be useful (min 10 chars)
                    if not content or len(content) < 10:
                        continue
                        
                    # Metadata Extraction: Extract features to allow for advanced filtering/filtering in RAG
                    filename = os.path.basename(file_path)
                    # Infer sector name from filename logic (e.g., Traffic_Master.csv -> Traffic)
                    sector_name = filename.replace("_Master.csv", "").replace(".csv", "")
                    
                    metadata = {
                        "source": filename,
                        "sector_eng": sector_name,
                        # Map Arabic columns safely to metadata keys
                        "service_name": str(row.get("اسم الخدمة", "Unknown")).strip(),
                        "sector_ar": str(row.get("القطاع", "Unknown")).strip(),
                        "audience": str(row.get("الجمهور المستهدف", "General")).strip()
                    }
                    
                    # Append the processed document to the final collection
                    documents.append(Document(page_content=content, metadata=metadata))

            except Exception as e:
                logger.error(f"❌ Error processing file {file_path}: {e}")

        logger.info(f"✅ Ingestion Complete. Prepared {len(documents)} documents for Vector DB.")
        return documents

# Logic for direct execution / testing
if __name__ == "__main__":
    # Test Ingestion Flow
    ingestor = DataIngestor()
    docs = ingestor.load_and_process()
    if docs:
        print(f"\n--- Sample Ingested Document ---\nContent: {docs[0].page_content[:150]}...\nMetadata: {docs[0].metadata}")