# =========================================================================
# File Name: data/ingestion.py
# Project: Absher Smart Assistant (MOI ChatBot)
# Architecture: Cross-Lingual Hybrid RAG (BGE-M3 + BM25 + ALLaM-7B)
#
# Affiliation: King Abdullah University of Science and Technology (KAUST)
# Team: Ahmed AlRashidi, Sultan Alshaibani, Fahad Alqahtani, 
#       Rakan Alharbi, Sultan Alotaibi, Abdulaziz Almutairi.
# Advisors: Prof. Naeemullah Khan & Dr. Salman Khan
# =========================================================================

import os
import glob
import pandas as pd
from typing import List
from tqdm import tqdm
from langchain.schema import Document
from config import Config
from utils.logger import setup_logger

# Initialize module logger
logger = setup_logger(__name__)

class DataIngestor:
    """
    Handles the ETL (Extract, Transform, Load) process specifically for the Vector Store.
    
    Responsibility:
    - Reads 'Master' CSV files from the configured directory.
    - Extracts the pre-formatted 'RAG_Content' column.
    - Converts rows into LangChain Document objects with rich metadata.
    """
    
    def __init__(self):
        self.master_dir = Config.DATA_MASTER_DIR
        
    def load_master_documents(self) -> List[Document]:
        """
        Loads and converts Master CSV data into Documents for the Dense Vector Store.
        
        Returns:
            List[Document]: A list of documents ready for BGE-M3 embedding.
        """
        documents = []
        
        # Verify directory exists
        if not os.path.exists(self.master_dir):
            logger.error(f"❌ Master Data directory not found: {self.master_dir}")
            return []

        # Get all CSV files in the Master directory
        # Uses recursive glob if needed, currently flat structure
        csv_files = glob.glob(os.path.join(self.master_dir, "*.csv"))
        
        if not csv_files:
            logger.warning(f"⚠️ No CSV files found in {self.master_dir}")
            return []
            
        logger.info(f"📂 Found {len(csv_files)} Master CSV files to process.")

        # Iterate over files with a progress bar
        for file_path in tqdm(csv_files, desc="Ingesting Master Data"):
            try:
                # Read CSV (handling potential encoding issues automatically via pandas)
                df = pd.read_csv(file_path)
                
                # Validation: Ensure the critical RAG column exists
                if "RAG_Content" not in df.columns:
                    logger.warning(f"⚠️ Skipping {os.path.basename(file_path)}: Missing 'RAG_Content' column.")
                    continue

                # Process each row
                for _, row in df.iterrows():
                    content = row["RAG_Content"]
                    
                    # Skip empty or non-string content
                    if not isinstance(content, str) or not content.strip():
                        continue
                        
                    # Extract Metadata from filename and columns
                    # Filename format expected: "SectorName_Master.csv"
                    filename = os.path.basename(file_path)
                    sector_from_filename = filename.replace("_Master.csv", "").replace(".csv", "")
                    
                    metadata = {
                        "source": filename,
                        "english_sector": sector_from_filename,
                        "service_name": row.get("اسم الخدمة", "Unknown"),
                        "arabic_sector": row.get("القطاع", "Unknown"),
                        "target_audience": row.get("الجمهور المستهدف", "General")
                    }
                    
                    # Create the Document object
                    doc = Document(page_content=content, metadata=metadata)
                    documents.append(doc)

            except Exception as e:
                logger.error(f"❌ Error processing file {file_path}: {e}")

        logger.info(f"✅ Ingestion Complete. Prepared {len(documents)} documents for Vector Store.")
        return documents

# Helper block for standalone testing
if __name__ == "__main__":
    ingestor = DataIngestor()
    docs = ingestor.load_master_documents()
    if docs:
        print(f"\n--- Sample Document ---\nContent: {docs[0].page_content[:100]}...\nMetadata: {docs[0].metadata}")