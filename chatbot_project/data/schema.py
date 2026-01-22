# =========================================================================
# File Name: data/schema.py
# Project: Absher Smart Assistant (MOI ChatBot)
# Architecture: Cross-Lingual Hybrid RAG (BGE-M3 + BM25 + ALLaM-7B)
#
# Affiliation: King Abdullah University of Science and Technology (KAUST)
# Team: Ahmed AlRashidi, Sultan Alshaibani, Fahad Alqahtani, 
#       Rakan Alharbi, Sultan Alotaibi, Abdulaziz Almutairi.
# Advisors: Prof. Naeemullah Khan & Dr. Salman Khan
# =========================================================================

import pandas as pd
from typing import List
from utils.logger import setup_logger

# Initialize module logger
logger = setup_logger(__name__)

# --- 1. Master Data Schema (For Vector Store) ---
# These columns MUST exist in 'Data_Master/*.csv' files
MASTER_REQUIRED: List[str] = [
    "اسم الخدمة",      # Service Name (Primary Key)
    "القطاع",          # Sector Name
    "RAG_Content"      # The Critical Column for Embeddings
]

# --- 2. Chunk Data Schema (For BM25/Sparse Search) ---
# These columns MUST exist in 'Data_Chunk/*.csv' files
CHUNK_REQUIRED: List[str] = [
    "اسم الخدمة",          # Service Name
    "خطوات الخدمة",        # Service Steps (Procedural details)
    "المستندات المطلوبة"   # Required Documents
]

def validate_schema(df: pd.DataFrame, required_cols: List[str], filename: str) -> None:
    """
    Performs strict validation on DataFrames before ingestion.
    Ensures that the CSV files strictly follow the defined structure.

    Args:
        df (pd.DataFrame): The loaded data.
        required_cols (List[str]): List of mandatory columns.
        filename (str): Filename for error context.

    Raises:
        ValueError: If critical columns are missing, stopping the pipeline.
    """
    
    # 1. Check Missing Columns
    missing = [c for c in required_cols if c not in df.columns]
    
    if missing:
        error_msg = f"❌ Schema Validation Failed in '{filename}': Missing columns {missing}"
        logger.error(error_msg)
        # Stop processing immediately to prevent corrupted index
        raise ValueError(error_msg)

    # 2. Critical Content Check
    # Ensure the dataframe is not completely empty
    if df.empty:
        logger.warning(f"⚠️ Warning: File '{filename}' is empty.")
        return

    # 3. Check for Nulls in Critical Columns (Optional but recommended)
    # We check if 'اسم الخدمة' is missing, as it's the link key
    if "اسم الخدمة" in df.columns:
        null_count = df["اسم الخدمة"].isnull().sum()
        if null_count > 0:
            logger.warning(f"⚠️ Data Integrity: '{filename}' has {null_count} rows with missing 'Service Name'.")

    logger.info(f"✅ Schema Validated for: {filename}")