import pandas as pd
from typing import List
from utils.logger import setup_logger

logger = setup_logger(__name__)

# --- 1. Master Data Schema ---
MASTER_REQUIRED: List[str] = [
    "service_id",       # Critical for linking
    "service_title_ar", # Critical for semantics
    "description_full", # Critical for semantics
    "beneficiaries",
    "fees",
    "conditions",
    "access_path"
]

# --- 2. Chunk Data Schema ---
CHUNK_REQUIRED: List[str] = [
    "chunk_id",
    "service_id",
    "chunk_title",
    "chunk_text",       # THE most important field
    "chunk_type",
    "language"
]

# Defines fields that MUST NOT be empty/null/whitespace
CRITICAL_FIELDS = {
    "service_id", 
    "service_title_ar", 
    "description_full", 
    "chunk_text", 
    "chunk_id"
}

def validate_schema(df: pd.DataFrame, schema: List[str], filename: str) -> None:
    """
    Performs Strict Validation on DataFrames.
    Checks for:
    1. Column Existence (Schema Match).
    2. Data Types (Enforces ID string format).
    3. Content Integrity (No empty critical fields).

    Raises:
        ValueError: If critical columns are missing (stops processing).
    """
    # 1. Check Missing Columns
    missing = [c for c in schema if c not in df.columns]
    if missing:
        error_msg = f"❌ Schema Validation Failed in '{filename}': Missing columns {missing}"
        logger.error(error_msg)
        # CRITICAL FIX: Raise error to stop ingestion for this file
        raise ValueError(error_msg)

    # 2. Type Enforcement (Sanitize IDs)
    # Ensure service_id is always string to prevent int/str mismatch issues
    if "service_id" in df.columns:
        df["service_id"] = df["service_id"].astype(str).str.strip()
        
    if "chunk_id" in df.columns:
        df["chunk_id"] = df["chunk_id"].astype(str).str.strip()

    # 3. Critical Content Check (No Empty Values)
    # Filter schema cols that are present in the Critical Set
    cols_to_check = [c for c in schema if c in CRITICAL_FIELDS]
    
    for col in cols_to_check:
        # Check for NaN or Empty Strings (after stripping whitespace)
        if df[col].isna().any() or (df[col].astype(str).str.strip() == "").any():
            invalid_count = df[col].isna().sum() + (df[col].astype(str).str.strip() == "").sum()
            if invalid_count > 0:
                logger.warning(
                    f"⚠️ Data Integrity Warning in '{filename}': "
                    f"Column '{col}' has {invalid_count} empty/null rows. These rows may be skipped during processing."
                )
                # We don't raise error here, we just warn. 
                # The ingestion loop filters out empty rows later.