# =========================================================================
# File Name: data/schema.py
# Project: Absher Smart Assistant (MOI ChatBot)
# Purpose: Data Validation & Structural Integrity Enforcement
# Features:
# - Strict Schema Enforcement: Validates both Vector (Dense) and BM25 (Sparse) inputs.
# - Quality Assurance: Detects null values, empty strings, and "ghost data".
# - Compliance Ready: Ensures datasets meet the minimum quality for AI reasoning.
# =========================================================================

import pandas as pd
from typing import List
from utils.logger import setup_logger

# Initialize a dedicated schema validator logger
logger = setup_logger("Schema_Validator")

# --- 1. Master Data Schema (For Vector Store / Dense Retrieval) ---
# 'RAG_Content' is the unified column containing the core descriptive text.
# It is mandatory for generating high-quality semantic embeddings.
MASTER_REQUIRED_COLS: List[str] = [
    "RAG_Content" 
]

# --- 2. Chunk Data Schema (For BM25 / Sparse Retrieval) ---
# These columns are essential for constructing the keyword-based search index.
# They provide the granular details needed for exact matching (Services, Steps, Docs).
CHUNK_REQUIRED_COLS: List[str] = [
    "اسم الخدمة", 
    "خطوات الخدمة", 
    "المستندات المطلوبة" 
]

def validate_schema(df: pd.DataFrame, required_cols: List[str], filename: str) -> bool:
    """
    Executes a rigorous validation suite on DataFrames prior to ingestion.
    
    This function serves as a 'Guardrail' to prevent corrupted or empty data from 
    polluting the Vector Database, which could lead to hallucinations.

    Args:
        df (pd.DataFrame): The input data to be validated.
        required_cols (List[str]): The list of mandatory column names.
        filename (str): The name of the file being processed for logging context.

    Returns:
        bool: True if the data passes all integrity checks, False otherwise.
    """
    
    # 1. Integrity Check: Verify the file is not empty
    if df.empty or len(df) == 0:
        logger.warning(f"⚠️ Validation Failed: File '{filename}' is empty or contains no records.")
        return False

    # 2. Structural Check: Verify all mandatory columns exist
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error(f"❌ Schema Violation in '{filename}': Missing mandatory columns {missing}")
        return False

    # 3. Qualitative Content Check: Ensure columns contain actionable data
    for col in required_cols:
        
        # A. Null/NaN Validation: Check if the entire column is unpopulated
        if df[col].isnull().all():
            logger.error(f"❌ Data Integrity Error in '{filename}': Column '{col}' is entirely Null/Empty.")
            return False
        
        # B. "Ghost Data" Validation: Detect columns containing only whitespace/empty strings
        # This is a critical check for data cleaned in Excel that might contain hidden spaces.
        if df[col].dtype == object: 
            # Strip whitespace and count resulting empty strings
            empty_or_space_count = df[col].astype(str).str.strip().eq("").sum()
            
            # Fail if every row in a required column is effectively empty
            if empty_or_space_count == len(df):
                logger.error(f"❌ Data Quality Error in '{filename}': Column '{col}' contains only whitespace strings.")
                return False
            
            # Warning: Log a quality alert if more than 50% of the data is missing
            # This allows the process to continue but flags the dataset for manual review.
            if empty_or_space_count > (len(df) * 0.5):
                logger.warning(f"⚠️ Data Quality Alert: '{filename}' column '{col}' is >50% sparse ({empty_or_space_count}/{len(df)} rows).")

    # Log successful validation
    logger.info(f"✅ Schema successfully validated for: {filename}")
    return True