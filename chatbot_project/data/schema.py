# =========================================================================
# File Name: data/schema.py
# Purpose: GRC-Grade Data Validation & Quality Enforcement.
# Project: Absher Smart Assistant (MOI ChatBot)
# Features:
# - V2 Schema Alignment: Supports full Master CSV headers.
# - Semantic Guard: Validates minimum content length for RAG quality.
# - Placeholder Detection: Flags "N/A", "TBD", or empty-meaning rows.
# - Comprehensive Auditing: Detailed logging for data engineering review.
# =========================================================================

import pandas as pd
from typing import List, Dict
from pandas.api.types import is_string_dtype
from utils.logger import setup_logger

# Initialize specialized logger for compliance auditing
logger = setup_logger("Schema_Validator")

# --- 1. DEFINITION OF TRUTH (Schema Mapping) ---
# Aligned with the production MOI_Master_Knowledge schema
SCHEMA_MAP: Dict[str, List[str]] = {
    "master": [
        "Sector", 
        "Service_Name", 
        "Target_Audience", 
        "Service_Description",
        "RAG_Content", 
        "Service_Steps", 
        "Requirements", 
        "Service_Fees", 
        "Official_URL"
    ],
    "chunk": [
        "اسم الخدمة", 
        "RAG_Content", 
        "القطاع", 
        "خطوات الخدمة", 
        "المستندات المطلوبة", 
        "سعر الخدمة", 
        "رابط الخدمة"
    ]
}

def validate_schema(df: pd.DataFrame, schema_type: str, filename: str) -> bool:
    """
    Performs a deep-scan validation of the DataFrame. 
    Enforces structural integrity and content quality to prevent 
    AI bias and retrieval of low-value information.
    """
    
    # [A] STRUCTURAL VALIDATION
    if schema_type not in SCHEMA_MAP:
        logger.error(f"❌ Configuration Error: Schema type '{schema_type}' is undefined.")
        return False

    required_cols = SCHEMA_MAP[schema_type]

    if df is None or df.empty:
        logger.error(f"❌ Input Error: '{filename}' is empty or null.")
        return False

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error(f"❌ Compliance Violation in '{filename}': Missing columns {missing}")
        return False

    # [B] QUALITATIVE VALIDATION
    placeholders = {'n/a', 'none', 'tbd', 'جاري العمل', 'لا يوجد', '-', 'null'}
    
    for col in required_cols:
        # 1. Check for total column failure (All NULL)
        if df[col].isnull().all():
            logger.error(f"❌ Integrity Failure: Column '{col}' in '{filename}' is 100% empty.")
            return False

        # 2. Content Quality Analysis (String columns only)
        if is_string_dtype(df[col]):
            series = df[col].fillna("").astype(str).str.strip()
            
            # Detect Placeholder/Dummy data
            placeholder_matches = series.str.lower().isin(placeholders).sum()
            empty_matches = (series == "").sum()
            total_invalid = placeholder_matches + empty_matches
            
            # Fatal Error: If the most critical columns are invalid
            if col in ["RAG_Content", "Service_Name"] and total_invalid == len(df):
                logger.error(f"❌ Critical Failure: Mandatory column '{col}' has no valid semantic data.")
                return False

            # Semantic Depth Check: RAG_Content should be informative
            if col == "RAG_Content":
                short_rows = (series.str.len() < 50).sum()
                if short_rows > (len(df) * 0.3):
                    logger.warning(f"⚠️ Content Quality Alert: {short_rows} rows in 'RAG_Content' are too short (<50 chars).")

            # Sparsity Alerting
            if total_invalid > (len(df) * 0.5):
                logger.warning(
                    f"⚠️ High Sparsity in '{filename}': Column '{col}' is {total_invalid/len(df):.1%} invalid/empty. "
                    "This may reduce search recall."
                )

    # Duplicate service name check
    if schema_type == "master" and "Service_Name" in df.columns:
        dupes = df["Service_Name"].duplicated().sum()
        if dupes > 0:
            logger.warning(f"⚠️ {dupes} duplicate Service_Name(s) detected in '{filename}'. May cause retrieval conflicts.")

    logger.info(f"✅ Schema validated: {filename} | {len(df)} rows, {len(required_cols)} columns")
    return True