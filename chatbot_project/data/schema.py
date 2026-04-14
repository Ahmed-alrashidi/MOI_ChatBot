# =========================================================================
# File Name: data/schema.py
# Purpose: GRC-Grade Data Validation & Quality Enforcement.
# Project: Absher Smart Assistant (MOI ChatBot)
# Version: 5.3.0 (Strict Tolerance + Dtype-Agnostic + Fatal Duplicates)
#
# Changelog v5.2.0 → v5.3.0:
#   - [FIX] Strict tolerance: Critical columns fail at >5% invalid, not 100%.
#           (Engineer Report §8A — "100% Garbage Loophole")
#   - [FIX] Dtype-agnostic validation: All columns force-cast to str before
#           quality checks, catching numeric cols that bypass is_string_dtype.
#           (Engineer Report §8B — "Type-Casting Evasion")
#   - [FIX] Fatal duplicate detection: Duplicate Service_Names now block
#           ingestion instead of just warning. Prevents vector collision
#           and KG hallucination. (Engineer Report §8C)
#   - [NEW] Configurable thresholds via constants for easy tuning.
#
# Features:
# - V2 Schema Alignment: Supports full Master CSV headers.
# - Semantic Guard: Validates minimum content length for RAG quality.
# - Placeholder Detection: Flags "N/A", "TBD", or empty-meaning rows.
# - Comprehensive Auditing: Detailed logging for data engineering review.
# =========================================================================

import pandas as pd
from typing import List, Dict
from utils.logger import setup_logger

logger = setup_logger("Schema_Validator")

# --- 1. VALIDATION THRESHOLDS (Configurable) ---

# Critical columns: fail if more than this % of rows are invalid
CRITICAL_INVALID_THRESHOLD = 0.05   # 5% — strict for RAG_Content, Service_Name

# Non-critical columns: warn if more than this % of rows are invalid
SPARSITY_WARN_THRESHOLD = 0.30      # 30% — alert for optional columns

# RAG_Content: minimum character length for semantic value
RAG_MIN_CHARS = 50

# RAG_Content: fail if more than this % of rows are too short
RAG_SHORT_THRESHOLD = 0.30          # 30%


# --- 2. SCHEMA DEFINITION ---

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

# Columns that MUST have high data quality (block ingestion if corrupted)
CRITICAL_COLUMNS = {"RAG_Content", "Service_Name", "اسم الخدمة"}

# Known placeholder values that indicate missing/dummy data
PLACEHOLDERS = {'n/a', 'none', 'tbd', 'جاري العمل', 'لا يوجد', '-', 'null', 'nan', ''}


def validate_schema(df: pd.DataFrame, schema_type: str, filename: str) -> bool:
    """
    Performs deep-scan validation of the DataFrame.
    Enforces structural integrity and content quality to prevent
    AI bias and retrieval of low-value information.

    Returns:
        bool: True if data passes all validation gates, False to block ingestion.
    """

    # ================================================================
    # [A] STRUCTURAL VALIDATION
    # ================================================================
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

    total_rows = len(df)

    # ================================================================
    # [B] QUALITATIVE VALIDATION (Dtype-Agnostic)
    # ================================================================
    for col in required_cols:

        # [FIX v5.3.0] Force-cast ALL columns to string for validation.
        # Previously gated behind is_string_dtype(), which let numeric columns
        # (e.g., Service_Fees with NaN → float64) bypass all quality checks.
        series = df[col].fillna("").astype(str).str.strip()

        # 1. Check for total column failure (All NULL/empty)
        all_empty = (series == "").all()
        if all_empty:
            logger.error(f"❌ Integrity Failure: Column '{col}' in '{filename}' is 100% empty.")
            return False

        # 2. Detect placeholder/dummy data
        placeholder_count = series.str.lower().isin(PLACEHOLDERS).sum()
        empty_count = (series == "").sum()
        total_invalid = placeholder_count + empty_count
        invalid_ratio = total_invalid / total_rows

        # 3. [FIX v5.3.0] Strict threshold for critical columns.
        # OLD: Only failed if total_invalid == len(df) (100% garbage passed!)
        # NEW: Fails if >5% of critical column data is invalid.
        if col in CRITICAL_COLUMNS:
            if invalid_ratio > CRITICAL_INVALID_THRESHOLD:
                logger.error(
                    f"❌ Critical Data Quality Failure: Column '{col}' in '{filename}' "
                    f"has {invalid_ratio:.1%} invalid rows ({total_invalid}/{total_rows}). "
                    f"Threshold: {CRITICAL_INVALID_THRESHOLD:.0%}. Ingestion blocked."
                )
                return False

        # 4. Semantic Depth Check for RAG_Content
        if col == "RAG_Content":
            short_rows = (series.str.len() < RAG_MIN_CHARS).sum()
            short_ratio = short_rows / total_rows
            if short_ratio > RAG_SHORT_THRESHOLD:
                logger.warning(
                    f"⚠️ Content Quality Alert: {short_rows}/{total_rows} rows in 'RAG_Content' "
                    f"are too short (<{RAG_MIN_CHARS} chars). Ratio: {short_ratio:.1%}"
                )

        # 5. Sparsity alerting for non-critical columns
        if col not in CRITICAL_COLUMNS and invalid_ratio > SPARSITY_WARN_THRESHOLD:
            logger.warning(
                f"⚠️ High Sparsity in '{filename}': Column '{col}' is "
                f"{invalid_ratio:.1%} invalid/empty. May reduce search recall."
            )

    # ================================================================
    # [C] DUPLICATE DETECTION (Fatal for Vector/BM25 environments)
    # ================================================================
    if schema_type == "master" and "Service_Name" in df.columns:
        dupes = df["Service_Name"].duplicated().sum()
        if dupes > 0:
            # [FIX v5.3.0] Upgraded from warning to FATAL error.
            # Duplicate Service_Names cause vector collision in FAISS,
            # BM25 scoring confusion, and KG lookup ambiguity.
            dupe_names = df.loc[df["Service_Name"].duplicated(keep=False), "Service_Name"].unique().tolist()
            logger.error(
                f"❌ Duplicate Service_Names in '{filename}': {dupes} duplicates found. "
                f"Names: {dupe_names[:5]}{'...' if len(dupe_names) > 5 else ''}. "
                f"Resolve by appending sector: e.g., 'تجديد رخصة (المرور)'. "
                f"Ingestion blocked to prevent vector collision."
            )
            return False

    logger.info(f"✅ Schema validated: {filename} | {total_rows} rows, {len(required_cols)} columns")
    return True