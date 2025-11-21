import pandas as pd
from typing import List

# Required columns for Master CSVs (Service Descriptions)
MASTER_REQUIRED = [
    "service_id", "service_title_ar", "description_full",
    "beneficiaries", "fees", "conditions", "access_path"
]

# Required columns for Chunk CSVs (Detailed Parts)
CHUNK_REQUIRED = [
    "chunk_id", "service_id", "chunk_title",
    "chunk_text", "chunk_type", "language", "meta"
]

def validate_schema(df: pd.DataFrame, schema: List[str], filename: str) -> None:
    """
    Checks if the DataFrame contains all required columns.
    Raises ValueError if columns are missing.
    """
    missing = [c for c in schema if c not in df.columns]
    if missing:
        raise ValueError(f"❌ Schema mismatch in {filename}: missing {missing}")