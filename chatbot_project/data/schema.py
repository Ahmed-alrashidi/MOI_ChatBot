import pandas as pd
from typing import List

# --- 1. Master Data Schema ---
# Defines the structure for high-level service descriptions.
# These columns capture the "General Knowledge" about a service.
MASTER_REQUIRED: List[str] = [
    "service_id",       # Unique identifier for the service
    "service_title_ar", # Arabic title of the service
    "description_full", # Detailed description (Main context source)
    "beneficiaries",    # Who can use this service?
    "fees",             # Cost/Fees information
    "conditions",       # Prerequisites or requirements
    "access_path"       # How to access (App, Portal, etc.)
]

# --- 2. Chunk Data Schema ---
# Defines the structure for granular/detailed parts of a service.
# These are used for specific Q&A (e.g., "Step 3 in renewing passport").
CHUNK_REQUIRED: List[str] = [
    "chunk_id",    # Unique ID for the specific chunk
    "service_id",  # Foreign key linking back to the Master service
    "chunk_title", # Title of the specific section/step
    "chunk_text",  # The actual content text
    "chunk_type",  # Type classification (e.g., 'procedure', 'faq')
    "language",    # Language code (ar/en)
    "meta"         # Extra metadata JSON (optional)
]

def validate_schema(df: pd.DataFrame, schema: List[str], filename: str) -> None:
    """
    Validates that a DataFrame conforms to the expected schema requirements.
    
    This function acts as a gatekeeper in the ingestion pipeline. It checks 
    if all mandatory columns exist in the loaded CSV. If any are missing, 
    it blocks execution to prevent downstream errors during embedding.

    Args:
        df (pd.DataFrame): The pandas DataFrame loaded from the CSV.
        schema (List[str]): A list of column names that MUST be present.
        filename (str): The name of the file being validated (for error logging).

    Raises:
        ValueError: If one or more required columns are missing from the DataFrame.
    """
    # Identify missing columns by checking schema against DataFrame columns
    missing = [c for c in schema if c not in df.columns]
    
    if missing:
        # Raise a blocking error to stop processing of invalid files immediately
        raise ValueError(f"❌ Schema mismatch in {filename}: missing columns {missing}")