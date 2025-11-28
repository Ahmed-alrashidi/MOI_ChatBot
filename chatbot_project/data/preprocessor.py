import os
import pandas as pd
from typing import List
from utils.text_utils import normalize_arabic, soft_clean

# Mapping dictionary to convert file prefixes (e.g., 'jawazat') to official Arabic Sector names.
# This ensures consistent naming across the application regardless of the input filename.
SECTOR_MAP = {
    "jawazat": "الجوازات",
    "muroor": "المرور",
    "ahwal": "الأحوال المدنية",
    "waffedeen": "شؤون الوافدين",
    "tafweed": "إدارة التفاويض",
    "prisons": "المديرية العامة للسجون",
    "amn": "الأمن العام",
    "niyaba": "النيابة العامة",
    "moiministry": "وزارة الداخلية",
    "hajj": "وزارة الحج والعمرة"
}

def infer_sector(filepath: str) -> str:
    """
    Extracts and maps the sector name from the CSV filename.
    
    It expects filenames in the format 'prefix_rest.csv' (e.g., 'jawazat_master.csv').
    If the prefix isn't found in the map, it returns 'Non-specified'.

    Args:
        filepath (str): The full path to the input CSV file.

    Returns:
        str: The official Arabic name of the sector.
    """
    base = os.path.basename(filepath).lower()
    # Extract the first part of the filename before the underscore
    key = base.split("_")[0]
    return SECTOR_MAP.get(key, "غير محدد")

def clean_dataframe(df: pd.DataFrame, columns_to_clean: List[str]) -> pd.DataFrame:
    """
    Iterates over specific columns in a DataFrame and applies text normalization.
    
    It uses helper functions to:
    1. Remove noise (soft_clean).
    2. Normalize Arabic characters (unify Alef, Ya, Ta-Marbuta, etc.).

    Args:
        df (pd.DataFrame): The input dataframe.
        columns_to_clean (List[str]): List of column names to apply cleaning to.

    Returns:
        pd.DataFrame: The dataframe with cleaned text columns.
    """
    for col in columns_to_clean:
        if col in df.columns:
            # Apply cleaning pipeline: ensure string -> soft clean -> normalize arabic
            df[col] = df[col].astype(str).map(soft_clean).map(normalize_arabic)
    return df

def build_master_text_representation(row: pd.Series) -> str:
    """
    Constructs a single, rich-text string from multiple columns of a Master record.
    
    This function flattens the structured data (title, description, fees, etc.) into 
    one context string. This is crucial for Embedding models to capture the 
    full context of a service in a single vector.

    Args:
        row (pd.Series): A single row from the Master DataFrame.

    Returns:
        str: A concatenated string combining all relevant fields with separators.
    """
    parts = [
        f"اسم الخدمة: {row.get('service_title_ar', '')}",
        f"الوصف: {row.get('description_full', '')}"
    ]
    
    # Conditionally add optional fields if they exist and are not empty
    if row.get("beneficiaries"): 
        parts.append(f"المستفيدون: {row['beneficiaries']}")
        
    if row.get("fees"): 
        parts.append(f"الرسوم: {row['fees']}")
        
    if row.get("conditions"): 
        parts.append(f"الشروط: {row['conditions']}")
        
    if row.get("access_path"): 
        parts.append(f"طريقة الوصول: {row['access_path']}")
    
    # Join all parts with a separator for clear distinction in the text chunk
    return " | ".join(parts)