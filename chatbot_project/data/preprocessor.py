import os
import pandas as pd
from typing import List, Optional
from utils.text_utils import normalize_arabic, soft_clean

# Mapping dictionary to convert file prefixes to official Arabic Sector names.
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
    Handles case-insensitivity and ensures a default fallback.
    """
    try:
        base = os.path.basename(filepath).lower()
        # Extract the first part of the filename (e.g., 'jawazat' from 'jawazat_master.csv')
        key = base.split("_")[0]
        return SECTOR_MAP.get(key, "قطاع حكومي عام")
    except Exception:
        return "غير محدد"

def clean_dataframe(df: pd.DataFrame, cols_to_clean: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Applies text cleaning pipeline to specific columns in the DataFrame.
    Optimized for memory efficiency by modifying in-place where possible.
    
    Updated to accept 'cols_to_clean' argument to match ingestion.py calls.
    """
    # If explicit columns are not provided, fall back to this comprehensive list
    if cols_to_clean is None:
        cols_to_clean = [
            "service_title_ar", "description_full", "beneficiaries", 
            "fees", "conditions", "chunk_text", "chunk_title"
        ]
    
    for col in cols_to_clean:
        if col in df.columns:
            # Drop NaNs to prevent errors during mapping
            df[col] = df[col].fillna("")
            # Apply cleaning: Ensure String -> Soft Clean -> Normalize Arabic
            df[col] = df[col].astype(str).map(soft_clean).map(normalize_arabic)
            
    return df

def build_master_text_representation(row: pd.Series) -> str:
    """
    Constructs a 'Semantic Rich Text' representation for Embedding.
    Uses natural language connectors to help the Embedding Model (BGE-M3) understand context.
    """
    # Base Sentence
    title = row.get('service_title_ar', 'خدمة غير معنونة')
    desc = row.get('description_full', '')
    
    # Natural Language Construction
    text_parts = [f"خدمة {title}: {desc}"]
    
    # Add details only if they contain meaningful content (checking length > 2 filters out "nan" or "-")
    if row.get("beneficiaries") and len(str(row["beneficiaries"])) > 2: 
        text_parts.append(f"الفئات المستفيدة من الخدمة هي: {row['beneficiaries']}.")
        
    if row.get("fees") and len(str(row["fees"])) > 1: 
        text_parts.append(f"تبلغ رسوم الخدمة: {row['fees']}.")
        
    if row.get("conditions") and len(str(row["conditions"])) > 2: 
        text_parts.append(f"الشروط والمتطلبات تشمل: {row['conditions']}.")
        
    if row.get("access_path") and len(str(row["access_path"])) > 2: 
        text_parts.append(f"طريقة الوصول للخدمة عبر أبشر: {row['access_path']}.")

    # Join with newlines to separate distinct semantic blocks
    return "\n".join(text_parts)