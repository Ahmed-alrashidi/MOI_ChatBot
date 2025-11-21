import os
import pandas as pd
from utils.text_utils import normalize_arabic, soft_clean

# Map filename prefixes to Sector Names
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
    """Infers the sector name based on the filename (e.g., 'jawazat_master.csv')."""
    base = os.path.basename(filepath).lower()
    key = base.split("_")[0]
    return SECTOR_MAP.get(key, "غير محدد")

def clean_dataframe(df: pd.DataFrame, columns_to_clean: list) -> pd.DataFrame:
    """Applies text normalization to specific columns."""
    for col in columns_to_clean:
        if col in df.columns:
            df[col] = df[col].astype(str).map(soft_clean).map(normalize_arabic)
    return df

def build_master_text_representation(row: pd.Series) -> str:
    """
    Combines multiple columns into a single text block for embedding.
    Used for Master-Level documents.
    """
    parts = [
        f"اسم الخدمة: {row.get('service_title_ar', '')}",
        f"الوصف: {row.get('description_full', '')}"
    ]
    if row.get("beneficiaries"): parts.append(f"المستفيدون: {row['beneficiaries']}")
    if row.get("fees"): parts.append(f"الرسوم: {row['fees']}")
    if row.get("conditions"): parts.append(f"الشروط: {row['conditions']}")
    if row.get("access_path"): parts.append(f"طريقة الوصول: {row['access_path']}")
    
    return " | ".join(parts)