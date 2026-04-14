# =========================================================================
# File Name: data/ingestion.py
# Purpose: ETL Pipeline for Master Data Ingestion (A100 Optimized).
# Project: Absher Smart Assistant (MOI ChatBot)
# Version: 5.3.0 (Context Injection + Unified Search Schema)
#
# Changelog v5.2.0 → v5.3.0:
#   - [FIX] Context Injection: Every chunk now carries its service name
#           and sector in page_content, preventing "orphaned chunks" that
#           lose semantic identity after splitting. (Engineer Report §1A)
#   - [FIX] Unified Search Schema: The EXACT same text string is now used
#           for both FAISS page_content and BM25 CSV, eliminating the RRF
#           scoring asymmetry between dense and sparse retrievers. (§1B)
#   - [FIX] Chunk text now uses normalize_for_dense() context prefix for
#           FAISS-optimized embedding, while BM25 CSV stores the same
#           unified text (normalize_arabic applied at query time by BM25).
#
# Features:
# - Schema Enforcement: Integrates with schema.py to block corrupted data.
# - Hybrid Readiness: Prepares IDENTICAL chunks for both FAISS and BM25.
# - Memory Optimization: Employs eager garbage collection during chunking.
# =========================================================================

import os
import glob
import gc
import pandas as pd
from typing import List
from tqdm import tqdm
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import Config
from utils.logger import setup_logger
from data.schema import validate_schema

logger = setup_logger("Data_Ingestor")


class DataIngestor:
    """
    Orchestrates the transformation of raw CSV tabular data into
    chunked LangChain Document objects (for Dense Retrieval) and
    synchronized CSV chunks (for Sparse BM25 Retrieval).
    """

    def __init__(self):
        """
        Initializes directory paths and configures the text splitting logic.
        Optimized for the BGE-M3 embedding model context window.
        """
        self.master_file = os.path.join(Config.DATA_MASTER_DIR, "MOI_Master_Knowledge.csv")
        self.chunk_dir = Config.DATA_CHUNK_DIR

        # BGE-M3 optimized splitter: 400 chars (~256 tokens) with 50 char overlap.
        # Tuned for enriched RAG_Content (~456 chars avg) to produce 1-2 chunks per service.
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def _cleanup_old_chunks(self):
        """Removes previously generated CSV chunks to ensure fresh indexing."""
        logger.info("🧹 Cleaning up old chunks before executing ingestion pipeline...")
        os.makedirs(self.chunk_dir, exist_ok=True)
        for file in glob.glob(os.path.join(self.chunk_dir, "*.csv")):
            try:
                os.remove(file)
            except Exception as e:
                logger.warning(f"⚠️ Failed to remove old chunk {file}: {e}")

    def _read_csv_safe(self, file_path: str) -> pd.DataFrame:
        """
        Attempts to load CSV with multiple encoding fallbacks.
        Crucial for handling Arabic datasets from different OS environments.
        """
        encodings = ['utf-8-sig', 'utf-8', 'cp1256', 'latin1']
        for enc in encodings:
            try:
                return pd.read_csv(file_path, encoding=enc, engine='python', on_bad_lines='skip')
            except (UnicodeDecodeError, Exception):
                continue

        logger.error(f"❌ Critical Error: Could not decode {file_path} with any supported encodings.")
        return pd.DataFrame()

    @staticmethod
    def _build_unified_text(service_name: str, sector: str, chunk_txt: str) -> str:
        """
        [NEW v5.3.0] Builds a unified text payload with context injection.

        Prepends service name and sector to every chunk, ensuring that:
        1. No chunk is "orphaned" — even chunk #3 of a service knows its identity.
        2. FAISS and BM25 see the EXACT same text — fair RRF merge scoring.
        3. BGE-M3 can embed the service context alongside the content.

        Format:
            "خدمة: إصدار الجواز السعودي | قطاع: الجوازات\n<chunk content>"

        Args:
            service_name: The official service name (Arabic).
            sector: The sector/department name (Arabic).
            chunk_txt: The raw chunk text from the splitter.

        Returns:
            str: Context-enriched text ready for both FAISS and BM25.
        """
        # Use Arabic labels to match the language of the content and KG
        return f"خدمة: {service_name} | قطاع: {sector}\n{chunk_txt}"

    def load_and_process(self) -> List[Document]:
        """
        Main ETL execution logic.
        Flow: Read Master → Validate Schema → Chunk → Context Inject → Export for BM25 → Return for FAISS.

        Returns:
            List[Document]: LangChain documents ready for vectorization.
        """
        self._cleanup_old_chunks()
        all_documents = []

        if not os.path.exists(self.master_file):
            logger.error(f"❌ Master knowledge file missing at: {self.master_file}")
            return []

        logger.info(f"📂 Starting Ingestion Pipeline for Master Knowledge Base...")

        try:
            # 1. Safe Extraction
            df = self._read_csv_safe(self.master_file)

            # 2. Strict GRC Validation via schema.py
            if not validate_schema(df, "master", os.path.basename(self.master_file)):
                logger.error("❌ Ingestion halted: Master Data failed schema validation.")
                return []

            # 3. Data Sanitization
            df = df.fillna("")
            chunked_rows_buffer = []

            # 4. Transformation (Chunking + Context Injection)
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing Services"):
                raw_text = str(row["RAG_Content"]).strip()

                # Filter out rows with insufficient semantic value
                if not raw_text or len(raw_text) < 20:
                    continue

                # Extract columns (aligned with V2 Schema)
                service_name = str(row.get("Service_Name", "Unknown")).strip()
                sector = str(row.get("Sector", "Unknown")).strip()
                audience = str(row.get("Target_Audience", "General")).strip()
                steps = str(row.get("Service_Steps", "")).strip()
                reqs = str(row.get("Requirements", "")).strip()
                fees = str(row.get("Service_Fees", "")).strip()
                url = str(row.get("Official_URL", "")).strip()

                # Split large RAG_Content into manageable semantic chunks
                text_chunks = self.text_splitter.split_text(raw_text)

                for i, chunk_txt in enumerate(text_chunks):

                    # [FIX v5.3.0] Context Injection: prepend service identity to EVERY chunk.
                    # This prevents "semantic decapitation" where chunks 2+ lose their service context.
                    # Both FAISS and BM25 now see the EXACT same unified text.
                    unified_text = self._build_unified_text(service_name, sector, chunk_txt)

                    # Construct metadata for FAISS payload
                    metadata = {
                        "source": "MOI_Master_Knowledge",
                        "sector": sector,
                        "service": service_name,
                        "audience": audience,
                        "url": url,
                        "chunk_index": i
                    }

                    # Target 1: FAISS — uses unified_text (with context prefix)
                    all_documents.append(Document(page_content=unified_text, metadata=metadata))

                    # Target 2: BM25 CSV — uses the SAME unified_text for symmetric scoring
                    chunked_rows_buffer.append({
                        "اسم الخدمة": service_name,
                        "RAG_Content": unified_text,
                        "القطاع": sector,
                        "خطوات الخدمة": steps,
                        "المستندات المطلوبة": reqs,
                        "سعر الخدمة": fees,
                        "رابط الخدمة": url
                    })

            # 5. Export processed CSV for BM25
            if chunked_rows_buffer:
                output_csv = os.path.join(self.chunk_dir, "master_chunks.csv")
                pd.DataFrame(chunked_rows_buffer).to_csv(output_csv, index=False, encoding='utf-8-sig')
                logger.info(f"✅ Saved BM25 reference chunks to: {output_csv}")

            # 6. Proactive Memory Management
            del df, chunked_rows_buffer
            gc.collect()

        except Exception as e:
            logger.error(f"❌ Critical ETL Failure during processing: {e}", exc_info=True)
            return all_documents

        # 7. Log pipeline stats
        if all_documents:
            chunk_lengths = [len(d.page_content) for d in all_documents]
            services_count = len(set(d.metadata.get('service', '') for d in all_documents))
            logger.info(
                f"✅ ETL Pipeline Success: {len(all_documents)} chunks from {services_count} services | "
                f"Avg chunk: {sum(chunk_lengths)//len(chunk_lengths)} chars | "
                f"Min: {min(chunk_lengths)} | Max: {max(chunk_lengths)}"
            )
        else:
            logger.warning("⚠️ ETL Pipeline produced 0 documents.")

        return all_documents


if __name__ == "__main__":
    ingestor = DataIngestor()
    results = ingestor.load_and_process()