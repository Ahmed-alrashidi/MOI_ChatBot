# =========================================================================
# File Name: core/rag_pipeline.py
# Project: Absher Smart Assistant (MOI ChatBot)
# Architecture: Cross-Lingual Hybrid RAG (BGE-M3 + BM25 + ALLaM-7B)
#
# Affiliation: King Abdullah University of Science and Technology (KAUST)
# Team: Ahmed AlRashidi, Sultan Alshaibani, Fahad Alqahtani, 
#       Rakan Alharbi, Sultan Alotaibi, Abdulaziz Almutairi.
# Advisors: Prof. Naeemullah Khan & Dr. Salman Khan
# =========================================================================

import os
import torch
import pandas as pd
from typing import List, Tuple, Any
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever

from config import Config
from utils.logger import setup_logger
from utils.text_utils import normalize_arabic  # Added for query preprocessing
from core.model_loader import ModelManager
from core.vector_store import VectorStoreManager

logger = setup_logger(__name__)

class RAGPipeline:
    """
    Orchestrates the Hybrid Retrieval-Augmented Generation pipeline.
    Combines Dense Vector Search (BGE-M3) and Sparse Keyword Search (BM25)
    using Reciprocal Rank Fusion (RRF), then generates answers via ALLaM-7B.
    """
    
    def __init__(self):
        logger.info("🚀 Initializing Hybrid RAG Pipeline...")
        
        # 1. Load Models (Embedding + LLM)
        self.embed_model = ModelManager.get_embedding_model()
        self.llm, self.tokenizer = ModelManager.get_llm()
        
        # 2. Initialize Dense Retrieval (Vector Store)
        # Relies on FAISS built from 'Data_Master' files
        self.vector_db = VectorStoreManager.load_or_build(self.embed_model)
        self.dense_retriever = self.vector_db.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": Config.RETRIEVAL_K}
        )
        
        # 3. Initialize Sparse Retrieval (Keyword Matching)
        # Built dynamically from 'Data_Chunk' files to capture specific details
        self.bm25_retriever = self._build_bm25_from_chunks()
        
        logger.info("✅ Pipeline Ready: Hybrid Retrieval (BGE-M3 + BM25) Enabled.")

    def _build_bm25_from_chunks(self) -> BM25Retriever:
        """
        Loads detailed procedural data from 'Data_Chunk/' directory to build BM25 index.
        Uses absolute paths from Config to prevent FileNotFoundError.
        """
        # FIX: Use the absolute path from Config
        chunk_dir = Config.DATA_CHUNK_DIR
        documents = []
        
        logger.info(f"📂 Loading Detailed Chunks for BM25 from: {chunk_dir}")
        
        if not os.path.exists(chunk_dir):
            logger.warning(f"⚠️ Directory {chunk_dir} not found. Sparse retrieval will be disabled.")
            # We don't exit, just return None so the pipeline continues with Dense only
            return None

        # Iterate over all detail CSV files
        files_processed = 0
        for filename in os.listdir(chunk_dir):
            if filename.endswith(".csv"): # Accept any CSV in the chunk dir
                files_processed += 1
                file_path = os.path.join(chunk_dir, filename)
                try:
                    df = pd.read_csv(file_path)
                    # Convert each row into a rich text Document
                    for _, row in df.iterrows():
                        # Construct content specifically for keyword matching (BM25)
                        # We use .get() to avoid errors if columns are missing
                        content = (
                            f"الخدمة: {row.get('اسم الخدمة', '')}\n"
                            f"الخطوات: {row.get('خطوات الخدمة', '')}\n"
                            f"المستندات: {row.get('المستندات المطلوبة', '')}\n"
                            f"السعر: {row.get('سعر الخدمة', '')}\n"
                            f"المتطلبات: {row.get('متطلبات الخدمة', '')}"
                        )
                        documents.append(Document(page_content=content, metadata={"source": filename}))
                except Exception as e:
                    logger.error(f"Error reading {filename}: {e}")

        if not documents:
            logger.warning(f"⚠️ No documents found in {chunk_dir} (Scanned {files_processed} files).")
            return None

        logger.info(f"✅ Built BM25 Index with {len(documents)} detailed procedures from {files_processed} files.")
        return BM25Retriever.from_documents(documents)

    def _rrf_merge(self, dense_docs: List[Document], sparse_docs: List[Document], k=60) -> List[Document]:
        """
        Reciprocal Rank Fusion (RRF) algorithm to merge results from Dense and Sparse retrievers.
        """
        scores = {}
        
        # Calculate scores for Dense results
        for rank, doc in enumerate(dense_docs):
            content = doc.page_content
            if content not in scores:
                scores[content] = {"doc": doc, "score": 0.0}
            scores[content]["score"] += 1.0 / (k + rank + 1)
            
        # Calculate scores for Sparse results (if available)
        if sparse_docs:
            for rank, doc in enumerate(sparse_docs):
                content = doc.page_content
                if content not in scores:
                    scores[content] = {"doc": doc, "score": 0.0}
                scores[content]["score"] += 1.0 / (k + rank + 1)
        
        # Sort final results by accumulated RRF score
        sorted_docs = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_docs[:Config.RETRIEVAL_K]]

    def run(self, query: str, history: List[Tuple[str, str]] = []) -> str:
        """
        Executes the RAG pipeline:
        1. Query Normalization
        2. Hybrid Retrieval (Dense + Sparse)
        3. RRF Fusion
        4. LLM Generation
        """
        # Step 0: Normalize Query (Crucial for Arabic Search)
        # This unifies Alef forms, removes diacritics, etc.
        clean_query = normalize_arabic(query)
        # logger.debug(f"🔍 Original Query: {query} | Normalized: {clean_query}")

        # Step 1: Retrieval
        dense_results = self.dense_retriever.invoke(clean_query)
        sparse_results = self.bm25_retriever.invoke(clean_query) if self.bm25_retriever else []
        
        # Step 2: Fusion (RRF)
        final_docs = self._rrf_merge(dense_results, sparse_results)
        
        # Guard Clause: If no documents found, return fallback immediately
        if not final_docs:
            logger.warning(f"⚠️ No relevant documents found for query: {clean_query}")
            return "عذراً، لا تتوفر لدي معلومات كافية حول هذا الموضوع في الوثائق الرسمية الحالية."

        # Step 3: Construct Context
        context_text = "\n\n".join([f"- {d.page_content}" for d in final_docs])
        
        # Step 4: Construct Prompt with Short-term Memory
        chat_history_text = "\n".join([f"User: {h[0]}\nAssistant: {h[1]}" for h in history[-3:]])
        
        # Use Centralized System Prompt from Config
        full_prompt = Config.SYSTEM_PROMPT_TEMPLATE.format(
            context=context_text,
            chat_history=chat_history_text,
            question=query # We pass original query to LLM for naturalness
        )
        
        # Step 5: Generation
        logger.info("🤖 Generating response with ALLaM-7B...")
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.llm.device)
        
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                repetition_penalty=1.1
            )
            
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Post-processing to extract only the assistant's reply
        if "[/INST]" in response:
            response = response.split("[/INST]")[-1].strip()
            
        return response