import numpy as np
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain.prompts import PromptTemplate
from transformers import pipeline

from config import Config
from utils.logger import setup_logger
from utils.text_utils import normalize_arabic, looks_english, is_arabic
from core.model_loader import ModelManager

logger = setup_logger(__name__)

# --- Prompts ---
SYSTEM_PROMPT = """
<s>[INST] <<SYS>>
أنت مساعد ذكي لخدمات وزارة الداخلية ومنصة أبشر.

السياسات:
- أجب بالعربية إلا إذا طلب المستخدم الإنجليزية أو سأل بها.
- لا تذكر رسوم/شروط/مدد إلا إذا ظهرت صراحة في "السياق".
- إن كان السياق غير كافٍ، قل: "المعلومة غير متوفرة في المستند."
- يسمح بالدردشة الخفيفة بنبرة سعودية لبقة.
<</SYS>>

[Context]
{context}

[User Question]
{question}
[/INST]
"""

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"], 
    template=SYSTEM_PROMPT
)

class ProRAGChain:
    """
    Advanced RAG Pipeline that handles:
    1. Query Rewriting & Translation
    2. Hybrid Retrieval (Dense + BM25)
    3. Reciprocal Rank Fusion (RRF)
    4. Semantic Reranking
    5. Generation with ALLaM-7B
    6. Post-hoc Translation
    """
    def __init__(self, vector_store, all_documents: List[Document]):
        self.vector_store = vector_store
        
        # Load Singleton Models
        self.llm_model, self.llm_tokenizer = ModelManager.get_llm()
        self.embed_model = ModelManager.get_embedding_model()
        
        # Initialize Retrievers
        logger.info("🔹 Initializing Retrievers (Dense + BM25)...")
        self.dense_retriever = vector_store.as_retriever(
            search_kwargs={"k": Config.RETRIEVAL_K}
        )
        self.bm25_retriever = BM25Retriever.from_documents(all_documents)
        self.bm25_retriever.k = Config.RETRIEVAL_K

        # Initialize Generation Pipelines
        logger.info("🔹 Initializing Generation Pipelines...")
        self.gen_pipeline = self._create_gen_pipeline()
        self.trans_pipeline = self._create_trans_pipeline()

    def _create_gen_pipeline(self):
        """Creates the main generation pipeline with sampling enabled."""
        return pipeline(
            "text-generation",
            model=self.llm_model,
            tokenizer=self.llm_tokenizer,
            max_new_tokens=Config.MAX_NEW_TOKENS,
            temperature=Config.TEMPERATURE,
            top_p=Config.TOP_P,
            do_sample=True,
            repetition_penalty=Config.REPETITION_PENALTY,
            pad_token_id=self.llm_tokenizer.eos_token_id
        )

    def _create_trans_pipeline(self):
        """Creates a greedy pipeline specifically for translation tasks."""
        return pipeline(
            "text-generation",
            model=self.llm_model,
            tokenizer=self.llm_tokenizer,
            max_new_tokens=1024,
            do_sample=False, # Greedy for accuracy
            pad_token_id=self.llm_tokenizer.eos_token_id
        )

    def _rrf_merge(self, dense_docs: List[Document], bm25_docs: List[Document], k=60) -> List[Document]:
        """
        Combines results from multiple retrievers using Reciprocal Rank Fusion.
        """
        scores = {}
        store = {}
        
        # Helper to create unique key for deduplication
        def get_key(d): 
            return f"{d.metadata.get('service_id','')}::{d.page_content[:50]}"
        
        # Score Dense Results
        for r, d in enumerate(dense_docs, 1):
            key = get_key(d)
            scores[key] = scores.get(key, 0) + 1 / (k + r)
            store[key] = d
            
        # Score BM25 Results
        for r, d in enumerate(bm25_docs, 1):
            key = get_key(d)
            scores[key] = scores.get(key, 0) + 1 / (k + r)
            store[key] = d
            
        # Sort by score descending
        ordered_keys = sorted(scores, key=scores.get, reverse=True)
        return [store[k] for k in ordered_keys]

    def _dense_rerank(self, query: str, docs: List[Document], top_k=6) -> List[Document]:
        """
        Re-ranks the merged documents using Cosine Similarity between query and document embeddings.
        """
        if not docs: return []
        
        try:
            query_vec = self.embed_model.embed_query(query)
            scored = []
            
            # Rerank only top 20 candidates to save compute
            candidates = docs[:20]
            
            for d in candidates:
                doc_vec = self.embed_model.embed_query(d.page_content)
                
                # Calculate Cosine Similarity
                norm_q = np.linalg.norm(query_vec)
                norm_d = np.linalg.norm(doc_vec)
                
                if norm_q == 0 or norm_d == 0:
                    score = 0
                else:
                    score = np.dot(query_vec, doc_vec) / (norm_q * norm_d)
                    
                scored.append((score, d))
                
            # Sort by similarity score
            scored.sort(key=lambda x: x[0], reverse=True)
            return [d for _, d in scored[:top_k]]
            
        except Exception as e:
            logger.error(f"⚠️ Reranking failed: {e}. Returning original order.")
            return docs[:top_k]

    def _translate(self, text: str, target="ar") -> str:
        """
        Translates text using the LLM. 
        """
        if not text or not text.strip(): return text
        
        lang_name = 'Arabic' if target == 'ar' else 'English'
        prompt = f"Translate the following text to {lang_name}:\nText: {text}\nTranslation:"
        
        try:
            out = self.trans_pipeline(prompt)[0]["generated_text"]
            # Extract text after "Translation:"
            if "Translation:" in out:
                return out.split("Translation:", 1)[-1].strip()
            return out.strip()
        except Exception as e:
            logger.warning(f"⚠️ Translation failed: {e}")
            return text

    def answer(self, query: str) -> str:
        """
        Main RAG entry point.
        """
        # 1. Detect Language & Normalize
        is_eng_query = looks_english(query)
        
        if is_eng_query:
            logger.info(f"🇬🇧 English query detected. Translating to Arabic...")
            q_ar = self._translate(query, "ar")
            logger.info(f"🔄 Translated Query: {q_ar}")
        else:
            q_ar = normalize_arabic(query)

        # 2. Retrieve (Hybrid)
        dense_res = self.dense_retriever.invoke(q_ar)
        bm25_res = self.bm25_retriever.invoke(q_ar)
        merged_docs = self._rrf_merge(dense_res, bm25_res)
        logger.info(f"🔍 Retrieved {len(merged_docs)} candidates (Dense+BM25)")

        # 3. Re-rank
        final_docs = self._dense_rerank(q_ar, merged_docs, top_k=Config.RERANK_TOP_K)
        logger.info(f"🔝 Reranked top {len(final_docs)} documents")
        
        # 4. Build Context
        context_text = "\n\n".join([f"- {d.page_content}" for d in final_docs])
        
        # 5. Generate Answer
        lang_instruction = "Answer in English." if is_eng_query else "أجب بالعربية."
        final_prompt = QA_PROMPT.format(
            context=context_text, 
            question=f"{query}\n{lang_instruction}"
        )
        
        logger.info("🤖 Generating answer...")
        try:
            raw_response = self.gen_pipeline(final_prompt)[0]["generated_text"]
            # Extract answer after [/INST]
            answer = raw_response.split("[/INST]", 1)[-1].strip()
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            return "عذراً، حدث خطأ أثناء توليد الإجابة."

        # 6. Post-hoc Translation Check (The "Language Sandwich")
        # If user asked in English but model answered in Arabic -> Translate back
        if is_eng_query and is_arabic(answer):
            logger.info("🔄 Translating Arabic answer back to English...")
            answer = self._translate(answer, "en")
            
        return answer