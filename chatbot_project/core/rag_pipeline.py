import numpy as np
from typing import List, Dict
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
QA_PROMPT = PromptTemplate(input_variables=["context", "question"], template=SYSTEM_PROMPT)

class ProRAGChain:
    def __init__(self, vector_store, all_documents):
        self.vector_store = vector_store
        self.llm_model, self.llm_tokenizer = ModelManager.get_llm()
        self.embed_model = ModelManager.get_embedding_model()
        
        # Retrievers
        self.dense_retriever = vector_store.as_retriever(search_kwargs={"k": Config.RETRIEVAL_K})
        self.bm25_retriever = BM25Retriever.from_documents(all_documents)
        self.bm25_retriever.k = Config.RETRIEVAL_K

        # Pipelines
        self.gen_pipeline = self._create_gen_pipeline()
        self.trans_pipeline = self._create_trans_pipeline()

    def _create_gen_pipeline(self):
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
        # Greedy decoding for translation (More accurate)
        return pipeline(
            "text-generation",
            model=self.llm_model,
            tokenizer=self.llm_tokenizer,
            max_new_tokens=1024,
            do_sample=False, # Greedy
            pad_token_id=self.llm_tokenizer.eos_token_id
        )

    def _rrf_merge(self, dense_docs, bm25_docs, k=60):
        """Reciprocal Rank Fusion to merge hybrid results."""
        scores, store = {}, {}
        def get_key(d): return f"{d.metadata.get('service_id','')}::{d.page_content[:50]}"
        
        for r, d in enumerate(dense_docs, 1):
            key = get_key(d)
            scores[key] = scores.get(key, 0) + 1 / (k + r)
            store[key] = d
            
        for r, d in enumerate(bm25_docs, 1):
            key = get_key(d)
            scores[key] = scores.get(key, 0) + 1 / (k + r)
            store[key] = d
            
        ordered_keys = sorted(scores, key=scores.get, reverse=True)
        return [store[k] for k in ordered_keys]

    def _dense_rerank(self, query, docs, top_k=6):
        """Re-rank results using Cosine Similarity."""
        if not docs: return []
        
        query_vec = self.embed_model.embed_query(query)
        scored = []
        for d in docs[:20]: # Only rerank top 20
            doc_vec = self.embed_model.embed_query(d.page_content)
            # Cosine Sim
            score = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
            scored.append((score, d))
            
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:top_k]]

    def _translate(self, text, target="ar"):
        """Post-hoc translation using the greedy pipeline."""
        prompt = f"Translate to {'Arabic' if target=='ar' else 'English'}:\n{text}\nTranslation:"
        try:
            out = self.trans_pipeline(prompt)[0]["generated_text"]
            return out.split("Translation:", 1)[-1].strip()
        except:
            return text

    def answer(self, query):
        # 1. Detect Language & Normalize
        is_eng_query = looks_english(query)
        q_ar = self._translate(query, "ar") if is_eng_query else normalize_arabic(query)

        # 2. Retrieve (Hybrid)
        dense_res = self.dense_retriever.invoke(q_ar)
        bm25_res = self.bm25_retriever.invoke(q_ar)
        merged_docs = self._rrf_merge(dense_res, bm25_res)

        # 3. Re-rank
        final_docs = self._dense_rerank(q_ar, merged_docs, top_k=Config.RERANK_TOP_K)
        
        # 4. Build Context
        context_text = "\n\n".join([f"- {d.page_content}" for d in final_docs])
        
        # 5. Generate
        lang_instruction = "Answer in English." if is_eng_query else "أجب بالعربية."
        final_prompt = QA_PROMPT.format(context=context_text, question=f"{query}\n{lang_instruction}")
        
        raw_response = self.gen_pipeline(final_prompt)[0]["generated_text"]
        answer = raw_response.split("[/INST]", 1)[-1].strip()

        # 6. Post-hoc Translation Check
        ans_is_arabic = is_arabic(answer)
        if is_eng_query and ans_is_arabic:
            logger.info("🔄 Translating Arabic answer to English...")
            answer = self._translate(answer, "en")
            
        return answer