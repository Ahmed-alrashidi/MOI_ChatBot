import torch
import gc
import numpy as np
from typing import List, Tuple, Any, Optional
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from transformers import pipeline, Pipeline
from langdetect import detect

from config import Config
from utils.logger import setup_logger
from utils.text_utils import normalize_arabic
from core.model_loader import ModelManager

# Initialize module logger
logger = setup_logger(__name__)

# --- 1. Prompts for Memory & Context ---

# Prompt to summarize long history to keep memory fresh
SUMMARY_PROMPT = """
لخص المحادثة التالية في 3 جمل قصيرة جداً ومركزة، واحتفظ بالمعلومات الرئيسية (الأرقام، الخدمات المطلوبة).
المحادثة:
{history}
الملخص:
"""

# Prompt for Query Rewriting (Context Awareness)
CONDENSE_Q_PROMPT = """
بناءً على "ملخص المحادثة" و"السجل الحديث"، أعد صياغة السؤال الأخير ليكون مستقلاً ومفهوماً.
ملخص سابق: {summary}
سجل حديث: {chat_history}
السؤال: {question}
السؤال المعاد صياغته:
"""

# Main System Prompt (Optimized for Summary + Context)
SYSTEM_PROMPT = """
<s>[INST] <<SYS>>
أنت مساعد ذكي لخدمات وزارة الداخلية (MOI Universal Assistant).

التعليمات الصارمة:
1. **اللغة:** أجب **دائماً** باللغة العربية الفصحى.
2. **الأسلوب:** كن **موجزاً جداً ومباشراً**. اذكر الإجابة (الرسوم، الشروط، الخطوات) فوراً.
3. **المصدر:** اعتمد حصرياً على [Context].
4. **الذاكرة:** استفد من [Memory Summary] لفهم سياق الحديث الطويل.
<</SYS>>

[Memory Summary]
{summary}

[Context]
{context}

[Recent Chat]
{chat_history}

[User Question]
{question}
[/INST]
"""

class ProRAGChain:
    """
    Advanced RAG Pipeline optimized for A100.
    Features:
    - Auto-Summarization (Infinite Memory)
    - Aggressive VRAM Cleanup
    - Input/Output Translation
    """

    def __init__(self, vector_store, all_documents: List[Document]):
        self.vector_store = vector_store
        
        # Load Singleton Models (A100 Optimized)
        self.llm_model, self.llm_tokenizer = ModelManager.get_llm()
        self.embed_model = ModelManager.get_embedding_model()
        
        # Initialize Retrievers
        self.dense_retriever = vector_store.as_retriever(search_kwargs={"k": Config.RETRIEVAL_K})
        
        if all_documents:
            self.bm25_retriever = BM25Retriever.from_documents(all_documents)
            self.bm25_retriever.k = Config.RETRIEVAL_K
        else:
            self.bm25_retriever = None

        # Initialize Pipelines with A100 Precision
        self.gen_pipeline = self._create_pipeline(max_new_tokens=Config.MAX_NEW_TOKENS, sample=True)
        self.trans_pipeline = self._create_pipeline(max_new_tokens=256, sample=False)
        
        # Memory State
        self.conversation_summary = ""
        self.turn_count = 0

    def _create_pipeline(self, max_new_tokens: int, sample: bool = True) -> Pipeline:
        """Creates a pipeline optimized for A100 (bfloat16)."""
        return pipeline(
            "text-generation",
            model=self.llm_model,
            tokenizer=self.llm_tokenizer,
            max_new_tokens=max_new_tokens,
            do_sample=sample,
            temperature=Config.TEMPERATURE if sample else 0.1,
            top_p=Config.TOP_P if sample else 1.0,
            repetition_penalty=Config.REPETITION_PENALTY if sample else 1.0,
            pad_token_id=self.llm_tokenizer.eos_token_id,
            # Ensure pipeline uses the model's device/dtype
            device_map="auto"
        )

    def _clean_memory(self):
        """Forces garbage collection and clears CUDA cache to prevent OOM."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _update_summary(self, history: List[Tuple[str, str]]):
        """
        Summarizes chat history every 3 turns to maintain long-term context 
        without exceeding token limits.
        """
        self.turn_count += 1
        # Trigger summary every 3 turns if there is history
        if self.turn_count % 3 == 0 and history:
            logger.info("🧠 Summarizing conversation history...")
            
            # Use the last 3 turns for the new summary update
            recent_turns = history[-3:]
            hist_text = "\n".join([f"User: {h[0]}\nAI: {h[1]}" for h in recent_turns])
            
            # Combine old summary with recent chat to update context
            full_context = f"ملخص سابق: {self.conversation_summary}\n\nمحادثة جديدة:\n{hist_text}"
            prompt = SUMMARY_PROMPT.format(history=full_context)
            
            try:
                # Generate summary
                out = self.trans_pipeline(prompt)[0]["generated_text"]
                # Extract summary assuming the model follows prompt structure
                if "الملخص:" in out:
                    new_summary = out.split("الملخص:")[-1].strip()
                else:
                    new_summary = out.strip() # Fallback

                if new_summary:
                    self.conversation_summary = new_summary
                    logger.info(f"📝 Memory Updated: {self.conversation_summary[:50]}...")
            except Exception as e:
                logger.warning(f"Summarization failed: {e}")
            
            # Heavy cleanup after summarization
            self._clean_memory()

    def _translate(self, text: str, source: str, target: str) -> str:
        """Helper for both Input and Output translation."""
        if not text or source == target: return text
        
        lang_map = {
            "ar": "Arabic", "en": "English", "fr": "French", "es": "Spanish",
            "de": "German", "ru": "Russian", "zh-cn": "Chinese", "hi": "Hindi"
        }
        src_name = lang_map.get(source, source)
        tgt_name = lang_map.get(target, target)

        prompt = f"Translate the following {src_name} text to {tgt_name}. Provide ONLY the translation.\n\nText: {text}\n\nTranslation:"
        
        try:
            out = self.trans_pipeline(prompt)[0]["generated_text"]
            if "Translation:" in out:
                return out.split("Translation:", 1)[-1].strip()
            return out.replace(prompt, "").strip()
        except: 
            return text

    def _rewrite_query(self, query: str, history: List[Tuple[str, str]]) -> str:
        """Rewrites user query based on summary and recent history."""
        if not history and not self.conversation_summary: return query
        
        hist_str = "\n".join([f"User: {h[0]}\nAI: {h[1]}" for h in history[-2:]])
        prompt = CONDENSE_Q_PROMPT.format(
            summary=self.conversation_summary,
            chat_history=hist_str,
            question=query
        )
        try:
            out = self.trans_pipeline(prompt)[0]["generated_text"]
            if "صياغته:" in out:
                return out.split("صياغته:")[-1].strip() or query
            return query
        except: return query

    def _rrf_merge(self, dense_docs: List[Document], bm25_docs: List[Document], k: int = 60) -> List[Document]:
        scores = {}
        store = {}
        
        def get_key(d):
            # Safer key generation to handle missing metadata
            s_id = d.metadata.get('service_id', 'unknown') if d.metadata else 'unknown'
            content_sig = d.page_content[:50]
            return f"{s_id}::{content_sig}"
        
        for r, d in enumerate(dense_docs, 1):
            key = get_key(d)
            scores[key] = scores.get(key, 0) + 1 / (k + r)
            store[key] = d
            
        # Handle case where BM25 might be None or empty
        if bm25_docs:
            for r, d in enumerate(bm25_docs, 1):
                key = get_key(d)
                scores[key] = scores.get(key, 0) + 1 / (k + r)
                store[key] = d
                
        return [store[k] for k in sorted(scores, key=scores.get, reverse=True)]

    def _dense_rerank(self, query: str, docs: List[Document], top_k: int = 6) -> List[Document]:
        if not docs: return []
        try:
            query_vec = self.embed_model.embed_query(query)
            scored = []
            for d in docs[:20]: # Only rerank top 20
                doc_vec = self.embed_model.embed_query(d.page_content)
                norm_q = np.linalg.norm(query_vec)
                norm_d = np.linalg.norm(doc_vec)
                score = 0 if (norm_q == 0 or norm_d == 0) else np.dot(query_vec, doc_vec) / (norm_q * norm_d)
                scored.append((score, d))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [d for _, d in scored[:top_k]]
        except: return docs[:top_k]

    def _apply_text_direction(self, text: str, lang_code: str) -> str:
        rtl_langs = ['ar', 'he', 'ur', 'fa']
        direction = 'rtl' if lang_code in rtl_langs else 'ltr'
        alignment = 'right' if lang_code in rtl_langs else 'left'
        return f"<div dir='{direction}' style='text-align: {alignment};'>{text}</div>"

    def answer(self, query: str, history: List[Tuple[str, str]] = []) -> str:
        # --- Step 0: Hygiene ---
        self._clean_memory()
        
        # --- Step 1: Update Memory (Summarize if needed) ---
        self._update_summary(history)

        # --- Step 2: Language Detection ---
        if any('\u0600' <= char <= '\u06FF' for char in query):
            user_lang = 'ar'
        else:
            try: user_lang = detect(query)
            except: user_lang = 'ar'
        
        logger.info(f"🌍 User Language: {user_lang}")

        # --- Step 3: Input Translation -> Arabic ---
        q_ar = self._translate(query, source=user_lang, target='ar')

        # --- Step 4: Retrieval ---
        # Rewrite query using both Summary and Recent History
        search_query = self._rewrite_query(q_ar, history)
        
        dense_res = self.dense_retriever.invoke(search_query)
        bm25_res = self.bm25_retriever.invoke(search_query) if self.bm25_retriever else []
        
        merged_docs = self._rrf_merge(dense_res, bm25_res)
        final_docs = self._dense_rerank(search_query, merged_docs, top_k=Config.RETRIEVAL_K)
        
        # --- Step 5: Generation (Arabic) ---
        context_text = "\n\n".join([f"- {d.page_content}" for d in final_docs])
        
        # Only pass the last 3 turns + the summarized memory
        recent_history = history[-3:] if history else []
        history_text = "\n".join([f"User: {h[0]}\nAI: {h[1]}" for h in recent_history])
        
        final_prompt = SYSTEM_PROMPT.format(
            summary=self.conversation_summary, # The long-term memory
            context=context_text, 
            chat_history=history_text,         # The short-term memory
            question=f"{search_query}"
        )
        
        logger.info("🤖 Generating Arabic Answer...")
        try:
            raw_response = self.gen_pipeline(final_prompt)[0]["generated_text"]
            # Robust split
            if "[/INST]" in raw_response:
                answer_ar = raw_response.split("[/INST]", 1)[-1].strip()
            else:
                answer_ar = raw_response.strip()
        except Exception as e:
            logger.error(f"Generation Error: {e}")
            return "عذراً، حدث خطأ في النظام."

        # --- Step 6: Output Translation ---
        final_response = self._translate(answer_ar, source='ar', target=user_lang)

        return self._apply_text_direction(final_response, user_lang)