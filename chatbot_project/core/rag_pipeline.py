import numpy as np
from typing import List, Tuple, Any
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langdetect import detect, LangDetectException

from config import Config
from utils.logger import setup_logger
from utils.text_utils import normalize_arabic
from core.model_loader import ModelManager

logger = setup_logger(__name__)

# --- 1. Prompt for Rewriting Query (Smart Memory) ---
CONDENSE_Q_PROMPT = """
مهمتك هي تحليل "الجملة الحالية" بناءً على "سجل المحادثة":

1. **حالة المتابعة:** إذا كانت الجملة تعتمد على ما قبلها لفهم معناها (مثلاً: "كم يكلف؟"، "طيب وللأطفال؟"، "كيف الطريقة؟") -> أعد صياغتها لتصبح سؤالاً كاملاً ومستقلاً يجمع التفاصيل من السجل.
2. **حالة موضوع جديد/رد فعل:** إذا كانت الجملة موضوعاً جديداً كلياً، أو مجرد رد فعل عاطفي أو اجتماعي (مثلاً: "شكراً"، "هههههه"، "أحبك"، "ممتاز"، "السلام عليكم") -> **اتركها كما هي تماماً** ولا تضف عليها أي شيء من السجل السابق.

سجل المحادثة:
{chat_history}

الجملة الحالية:
{question}

النتيجة (اكتب الجملة النهائية فقط):
"""

# --- 2. Main System Prompt (Final, Robust Version) ---
SYSTEM_PROMPT = """
<s>[INST] <<SYS>>
أنت مساعد ذكي لخدمات وزارة الداخلية (MOI Universal Assistant).

التعليمات الأساسية:
1. **الرد على التحية (Smart Greeting):** رد بتحية مهذبة *فقط* في بداية المحادثة أو عندما يسألك المستخدم عن أحوالك، وليس في منتصف الإجابات أو عند سؤالك عن خدمة مباشرة.
2. **المعلومات الرسمية:** اعتمد كلياً على السياق المرفق.
3. **موافقة الوالدين:** بناءً على المصادر، الموافقة مطلوبة للأفراد دون 18 عاماً. استخدم هذا الحد (دون 18 عاماً) في إجابتك.
4. **العملة:** استخدم "ريال سعودي" دائماً.
5. **اللغة:** أجب دائماً بالعربية الفصحى.
<</SYS>>

[Context]
{context}

[Chat History]
{chat_history}

[User Question]
{question}
[/INST]
"""

class ProRAGChain:
    def __init__(self, vector_store, all_documents: List[Document]):
        self.vector_store = vector_store
        self.llm_model, self.llm_tokenizer = ModelManager.get_llm()
        self.embed_model = ModelManager.get_embedding_model()
        
        self.dense_retriever = vector_store.as_retriever(search_kwargs={"k": Config.RETRIEVAL_K})
        self.bm25_retriever = BM25Retriever.from_documents(all_documents)
        self.bm25_retriever.k = Config.RETRIEVAL_K

        self.gen_pipeline = self._create_pipeline(max_new_tokens=Config.MAX_NEW_TOKENS, sample=True)
        self.trans_pipeline = self._create_pipeline(max_new_tokens=1024, sample=False)

    def _create_pipeline(self, max_new_tokens, sample=True):
        return pipeline(
            "text-generation",
            model=self.llm_model,
            tokenizer=self.llm_tokenizer,
            max_new_tokens=max_new_tokens,
            do_sample=sample,
            temperature=Config.TEMPERATURE if sample else 0.1,
            top_p=Config.TOP_P if sample else 1.0,
            repetition_penalty=Config.REPETITION_PENALTY if sample else 1.0,
            pad_token_id=self.llm_tokenizer.eos_token_id
        )

    def _rewrite_query(self, query: str, history: List[Tuple[str, str]]) -> str:
        if not history: return query
        
        hist_str = "\n".join([f"User: {h[0]}\nAI: {h[1]}" for h in history[-2:]])
        
        prompt = CONDENSE_Q_PROMPT.format(chat_history=hist_str, question=query)
        
        try:
            out = self.trans_pipeline(prompt)[0]["generated_text"]
            rewritten = out.split("\n")[-1].strip()
            
            logger.info(f"🧠 Rewriting Logic: '{query}' -> '{rewritten}'")
            return rewritten if rewritten else query
        except Exception as e:
            logger.warning(f"⚠️ Query rewriting failed: {e}")
            return query

    def _rrf_merge(self, dense_docs: List[Document], bm25_docs: List[Document], k=60) -> List[Document]:
        scores = {}
        store = {}
        def get_key(d): return f"{d.metadata.get('service_id','')}::{d.page_content[:50]}"
        for r, d in enumerate(dense_docs, 1):
            key = get_key(d)
            scores[key] = scores.get(key, 0) + 1 / (k + r)
            store[key] = d
        for r, d in enumerate(bm25_docs, 1):
            key = get_key(d)
            scores[key] = scores.get(key, 0) + 1 / (k + r)
            store[key] = d
        return [store[k] for k in sorted(scores, key=scores.get, reverse=True)]

    def _dense_rerank(self, query: str, docs: List[Document], top_k=6) -> List[Document]:
        if not docs: return []
        try:
            query_vec = self.embed_model.embed_query(query)
            scored = []
            for d in docs[:20]:
                doc_vec = self.embed_model.embed_query(d.page_content)
                norm_q = np.linalg.norm(query_vec)
                norm_d = np.linalg.norm(doc_vec)
                score = 0 if (norm_q == 0 or norm_d == 0) else np.dot(query_vec, doc_vec) / (norm_q * norm_d)
                scored.append((score, d))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [d for _, d in scored[:top_k]]
        except Exception as e:
            logger.error(f"⚠️ Reranking failed: {e}")
            return docs[:top_k]

    def _translate(self, text: str, source_lang: str, target_lang: str) -> str:
        if not text or not text.strip(): return text
        if source_lang == target_lang: return text
        
        lang_map = {
            "ar": "Arabic", "en": "English", "zh-cn": "Chinese", 
            "hi": "Hindi", "ru": "Russian", "ur": "Urdu", "fr": "French"
        }
        src_name = lang_map.get(source_lang, source_lang)
        tgt_name = lang_map.get(target_lang, target_lang)
        
        prompt = f"Translate from {src_name} to {tgt_name}.\nText: {text}\nTranslation:"
        
        try:
            out = self.trans_pipeline(prompt, temperature=0.1)[0]["generated_text"]
            if "Translation:" in out: return out.split("Translation:", 1)[-1].strip()
            return out.replace(prompt, "").strip()
        except: return text

    def _apply_text_direction(self, text: str, lang_code: str) -> str:
        rtl_langs = ['ar', 'he', 'ur', 'fa']
        return f"<div dir='{'rtl' if lang_code in rtl_langs else 'ltr'}' style='text-align: {'right' if lang_code in rtl_langs else 'left'};'>{text}</div>"

    def answer(self, query: str, history: List[Tuple[str, str]] = []) -> str:
        
        # 1. Detect Language (Smart Arabic Priority)
        # 🟢 FIX: Prioritize Arabic based on character set
        if any('\u0600' <= char <= '\u06FF' for char in query):
            user_lang = 'ar'
        else:
            try: user_lang = detect(query)
            except: user_lang = 'ar'
        
        logger.info(f"🌍 Detected User Language: {user_lang}")

        # 2. Pre-Translation
        if user_lang != 'ar':
            q_ar = self._translate(query, source_lang=user_lang, target_lang='ar')
        else:
            q_ar = normalize_arabic(query)

        # 3. Contextual Rewrite (Smart Memory)
        if history:
            search_query = self._rewrite_query(q_ar, history)
        else:
            search_query = q_ar

        # 4. Retrieve
        dense_res = self.dense_retriever.invoke(search_query)
        bm25_res = self.bm25_retriever.invoke(search_query)
        merged_docs = self._rrf_merge(dense_res, bm25_res)
        final_docs = self._dense_rerank(search_query, merged_docs, top_k=Config.RETRIEVAL_K) # Keep higher K for final check
        
        # 5. Generate Answer
        context_text = "\n\n".join([f"- {d.page_content}" for d in final_docs])
        history_text = "\n".join([f"User: {h[0]}\nAI: {h[1]}" for h in history[-3:]])
        
        # Pass the Rewritten Query (search_query) to the context
        final_prompt = SYSTEM_PROMPT.format(
            context=context_text, 
            chat_history=history_text,
            question=f"{search_query}" 
        )
        
        logger.info("🤖 Generating answer...")
        try:
            raw_response = self.gen_pipeline(final_prompt)[0]["generated_text"]
            answer_ar = raw_response.split("[/INST]", 1)[-1].strip()
        except Exception as e:
            return "عذراً، حدث خطأ."

        # 6. Post-Translation
        final_response_text = answer_ar
        if user_lang != 'ar':
            final_response_text = self._translate(answer_ar, source_lang='ar', target_lang=user_lang)

        return self._apply_text_direction(final_response_text, user_lang)