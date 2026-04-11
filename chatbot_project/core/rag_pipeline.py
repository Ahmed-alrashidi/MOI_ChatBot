# =========================================================================
# File Name: core/rag_pipeline.py
# Purpose: Master Intelligence Orchestrator (The Sovereign AI Engine).
# Project: Absher Smart Assistant (MOI ChatBot)
# Version: 5.0 (KG Price Authority, Intent Guard v5, Rewrite Validation, Dynamic Token Cap)
# Features: 
# - Intent Guard 3.0: Expanded social detection with fuzzy matching & abuse filter.
# - T-S-T Logic: Full Multilingual depth (Translate-Search-Translate).
# - KG Enrichment: Precision matching using query + context intersection.
# - Query Rewriting: Fixed short-query skip to preserve multi-turn context.
# - Inference Mode: Optimized for maximum throughput on NVIDIA A100.
# =========================================================================

import os
import torch
import json
import time
import pandas as pd
from difflib import SequenceMatcher
from typing import List, Tuple, Optional, Any, Dict
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langdetect import detect

from config import Config
from utils.logger import setup_logger
from utils.text_utils import normalize_arabic
from core.model_loader import ModelManager
from core.vector_store import VectorStoreManager

# Initialize the main Reasoning Engine Logger
logger = setup_logger("RAG_Engine")

class RAGPipeline:
    """
    The central intelligence core that manages the query lifecycle:
    Intent Detection -> Translation -> Retrieval -> KG Enrichment -> Generation.
    """
    
    def __init__(self, llm: Optional[Any] = None, tokenizer: Optional[Any] = None):
        """Initializes all intelligence components and retrievers."""
        logger.info("🚀 System: Booting Secure RAG Intelligence...")
        
        self.embed_model = ModelManager.get_embedding_model()
        
        # Load LLM (Supports dependency injection for benchmarking)
        if llm is not None and tokenizer is not None:
            self.llm, self.tokenizer = llm, tokenizer
        else:
            self.llm, self.tokenizer = ModelManager.get_llm()
            
        # Initialize verified Knowledge Graph
        self.kg_path = os.path.join(Config.DATA_PROCESSED_DIR, "services_knowledge_graph.json")
        self.knowledge_graph = self._load_knowledge_graph()

        # [NEW] Build a flat KG lookup for precise query-based matching
        self._kg_flat_index = self._build_kg_flat_index()

        # Build Hybrid Retrievers
        self.vector_db = VectorStoreManager.load_or_build(self.embed_model)
        self.dense_retriever = self.vector_db.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": Config.RETRIEVAL_K}
        )
        self.bm25_retriever = self._build_bm25_from_chunks()
        
        self.model_family = self._detect_model_family()
        self.PRIMARY_LANGS = ['ar', 'en']
        
        logger.info(f"✅ Status: Reasoning Engine ready for [{self.model_family}]")

    # --- 1. CORE INTELLIGENCE METHODS ---

    def _is_social_intent(self, query: str) -> Tuple[bool, str]:
        """
        [GUARD 3.0]: Detects social turns, short conversational noise, and abuse
        to bypass unnecessary database load.
        
        Improvements over v2.0:
        - Fuzzy matching for typos (e.g., 'ممع السلامة' -> 'مع السلامة')
        - Short conversational words ('لا', 'نعم', 'اوكي', 'تمام')
        - Abuse/off-topic filter for non-service queries
        - Farewell phrases expanded
        """
        q = query.lower().strip().replace("؟", "").replace("?", "")
        words = q.split()
        num_words = len(words)

        # Priority 0: Short conversational noise (1-2 words, non-technical)
        short_noise = {
            'لا', 'نعم', 'اوكي', 'تمام', 'ماشي', 'اها', 'اوك', 'ok', 'okay',
            'اي', 'ايه', 'يب', 'لا شكرا', 'yes', 'no', 'yep', 'nope', 'nah',
            'خلاص', 'بس', 'كفاية', 'enough', 'done', 'طيب'
        }
        if num_words <= 2 and q in short_noise:
            return True, "closing"

        # Priority 1: Greetings (MUST be checked before closings)
        # [FIX v5.0]: Removed standalone 'سلام' — it matches 'السلامة' in closings.
        # Use startswith for short greetings to avoid substring conflicts.
        greetings_exact = {
            'سلام عليكم', 'السلام عليكم', 'عليكم السلام',
            'هلا والله', 'السلام عليكم ورحمة الله'
        }
        greetings_starts = {
            'السلام', 'مرحبا', 'هلا', 'صباح', 'مساء',
            'hi', 'hello', 'hey', 'اهلا', 'مرحب'
        }
        if num_words <= 5:
            if any(g in q for g in greetings_exact):
                return True, "greeting"
            if any(q.startswith(g) for g in greetings_starts):
                return True, "greeting"

        # Priority 2: Farewell / Closing / Thanks (exact + fuzzy)
        # [FIX v4.0]: Only match on short messages (< 8 words) to avoid catching
        # real questions that happen to contain "شكرا" or similar words.
        closings_exact = {
            'مع السلامة', 'في امان الله', 'بمان الله', 'الله يعينك',
            'شكرا', 'مشكور', 'جزاك', 'جزاك الله', 'جزاك الله خير',
            'thanks', 'thank', 'thank you', 'thnx', 'thx', 'bye', 'goodbye', 'see you',
            'الله يعطيك العافية', 'يعطيك العافية', 'الله يجزاك خير',
            'انهي المحادثة', 'خلاص مع السلامة', 'باي'
        }
        # Exact substring check (only short messages)
        if num_words <= 7 and any(p in q for p in closings_exact):
            return True, "closing"
        
        # [NEW] Fuzzy check for typos (e.g., 'ممع السلامة', 'شكررا')
        if num_words <= 7 and self._fuzzy_match_any(q, closings_exact, threshold=0.80):
            return True, "closing"

        # Priority 3: Abuse / Off-topic filter
        abuse_keywords = {
            'حمار', 'غبي', 'كلب', 'حيوان', 'تافه', 'احمق', 'معفن',
            'ابله', 'منيح', 'زبالة', 'وسخ', 'كذاب', 'خرا',
            'stupid', 'idiot', 'dumb', 'useless', 'trash', 'fool'
        }
        if any(w in q for w in abuse_keywords):
            return True, "abuse"
        
        # Priority 4: Positive reactions / praise (not real questions)
        praise_patterns = {
            'ممتاز', 'رائع', 'حلو', 'جميل', 'عظيم', 'مبدع', 'احسنت', 'يسلمو',
            'ابداع', 'بارك الله', 'الله يعطيك', 'كفو', 'يا سلام', 'تسلم',
            'great', 'awesome', 'amazing', 'perfect', 'nice', 'good job', 'cool',
            'excellent', 'wonderful', 'fantastic', 'well done', 'impressive'
        }
        if num_words <= 5 and any(p in q for p in praise_patterns):
            return True, "closing"

        # Priority 5: Short emotional noise (e.g., "احبك", "حب")
        if num_words < 3 and any(w in q for w in ['احبك', 'حب', 'بطل', 'love']):
            return True, "closing"

        return False, "technical"

    def _fuzzy_match_any(self, text: str, patterns: set, threshold: float = 0.80) -> bool:
        """
        [NEW] Checks if the text fuzzy-matches any pattern in the set.
        Uses SequenceMatcher for O(n) approximate string matching.
        Only activates for short texts (< 8 words) to avoid false positives.
        """
        if len(text.split()) > 7:
            return False
        for pattern in patterns:
            ratio = SequenceMatcher(None, text, pattern).ratio()
            if ratio >= threshold:
                return True
        return False

    def _enrich_with_kg(self, context: str, query: str) -> str:
        """
        [FACT-CHECK v3.0]: KG is the SINGLE SOURCE OF TRUTH for prices/fees.
        
        Fix v3.0: 
        - Match on query OR context (was AND — too strict, missed many services)
        - Strip Arabic definite article (ال) for better keyword matching
        - Lower keyword overlap threshold to 1 for recall
        - KG prices override anything in the RAG context
        """
        enriched_buffer = context
        found_facts = []
        query_normalized = normalize_arabic(query)
        context_normalized = normalize_arabic(context)
        
        for sector, services in self.knowledge_graph.items():
            for svc_name, details in services.items():
                svc_normalized = normalize_arabic(svc_name)
                
                # Match if service is relevant to query OR appears in retrieved context
                query_match = svc_normalized in query_normalized or self._kg_keyword_match(query_normalized, svc_normalized)
                context_match = svc_normalized in context_normalized or self._kg_keyword_match(context_normalized, svc_normalized)
                
                if query_match or context_match:
                    price = details.get('price', '')
                    if price in ('متغيرة', 'رسوم التوصيل (متغيرة)', 'رسوم نقل الملكية المقررة'):
                        continue
                    
                    # Prioritize query matches over context-only matches
                    priority = 0 if query_match else 1
                    found_facts.append((svc_name, details, priority))
        
        # Sort: query matches first, then context matches. Max 3 facts.
        found_facts.sort(key=lambda x: x[2])
        
        for svc_name, details, _ in found_facts[:3]:
            fact_sheet = (
                f"\n\n[المصدر الرسمي - {svc_name}]:\n"
                f"- الرسوم: {details['price']}\n"
                f"- الخطوات: {details['steps']}\n"
            )
            enriched_buffer += fact_sheet
        
        if found_facts:
            logger.info(f"🎯 Fact-Check Injection: {len(found_facts[:3])} fact(s) for [{', '.join(f[0] for f in found_facts[:3])}]")
        
        return enriched_buffer

    def _kg_keyword_match(self, query: str, service_name: str) -> bool:
        """
        [FIX v5.0] Tighter keyword matching. 
        Generic action words like إصدار/تجديد/استعلام are excluded because
        they appear in almost every service name and cause false matches.
        Only SPECIFIC content words (جواز, مرور, هوية, etc.) trigger matches.
        """
        skip_words = {
            # Stopwords
            'من', 'في', 'عن', 'على', 'الى', 'هل', 'ما', 'كم', 'كيف', 'هي', 'هو', 'او', 'ثم',
            # Generic action words that appear in most service names
            'خدمه', 'خدمة', 'استعلام', 'عام', 'عامه', 'اصدار', 'تجديد', 'الغاء',
            'طلب', 'تحديث', 'نقل', 'بيانات', 'حالة', 'صلاحيه', 'صلاحية',
            'ابشر', 'اعمال', 'منصه', 'منصة', 'رقم', 'معلومات',
        }
        
        def strip_article(word):
            if word.startswith('ال') and len(word) > 3:
                return word[2:]
            return word
        
        query_words = {strip_article(w) for w in query.split()} - skip_words
        svc_words = {strip_article(w) for w in service_name.split()} - skip_words
        
        # Remove very short words (< 3 chars)
        query_words = {w for w in query_words if len(w) >= 3}
        svc_words = {w for w in svc_words if len(w) >= 3}
        
        overlap = query_words & svc_words
        return len(overlap) >= 1

    def _build_kg_flat_index(self) -> Dict[str, Dict[str, Any]]:
        """
        [NEW] Builds a flat lookup of normalized service names -> KG data.
        Used for precise query-time matching.
        """
        flat = {}
        for sector, services in self.knowledge_graph.items():
            for svc_name, details in services.items():
                key = normalize_arabic(svc_name)
                flat[key] = {"name": svc_name, "sector": sector, **details}
        return flat

    def _rewrite_query(self, query: str, history: List[Tuple[str, str]]) -> str:
        """
        [CONTEXT-AWARE v3.0]: Resolves pronouns and implicit references using history.
        
        Fix v3.0: Added intent-preservation validation. If the rewrite drops ALL 
        meaningful words from the original query, it means the LLM hallucinated a 
        different topic from history. In that case, we reject the rewrite and use 
        the original query to prevent wrong-topic answers.
        
        Also: Skip rewriting if the query is already self-contained (has enough 
        content words and no pronouns/references that need resolution).
        """
        if not history:
            return query
        
        # Don't rewrite if query is social
        is_social, _ = self._is_social_intent(query)
        if is_social:
            return query

        # [NEW] Skip rewriting if the query is already self-contained
        # A query with 5+ words and no referential pronouns doesn't need rewriting
        referential_markers = {
            'هو', 'هي', 'هم', 'ها', 'ذلك', 'هذا', 'هذه', 'نفسها', 'نفسه',
            'كذلك', 'ايضا', 'أيضا', 'وكم', 'وماذا', 'وكيف',
            'it', 'its', 'that', 'this', 'those', 'them', 'same', 'also', 'too',
            'اجدده', 'اجددها', 'تجديده', 'رسومه', 'رسومها', 'خطواته', 'خطواتها',
        }
        query_words = set(query.lower().split())
        has_reference = bool(query_words & referential_markers)
        
        if len(query_words) >= 5 and not has_reference:
            return query

        sys_msg = (
            "You are a query rewriter for a Saudi government services chatbot. "
            "Rewrite the user's new question to be fully standalone and self-contained, "
            "incorporating relevant context from the conversation history. "
            "If the user refers to a previous topic (e.g., 'كم الرسوم' after asking about passports), "
            "make the reference explicit (e.g., 'كم رسوم تجديد جواز السفر'). "
            "IMPORTANT: If the new question introduces a NEW topic not in the history, "
            "keep it as-is. Do NOT replace it with a previous topic. "
            "Return ONLY the rewritten question in the same language, nothing else."
        )
        hist_txt = "\n".join([f"User: {h[0]}\nAI: {h[1]}" for h in history[-2:]])
        prompt = self._apply_template(sys_msg, f"History:\n{hist_txt}\nNew Question: {query}")
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(Config.DEVICE)
            with torch.inference_mode():
                out = self.llm.generate(**inputs, max_new_tokens=100, temperature=0.1)
            rewritten = self.tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            
            # Sanity check: if rewrite is empty or too long, use original
            if not rewritten or len(rewritten) > len(query) * 5:
                return query
            
            # [NEW] Intent-preservation validation:
            # If the original query has meaningful content words, at least some must 
            # survive in the rewrite. Otherwise the LLM replaced the topic entirely.
            if not self._rewrite_preserves_intent(query, rewritten):
                logger.warning(f"⚠️ Rewrite rejected (intent lost): '{query}' → '{rewritten}' | Using original.")
                return query
            
            logger.info(f"🔄 Query Rewrite: '{query}' → '{rewritten}'")
            return rewritten
        except Exception as e:
            logger.warning(f"⚠️ Query rewrite failed: {e}")
            return query

    def _rewrite_preserves_intent(self, original: str, rewritten: str) -> bool:
        """
        [NEW] Validates that the rewrite didn't hallucinate a completely different topic.
        
        Logic: Extract meaningful content words from the original query (skip stopwords).
        If the original has 2+ content words and NONE appear in the rewrite, the rewrite 
        is likely hallucinated from history and should be rejected.
        """
        stopwords = {
            # Arabic stopwords
            'من', 'في', 'عن', 'على', 'الى', 'هل', 'ما', 'كم', 'كيف', 'هي', 'هو',
            'او', 'ثم', 'لا', 'نعم', 'ان', 'لي', 'لك', 'قبل', 'بعد', 'عادي', 'يا',
            'اذا', 'لو', 'مع', 'حتى', 'بس', 'اللي', 'الي', 'ايش', 'وش', 'ليش',
            'اسال', 'سؤال', 'اخر', 'اريد', 'ابغى', 'ابي', 'ممكن', 'لو', 'سمحت',
            'امشي', 'ماهي', 'ماهو', 'شنو',
            # English stopwords  
            'the', 'a', 'an', 'is', 'are', 'was', 'what', 'how', 'can', 'i', 'you',
            'me', 'my', 'do', 'does', 'will', 'would', 'could', 'should', 'please',
            'want', 'need', 'know', 'tell', 'about', 'also', 'another', 'question',
            'before', 'after', 'yes', 'no', 'ok', 'okay',
        }
        
        orig_normalized = normalize_arabic(original.lower())
        rewrite_normalized = normalize_arabic(rewritten.lower())
        
        orig_words = set(orig_normalized.split()) - stopwords
        rewrite_words = set(rewrite_normalized.split()) - stopwords
        
        # If original has fewer than 2 content words, it's likely a short follow-up
        # that genuinely needs context (e.g., "و 10 سنوات كم") — allow the rewrite
        if len(orig_words) < 2:
            return True
        
        # If original has 2+ content words but NONE survived in the rewrite,
        # the LLM replaced the topic entirely — reject
        overlap = orig_words & rewrite_words
        if len(overlap) == 0:
            return False
        
        # [FIX v4.0] Question-type preservation: if the original asks "how/steps" 
        # but the rewrite asks "how much/fees", the question type changed — reject.
        how_words = {'كيف', 'خطوات', 'طريقه', 'طريقة', 'how', 'steps', 'ماهي', 'ماهو'}
        price_words = {'كم', 'رسوم', 'سعر', 'تكلفه', 'تكلفة', 'fees', 'cost', 'price', 'much'}
        
        orig_all = set(orig_normalized.split())
        rewrite_all = set(rewrite_normalized.split())
        
        orig_asks_how = bool(orig_all & how_words)
        rewrite_asks_price = bool(rewrite_all & price_words)
        orig_asks_price = bool(orig_all & price_words)
        rewrite_asks_how = bool(rewrite_all & how_words)
        
        # If question type flipped (how→price or price→how), reject
        if orig_asks_how and not orig_asks_price and rewrite_asks_price and not rewrite_asks_how:
            return False
        if orig_asks_price and not orig_asks_how and rewrite_asks_how and not rewrite_asks_price:
            return False
        
        return True

    def translate_text(self, text: str, source: str, target: str) -> str:
        """[TST-UTILITY]: Translates text between any languages for global access."""
        if source == target: return text
        sys_msg = f"Translate from {source} to {target}. Maintain technical terms like 'Absher'."
        prompt = self._apply_template(sys_msg, f"Text: {text}")
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(Config.DEVICE)
            with torch.inference_mode():
                out = self.llm.generate(**inputs, max_new_tokens=256, temperature=0.1)
            return self.tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        except Exception as e:
            logger.warning(f"⚠️ Translation failed: {e}")
            return text

    # --- 2. EXECUTION PIPELINE ---

    def run(self, query: str, history: Optional[List[Tuple[str, str]]] = None) -> str:
        """Master Orchestration Flow."""
        if history is None:
            history = []
        start_time = time.time()
        
        # 1. Language Detection & Intent Guard
        user_lang = self._detect_real_lang(query)
        is_social, intent_type = self._is_social_intent(query)
        
        if is_social:
            logger.info(f"🎭 Social Interaction ({intent_type}) Detected.")
            
            if intent_type == "greeting":
                return ("وعليكم السلام ورحمة الله وبركاته، كيف يمكنني مساعدتك اليوم؟" 
                        if user_lang == 'ar' else "Hello! How can I assist you today?")
            
            if intent_type == "abuse":
                return ("عذراً، أنا مساعد أبشر الذكي ومهمتي مساعدتك في خدمات وزارة الداخلية. كيف يمكنني خدمتك؟" 
                        if user_lang == 'ar' else 
                        "I'm the Absher Smart Assistant, here to help with Ministry of Interior services. How can I assist you?")
            
            # closing / thanks / noise
            return ("شكراً لتواصلك مع مساعد أبشر الذكي. نتمنى لك يوماً سعيداً!" 
                    if user_lang == 'ar' else "Thank you for using Absher Smart Assistant. Have a great day!")

        # 2. T-S-T Logic (Translate to English if language is weak)
        is_weak = user_lang not in self.PRIMARY_LANGS
        search_query = self.translate_text(query, user_lang, "en") if is_weak else query
        
        # 3. Query Rewriting & Normalization
        processed_query = self._rewrite_query(search_query, history)
        clean_search = normalize_arabic(processed_query)

        # 4. Hybrid Retrieval
        dense_res = self.dense_retriever.invoke(clean_search)
        sparse_res = self.bm25_retriever.invoke(clean_search) if self.bm25_retriever else []
        final_docs = self._rrf_merge(dense_res, sparse_res)

        if not final_docs:
            return ("عذراً، لا تتوفر معلومات رسمية لهذه الحالة." 
                    if user_lang == 'ar' else "Sorry, no official records found.")

        # 5. Enrichment & Prompt Construction
        # [FIX] Pass query to KG enrichment for precision matching
        raw_context = "\n".join([d.page_content for d in final_docs])
        context = self._enrich_with_kg(raw_context, processed_query)
        target_gen_lang = "English" if is_weak else ("Arabic" if user_lang == 'ar' else "English")
        
        system_instr = Config.SYSTEM_PROMPT_CONTENT.format(target_lang=target_gen_lang)
        full_prompt = self._apply_template(system_instr, f"Context:\n{context}\n\nQ: {search_query}")

        # 6. LLM Generation (Optimized)
        # [FIX v5.0] Dynamic token cap: more aggressive to prevent hallucination.
        # Short context (~500 chars, 1 service) → 256 tokens max.
        # Medium context (~1500 chars) → ~500 tokens.
        # Cap at 512 tokens to prevent runaway generation on any query.
        context_len = len(context)
        dynamic_max_tokens = min(512, max(256, context_len // 3))
        
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(Config.DEVICE)
        if "token_type_ids" in inputs: del inputs["token_type_ids"]

        with torch.inference_mode():
            outputs = self.llm.generate(
                **inputs, 
                max_new_tokens=dynamic_max_tokens, 
                temperature=Config.TEMPERATURE, 
                repetition_penalty=Config.REPETITION_PENALTY,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

        # 7. Final Output Translation (Back to user's native tongue if weak)
        final_output = self.translate_text(response, "en", user_lang) if is_weak else response
        
        logger.info(f"✅ RAG Cycle Complete | Latency: {time.time() - start_time:.2f}s")
        return final_output

    # --- 3. HELPER UTILITIES ---

    def _detect_real_lang(self, text: str) -> str:
        """
        Heuristic language detection with Urdu/Arabic disambiguation.
        Urdu-specific chars (پ چ ڈ ڑ ٹ ں ہ ے) distinguish it from Arabic.
        """
        try:
            has_arabic_script = any("\u0600" <= c <= "\u06FF" for c in text)
            if has_arabic_script:
                # Check for Urdu-specific characters before defaulting to Arabic
                urdu_chars = set('پچڈڑٹںھہےۓکگ')
                if any(c in urdu_chars for c in text):
                    return 'ur'
                return 'ar'
            lang = detect(text)
            return 'ur' if lang in ('hi', 'ur') else lang
        except:
            return 'en'

    def _detect_model_family(self) -> str:
        """Infers architecture from config."""
        name = str(getattr(self.llm.config, "_name_or_path", "")).lower()
        if any(x in name for x in ["allam", "llama-2", "mistral"]): return "llama2"
        if "llama-3" in name: return "llama3"
        if "qwen" in name: return "qwen"
        return "default"

    def _apply_template(self, system_msg: str, user_msg: str) -> str:
        """Wraps messages in native model tokens."""
        if self.model_family == "llama2":
            return f"<s>[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n\n{user_msg} [/INST]"
        elif self.model_family == "llama3":
            return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        elif self.model_family == "qwen":
            return f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
        return f"System: {system_msg}\nUser: {user_msg}\nAssistant: "

    def _load_knowledge_graph(self) -> Dict[str, Any]:
        """Loads deterministic KG data."""
        if os.path.exists(self.kg_path):
            try:
                with open(self.kg_path, 'r', encoding='utf-8') as f: return json.load(f)
            except Exception as e: logger.error(f"KG Error: {e}")
        return {}

    def _build_bm25_from_chunks(self) -> Optional[BM25Retriever]:
        """Constructs Sparse BM25 index."""
        docs = []
        if not os.path.exists(Config.DATA_CHUNK_DIR): return None
        for fn in os.listdir(Config.DATA_CHUNK_DIR):
            if fn.endswith(".csv"):
                try:
                    df = pd.read_csv(os.path.join(Config.DATA_CHUNK_DIR, fn))
                    for _, r in df.iterrows():
                        content = f"Svc: {r.get('اسم الخدمة','')}\nCont: {r.get('RAG_Content','')}"
                        docs.append(Document(page_content=normalize_arabic(content)))
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load BM25 chunk {fn}: {e}")
                    continue
        if docs:
            retriever = BM25Retriever.from_documents(docs)
            retriever.k = Config.RETRIEVAL_K  # Match FAISS top-K
            return retriever
        return None

    def _rrf_merge(self, dense_docs: List[Document], sparse_docs: List[Document]) -> List[Document]:
        """Blends FAISS and BM25 results."""
        scores = {}
        for rank, d in enumerate(dense_docs):
            txt = d.page_content
            scores[txt] = scores.get(txt, 0.0) + (1.0 / (Config.RRF_K + rank + 1))
        if sparse_docs:
            for rank, d in enumerate(sparse_docs):
                txt = d.page_content
                scores[txt] = scores.get(txt, 0.0) + (1.0 / (Config.RRF_K + rank + 1))
        sorted_res = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [Document(page_content=k) for k, v in sorted_res[:Config.RETRIEVAL_K]]