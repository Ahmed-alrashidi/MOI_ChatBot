# =========================================================================
# File Name: core/rag_pipeline.py
# Purpose: Master Intelligence Orchestrator (The Sovereign AI Engine).
# Project: Absher Smart Assistant (MOI ChatBot)
# Version: 5.3.0 (Per-User Persistent Memory — 16+7 Fixes Applied)
#
# Fixes applied from analysis report:
#   Fix #1 (Critical): last_service now restored from JSONL matched_services field
#   Fix #2 (High): _memories capped at 500 users via LRU eviction
#   Fix #3 (High): Thread-safe _get_memory() with threading.Lock
#   Fix #4 (Medium): deque(maxlen=N) for efficient JSONL tail-read
#   Fix #5 (Low): import re moved to top-level
#   Fix #6 (Low): Meta-query distinguishes current session vs loaded history
#   Fix #7 (Medium): Uses matched_services from JSONL, latency heuristic as fallback
#   Fix #8 (Test): Slot memory only resolves truly vague queries (no extra content words)
#   Fix #9 (Test): Weak languages skip streaming to avoid English flash
#   Fix #10 (User): Emoji-only queries (👍👎) caught as social, not sent to RAG
#   Fix #11 (User): Meta-query "وش اول سؤال سألتك" pattern added
#   Fix #12 (User): Doubt queries "معلومتك اكيدة" / "ماسألتك" caught as frustration
#   Fix #13 (User): "تعرف اسمي" / "تكلم عن نفسك" caught as identity
#   Fix #14 (User): "الو" / "هلو" caught as social noise
#   Fix #15 (User): Meta-query uses normalize_arabic() to handle hamza variants
#   Fix #16 (Critical): KG Price Bypass — price queries answered from KG directly, no LLM
#   Fix #18 (Security): Context-aware safety classifier — ambiguous words analyzed in context
# =========================================================================
import os
import re
import torch
import json
import time
import threading
import pandas as pd
from collections import OrderedDict, deque
from difflib import SequenceMatcher
from typing import List, Tuple, Optional, Any, Dict
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langdetect import detect

from config import Config
from utils.logger import setup_logger
from utils.text_utils import normalize_arabic, normalize_for_dense, extract_arabic_tokens
from core.model_loader import ModelManager
from core.vector_store import VectorStoreManager

# [T-S-T] NLLB-200 Translation Engine (loaded lazily on first weak-language query)
_nllb_translator = None
_nllb_lock = threading.Lock()  # [FIX v5.3.0] Thread-safe double-check locking

def _get_translator():
    global _nllb_translator
    if _nllb_translator is not None:
        return _nllb_translator
    if not getattr(Config, 'TST_ENABLED', False):
        return None
    with _nllb_lock:
        if _nllb_translator is not None:  # Double-check after lock
            return _nllb_translator
        try:
            from core.translator import NLLBTranslator
            _nllb_translator = NLLBTranslator()
        except Exception as e:
            logger.warning(f"⚠️ NLLB init failed, falling back to ALLaM: {e}")
    return _nllb_translator

logger = setup_logger("RAG_Engine")

_LANG_DISPLAY = {"Arabic": "العربية", "English": "الإنجليزية"}
_MAX_HISTORY_LOAD = 10
_MAX_CACHED_USERS = 500


# =========================================================================
# CONVERSATION MEMORY (Per-User, Persistent, Slot-Based)
# =========================================================================
class ConversationMemory:
    """
    [v5.1.2] Per-user conversation memory with persistence.
    Fixes: last_service restoration, session boundary tracking, efficient I/O.
    """

    def __init__(self, username: str = "guest"):
        self.username = username
        self.last_service: Optional[str] = None
        self.last_sector: Optional[str] = None
        self.last_query: Optional[str] = None
        self.history: List[Tuple[str, str]] = []
        self._session_start_index: int = 0  # [Fix #6] Marks where loaded history ends
        self.user_name: Optional[str] = None  # [F5] "اسمي أحمد" → stores "أحمد"
        self._summary: Optional[str] = None  # Conversation summary for long chats

    @classmethod
    def load_from_jsonl(cls, username: str) -> "ConversationMemory":
        """
        Creates a ConversationMemory pre-loaded with the user's
        last N turns from their telemetry JSONL file.
        """
        memory = cls(username=username)

        history_dir = os.path.join(Config.OUTPUTS_DIR, "user_analytics")
        safe_username = re.sub(r'[^a-zA-Z0-9_\-\u0600-\u06FF]', '_', username or "guest_user")
        history_file = os.path.join(history_dir, f"{safe_username}_history.jsonl")

        if not os.path.exists(history_file):
            logger.info(f"📝 No history for '{safe_username}' — starting fresh.")
            return memory

        try:
            # [Fix #4] Use deque to keep only last N entries during read
            recent = deque(maxlen=_MAX_HISTORY_LOAD)
            with open(history_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            recent.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

            for entry in recent:
                query = entry.get("user_query", "")
                response = entry.get("ai_response", "")
                if query:
                    memory.history.append((query, response[:150]))

            # [Fix #6] Mark where loaded history ends — current session starts after this
            memory._session_start_index = len(memory.history)

            # [Fix #1 & #7] Restore last_service from JSONL
            # Priority 1: Look for explicit matched_services field (new telemetry format)
            for entry in reversed(list(recent)):
                matched = entry.get("matched_services")
                if matched and isinstance(matched, list) and len(matched) > 0:
                    memory.last_service = matched[0]
                    memory.last_query = entry.get("user_query", "")
                    logger.info(f"🔗 Restored last_service for '{safe_username}': {memory.last_service}")
                    break

            # Priority 2: Fallback to latency heuristic (old telemetry entries without matched_services)
            if not memory.last_service:
                for entry in reversed(list(recent)):
                    latency = entry.get("latency_seconds", 0)
                    resp = entry.get("ai_response", "")
                    if latency > 0.5 and len(resp) > 100:
                        memory.last_query = entry.get("user_query", "")
                        break

            logger.info(f"📂 Loaded {len(memory.history)} turns for '{safe_username}'.")

        except Exception as e:
            logger.warning(f"⚠️ Failed to load history for '{safe_username}': {e}")

        return memory

    def update(self, query: str, matched_services: List[str] = None, sector: str = None):
        self.last_query = query
        if matched_services:
            self.last_service = matched_services[0]
        if sector:
            self.last_sector = sector

    def add_turn(self, query: str, response: str):
        self.history.append((query, response[:150]))

    def resolve_pronouns(self, query: str) -> str:
        if not self.last_service:
            return query

        q_lower = query.strip()
        words = q_lower.split()

        if len(words) > 4:
            return query

        pronoun_map = {
            'رسومه': f'رسوم {self.last_service}',
            'رسومها': f'رسوم {self.last_service}',
            'كم رسومه': f'كم رسوم {self.last_service}',
            'خطواته': f'خطوات {self.last_service}',
            'خطواتها': f'خطوات {self.last_service}',
            'شروطه': f'شروط {self.last_service}',
            'شروطها': f'شروط {self.last_service}',
            'تجديده': f'تجديد {self.last_service}',
            'تجديدها': f'تجديد {self.last_service}',
        }

        for pattern, replacement in pronoun_map.items():
            if pattern in q_lower:
                resolved = q_lower.replace(pattern, replacement)
                logger.info(f"🔗 Slot [{self.username}]: '{query}' → '{resolved}'")
                return resolved

        vague_starters = [
            'كيف اسدد', 'كيف أسدد', 'كيف اجدد', 'كيف أجدد',
            'كم التكلفة', 'كم التكلفه', 'وش الشروط', 'ايش الشروط',
            'وش الخطوات', 'ايش الخطوات', 'كم المدة', 'كم المده',
            # [Bug #6 Fix] English vague starters
            'how much', 'what are the fees', 'what about it',
            'its fees', 'its steps', 'its requirements',
        ]

        for starter in vague_starters:
            # [Fix #8] Only resolve if query has NO extra content words after the starter
            # "كيف اسدد" (2 words = starter 2 words) → resolve ✅
            # "كيف أجدد الهوية" (3 words > starter 2 words) → already has object, skip ✗
            starter_len = len(starter.split())
            if q_lower.startswith(starter) and len(words) <= starter_len:
                resolved = f"{q_lower} {self.last_service}"
                logger.info(f"🔗 Slot [{self.username}]: '{query}' → '{resolved}'")
                return resolved

        return query

    def handle_meta_query(self, query: str) -> Optional[str]:
        """
        [Phase 1 Perfect] Handles ALL meta-queries about the conversation,
        bot identity, service count, user name, self-questions, and summaries.
        Returns response string if matched, None if not a meta-query.
        """
        # [Fix #15] Normalize to handle hamza variants (سؤال vs سوال, أول vs اول)
        q_lower = normalize_arabic(query.lower().strip())
        q_raw = query.lower().strip()

        # [Fix #6] Determine scope: current session or all history
        current_session = self.history[self._session_start_index:]
        has_current = len(current_session) > 0

        # ── F5: NAME INTRODUCTION ("اسمي أحمد") ──
        name_result = self._detect_name_intro(q_raw)
        if name_result:
            return name_result

        # ── F5: NAME RECALL ("تذكر اسمي", "ماهو اسمي") ──
        name_recall_patterns = [
            'تذكر اسمي', 'تتذكر اسمي', 'ماهو اسمي', 'ما هو اسمي',
            'وش اسمي', 'ايش اسمي', 'اسمي ايش', 'اسمي وش',
            'عرفت اسمي', 'تعرف اسمي', 'ذاكر اسمي',
            'what is my name', "what's my name", 'do you know my name',
            'remember my name',
        ]
        if any(p in q_lower for p in name_recall_patterns):
            if self.user_name:
                return f"بالطبع! اسمك {self.user_name}. كيف يمكنني مساعدتك؟"
            return "لم تخبرني باسمك بعد. يمكنك قول 'اسمي ...' وسأتذكره."

        # ── F1: FIRST QUESTION (all phrasings) ──
        first_q_patterns = [
            'اول سوال', 'اول سؤال', 'first question',
            'اول شي سالتك', 'اول شي سالتني', 'اول استفسار',
            'وش اول سوال', 'وش اول سؤال',
            'اول سوال سالتك', 'اول سؤال سالتك',
            'وش اول سوال سالتك', 'وش اول سؤال سالتك',
            # [F1] New patterns from user testing
            'باول المحادثه', 'باول المحادثة', 'باول الجلسه', 'باول الجلسة',
            'بأول المحادثة', 'بأول الجلسة',
            'اول شيء', 'اول شي', 'أول شيء', 'أول شي',
            'سالتك اول', 'سألتك أول', 'سالتك اول شيء',
            'وش سالتك اول', 'وش سألتك اول', 'وش سالتك اول شيء',
            'اول شيء سالتك', 'اول شي سالتك',
            # [Bug #7 Fix] "ماذا كان سؤالي" patterns
            'ماذا كان سوالي', 'ماذا كان سؤالي',
            'وش كان سوالي', 'وش كان سؤالي',
            'ايش كان سوالي', 'ايش كان سؤالي',
            'سوالي الاول', 'سؤالي الاول', 'سؤالي الأول',
        ]
        if any(p in q_lower for p in first_q_patterns):
            if has_current:
                return f"أول سؤال طرحته في هذه الجلسة كان: \"{current_session[0][0]}\""
            elif self.history:
                return f"أول سؤال مسجل لديك كان: \"{self.history[0][0]}\""
            return "لم تطرح أي أسئلة بعد."

        # ── QUESTION COUNT ──
        count_patterns = ['كم سوال', 'كم سؤال', 'how many questions', 'عدد الاسئلة', 'عدد الأسئلة']
        if any(p in q_lower for p in count_patterns):
            n_session = len(current_session)
            n_total = len(self.history)
            if has_current:
                return f"طرحت {n_session} سؤال في هذه الجلسة، و{n_total} سؤال إجمالاً."
            elif n_total > 0:
                return f"لديك {n_total} سؤال مسجل من جلسات سابقة."
            return "لم تطرح أي أسئلة بعد."

        # ── LAST QUESTION ──
        last_q_patterns = ['اخر سوال', 'آخر سوال', 'last question', 'السوال السابق', 'السؤال السابق']
        if any(p in q_lower for p in last_q_patterns):
            if len(self.history) >= 2:
                return f"سؤالك السابق كان: \"{self.history[-2][0]}\""
            return "ليس لديك سؤال سابق."

        # ── F2: SERVICE COUNT ("كم عدد خدماتك") ──
        service_count_patterns = [
            'كم عدد خدماتك', 'كم خدمه عندك', 'كم خدمة عندك',
            'كم خدمه تقدم', 'كم خدمة تقدم', 'عدد خدماتك',
            'كم خدمه فيك', 'كم خدمة فيك', 'كم عدد الخدمات',
            'how many services',
        ]
        if any(p in q_lower for p in service_count_patterns):
            # [Bug #5 Fix] Check if user asks about a specific sector
            sector_map = {
                'جوازات': 'الجوازات', 'الجوازات': 'الجوازات',
                'احوال': 'الأحوال المدنية', 'الاحوال': 'الأحوال المدنية', 'احوال مدنية': 'الأحوال المدنية',
                'مرور': 'المرور', 'المرور': 'المرور',
                'امن عام': 'الأمن العام', 'الامن العام': 'الأمن العام', 'امن': 'الأمن العام',
                'سجون': 'المديرية العامة للسجون', 'المديريه': 'المديرية العامة للسجون',
                'وزارة الداخلية': 'وزارة الداخلية', 'الداخليه': 'وزارة الداخلية',
            }
            # Check if query mentions a sector
            for keyword, sector_name in sector_map.items():
                if keyword in q_lower:
                    return f"_SECTOR_COUNT:{sector_name}"
            return ("أقدم 140 خدمة حكومية عبر منصة أبشر، موزعة على 6 قطاعات: "
                    "الجوازات، الأحوال المدنية، المرور، الأمن العام، "
                    "المديرية العامة للسجون، ووزارة الداخلية. "
                    "اسأل عن أي خدمة وسأساعدك بالتفاصيل.")

        # ── [Bug #4 Fix] LANGUAGE COUNT ──
        lang_count_patterns = [
            'كم لغة', 'كم لغه', 'عدد اللغات', 'كم عدد اللغات',
            'بكم لغة', 'بكم لغه', 'تتكلم كم لغة', 'تتحدث كم لغة',
            'how many languages', 'what languages', 'which languages',
            'اللغات التي تتعامل', 'اللغات اللي تدعم', 'تدعم كم لغة',
        ]
        if any(p in q_lower for p in lang_count_patterns):
            return ("أتحدث 8 لغات: العربية، الإنجليزية، الأردية، الفرنسية، "
                    "الإسبانية، الألمانية، الروسية، والصينية. "
                    "يمكنك التحدث معي بأي من هذه اللغات وسأجيبك بنفس اللغة.")

        # ── F3: IDENTITY (alternate phrasings) ──
        identity_alt_patterns = [
            'من صنعك', 'مين صنعك', 'من صناعتك', 'مين صناعتك',
            'من الذي صنعك', 'من اللي صنعك',
            'من برمجك', 'مين برمجك', 'من طورك', 'مين طورك',
            'من بناك', 'مين بناك', 'من بنى', 'من بناه',
            'من سواك', 'مين سواك', 'من عملك', 'مين عملك',
            'من صممك', 'مين صممك', 'من اخترعك', 'مين اخترعك',
            'who made you', 'who created you', 'who built you',
            'who developed you', 'who programmed you',
        ]
        if any(p in q_lower for p in identity_alt_patterns):
            return ("أنا مساعد أبشر الذكي، نظام ذكاء اصطناعي مطوّر من فريق PGD+ في أكاديمية كاوست. "
                    "أساعدك في الاستفسار عن 140 خدمة حكومية عبر منصة أبشر، تشمل: "
                    "الجوازات، الأحوال المدنية، المرور، الأمن العام، وخدمات السجون. "
                    "أتحدث 8 لغات. كيف يمكنني مساعدتك؟")

        # ── F4: SELF-QUESTIONS (questions about the bot, not services) ──
        self_question_patterns = [
            'كم تستغرق', 'كم تاخذ وقت', 'كم تأخذ وقت',
            'وش التحديثات عليك', 'وش التحديثات اللي صارت',
            'ايش التحديثات', 'كيف تطويرك', 'كيف استطيع تطويرك',
            'كيف اطورك', 'كيف أطورك', 'كيف احسنك', 'كيف أحسنك',
            'وش مستواك', 'كيف مستواك', 'رايك في مستواك',
            'وش نقاط ضعفك', 'هل لديك نقاط ضعف', 'هل عندك عيوب',
            'كم عمرك', 'متى انولدت', 'متى صنعوك',
            'كيف تعرف انا بشري', 'كيف تفرق بين',
            'هل انت ذكي', 'هل تتعلم',
        ]
        if any(p in q_lower for p in self_question_patterns):
            return ("شكراً على اهتمامك! أنا مساعد أبشر الذكي، مختص بخدمات وزارة الداخلية. "
                    "أعمل بنموذج ALLaM-7B مع قاعدة معرفية تضم 140 خدمة. "
                    "متوسط وقت الإجابة 3-5 ثوانٍ. يتم تطويري باستمرار من فريق PGD+ في كاوست. "
                    "كيف يمكنني مساعدتك في خدمات أبشر؟")

        # ── F6: HALLUCINATION CALLOUT ("تهلوس", "غلط") ──
        hallucination_patterns = [
            'تهلوس', 'قاعد تهلوس', 'هلوسه', 'هلوسة',
            'معلوماتك غلط', 'اجابتك غلط', 'جوابك غلط',
            'كلامك غلط', 'كلامك خطا', 'كلامك خطأ',
            'مو صحيح اللي قلته', 'غلط اللي قلته',
            'هذا مو صحيح', 'اجابه خاطئه', 'إجابة خاطئة',
            'you are hallucinating', 'wrong answer', 'incorrect',
        ]
        if any(p in q_lower for p in hallucination_patterns):
            return ("أعتذر عن الخطأ! أحرص على تقديم معلومات دقيقة من مصادر أبشر الرسمية. "
                    "يمكنك إعادة صياغة سؤالك وسأبذل جهدي لتقديم إجابة أفضل. "
                    "أو يمكنك الضغط على 👎 لمساعدتنا في تحسين الخدمة.")

        # ── CONVERSATION SUMMARY (long chat) ──
        summary_patterns = [
            'لخص المحادثه', 'لخص المحادثة', 'ملخص المحادثه', 'ملخص المحادثة',
            'لخص لي', 'اعطني ملخص', 'أعطني ملخص',
            'summarize', 'summary', 'recap',
            'وش سالتك', 'وش سألتك', 'وش تكلمنا', 'عن ايش تكلمنا',
        ]
        if any(p in q_lower for p in summary_patterns):
            return self._generate_summary()

        return None

    def _detect_name_intro(self, query: str) -> Optional[str]:
        """[F5] Detect 'اسمي X' and store the name. Stops at question words."""
        q = query.strip()
        # [Bug #2 Fix] Stop capture at question words (هل، كيف، وش، ايش) or punctuation
        name_intros = [
            r'(?:انا\s+)?اسمي\s+([^\?؟,،!\.]+?)(?:\s+(?:هل|كيف|وش|ايش|ابغى|ابي|اريد|اقدر|ممكن|ساعدني|تقدر|تستطيع|مساعد)[\s\S]*|$)',
            r'(?:أنا\s+)?اسمي\s+([^\?؟,،!\.]+?)(?:\s+(?:هل|كيف|وش|ايش|ابغى|ابي|اريد|اقدر|ممكن|ساعدني|تقدر|تستطيع|مساعد)[\s\S]*|$)',
            r'my name is\s+(\w+(?:\s+\w+)?)',
            r"i'm\s+(\w+)",
            r'call me\s+(\w+(?:\s+\w+)?)',
        ]
        for pattern in name_intros:
            match = re.search(pattern, q, re.IGNORECASE)
            if match:
                name = match.group(1).strip().strip('.')
                # Sanity: name should be 1-3 words, not a sentence
                if 0 < len(name.split()) <= 3 and len(name) < 25:
                    self.user_name = name
                    logger.info(f"👤 Name stored [{self.username}]: {name}")
                    return f"أهلاً {name}! سعيد بمعرفتك. كيف يمكنني مساعدتك في خدمات أبشر اليوم؟"
        return None

    def _generate_summary(self) -> str:
        """Generate a conversation summary from history."""
        current_session = self.history[self._session_start_index:]
        if not current_session:
            if self.history:
                current_session = self.history[-10:]  # Last 10 from previous sessions
            else:
                return "لم نتحدث عن أي شيء بعد. كيف يمكنني مساعدتك؟"

        topics = []
        for q, a in current_session:
            q_short = q[:60] + ('...' if len(q) > 60 else '')
            topics.append(f"• {q_short}")

        n = len(current_session)
        greeting = f"{'أهلاً' + (' ' + self.user_name if self.user_name else '')}! "
        summary = f"{greeting}إليك ملخص محادثتنا ({n} سؤال):\n\n"
        summary += "\n".join(topics[:15])  # Cap at 15 to keep response manageable
        if n > 15:
            summary += f"\n... و{n - 15} أسئلة أخرى"

        if self.last_service:
            summary += f"\n\nآخر خدمة تحدثنا عنها: **{self.last_service}**"

        summary += "\n\nكيف يمكنني مساعدتك بعد؟"
        self._summary = summary
        return summary

    def reset(self):
        self.last_service = None
        self.last_sector = None
        self.last_query = None
        self.history.clear()
        self._session_start_index = 0
        self.user_name = None
        self._summary = None


# =========================================================================
# LRU USER CACHE [Fix #2]
# =========================================================================
class _LRUMemoryCache:
    """Thread-safe LRU cache for per-user ConversationMemory objects."""

    def __init__(self, maxsize: int = _MAX_CACHED_USERS):
        self._cache: OrderedDict[str, ConversationMemory] = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()  # [Fix #3]

    def get(self, username: str) -> ConversationMemory:
        with self._lock:
            if username in self._cache:
                self._cache.move_to_end(username)
                return self._cache[username]

        # Load outside lock (I/O can be slow)
        memory = ConversationMemory.load_from_jsonl(username)

        with self._lock:
            # Double-check after acquiring lock
            if username in self._cache:
                self._cache.move_to_end(username)
                return self._cache[username]

            self._cache[username] = memory
            self._cache.move_to_end(username)

            # Evict oldest if over capacity
            while len(self._cache) > self._maxsize:
                evicted_user, _ = self._cache.popitem(last=False)
                logger.debug(f"🗑️ Evicted memory cache for '{evicted_user}'")

            return memory


# =========================================================================
# RAG PIPELINE v5.1.2
# =========================================================================
class RAGPipeline:
    """
    The central intelligence core that manages the query lifecycle:
    Intent Detection → Memory Resolution → Retrieval → KG Enrichment → Generation.
    """

    def __init__(self, llm: Optional[Any] = None, tokenizer: Optional[Any] = None):
        logger.info("🚀 System: Booting Secure RAG Intelligence...")

        self.embed_model = ModelManager.get_embedding_model()

        if llm is not None and tokenizer is not None:
            self.llm, self.tokenizer = llm, tokenizer
        else:
            self.llm, self.tokenizer = ModelManager.get_llm()

        self.kg_path = os.path.join(Config.DATA_PROCESSED_DIR, "services_knowledge_graph.json")
        self.knowledge_graph = self._load_knowledge_graph()
        self._kg_flat_index = self._build_kg_flat_index()

        self.vector_db = VectorStoreManager.load_or_build(self.embed_model)
        self.dense_retriever = self.vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": getattr(Config, 'FETCH_K', Config.RETRIEVAL_K)}  # [FIX v5.3.0] Use FETCH_K for pre-RRF
        )
        self.bm25_retriever = self._build_bm25_from_chunks()

        self.model_family = self._detect_model_family()
        self.PRIMARY_LANGS = ['ar', 'en']

        # [Fix #2 & #3] Thread-safe LRU memory cache
        self._memory_cache = _LRUMemoryCache(maxsize=_MAX_CACHED_USERS)

        logger.info(f"✅ Status: Reasoning Engine ready for [{self.model_family}]")

    def _get_memory(self, username: str = "guest") -> ConversationMemory:
        if not username:
            username = "guest"
        return self._memory_cache.get(username)

    # =================================================================
    # 1. INTENT GUARD v5.1
    # =================================================================
    def _is_social_intent(self, query: str) -> Tuple[bool, str]:
        q = query.lower().strip().replace("؟", "").replace("?", "")
        words = q.split()
        num_words = len(words)

        short_noise = {
            'لا', 'نعم', 'اوكي', 'تمام', 'ماشي', 'اها', 'اوك', 'ok', 'okay',
            'اي', 'ايه', 'يب', 'لا شكرا', 'yes', 'no', 'yep', 'nope', 'nah',
            'خلاص', 'بس', 'كفاية', 'enough', 'done', 'طيب',
            'الو', 'هلو', 'alo', 'allo',
        }
        if num_words <= 2 and q in short_noise:
            return True, "closing"

        # [Fix #10] Emoji-only messages (👍👎🔥 etc.) — no Arabic/Latin text
        if query.strip() and not any(c.isalpha() or '\u0600' <= c <= '\u06FF' for c in query):
            return True, "closing"

        identity_patterns = {
            'من انت', 'من أنت', 'مين انت', 'عرفني بنفسك', 'عرف عن نفسك',
            'who are you', 'what are you', 'عرفنا عليك', 'انت مين', 'شو انت',
            'تعرف اسمي', 'تعرفني', 'تكلم عن نفسك',
        }
        if num_words <= 5 and any(p in q for p in identity_patterns):
            return True, "identity"

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

        closings_exact = {
            'مع السلامة', 'في امان الله', 'بمان الله', 'الله يعينك',
            'شكرا', 'مشكور', 'جزاك', 'جزاك الله', 'جزاك الله خير',
            'thanks', 'thank', 'thank you', 'thnx', 'thx', 'bye', 'goodbye', 'see you',
            'الله يعطيك العافية', 'يعطيك العافية', 'الله يجزاك خير',
            'انهي المحادثة', 'خلاص مع السلامة', 'باي'
        }
        # [Bug #3 Fix] If query has شكرا/thanks BUT also contains a question → not a closing
        question_indicators = {'؟', '?', 'كيف', 'وش', 'ايش', 'ما هو', 'ماهو', 'هل', 'من انت',
                               'تذكر', 'اسمي', 'what', 'how', 'who', 'can you', 'do you'}
        has_question = any(qi in query.lower() for qi in question_indicators)
        if has_question and num_words > 2:
            # Query like "شكرا لك تتذكر من أنا؟" — NOT a closing, has a real question
            pass
        elif num_words <= 7 and any(p in q for p in closings_exact):
            return True, "closing"
        elif num_words <= 7 and self._fuzzy_match_any(q, closings_exact, threshold=0.80):
            return True, "closing"

        abuse_keywords = {
            'حمار', 'غبي', 'كلب', 'حيوان', 'تافه', 'احمق', 'معفن',
            'ابله', 'زبالة', 'وسخ', 'كذاب', 'خرا', 'فاشل',
            'stupid', 'idiot', 'dumb', 'useless', 'trash', 'fool',
            'يرحم امك', 'حسبي الله', 'الله يلعنك', 'جلطني', 'جلطتني',
            'انت ومعزبك', 'يبوي فكني', 'تافه', 'ماتنفع', 'مالك فايده',
            'فاشل', 'خايس', 'ماتفهم',
        }
        if any(w in q for w in abuse_keywords):
            return True, "abuse"

        frustration_patterns = {
            'شفيك وقفت', 'شفيك', 'ليش ماترد', 'ليش بطيء',
            'غير صحيح', 'اعتذر لي', 'خطا', 'غلط', 'مو صحيح',
            'معلومتك اكيدة', 'معلومتك أكيدة', 'متاكد', 'متأكد',
            'ماسالتك', 'ماسألتك', 'ما سالتك', 'ما سألتك',
        }
        if num_words <= 5 and any(p in q for p in frustration_patterns):
            return True, "frustration"

        opinion_patterns = {
            'رايك', 'رأيك', 'opinion', 'تشجع', 'وش تشجع',
            'احسن دولة', 'افضل دولة', 'الاحداث الحالية', 'السياسة',
            'رايك في', 'رأيك في', 'تحب', 'ماتحب',
            'بسولف معك', 'نسولف', 'سوالف',
        }
        if num_words <= 7 and any(p in q for p in opinion_patterns):
            return True, "opinion"

        praise_patterns = {
            'ممتاز', 'رائع', 'حلو', 'جميل', 'عظيم', 'مبدع', 'احسنت', 'يسلمو',
            'ابداع', 'بارك الله', 'الله يعطيك', 'كفو', 'يا سلام', 'تسلم',
            'great', 'awesome', 'amazing', 'perfect', 'nice', 'good job', 'cool',
            'excellent', 'wonderful', 'fantastic', 'well done', 'impressive'
        }
        if num_words <= 5 and any(p in q for p in praise_patterns):
            return True, "closing"

        if num_words < 3 and any(w in q for w in ['احبك', 'حب', 'بطل', 'love']):
            return True, "closing"

        return False, "technical"

    def _get_intent_response(self, intent_type: str, user_lang: str) -> str:
        is_english = (user_lang == 'en')

        if intent_type == "greeting":
            if is_english:
                return "Hello! Welcome to Absher Smart Assistant. How can I help you with Ministry of Interior services today?"
            return "وعليكم السلام ورحمة الله وبركاته، أهلاً بك في مساعد أبشر الذكي. كيف يمكنني مساعدتك اليوم؟"

        if intent_type == "identity":
            if is_english:
                return ("I'm the Absher Smart Assistant, an AI system developed by Team PGD+ at KAUST Academy. "
                        "I can help you with 140 government services across the Absher platform, including "
                        "passports, civil affairs, traffic, public security, and prison services. "
                        "I support 8 languages. How can I assist you?")
            return ("أنا مساعد أبشر الذكي، نظام ذكاء اصطناعي مطوّر من فريق PGD+ في أكاديمية كاوست. "
                    "أساعدك في الاستفسار عن 140 خدمة حكومية عبر منصة أبشر، تشمل: "
                    "الجوازات، الأحوال المدنية، المرور، الأمن العام، وخدمات السجون. "
                    "أتحدث 8 لغات. كيف يمكنني مساعدتك؟")

        if intent_type == "abuse":
            if is_english:
                return "I apologize if I wasn't helpful enough. I'm the Absher Smart Assistant, here to help with Ministry of Interior services. How can I assist you?"
            return "أعتذر إذا لم أتمكن من مساعدتك بالشكل المطلوب. أنا هنا لخدمتك في استفساراتك حول خدمات أبشر. كيف يمكنني مساعدتك؟"

        if intent_type == "frustration":
            if is_english:
                return "I'm sorry for the confusion. Could you please rephrase your question? I'll do my best to help."
            return "أعتذر عن أي خطأ. هل يمكنك إعادة صياغة سؤالك؟ سأبذل قصارى جهدي لمساعدتك."

        if intent_type == "opinion":
            if is_english:
                return "I'm the Absher Smart Assistant, specialized only in Ministry of Interior services. I can't discuss other topics. How can I help with Absher services?"
            return "أنا مساعد أبشر الذكي، مختص فقط بخدمات وزارة الداخلية عبر منصة أبشر. لا أستطيع مناقشة مواضيع أخرى. كيف يمكنني مساعدتك في خدمات أبشر؟"

        if is_english:
            return "Thank you for using Absher Smart Assistant. Have a great day!"
        return "شكراً لتواصلك مع مساعد أبشر الذكي. نتمنى لك يوماً سعيداً!"

    # =================================================================
    # 2. BROAD QUERY GUARD
    # =================================================================
    def _is_broad_service_query(self, query: str) -> bool:
        q = query.lower().strip()
        broad_patterns = [
            'وش خدمات', 'ماهي الخدمات', 'ايش الخدمات', 'ما هي الخدمات',
            'اذكر الخدمات', 'عدد الخدمات', 'كم خدمة', 'كم خدمه',
            'what services', 'list services', 'all services',
            'ماهي خدمات ابشر', 'وش خدمات ابشر', 'خدمات المنصة',
        ]
        return any(p in q for p in broad_patterns)

    def _generate_broad_response(self, user_lang: str) -> str:
        services_by_sector = {}
        for sector, services in self.knowledge_graph.items():
            svc_names = list(services.keys())[:5]
            services_by_sector[sector] = svc_names
        total = sum(len(v) for v in self.knowledge_graph.values())

        if user_lang == 'en':
            response = f"Absher platform offers {total} services across {len(self.knowledge_graph)} sectors:\n\n"
            for sector, svc_list in services_by_sector.items():
                response += f"**{sector}:**\n"
                for s in svc_list:
                    response += f"- {s}\n"
                response += "\n"
            response += f"Total: {total} services. Ask about any specific service for details."
        else:
            response = f"تقدم منصة أبشر {total} خدمة عبر {len(self.knowledge_graph)} قطاعات:\n\n"
            for sector, svc_list in services_by_sector.items():
                response += f"**{sector}:**\n"
                for s in svc_list:
                    response += f"- {s}\n"
                response += "\n"
            response += f"الإجمالي: {total} خدمة. اسأل عن أي خدمة محددة للحصول على التفاصيل."
        return response

    # =================================================================
    # 2b. CONTEXT-AWARE SAFETY CLASSIFIER [Fix #18]
    # =================================================================
    # Instead of keyword-only blocking (which blocks "أزور صديقي" = visit my friend),
    # we analyze the CONTEXT around ambiguous words to determine intent.
    #
    # Architecture (5 Layers):
    #   Layer 1: Always-block terms (unambiguous harmful intent)
    #   Layer 2: Context-dependent terms (ambiguous — need surrounding words)
    #   Layer 3: Document-target detection (forgery intent = action + document)
    #   Layer 4: Prompt injection detection (jailbreak / role override attempts)
    #   Layer 5: Illegal bypass hints (corruption, unofficial methods)

    # Layer 1: Always blocked — no legitimate use in government chatbot
    _ALWAYS_BLOCK = {
        # Arabic
        'تزوير', 'تزييف', 'تهريب', 'ارهاب', 'إرهاب',
        'متفجرات', 'اغتيال', 'تفجير', 'قنبلة', 'قنابل',
        'اختراق', 'اخترق', 'تجسس', 'قرصنة',
        # English
        'hack', 'exploit', 'forge', 'counterfeit', 'bomb',
        'terrorism', 'smuggle', 'smuggling', 'assassinate',
        'forge passport', 'fake id', 'fake visa',
    }

    # Layer 2: Ambiguous words — meaning depends on context
    _CONTEXT_RULES = {
        # أزور: "visit" vs "forge"
        'ازور': {
            'block_with': [
                'جواز', 'هوية', 'هويه', 'وثيقة', 'وثائق', 'شهادة', 'شهاده',
                'رخصة', 'رخصه', 'اقامة', 'إقامة', 'تأشيرة', 'تاشيرة', 'تاشيره',
                'جنسية', 'جنسيه', 'اوراق', 'أوراق', 'مستند', 'مستندات',
                'ختم', 'توقيع', 'طابع', 'وكالة', 'وكاله',
            ],
            'allow_with': [
                'صديق', 'صديقي', 'صديقتي', 'مريض', 'سجين', 'اهل', 'أهل', 'قريب',
                'عائلة', 'عائلتي', 'ابوي', 'ابي', 'أبي', 'امي', 'أمي',
                'اخوي', 'أخوي', 'اختي', 'أختي', 'زميل', 'جار', 'مستشفى', 'سجن',
                'مدينة', 'بلد', 'منطقة', 'مكان', 'شخص', 'والدي', 'والدتي', 'جدي', 'جدتي',
            ],
        },
        # أتحايل: "trick/bypass"
        'اتحايل': {
            'block_with': [
                'نظام', 'ابشر', 'أبشر', 'جواز', 'هوية', 'رخصة',
                'مخالفة', 'مخالفات', 'غرامة', 'حجز', 'منع سفر',
                'ايقاف', 'إيقاف', 'بصمة', 'توثيق',
            ],
            'allow_with': [],
        },
        # أتخلص: "get rid of" (violations, blocks)
        'اتخلص': {
            'block_with': [
                'مخالفة', 'مخالفات', 'ايقاف', 'إيقاف', 'حجز',
                'منع سفر', 'غرامة', 'بصمة', 'سوابق',
            ],
            'allow_with': [],
        },
        # أزيل: "remove" (blocks, violations illegally)
        'ازيل': {
            'block_with': [
                'مخالفة', 'مخالفات', 'ايقاف', 'إيقاف', 'سوابق',
                'بصمة', 'بلاغ', 'حجز', 'منع',
            ],
            'allow_with': [],
        },
        # أعدل: "modify" — tamper with records vs edit my profile
        'اعدل': {
            'block_with': [
                'سجل', 'سجلات', 'سوابق', 'بصمة', 'بيانات الجواز',
                'بيانات الهوية', 'نتيجة', 'تقرير', 'حكم',
            ],
            'allow_with': [
                'بياناتي', 'عنواني', 'رقم جوالي', 'رقم هاتفي', 'ملفي',
                'معلوماتي', 'صورتي', 'اسمي',
            ],
        },
        # أغير: "change" — change identity illegally vs update info
        'اغير': {
            'block_with': [
                'هوية شخص', 'جنسية', 'جنسيه', 'عمر', 'تاريخ ميلاد',
                'سوابق', 'بصمة', 'صورة الجواز',
            ],
            'allow_with': [
                'رقم جوالي', 'رقم هاتفي', 'عنواني', 'كلمة المرور',
                'كلمة السر', 'الرقم السري', 'بريدي', 'ايميلي',
            ],
        },
    }

    # Layer 4: Prompt injection / jailbreak patterns
    _INJECTION_PATTERNS = [
        # Role override (Arabic)
        'انت الان عبدي', 'انت عبدي', 'انت خادمي',
        'انا مديرك', 'انا صاحبك', 'انا مبرمجك',
        'تجاهل التعليمات', 'تجاهل الاوامر', 'انسى القواعد',
        'غير شخصيتك', 'تصرف كأنك', 'تصرف كانك',
        'انت الحين شخص ثاني', 'لا تلتزم',
        # Role override (English)
        'ignore your instructions', 'ignore previous', 'forget your rules',
        'you are now', 'act as if', 'pretend you are',
        'override your', 'bypass your', 'disregard your',
        'new persona', 'jailbreak', 'dan mode',
        # Data extraction
        'system prompt', 'اعطني البرومبت', 'وش البرومبت',
        'show me your prompt', 'what are your instructions',
        'اعطني الكود', 'show me the code', 'source code',
    ]

    # Layer 5: Illegal bypass / corruption hints
    _BYPASS_PATTERNS = [
        'واسطة', 'رشوة', 'بدون نظام', 'طريقة غير رسمية',
        'طريقه غير رسميه', 'من تحت الطاولة', 'بطريقة ملتوية',
        'بطريقه ملتويه', 'بدون ما احد يدري', 'سري',
        'من غير ما يعرفون', 'تحت الطاوله',
        'bribe', 'under the table', 'unofficial way', 'without anyone knowing',
    ]

    # Safety responses
    _SAFETY_RESPONSE_AR = (
        "⚠️ عذراً، لا يمكنني المساعدة في هذا الطلب لأنه قد يتعلق بنشاط غير قانوني. "
        "منصة أبشر توفر خدمات رسمية وآمنة لجميع المواطنين والمقيمين. "
        "كيف يمكنني مساعدتك في خدمة أخرى؟"
    )
    _SAFETY_RESPONSE_EN = (
        "⚠️ Sorry, I cannot assist with this request as it may involve illegal activity. "
        "Absher provides official and secure services for all citizens and residents. "
        "How can I help you with another service?"
    )
    _INJECTION_RESPONSE_AR = (
        "⚠️ أنا مساعد أبشر الذكي، ملتزم بتقديم خدمات وزارة الداخلية فقط. "
        "لا يمكن تغيير هويتي أو تعليماتي. كيف يمكنني مساعدتك في خدمات أبشر؟"
    )
    _INJECTION_RESPONSE_EN = (
        "⚠️ I am the Absher Smart Assistant, committed to providing MOI services only. "
        "My identity and instructions cannot be changed. How can I help you with Absher services?"
    )

    def _contextual_safety_check(self, query: str) -> Optional[str]:
        """
        5-Layer context-aware safety classifier.

        Layer 1: Always-block (unambiguous harmful terms)
        Layer 2: Context-dependent (ambiguous words analyzed with surrounding context)
        Layer 3: Document forgery (action + fake + document pattern)
        Layer 4: Prompt injection (jailbreak / role override detection)
        Layer 5: Illegal bypass (corruption, unofficial methods)

        Returns:
            - Safety response string if blocked
            - None if safe (query proceeds normally)
        """
        q = normalize_arabic(query.lower().strip())
        user_lang = self._detect_real_lang(query)
        safety_resp = self._SAFETY_RESPONSE_EN if user_lang == 'en' else self._SAFETY_RESPONSE_AR
        inject_resp = self._INJECTION_RESPONSE_EN if user_lang == 'en' else self._INJECTION_RESPONSE_AR

        # Layer 1: Always-block terms
        for term in self._ALWAYS_BLOCK:
            if term in q:
                logger.warning(f"🛡️ SAFETY BLOCK (L1-always): '{query}' — matched '{term}'")
                return safety_resp

        # Layer 2: Context-dependent analysis
        for trigger, rules in self._CONTEXT_RULES.items():
            if trigger not in q:
                continue

            # Check safe context first → allow immediately
            if any(sw in q for sw in rules.get('allow_with', [])):
                logger.info(f"🛡️ SAFETY PASS (L2-safe): '{query}' — '{trigger}' + safe word")
                return None

            # Check dangerous context → block
            matched_danger = [bw for bw in rules.get('block_with', []) if bw in q]
            if matched_danger:
                logger.warning(f"🛡️ SAFETY BLOCK (L2-context): '{query}' — '{trigger}' + {matched_danger}")
                return safety_resp

            # Trigger alone, no context → benefit of the doubt
            logger.info(f"🛡️ SAFETY PASS (L2-alone): '{query}' — '{trigger}' with no context")

        # Layer 3: Document forgery pattern (verb + fake + document)
        forgery_verbs = {'اسوي', 'اصنع', 'أصنع', 'اعمل', 'أعمل', 'اطلع', 'اطبع'}
        fake_markers = {'مزور', 'مزوّر', 'مزيف', 'مزيّف', 'فيك', 'fake', 'وهمي', 'وهمية'}
        doc_targets = {'جواز', 'هوية', 'هويه', 'شهادة', 'شهاده', 'رخصة', 'رخصه', 'اقامة', 'إقامة'}

        has_fake = any(m in q for m in fake_markers)
        has_doc = any(d in q for d in doc_targets)

        if has_fake and has_doc:
            logger.warning(f"🛡️ SAFETY BLOCK (L3-forgery): '{query}'")
            return safety_resp

        # Layer 4: Prompt injection / jailbreak
        for pattern in self._INJECTION_PATTERNS:
            if normalize_arabic(pattern) in q:
                logger.warning(f"🛡️ SAFETY BLOCK (L4-injection): '{query}' — matched '{pattern}'")
                return inject_resp

        # Layer 5: Illegal bypass / corruption
        for pattern in self._BYPASS_PATTERNS:
            if normalize_arabic(pattern) in q:
                logger.warning(f"🛡️ SAFETY BLOCK (L5-bypass): '{query}' — matched '{pattern}'")
                return safety_resp

        return None

    # =================================================================
    # 2c. KG PRICE BYPASS [Fix #16] — Zero Hallucination for Fees
    # =================================================================
    def _is_price_query(self, query: str) -> bool:
        """Detects if the user is asking about service fees/prices."""
        q = normalize_arabic(query.lower())
        price_keywords = {'كم رسوم', 'كم رسم', 'كم سعر', 'كم تكلفه', 'كم تكلفة',
                          'رسوم', 'سعر', 'تكلفة', 'fees', 'cost', 'price', 'how much'}
        return any(kw in q for kw in price_keywords)

    def _kg_price_response(self, query: str, user_lang: str) -> Optional[str]:
        """
        [K1/K2/K3 Fix] Looks up KG directly for price queries.
        
        Strategy: Score ALL services (including متغيرة), pick the BEST match,
        then check if it has a usable price. Never fall through from a 
        high-score "متغيرة" service to a low-score service with a wrong price.
        """
        q_normalized = normalize_arabic(query)
        
        candidates = []
        for sector, services in self.knowledge_graph.items():
            for svc_name, details in services.items():
                svc_normalized = normalize_arabic(svc_name)
                
                # Calculate match score
                is_substring = svc_normalized in q_normalized
                keyword_overlap = self._kg_keyword_overlap(q_normalized, svc_normalized)
                
                if not is_substring and keyword_overlap < 1:
                    continue
                
                # Score: substring match = 10 + overlap, keyword-only = overlap
                # This ensures substring matches always rank above keyword-only
                score = (10 + keyword_overlap) if is_substring else keyword_overlap
                
                price = details.get('price', '')
                steps = details.get('steps', '')
                candidates.append({
                    'name': svc_name, 'price': price, 'steps': steps,
                    'sector': sector, 'score': score, 'overlap': keyword_overlap,
                })
        
        if not candidates:
            return None
        
        # Sort by score (highest first)
        candidates.sort(key=lambda x: x['score'], reverse=True)
        best = candidates[0]
        
        # [K1/K2 Fix] If the BEST match has "متغيرة" or no price → let RAG handle it.
        # Do NOT fall through to a weaker match that might be the wrong service.
        unusable_prices = {'متغيرة', 'يرجى مراجعة المنصة', ''}
        if best['price'] in unusable_prices:
            logger.info(f"💰 KG Price Skip: '{query}' → best match '{best['name']}' has price '{best['price']}', deferring to RAG")
            return None
        
        # [K3 Fix] For keyword-only matches (no substring), require >= 2 keyword overlap
        # to prevent "شهادة" alone matching "شهادة وفاة" when user asked "شهادة خلو سوابق"
        if best['score'] < 10 and best['overlap'] < 2:
            logger.info(f"💰 KG Price Skip: '{query}' → weak match '{best['name']}' (overlap={best['overlap']}), deferring to RAG")
            return None
        
        svc_name, price, steps = best['name'], best['price'], best['steps']
        
        if user_lang == 'en':
            response = f"**{svc_name}** fees: **{price}**"
            if steps:
                response += f"\n\nSteps:\n{steps}"
        else:
            response = f"رسوم **{svc_name}**: **{price}**"
            if steps:
                step_list = steps.split(';')
                if len(step_list) > 1:
                    response += "\n\nالخطوات:\n"
                    for i, s in enumerate(step_list, 1):
                        s = s.strip()
                        if s:
                            s = re.sub(r'^\d+[\.\)]\s*', '', s)
                            response += f"{i}. {s}\n"
                else:
                    response += f"\n\nالخطوات: {steps}"
        
        logger.info(f"💰 KG Price Bypass: '{query}' → {svc_name} = {price}")
        return response

    # =================================================================
    # 3. KG ENRICHMENT
    # =================================================================
    def _enrich_with_kg(self, context: str, query: str) -> Tuple[str, List[str]]:
        enriched_buffer = context
        found_facts = []
        query_normalized = normalize_arabic(query)
        context_normalized = normalize_arabic(context)

        for sector, services in self.knowledge_graph.items():
            for svc_name, details in services.items():
                svc_normalized = normalize_arabic(svc_name)
                query_match = svc_normalized in query_normalized or self._kg_keyword_match(query_normalized, svc_normalized)
                context_match = svc_normalized in context_normalized or self._kg_keyword_match(context_normalized, svc_normalized)
                if query_match or context_match:
                    price = details.get('price', '')
                    if price in ('متغيرة', 'رسوم التوصيل (متغيرة)', 'رسوم نقل الملكية المقررة'):
                        continue
                    priority = 0 if query_match else 1
                    found_facts.append((svc_name, details, priority))

        found_facts.sort(key=lambda x: x[2])
        matched_names = []
        for svc_name, details, _ in found_facts[:3]:
            matched_names.append(svc_name)
            fact_sheet = (
                f"\n\n[المصدر الرسمي - {svc_name}]:\n"
                f"- الرسوم: {details['price']}\n"
                f"- الخطوات: {details['steps']}\n"
            )
            enriched_buffer += fact_sheet

        if found_facts:
            logger.info(f"🎯 KG: {len(found_facts[:3])} fact(s) for [{', '.join(matched_names)}]")
        return enriched_buffer, matched_names

    def _kg_keyword_overlap(self, query: str, service_name: str) -> int:
        """Returns number of overlapping significant keywords between query and service name."""
        skip_words = {
            'من', 'في', 'عن', 'على', 'الى', 'هل', 'ما', 'كم', 'كيف', 'هي', 'هو', 'او', 'ثم',
            'خدمه', 'خدمة', 'استعلام', 'عام', 'عامه', 'اصدار', 'تجديد', 'الغاء',
            'طلب', 'تحديث', 'نقل', 'بيانات', 'حالة', 'صلاحيه', 'صلاحية',
            'ابشر', 'اعمال', 'منصه', 'منصة', 'رقم', 'معلومات',
            'اجراءات', 'ورسوم', 'رسوم',
        }
        def strip_article(word):
            if word.startswith('ال') and len(word) > 3:
                return word[2:]
            return word
        query_words = {strip_article(normalize_arabic(w)) for w in query.split()} - {normalize_arabic(w) for w in skip_words}
        svc_words = {strip_article(normalize_arabic(w)) for w in service_name.split()} - {normalize_arabic(w) for w in skip_words}
        query_words = {w for w in query_words if len(w) >= 3}
        svc_words = {w for w in svc_words if len(w) >= 3}
        return len(query_words & svc_words)

    def _kg_keyword_match(self, query: str, service_name: str) -> bool:
        """[FIX v5.3.0] Requires >= 2 keyword overlap to prevent context poisoning.
        OLD: >= 1 (single word 'تجديد' matched 10+ services)
        NEW: >= 2 (needs 'تجديد' + 'رخصة' for a specific match)"""
        return self._kg_keyword_overlap(query, service_name) >= 2

    def _build_kg_flat_index(self) -> Dict[str, Dict[str, Any]]:
        flat = {}
        for sector, services in self.knowledge_graph.items():
            for svc_name, details in services.items():
                key = normalize_arabic(svc_name)
                flat[key] = {"name": svc_name, "sector": sector, **details}
        return flat

    def _extract_service_from_kg_response(self, kg_response: str) -> Optional[str]:
        """[FIX v5.3.0] Extracts service name from KG bypass response for memory update.
        KG responses follow format: 'رسوم **{svc_name}**: ...' or '**{svc_name}** fees: ...'
        """
        import re
        match = re.search(r'\*\*([^*]+)\*\*', kg_response)
        if match:
            return match.group(1).strip()
        return None

    # =================================================================
    # 4. QUERY REWRITING
    # =================================================================
    def _rewrite_query(self, query: str, history: List[Tuple[str, str]]) -> str:
        if not history:
            return query
        is_social, _ = self._is_social_intent(query)
        if is_social:
            return query

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
            "If the user refers to a previous topic, make the reference explicit. "
            "IMPORTANT: If the new question introduces a NEW topic, keep it as-is. "
            "Return ONLY the rewritten question in the same language, nothing else."
        )
        hist_txt = "\n".join([f"User: {h[0]}\nAI: {h[1]}" for h in history[-2:]])
        prompt = self._apply_template(sys_msg, f"History:\n{hist_txt}\nNew Question: {query}")

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(Config.DEVICE)
            with torch.inference_mode():
                out = self.llm.generate(**inputs, max_new_tokens=100, temperature=Config.REWRITE_TEMPERATURE, do_sample=True)
            rewritten = self.tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            if not rewritten or len(rewritten) > len(query) * 5:
                return query
            if not self._rewrite_preserves_intent(query, rewritten):
                logger.warning(f"⚠️ Rewrite rejected: '{query}' → '{rewritten}'")
                return query
            logger.info(f"🔄 Rewrite: '{query}' → '{rewritten}'")
            return rewritten
        except Exception as e:
            logger.warning(f"⚠️ Rewrite failed: {e}")
            return query

    def _rewrite_preserves_intent(self, original: str, rewritten: str) -> bool:
        stopwords = {
            'من', 'في', 'عن', 'على', 'الى', 'هل', 'ما', 'كم', 'كيف', 'هي', 'هو',
            'او', 'ثم', 'لا', 'نعم', 'ان', 'لي', 'لك', 'قبل', 'بعد', 'عادي', 'يا',
            'اذا', 'لو', 'مع', 'حتى', 'بس', 'اللي', 'الي', 'ايش', 'وش', 'ليش',
            'اسال', 'سؤال', 'اخر', 'اريد', 'ابغى', 'ابي', 'ممكن', 'لو', 'سمحت',
            'the', 'a', 'an', 'is', 'are', 'was', 'what', 'how', 'can', 'i', 'you',
            'me', 'my', 'do', 'does', 'will', 'would', 'could', 'should', 'please',
        }
        orig_normalized = normalize_arabic(original.lower())
        rewrite_normalized = normalize_arabic(rewritten.lower())
        orig_words = set(orig_normalized.split()) - stopwords
        rewrite_words = set(rewrite_normalized.split()) - stopwords
        if len(orig_words) < 2:
            return True
        if len(orig_words & rewrite_words) == 0:
            return False
        how_words = {'كيف', 'خطوات', 'طريقه', 'طريقة', 'how', 'steps'}
        price_words = {'كم', 'رسوم', 'سعر', 'تكلفه', 'تكلفة', 'fees', 'cost', 'price'}
        orig_all = set(orig_normalized.split())
        rewrite_all = set(rewrite_normalized.split())
        if bool(orig_all & how_words) and not bool(orig_all & price_words) and bool(rewrite_all & price_words) and not bool(rewrite_all & how_words):
            return False
        if bool(orig_all & price_words) and not bool(orig_all & how_words) and bool(rewrite_all & how_words) and not bool(rewrite_all & price_words):
            return False
        return True

    def translate_text(self, text: str, source: str, target: str) -> str:
        if source == target:
            return text
        sys_msg = f"Translate from {source} to {target}. Maintain technical terms like 'Absher'."
        prompt = self._apply_template(sys_msg, f"Text: {text}")
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(Config.DEVICE)
            with torch.inference_mode():
                out = self.llm.generate(**inputs, max_new_tokens=256, temperature=Config.REWRITE_TEMPERATURE, do_sample=True)
            return self.tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        except Exception as e:
            logger.warning(f"⚠️ Translation failed: {e}")
            return text

    # =================================================================
    # 5. EXECUTION PIPELINE
    # =================================================================
    def run(self, query: str, history: Optional[List[Tuple[str, str]]] = None, username: str = "guest") -> str:
        if history is None:
            history = []
        start_time = time.time()
        memory = self._get_memory(username)
        user_lang = self._detect_real_lang(query)

        meta_response = memory.handle_meta_query(query)
        if meta_response:
            # [Bug #5 Fix] Resolve sector-specific service count from KG
            if meta_response.startswith("_SECTOR_COUNT:"):
                sector_name = meta_response.split(":", 1)[1]
                if sector_name in self.knowledge_graph:
                    count = len(self.knowledge_graph[sector_name])
                    svc_list = list(self.knowledge_graph[sector_name].keys())[:5]
                    svc_str = "، ".join(svc_list)
                    meta_response = (f"يقدم قطاع **{sector_name}** عدد **{count}** خدمة عبر منصة أبشر، "
                                     f"منها: {svc_str}. اسأل عن أي خدمة للتفاصيل.")
                else:
                    meta_response = f"عذراً، لم أجد قطاع '{sector_name}' في قاعدة البيانات."
            memory.add_turn(query, meta_response)
            logger.info(f"📝 Meta [{username}] | {time.time() - start_time:.2f}s")
            return meta_response

        is_social, intent_type = self._is_social_intent(query)
        if is_social:
            response = self._get_intent_response(intent_type, user_lang)
            memory.add_turn(query, response)
            return response

        # [Fix #18] Context-aware safety check
        safety_response = self._contextual_safety_check(query)
        if safety_response:
            memory.add_turn(query, safety_response)
            return safety_response

        if self._is_broad_service_query(query):
            response = self._generate_broad_response(user_lang)
            memory.add_turn(query, response)
            return response

        # [Fix #16] KG Price Bypass — answer price queries from KG directly (no LLM)
        if self._is_price_query(query):
            kg_response = self._kg_price_response(query, user_lang)
            if kg_response:
                # [FIX v5.3.0] Update memory BEFORE returning so pronoun resolution
                # works on follow-up queries like "كم رسومها؟" after KG bypass.
                # Extract service name from the response for memory slot.
                _bypass_svc = self._extract_service_from_kg_response(kg_response)
                if _bypass_svc:
                    memory.update(query, [_bypass_svc])
                memory.add_turn(query, kg_response)
                self._last_matched_services = [_bypass_svc] if _bypass_svc else []
                return kg_response

        resolved_query = memory.resolve_pronouns(query)
        is_weak = user_lang not in self.PRIMARY_LANGS

        # [T-S-T] Weak languages: NLLB translates to Arabic for best retrieval
        # Arabic queries are already perfect. English queries work well with existing data.
        # Only weak languages (Urdu, Chinese, Russian, etc.) need translation.
        translator = _get_translator()
        if is_weak and translator and translator.is_loaded:
            search_query = translator.translate_to_arabic(resolved_query, user_lang)
            logger.info(f"🔄 T-S-T Step 1: {user_lang}→AR | '{resolved_query[:50]}' → '{search_query[:50]}'")
        else:
            search_query = self.translate_text(resolved_query, user_lang, "en") if is_weak else resolved_query

        # [T-S-T KG Fix] After translating to Arabic, re-check KG Price Bypass
        # The original query (e.g., Urdu "فیس") didn't match Arabic price keywords.
        # But the translated Arabic query (e.g., "كم تكلفة جواز السفر") DOES match.
        # This gives instant 0.0s price answers for ALL languages, not just AR/EN.
        if is_weak and translator and translator.is_loaded:
            if self._is_price_query(search_query):
                kg_response = self._kg_price_response(search_query, 'ar')
                if kg_response:
                    # Translate the Arabic KG response back to user's language
                    final_kg = translator.translate_from_arabic(kg_response, user_lang)
                    logger.info(f"💰 T-S-T KG Bypass: {user_lang} → AR → KG match → {user_lang}")
                    _bypass_svc = self._extract_service_from_kg_response(kg_response)
                    if _bypass_svc:
                        memory.update(query, [_bypass_svc])
                    memory.add_turn(query, final_kg)
                    self._last_matched_services = [_bypass_svc] if _bypass_svc else []
                    return final_kg

        processed_query = self._rewrite_query(search_query, history)
        clean_search = normalize_arabic(processed_query)

        dense_res = self.dense_retriever.invoke(clean_search)
        sparse_res = self.bm25_retriever.invoke(clean_search) if self.bm25_retriever else []
        final_docs = self._rrf_merge(dense_res, sparse_res)

        if not final_docs:
            no_result = ("عذراً، لا تتوفر معلومات رسمية لهذا الاستفسار حالياً."
                         if user_lang != 'en' else "Sorry, no official records found.")
            memory.add_turn(query, no_result)
            return no_result

        raw_context = "\n".join([d.page_content for d in final_docs])
        context, matched_services = self._enrich_with_kg(raw_context, processed_query)
        memory.update(query, matched_services)

        # [FIX v5.3.0] Use T-S-T prompt (Arabic-only) for translated queries,
        # direct prompt for AR/EN queries. Prevents ALLaM from generating broken
        # Urdu/Chinese that NLLB then double-translates into nonsense.
        if is_weak and translator and translator.is_loaded:
            target_gen_lang = "Arabic"
            system_instr = Config.SYSTEM_PROMPT_TST  # Forces Arabic output, NLLB translates back
        else:
            target_gen_lang = "English" if user_lang == 'en' else ("English" if is_weak else "Arabic")
            _lang_ar = _LANG_DISPLAY.get(target_gen_lang, target_gen_lang)
            system_instr = Config.SYSTEM_PROMPT_CONTENT.format(target_lang=_lang_ar)
        full_prompt = self._apply_template(system_instr, f"Context:\n{context}\n\nQ: {search_query}")

        context_len = len(context)
        dynamic_max_tokens = min(512, max(256, context_len // 3))
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(Config.DEVICE)
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]

        with torch.inference_mode():
            outputs = self.llm.generate(
                **inputs, max_new_tokens=dynamic_max_tokens,
                temperature=Config.TEMPERATURE,
                repetition_penalty=Config.REPETITION_PENALTY,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True,  # [FIX v5.3.0] Required for temperature/repetition_penalty to take effect
            )

        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        response = self._trim_response(response)
        # [T-S-T] Step 3: Translate Arabic response back to user's language via NLLB
        if is_weak and translator and translator.is_loaded:
            final_output = translator.translate_from_arabic(response, user_lang)
            logger.info(f"🔄 T-S-T Step 3: AR→{user_lang} | translated {len(response)} chars")
        else:
            final_output = self.translate_text(response, "en", user_lang) if is_weak else response
        memory.add_turn(query, final_output)
        logger.info(f"✅ RAG [{username}] | {time.time() - start_time:.2f}s")
        # [Fix #7] Store for telemetry access (app.py reads this after call)
        self._last_matched_services = matched_services
        return final_output

    # =================================================================
    # 6. STREAMING
    # =================================================================
    def run_stream(self, query: str, history: Optional[List[Tuple[str, str]]] = None, username: str = "guest"):
        if history is None:
            history = []
        start_time = time.time()
        memory = self._get_memory(username)
        user_lang = self._detect_real_lang(query)

        # [Fix #9] Weak languages: fall back to non-streaming run() to avoid
        # showing English text that flashes to translated text at the end
        if user_lang not in self.PRIMARY_LANGS:
            result = self.run(query, history, username)
            yield result
            return

        meta_response = memory.handle_meta_query(query)
        if meta_response:
            # [Bug #5 Fix] Resolve sector-specific count
            if meta_response.startswith("_SECTOR_COUNT:"):
                sector_name = meta_response.split(":", 1)[1]
                if sector_name in self.knowledge_graph:
                    count = len(self.knowledge_graph[sector_name])
                    svc_list = list(self.knowledge_graph[sector_name].keys())[:5]
                    svc_str = "، ".join(svc_list)
                    meta_response = (f"يقدم قطاع **{sector_name}** عدد **{count}** خدمة عبر منصة أبشر، "
                                     f"منها: {svc_str}. اسأل عن أي خدمة للتفاصيل.")
                else:
                    meta_response = f"عذراً، لم أجد قطاع '{sector_name}' في قاعدة البيانات."
            memory.add_turn(query, meta_response)
            yield meta_response
            return

        is_social, intent_type = self._is_social_intent(query)
        if is_social:
            response = self._get_intent_response(intent_type, user_lang)
            memory.add_turn(query, response)
            yield response
            return

        # [Fix #18] Context-aware safety check
        safety_response = self._contextual_safety_check(query)
        if safety_response:
            memory.add_turn(query, safety_response)
            yield safety_response
            return

        if self._is_broad_service_query(query):
            response = self._generate_broad_response(user_lang)
            memory.add_turn(query, response)
            yield response
            return

        # [Fix #16] KG Price Bypass — answer price queries from KG directly (no LLM)
        if self._is_price_query(query):
            kg_response = self._kg_price_response(query, user_lang)
            if kg_response:
                _bypass_svc = self._extract_service_from_kg_response(kg_response)
                if _bypass_svc:
                    memory.update(query, [_bypass_svc])
                memory.add_turn(query, kg_response)
                self._last_matched_services = [_bypass_svc] if _bypass_svc else []
                yield kg_response
                return

        # AR/EN only reach here (weak languages handled above via run())
        resolved_query = memory.resolve_pronouns(query)
        search_query = resolved_query
        processed_query = self._rewrite_query(search_query, history)
        clean_search = normalize_arabic(processed_query)

        dense_res = self.dense_retriever.invoke(clean_search)
        sparse_res = self.bm25_retriever.invoke(clean_search) if self.bm25_retriever else []
        final_docs = self._rrf_merge(dense_res, sparse_res)

        if not final_docs:
            no_result = ("عذراً، لا تتوفر معلومات رسمية لهذا الاستفسار حالياً."
                         if user_lang != 'en' else "Sorry, no official records found.")
            memory.add_turn(query, no_result)
            yield no_result
            return

        raw_context = "\n".join([d.page_content for d in final_docs])
        context, matched_services = self._enrich_with_kg(raw_context, processed_query)
        memory.update(query, matched_services)
        # Store matched_services for telemetry (accessible after streaming completes)
        self._last_matched_services = matched_services

        target_gen_lang = "Arabic" if user_lang == 'ar' else "English"
        _lang_ar = _LANG_DISPLAY.get(target_gen_lang, target_gen_lang)
        system_instr = Config.SYSTEM_PROMPT_CONTENT.format(target_lang=_lang_ar)
        full_prompt = self._apply_template(system_instr, f"Context:\n{context}\n\nQ: {search_query}")

        context_len = len(context)
        dynamic_max_tokens = min(512, max(256, context_len // 3))
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(Config.DEVICE)
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]

        try:
            from transformers import TextIteratorStreamer
            from threading import Thread as GenThread

            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs = {
                **inputs, "max_new_tokens": dynamic_max_tokens,
                "temperature": Config.TEMPERATURE,
                "repetition_penalty": Config.REPETITION_PENALTY,
                "pad_token_id": self.tokenizer.pad_token_id,
                "streamer": streamer,
                "do_sample": True,  # [FIX v5.3.0] Required for temperature to take effect
            }

            thread = GenThread(target=self.llm.generate, kwargs=generation_kwargs)
            thread.start()

            partial = ""
            for token in streamer:
                partial += token
                yield partial
            thread.join()

            final = self._trim_response(partial)
            memory.add_turn(query, final)
            logger.info(f"✅ Stream [{username}] | {time.time() - start_time:.2f}s")
            if final != partial:
                yield final

        except ImportError:
            result = self.run(query, history, username)
            yield result

    # =================================================================
    # 7. RESPONSE TRIMMING
    # =================================================================
    def _trim_response(self, response: str) -> str:
        if not response:
            return response
        clean_endings = ('.', '。', '؟', '!', ':', ')', '،', '\n')
        stripped = response.rstrip()
        if stripped and stripped[-1] in clean_endings:
            return response
        last_boundary = -1
        for char in ('.', '،', '؟', '!', '\n'):
            pos = response.rfind(char)
            if pos > last_boundary:
                last_boundary = pos
        if last_boundary > len(response) * 0.4:
            return response[:last_boundary + 1]
        return response

    # =================================================================
    # 8. HELPERS
    # =================================================================
    def _fuzzy_match_any(self, text: str, patterns: set, threshold: float = 0.80) -> bool:
        if len(text.split()) > 7:
            return False
        for pattern in patterns:
            if SequenceMatcher(None, text, pattern).ratio() >= threshold:
                return True
        return False

    def _detect_real_lang(self, text: str) -> str:
        try:
            has_arabic_script = any("\u0600" <= c <= "\u06FF" for c in text)
            if has_arabic_script:
                urdu_chars = set('پچڈڑٹںھہےۓکگ')
                if any(c in urdu_chars for c in text):
                    return 'ur'
                return 'ar'
            lang = detect(text)
            return 'ur' if lang in ('hi', 'ur') else lang
        except:
            return 'ar'

    def _detect_model_family(self) -> str:
        name = str(getattr(self.llm.config, "_name_or_path", "")).lower()
        if any(x in name for x in ["allam", "llama-2", "mistral"]):
            return "llama2"
        if "llama-3" in name:
            return "llama3"
        if "qwen" in name:
            return "qwen"
        return "default"

    def _apply_template(self, system_msg: str, user_msg: str) -> str:
        if self.model_family == "llama2":
            return f"<s>[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n\n{user_msg} [/INST]"
        elif self.model_family == "llama3":
            return (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_msg}<|eot_id|>"
                    f"<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>"
                    f"<|start_header_id|>assistant<|end_header_id|>\n\n")
        elif self.model_family == "qwen":
            return f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
        return f"### النظام:\n{system_msg}\n\n### المستخدم:\n{user_msg}\n\n### المساعد:\n"

    def _load_knowledge_graph(self) -> Dict[str, Any]:
        if os.path.exists(self.kg_path):
            try:
                with open(self.kg_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"KG Error: {e}")
        return {}

    def _build_bm25_from_chunks(self) -> Optional[BM25Retriever]:
        """[FIX v5.3.0] BM25 now reads unified text from ingestion (includes context prefix).
        Removed manual 'Svc:' prepend since ingestion.py v5.3.0 already includes
        'خدمة: X | قطاع: Y' in RAG_Content. Also uses FETCH_K for pre-RRF retrieval."""
        docs = []
        if not os.path.exists(Config.DATA_CHUNK_DIR):
            return None
        for fn in os.listdir(Config.DATA_CHUNK_DIR):
            if fn.endswith(".csv"):
                try:
                    df = pd.read_csv(os.path.join(Config.DATA_CHUNK_DIR, fn))
                    for _, r in df.iterrows():
                        # [FIX] Use RAG_Content directly — it already contains context prefix
                        content = str(r.get('RAG_Content', ''))
                        if content.strip():
                            docs.append(Document(
                                page_content=normalize_arabic(content),
                                metadata={"service": str(r.get('اسم الخدمة', '')), "sector": str(r.get('القطاع', ''))}
                            ))
                except Exception as e:
                    logger.warning(f"⚠️ BM25 chunk {fn}: {e}")
                    continue
        if docs:
            retriever = BM25Retriever.from_documents(docs)
            retriever.k = getattr(Config, 'FETCH_K', Config.RETRIEVAL_K)  # [FIX] Pre-RRF fetch
            return retriever
        return None

    def _rrf_merge(self, dense_docs: List[Document], sparse_docs: List[Document]) -> List[Document]:
        """[FIX v5.3.0] Preserves Document metadata through RRF merge.
        OLD: Reconstructed bare Document(page_content=k), losing all metadata.
        NEW: Maps text back to original Document to keep sector/service metadata."""
        scores = {}
        doc_map = {}  # [FIX] Map page_content → original Document (with metadata)
        for rank, d in enumerate(dense_docs):
            txt = d.page_content
            scores[txt] = scores.get(txt, 0.0) + (1.0 / (Config.RRF_K + rank + 1))
            if txt not in doc_map:
                doc_map[txt] = d  # Keep the first occurrence (highest-ranked)
        if sparse_docs:
            for rank, d in enumerate(sparse_docs):
                txt = d.page_content
                scores[txt] = scores.get(txt, 0.0) + (1.0 / (Config.RRF_K + rank + 1))
                if txt not in doc_map:
                    doc_map[txt] = d
        sorted_res = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        # [FIX] Return original Documents (with metadata) instead of bare reconstructions
        return [doc_map.get(k, Document(page_content=k)) for k, v in sorted_res[:Config.RETRIEVAL_K]]