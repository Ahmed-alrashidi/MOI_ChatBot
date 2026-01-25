# =========================================================================
# File Name: core/rag_pipeline.py
# Purpose: Orchestrates the Hybrid RAG (Retrieval-Augmented Generation) Logic.
# Location: Part of the Core logic responsible for processing user input, 
#           retrieving data, and generating final responses.
# Features:
# - Hybrid Search: Combines Dense (FAISS) and Sparse (BM25) retrieval.
# - RRF Fusion: Merges search results for higher precision.
# - Contextual Memory: Rewrites queries to handle follow-up questions.
# - Knowledge Graph (KG): Injects deterministic/verified data (Fees, Steps).
# - T-S-T Strategy: Translate-Search-Translate for non-primary languages.
# =========================================================================

import os
import torch
import json
import pandas as pd
from typing import List, Tuple
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langdetect import detect

from config import Config
from utils.logger import setup_logger
from utils.text_utils import normalize_arabic
from core.model_loader import ModelManager
from core.vector_store import VectorStoreManager

# Initialize the logger for the pipeline
logger = setup_logger(__name__)

class RAGPipeline:
    """
    The central orchestration engine for the Absher Smart Assistant.
    It manages the flow from user query to language detection, information 
    retrieval, context enrichment, and final response generation.
    """
    
    def __init__(self):
        """
        Initializes the pipeline by loading required models, the Knowledge Graph, 
        and initializing both semantic (FAISS) and keyword (BM25) retrievers.
        """
        logger.info("🚀 Initializing Smart RAG Pipeline...")
        
        # 1. Load Models through the ModelManager (Singleton)
        self.embed_model = ModelManager.get_embedding_model()
        self.llm, self.tokenizer = ModelManager.get_llm()
        
        # 2. Load Knowledge Graph (Deterministic JSON Data)
        # Used for high-stakes information like service fees and official steps.
        self.kg_path = os.path.join(Config.DATA_DIR, "data_processed", "services_knowledge_graph.json")
        self.knowledge_graph = self._load_knowledge_graph()

        # 3. Initialize Retrieval Components
        # Load the Vector Database (Dense Retriever)
        self.vector_db = VectorStoreManager.load_or_build(self.embed_model)
        self.dense_retriever = self.vector_db.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": Config.RETRIEVAL_K}
        )
        # Build the Keyword-based Retriever (Sparse Retriever)
        self.bm25_retriever = self._build_bm25_from_chunks()
        
        # 4. Define Primary Languages for direct processing
        self.PRIMARY_LANGUAGES = ['ar', 'en']
        
        logger.info("✅ Pipeline Ready.")

    def _load_knowledge_graph(self) -> dict:
        """
        Loads the structured JSON service map from the disk.
        
        Returns:
            dict: The loaded knowledge graph or an empty dict on failure.
        """
        if os.path.exists(self.kg_path):
            try:
                with open(self.kg_path, 'r', encoding='utf-8') as f:
                    logger.info("🧠 Knowledge Graph Loaded.")
                    return json.load(f)
            except Exception as e:
                logger.error(f"❌ Failed to load Knowledge Graph: {e}")
        return {}

    def _build_bm25_from_chunks(self) -> BM25Retriever:
        """
        Constructs a BM25 retriever using the processed CSV chunks.
        BM25 is critical for exact keyword matching (e.g., service names, specific fees).
        
        Returns:
            BM25Retriever: Initialized keyword retriever or None if no data exists.
        """
        chunk_dir = Config.DATA_CHUNK_DIR
        documents = []
        
        if not os.path.exists(chunk_dir):
            return None

        # Iterate through processed chunks to build the search corpus
        for filename in os.listdir(chunk_dir):
            if filename.endswith(".csv"):
                try:
                    df = pd.read_csv(os.path.join(chunk_dir, filename))
                    for _, row in df.iterrows():
                        # Construct a searchable string representation of the document
                        content = (
                            f"Service: {row.get('اسم الخدمة', '')}\n"
                            f"Steps: {row.get('خطوات الخدمة', '')}\n"
                            f"Docs: {row.get('المستندات المطلوبة', '')}\n"
                            f"Price: {row.get('سعر الخدمة', '')}"
                        )
                        documents.append(Document(page_content=content))
                except Exception as e:
                    logger.error(f"Error reading {filename}: {e}")

        if not documents:
            return None
        return BM25Retriever.from_documents(documents)

    def _rrf_merge(self, dense_docs: List[Document], sparse_docs: List[Document]) -> List[Document]:
        """
        Implements Reciprocal Rank Fusion (RRF) to combine results from 
        Dense (Vector) and Sparse (BM25) retrievers.
        
        Args:
            dense_docs: Documents retrieved via semantic search.
            sparse_docs: Documents retrieved via keyword search.
            
        Returns:
            List[Document]: Re-ranked top documents based on combined scores.
        """
        scores = {}
        k = Config.RRF_K
        
        # Calculate RRF scores for semantic search results
        for rank, doc in enumerate(dense_docs):
            content = doc.page_content
            if content not in scores: scores[content] = {"doc": doc, "score": 0.0}
            scores[content]["score"] += 1.0 / (k + rank + 1)
            
        # Calculate RRF scores for keyword search results and add to the total
        if sparse_docs:
            for rank, doc in enumerate(sparse_docs):
                content = doc.page_content
                if content not in scores: scores[content] = {"doc": doc, "score": 0.0}
                scores[content]["score"] += 1.0 / (k + rank + 1)
        
        # Sort documents by the combined RRF score
        sorted_docs = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_docs[:Config.RETRIEVAL_K]]

    def detect_language(self, text: str) -> str:
        """
        Identifies the language of the input text. Prioritizes Arabic/Urdu scripts.
        
        Args:
            text: The user's raw input.
            
        Returns:
            str: ISO language code (e.g., 'ar', 'en', 'ur').
        """
        try:
            # Check for Arabic script characters explicitly
            if any("\u0600" <= c <= "\u06FF" for c in text): return 'ar'
            lang = detect(text)
            # Map Hindi detection to Urdu for system consistency
            if lang == 'hi': return 'ur'
            return lang
        except: return 'en'

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Uses the internal LLM to translate text between languages.
        
        Args:
            text: Text to translate.
            source_lang: Detected source language.
            target_lang: Desired target language.
            
        Returns:
            str: Translated text.
        """
        if source_lang == target_lang: return text
        logger.info(f"🔤 Translating '{source_lang}' -> '{target_lang}'")
        
        # System instruction to maintain entity integrity during translation
        prompt = f"""[INST] <<SYS>>
        You are a professional translator. Translate from {source_lang} to {target_lang}.
        Keep technical terms like 'Absher' as they are. Preserve all numbers and prices.
        <</SYS>>
        Text: {text} [/INST]"""
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(Config.DEVICE)
            with torch.no_grad():
                out = self.llm.generate(**inputs, max_new_tokens=Config.MAX_NEW_TOKENS, temperature=0.1)
            return self.tokenizer.decode(out[0], skip_special_tokens=True).split("[/INST]")[-1].strip()
        except: return text

    def _rewrite_query(self, current_query: str, history: List[Tuple[str, str]]) -> str:
        """
        Analyzes conversation history to rewrite vague follow-up queries 
        into standalone, searchable questions.
        Example: "How much is the fee?" -> "How much is the Saudi Passport renewal fee?"
        
        Args:
            current_query: The latest user question.
            history: List of previous (Question, Answer) tuples.
            
        Returns:
            str: The contextualized standalone query.
        """
        if not history: return current_query
            
        logger.info("🧠 Rewriting query for context...")
        # Use the last two turns of history for context to avoid noise
        short_history = history[-2:]
        history_text = "\n".join([f"User: {h[0]}\nAI: {h[1]}" for h in short_history])
        
        prompt = f"""[INST] <<SYS>>
        Rewrite the last user question to be standalone based on the conversation history.
        ONLY output the rewritten question.
        <</SYS>>
        History: {history_text}
        Last Question: {current_query}
        Rewritten Question: [/INST]"""
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(Config.DEVICE)
            with torch.no_grad():
                out = self.llm.generate(**inputs, max_new_tokens=128, temperature=0.1)
            rewritten = self.tokenizer.decode(out[0], skip_special_tokens=True).split("[/INST]")[-1].strip()
            logger.info(f"🔄 Rewritten: '{current_query}' -> '{rewritten}'")
            return rewritten
        except: return current_query

    def _enrich_context_with_kg(self, retrieved_context: str) -> str:
        """
        Scans retrieved text and cross-references it with the Knowledge Graph.
        If a service match is found, it injects verified deterministic data (Fees, Steps).
        This is a critical anti-hallucination measure.
        
        Args:
            retrieved_context: The text retrieved from the vector/keyword search.
            
        Returns:
            str: The context enriched with verified KG data.
        """
        enriched_context = retrieved_context
        for sector, services in self.knowledge_graph.items():
            for service_name, details in services.items():
                # If a service from the KG is mentioned in the retrieved documents
                if service_name in retrieved_context:
                    logger.info(f"💡 KG Hit: {service_name}")
                    kg_snippet = (
                        f"\n\n[VERIFIED DATA: {service_name}]\n"
                        f"- Price: {details['price']}\n"
                        f"- Requirements: {details['requirements']}\n"
                        f"- Official Steps: {details['steps']}\n"
                    )
                    enriched_context += kg_snippet
        return enriched_context

    def run(self, query: str, history: List[Tuple[str, str]] = []) -> str:
        """
        Main Pipeline Entry Point.
        Executes the full logic: Language Detection -> Contextual Rewriting -> 
        T-S-T (if needed) -> Hybrid Retrieval -> KG Enrichment -> LLM Generation.
        
        Args:
            query: Raw user input.
            history: Conversation history for context.
            
        Returns:
            str: Final generated (and potentially translated) AI response.
        """
        # 
        
        # 1. Detect User's Language
        user_lang = self.detect_language(query)
        logger.info(f"🌍 Language: {user_lang}")
        
        # 2. Contextual Query Rewriting to handle follow-up ambiguity
        processed_query = self._rewrite_query(query, history) if history else query

        # 3. Translate-Search-Translate (TST) Strategy
        # For languages other than Arabic/English, we translate the query to English 
        # for higher-quality retrieval before translating the final answer back.
        search_query = processed_query
        is_weak_lang = user_lang not in self.PRIMARY_LANGUAGES
        
        if is_weak_lang:
            logger.info("🔄 Applying T-S-T Strategy...")
            search_query = self.translate_text(processed_query, user_lang, "English")

        # 4. Hybrid Retrieval (Combining Vector & Keyword Search)
        # Apply Arabic normalization to ensure search consistency
        clean_query = normalize_arabic(search_query) 
        dense_res = self.dense_retriever.invoke(clean_query)
        sparse_res = self.bm25_retriever.invoke(clean_query) if self.bm25_retriever else []
        final_docs = self._rrf_merge(dense_res, sparse_res)
        
        # Guardrail: Handle cases where no relevant information is found
        if not final_docs:
            return "عذراً، المعلومات غير متوفرة." if user_lang == 'ar' else "Sorry, info not found."

        # 5. Knowledge Graph (KG) Enrichment
        # Augment the retrieved documents with verified deterministic data from JSON
        raw_context = "\n".join([d.page_content for d in final_docs])
        enriched_context = self._enrich_context_with_kg(raw_context)

        # 6. Generation (ALLaM Reasoning Engine)
        # Determine the generation language (English for bridge or user's primary lang)
        target_gen_lang = "English" if is_weak_lang else ("Arabic" if user_lang == 'ar' else "English")
        
        # Format the system prompt using the centralized Config template
        full_prompt = Config.SYSTEM_PROMPT_TEMPLATE.format(
            context=enriched_context,
            chat_history="", # History is handled via query rewriting step
            question=search_query, 
            target_lang=target_gen_lang
        )
        
        logger.info(f"🤖 Generating in {target_gen_lang}...")
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(Config.DEVICE)
        
        # Generate the response using strict factual parameters
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=Config.MAX_NEW_TOKENS,
                temperature=Config.TEMPERATURE, # Nearly 0 for factuality
                do_sample=True,
                repetition_penalty=Config.REPETITION_PENALTY
            )
        
        # Decode response and strip input tokens
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("[/INST]")[-1].strip()

        # 7. Final Translation (TST Strategy Conclusion)
        # If the user spoke a weak language, translate the verified English response back to them.
        if is_weak_lang:
            logger.info(f"🔄 Translating Back -> {user_lang}")
            response = self.translate_text(response, "English", user_lang)

        return response