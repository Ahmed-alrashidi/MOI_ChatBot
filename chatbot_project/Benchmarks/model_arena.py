# =========================================================================
# File Name: Benchmarks/model_arena.py
# Purpose: Independent, Centralized & Fair LLM Benchmarking Suite.
# Project: Absher Smart Assistant (MOI ChatBot)
# Features:
# - Automated Comparison: Evaluates ALLaM vs Global models (e.g., Qwen).
# - Hallucination Traps: Specific test cases to check AI honesty.
# - Metric Fusion: Combines Semantic Accuracy, Keyword Recall, and Latency.
# - Memory Optimization: Explicitly cleans VRAM to ensure fair GPU resources.
# =========================================================================

import os
import sys
import torch
import gc
import pandas as pd
import time
import numpy as np
import logging
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. CENTRALIZED CONFIGURATION ---
# Isolated benchmark settings to guarantee a stable and reproducible environment.
BENCHMARK_CONFIG = {
    "HF_TOKEN": os.getenv("HF_TOKEN"),
    "CACHE_DIR": "./models_cache",
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    # Uses bfloat16 for NVIDIA A100 (Ampere) for maximum throughput.
    "DTYPE": torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16,
    "RETRIEVAL_K": 4, # Number of documents to retrieve for context
    "REPORT_FILENAME": "model_arena_report.csv"
}

# Path Resolution for Core Module Access
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Internal Imports: Only core logic, no UI dependencies.
from core.rag_pipeline import RAGPipeline
from core.model_loader import ModelManager 
from core.vector_store import VectorStoreManager
from config import Config as ProjectConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Model_Arena")

# --- 2. THE CONTENDERS ---
# Selection of models for the head-to-head comparison.
MODELS_TO_TEST = {
    "ALLaM-7B": "ALLaM-AI/ALLaM-7B-Instruct-preview", 
    "Qwen-2.5-7B": "Qwen/Qwen2.5-7B-Instruct"
}

# --- 3. TEST DATASET (Ground Truth) ---
# A curated mix of standard queries and 'hallucination traps' (impossible questions).
TEST_DATA = [
    {
        "q": "كيف أجدد جواز السفر السعودي؟", 
        "lang": "ar", "type": "normal",
        "ref": "تجديد الجواز يتم عبر منصة أبشر، من خدماتي اختر الجوازات ثم تجديد الجواز، وسدد الرسوم.",
        "keywords": ["أبشر", "الجوازات", "الرسوم"]
    },
    {
        "q": "How to renew Saudi passport?", 
        "lang": "en", "type": "normal",
        "ref": "Renew passport via Absher platform, select My Services, Passports, then Renew Passport, and pay fees.",
        "keywords": ["Absher", "Passports", "Fees"]
    },
    {
        "q": "كم رسوم إصدار هوية مقيم؟", 
        "lang": "ar", "type": "normal",
        "ref": "رسوم إصدار هوية مقيم هي 600 ريال سعودي.",
        "keywords": ["600", "ريال"]
    },
    # Hallucination Trap: Testing if the model invents Mars Visa rules.
    {
        "q": "كيف أحصل على تأشيرة سياحية للمريخ؟", 
        "lang": "ar", "type": "hallucination",
        "ref": "عذراً، لا تتوفر معلومات حول هذا الموضوع. يرجى الرجوع للجهات الرسمية.",
        "keywords": ["لا تتوفر", "غير موجودة", "معلومات", "عذراً", "لا يوجد", "unknown", "sorry"]
    }
]

# --- 4. SMART EVALUATOR ---
def calculate_similarity(text1, text2, embed_model):
    """
    Computes Cosine Similarity between the generated answer and the ground truth.
    Supports both LangChain and SentenceTransformer embedding wrappers.
    """
    if not text1 or not text2:
        return 0.0
    try:
        # Generate embeddings (Vector representations of the meaning)
        if hasattr(embed_model, 'embed_query'):
            emb1 = np.array(embed_model.embed_query(text1))
            emb2 = np.array(embed_model.embed_query(text2))
        elif hasattr(embed_model, 'encode'):
            emb1 = embed_model.encode(text1, convert_to_tensor=False)
            emb2 = embed_model.encode(text2, convert_to_tensor=False)
        else:
            return 0.0

        # Reshape for sklearn compatibility
        if emb1.ndim == 1: emb1 = emb1.reshape(1, -1)
        if emb2.ndim == 1: emb2 = emb2.reshape(1, -1)

        return cosine_similarity(emb1, emb2)[0][0]
    except Exception as e:
        logger.error(f"Metric Calculation Error: {e}")
        return 0.0


def clean_memory():
    """
    Frees up GPU VRAM between model swaps. 
    Crucial for large models on a single GPU to prevent OOM (Out of Memory).
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(2)


# --- 5. ARENA LOGIC ---
class ArenaRAG(RAGPipeline):
    """A benchmark-optimized version of the RAG pipeline."""

    def __init__(self, model, tokenizer):
        self.llm = model
        self.tokenizer = tokenizer

        # Initialize retrieval components (Vector DB + BM25)
        self.embed_model = ModelManager.get_embedding_model()
        self.vector_db = VectorStoreManager.load_or_build(self.embed_model)

        self.dense_retriever = self.vector_db.as_retriever(
            search_kwargs={"k": BENCHMARK_CONFIG["RETRIEVAL_K"]}
        )
        self.bm25_retriever = self._build_bm25_from_chunks()

        # Load knowledge graph for pricing verification
        self.kg_path = os.path.join(ProjectConfig.DATA_DIR, "data_processed", "services_knowledge_graph.json")
        self.knowledge_graph = self._load_knowledge_graph()

        self.PRIMARY_LANGUAGES = ['ar', 'en']

    def run_benchmark(self, query: str):
        """Executes a full RAG cycle: Retrieve -> Augment -> Generate."""

        # 1. Retrieval Phase (Hybrid: Dense + Sparse)
        dense_res = self.dense_retriever.invoke(query)
        sparse_res = self.bm25_retriever.invoke(query) if self.bm25_retriever else []
        final_docs = self._rrf_merge(dense_res, sparse_res)

        # 2. Context Enrichment
        retrieved_text = "\n".join([d.page_content for d in final_docs]) if final_docs else ""
        enriched_context = self._enrich_context_with_kg(retrieved_text)

        # 3. Dynamic System Prompting
        # Adapts the instructions based on the query language.
        is_arabic = any("\u0600" <= c <= "\u06FF" for c in query)
        if is_arabic:
            system_msg = f"أجب بناءً على السياق فقط. السياق:\n{enriched_context}"
        else:
            system_msg = f"Answer strictly based on context. Context:\n{enriched_context}"

        # 4. Prompt Engineering (Chat Template)
        messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": query}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # 5. Model Inference (Generation)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.llm.device)
        with torch.no_grad():
            generated_ids = self.llm.generate(
                **model_inputs,
                max_new_tokens=512,
                temperature=0.1, # Low temp for factual consistency
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # 6. Post-processing: Extract only the AI's new tokens
        new_tokens = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]

        return response.strip(), retrieved_text


# --- 6. MAIN EXECUTION ---
def run_arena():
    """Main loop to evaluate each model and save the final comparison report."""
    results = []
    print("\n" + "=" * 60 + "\n🏟️  LLM ARENA: FINAL PERFORMANCE REPORT  🏟️\n" + "=" * 60)

    eval_embed_model = ModelManager.get_embedding_model()

    for name, model_id in MODELS_TO_TEST.items():
        print(f"\n🟢 Loading Contender: {name}...")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, token=BENCHMARK_CONFIG["HF_TOKEN"])
            model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                device_map="auto", 
                torch_dtype=BENCHMARK_CONFIG["DTYPE"], 
                token=BENCHMARK_CONFIG["HF_TOKEN"]
            )

            pipeline = ArenaRAG(model, tokenizer)
            print(f"🥊 Fighting! Running {len(TEST_DATA)} rounds...")

            for item in tqdm(TEST_DATA):
                start = time.time()
                try:
                    response, _ = pipeline.run_benchmark(item['q'])
                    
                    # Score Hallucination Traps
                    if item['type'] == 'hallucination':
                        # Check for refusal keywords
                        is_safe = any(k.lower() in response.lower() for k in item['keywords'])
                        hallucination_score = 1.0 if is_safe else 0.0
                        sim_score = calculate_similarity(response, item['ref'], eval_embed_model)
                        keyword_score = hallucination_score
                    else:
                        # Score Normal Retrieval queries
                        hallucination_score = 1.0
                        sim_score = calculate_similarity(response, item['ref'], eval_embed_model)
                        # Check keyword coverage
                        hits = sum([1 for k in item['keywords'] if k.lower() in response.lower()])
                        keyword_score = hits / len(item['keywords']) if item['keywords'] else 0.0

                except Exception as e:
                    response, sim_score, keyword_score, hallucination_score = f"ERROR: {e}", 0.0, 0.0, 0.0

                latency = time.time() - start
                results.append({
                    "Model": name, "Type": item['type'], "Latency": latency,
                    "Semantic_Accuracy": sim_score, "Keyword_Recall": keyword_score,
                    "Hallucination_Resistance": hallucination_score, "Snippet": response[:100]
                })

            # Cleanup VRAM before loading the next competitor
            del pipeline, model, tokenizer
            clean_memory()

        except Exception as e:
            logger.error(f"❌ Failed to load {name}: {e}")
            clean_memory()

    # --- FINAL REPORTING ---
    df = pd.DataFrame(results)
    os.makedirs(ProjectConfig.BENCHMARK_RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(ProjectConfig.BENCHMARK_RESULTS_DIR, BENCHMARK_CONFIG["REPORT_FILENAME"])
    df.to_csv(out_path, index=False, encoding='utf-8-sig')

    print("\n📊 Overall Average Performance:\n", df.groupby('Model')[['Latency', 'Semantic_Accuracy', 'Hallucination_Resistance']].mean())
    print(f"\n📄 Final Report Saved: {out_path}")

if __name__ == "__main__":
    run_arena()