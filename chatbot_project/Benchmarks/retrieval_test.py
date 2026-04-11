# =========================================================================
# File Name: Benchmarks/retrieval_test.py
# Purpose: Retrieval Accuracy using Semantic Similarity.
# Version: 2.0 (Fixed paths + uses Config.DEFAULT_GROUND_TRUTH)
# =========================================================================
import time, os, sys, pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path: sys.path.append(project_root)

from config import Config
from utils.logger import setup_logger
from core.rag_pipeline import RAGPipeline
from utils.text_utils import normalize_arabic

logger = setup_logger("Retrieval_Benchmark")

class RetrievalBenchmark:
    def __init__(self):
        logger.info("🚀 Initializing Pipeline for Retrieval Evaluation...")
        self.rag = RAGPipeline()
        self.embed_model = self.rag.embed_model

    def _perform_hybrid_search(self, query, top_k):
        clean_query = normalize_arabic(query)
        try: dense_docs = self.rag.dense_retriever.invoke(clean_query)
        except: dense_docs = []
        try:
            sparse_docs = self.rag.bm25_retriever.invoke(clean_query) if self.rag.bm25_retriever else []
        except: sparse_docs = []
        seen = set()
        final = []
        for doc in dense_docs + sparse_docs:
            if doc.page_content not in seen:
                final.append(doc); seen.add(doc.page_content)
        return final[:top_k]

    def run_test(self, ground_truth_file=None, top_k=5, threshold=0.45):
        gt_file = ground_truth_file or Config.DEFAULT_GROUND_TRUTH
        data_path = os.path.join(Config.DATA_PROCESSED_DIR, gt_file)
        if not os.path.exists(data_path):
            logger.error(f"❌ Test data missing: {data_path}"); return

        df = pd.read_csv(data_path, encoding='utf-8-sig')
        logger.info(f"📂 Loaded {len(df)} queries.")
        results, hits = [], 0

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Retrieval Test"):
            query, target = str(row['question']), str(row['ground_truth'])
            start = time.time()
            top_docs = self._perform_hybrid_search(query, top_k)
            max_score, best_snippet = 0.0, ""

            if top_docs:
                target_vec = self.embed_model.embed_query(target) if hasattr(self.embed_model, 'embed_query') else self.embed_model.encode(target)
                for doc in top_docs:
                    doc_vec = self.embed_model.embed_query(doc.page_content) if hasattr(self.embed_model, 'embed_query') else self.embed_model.encode(doc.page_content)
                    score = float(cosine_similarity([target_vec], [doc_vec])[0][0])
                    if score > max_score: max_score = score; best_snippet = doc.page_content[:200]
                if max_score >= threshold: hits += 1

            results.append({"question": query, "lang": row.get('lang',''), "hit": 1 if max_score >= threshold else 0,
                           "similarity": round(max_score, 4), "latency": round(time.time()-start, 2), "snippet": best_snippet})

        hit_rate = (hits / len(df)) * 100
        avg_sim = sum(r['similarity'] for r in results) / len(results)

        out = os.path.join(Config.BENCHMARK_RESULTS_DIR, "retrieval_report.csv")
        os.makedirs(Config.BENCHMARK_RESULTS_DIR, exist_ok=True)
        pd.DataFrame(results).to_csv(out, index=False, encoding='utf-8-sig')

        print("\n" + "="*50)
        print(f"🎯 RETRIEVAL RESULTS (K={top_k}, threshold={threshold})")
        print(f"  Hit Rate:      {hit_rate:.1f}% ({hits}/{len(df)})")
        print(f"  Avg Similarity: {avg_sim:.4f}")
        print(f"  Report: {out}")
        print("="*50)

if __name__ == "__main__":
    RetrievalBenchmark().run_test()
