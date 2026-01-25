# =========================================================================
# File Name: Benchmarks/retrieval_test.py
# Purpose: Benchmarking Retrieval Accuracy using Semantic Similarity (Smart Match).
# Project: Absher Smart Assistant (MOI ChatBot)
# Logic: 
#   Calculates the Cosine Similarity between the 'Retrieved Chunk' and the 
#   'Ground Truth' answer. This proves the system is finding the right laws.
# =========================================================================

import time
import pandas as pd
import os
import sys
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# --- Path Configuration ---
# Ensures the script can find 'core' and 'utils' folders from the root directory.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from config import Config
from utils.logger import setup_logger
from core.rag_pipeline import RAGPipeline
from utils.text_utils import normalize_arabic

# Initialize module logger for benchmark tracking
logger = setup_logger("Retrieval_Benchmark")

class RetrievalBenchmark:
    """
    Evaluates the retrieval performance of the RAG system independently 
    from the generation phase. Focuses strictly on the 'Search' quality.
    """
    def __init__(self):
        try:
            logger.info("🚀 Initializing Pipeline for Semantic Evaluation...")
            self.rag = RAGPipeline()
            # Direct access to the embedding model (BGE-M3) for vector comparisons
            self.embed_model = self.rag.embed_model
        except Exception as e:
            logger.critical(f"Failed to init RAG: {e}")
            sys.exit(1)

    def _perform_hybrid_search(self, query: str, top_k: int):
        """
        Internal Engine: Replicates the Hybrid Retrieval (Dense + Sparse) 
        logic used in the main pipeline. 
        It extracts the best matching documents for a given query.
        """
        # 1. Query Pre-processing (Normalization)
        clean_query = normalize_arabic(query)
        
        # 2. Semantic Search: Using Vector DB (FAISS)
        try:
            dense_docs = self.rag.dense_retriever.invoke(clean_query)
        except Exception:
            dense_docs = []
            
        # 3. Keyword Search: Using BM25 (Sparse Retrieval)
        try:
            if self.rag.bm25_retriever:
                sparse_docs = self.rag.bm25_retriever.invoke(clean_query)
            else:
                sparse_docs = []
        except Exception:
            sparse_docs = []
            
        # 4. Fusion Logic: Deduplicate and merge results from both engines
        seen = set()
        final_docs = []
        for doc in dense_docs + sparse_docs:
            if doc.page_content not in seen:
                final_docs.append(doc)
                seen.add(doc.page_content)
        
        return final_docs[:top_k]

    def run_test(self, ground_truth_file="ground_truth_polyglot.csv", top_k=4, threshold=0.45):
        """
        Executes the Semantic Retrieval Test.
        
        Args:
            ground_truth_file: CSV containing queries and expected answers.
            top_k: Number of documents to inspect per query.
            threshold: Minimum similarity score (0.45) to consider the retrieval a 'Hit'.
        """
        data_path = os.path.join(Config.BENCHMARK_DATA_DIR, ground_truth_file)
        if not os.path.exists(data_path):
            logger.error(f"❌ Test Data missing at: {data_path}")
            return

        # Load the polyglot test set (8 languages)
        df = pd.read_csv(data_path)
        logger.info(f"📂 Loaded {len(df)} queries for validation.")

        results = []
        hits = 0

        print(f"\n⚡ Running Semantic Retrieval Test (@K={top_k})...")
        
        for index, row in tqdm(df.iterrows(), total=len(df)):
            query = str(row['question'])
            target_answer = str(row['ground_truth'])
            
            start_time = time.time()
            
            # --- PHASE 1: EXECUTE SEARCH ---
            # Retrieve the most relevant chunks from the database
            top_docs = self._perform_hybrid_search(query, top_k)
            
            # --- PHASE 2: SEMANTIC VERIFICATION ---
            is_hit = 0
            max_score = 0.0
            best_snippet = ""
            
            if top_docs:
                # Vectorize the Ground Truth Answer for comparison
                if hasattr(self.embed_model, 'embed_query'):
                    target_vec = self.embed_model.embed_query(target_answer)
                else:
                    target_vec = self.embed_model.encode(target_answer)
                
                for doc in top_docs:
                    # Vectorize each retrieved chunk
                    if hasattr(self.embed_model, 'embed_query'):
                        doc_vec = self.embed_model.embed_query(doc.page_content)
                    else:
                        doc_vec = self.embed_model.encode(doc.page_content)
                    
                    # Compute Cosine Similarity between vectors
                    score = float(cosine_similarity([target_vec], [doc_vec])[0][0])
                    
                    if score > max_score:
                        max_score = score
                        best_snippet = doc.page_content[:200].replace("\n", " ")
                
                # Check if the best retrieved chunk matches the 'Official Fact' above threshold
                if max_score >= threshold:
                    is_hit = 1
                    hits += 1
            
            latency = time.time() - start_time
            
            results.append({
                "question": query,
                "hit": is_hit,
                "similarity_score": round(max_score, 4),
                "latency": latency,
                "top_snippet": best_snippet
            })

        # --- PHASE 3: REPORT GENERATION ---
        hit_rate = (hits / len(df)) * 100
        avg_score = sum(r['similarity_score'] for r in results) / len(results) if results else 0
        
        # Save results for GRC (Governance, Risk, and Compliance) documentation
        out_file = os.path.join(Config.BENCHMARK_RESULTS_DIR, "retrieval_semantic_report.csv")
        os.makedirs(Config.BENCHMARK_RESULTS_DIR, exist_ok=True)
        pd.DataFrame(results).to_csv(out_file, index=False, encoding='utf-8-sig')

        print("\n" + "="*50)
        print("🎯 SEMANTIC RETRIEVAL RESULTS")
        print("="*50)
        print(f"✅ Total Queries:   {len(df)}")
        print(f"🎯 Hit Rate:         {hit_rate:.2f}% (Threshold > {threshold})")
        print(f"🧠 Avg Similarity:  {avg_score:.4f}")
        print(f"📄 Report saved:    {out_file}")
        print("="*50)

if __name__ == "__main__":
    tester = RetrievalBenchmark()
    # Runs the test using the polyglot ground truth file
    tester.run_test("ground_truth_polyglot.csv")