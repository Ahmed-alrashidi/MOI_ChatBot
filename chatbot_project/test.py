import time
import pandas as pd
import numpy as np
import os
import sys
import re
import collections
from typing import List, Dict
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# Import Project Modules
from config import Config
from utils.logger import setup_logger
from core.model_loader import ModelManager
from core.vector_store import VectorStoreManager
from core.rag_pipeline import ProRAGChain
from data.ingestion import DataIngestor

# Setup Logger
logger = setup_logger("Global_Benchmark")

class RAGBenchmark:
    """
    Advanced Multi-Lingual Benchmarking Suite for MOI Chatbot (v4.0).
    
    New Metrics:
    1. Response Fidelity (Exact Match - EM)
    2. Semantic Similarity (Cosine)
    3. Keyword Overlap (F1 Score)
    4. Hallucination Rate (Based on low semantic confidence)
    5. Performance Breakdown by Language (AR, EN, FR, ZH, DE, HI)
    """
    
    def __init__(self):
        self.rag_chain = None
        self.embed_model = None
        self.success_threshold = 0.75  # Threshold to consider an answer "Factually Correct"
        
    def initialize_system(self):
        logger.info("⚙️ Initializing System for Global Benchmarking...")
        self.embed_model = ModelManager.get_embedding_model()
        ingestor = DataIngestor()
        docs = ingestor.load_and_process()
        
        if not docs:
            raise ValueError("No documents found! Verification aborted.")
            
        vector_store = VectorStoreManager.load_or_build(self.embed_model, docs)
        self.rag_chain = ProRAGChain(vector_store, docs)
        logger.info("✅ System Ready.")

    # --- Metric Calculation Helpers ---

    def normalize_text(self, s: str) -> str:
        """Lower text and remove punctuation, articles and extra whitespace."""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        def white_space_fix(text):
            return ' '.join(text.split())
        def remove_punc(text):
            exclude = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
            return ''.join(ch for ch in text if ch not in exclude)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def calculate_exact_match(self, generated: str, reference: str) -> int:
        """Returns 1 if the normalized text matches exactly, else 0."""
        return int(self.normalize_text(generated) == self.normalize_text(reference))

    def calculate_f1(self, generated: str, reference: str) -> float:
        """Calculates word overlap F1 score."""
        pred_tokens = self.normalize_text(generated).split()
        truth_tokens = self.normalize_text(reference).split()
        
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return int(pred_tokens == truth_tokens)
        
        common_tokens = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
        num_same = sum(common_tokens.values())
        
        if num_same == 0:
            return 0.0
        
        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def calculate_semantic_score(self, generated: str, reference: str) -> float:
        """Computes Cosine Similarity using BGE-M3 Embeddings."""
        if not generated or not reference:
            return 0.0
        vec_gen = self.embed_model.embed_query(generated)
        vec_ref = self.embed_model.embed_query(reference)
        vec_gen = np.array(vec_gen).reshape(1, -1)
        vec_ref = np.array(vec_ref).reshape(1, -1)
        score = cosine_similarity(vec_gen, vec_ref)[0][0]
        return float(score)

    def run_test(self, test_file_path: str = "data/test_set_global.csv"):
        # Force fresh test set creation
        if os.path.exists(test_file_path):
            os.remove(test_file_path)

        logger.info(f"📝 Generating Multi-Lingual Test Dataset...")
        
        # --- Multi-Lingual Dataset (Same Fact, Different Languages) ---
        # Fact: Passport renewal fee for 10 years is 600 SAR.
        data_records = [
            # Arabic (The Core)
            {"lang": "Arabic", "question": "كم سعر تجديد جواز السفر 10 سنوات؟", "ground_truth": "سعر تجديد جواز السفر السعودي لمدة عشر سنوات هو 600 ريال سعودي."},
            {"lang": "Arabic", "question": "كيف اسدد مخالفاتي المرورية؟", "ground_truth": "يمكن تسديد المخالفات المرورية عبر منصة أبشر من خلال الخدمات الإلكترونية."},
            
            # English
            {"lang": "English", "question": "What is the fee for renewing a passport for 10 years?", "ground_truth": "The fee for renewing the Saudi passport for 10 years is 600 SAR."},
            
            # French
            {"lang": "French", "question": "Quel est le tarif de renouvellement du passeport pour 10 ans ?", "ground_truth": "Les frais de renouvellement du passeport saoudien pour 10 ans sont de 600 SAR."},
            
            # Chinese (Simplified)
            {"lang": "Chinese", "question": "续签护照10年的费用是多少？", "ground_truth": "续签沙特护照10年的费用为600沙特里亚尔。"},
            
            # German
            {"lang": "German", "question": "Wie hoch ist die Gebühr für die Erneuerung des Reisepasses für 10 Jahre?", "ground_truth": "Die Gebühr für die Erneuerung des saudischen Reisepasses für 10 Jahre beträgt 600 SAR."},
            
            # Hindi
            {"lang": "Hindi", "question": "10 साल के लिए पासपोर्ट नवीनीकरण शुल्क क्या है?", "ground_truth": "10 वर्षों के लिए सऊदी पासपोर्ट के नवीनीकरण का शुल्क 600 रियाल है।"}
        ]
        
        df = pd.DataFrame(data_records)
        df.to_csv(test_file_path, index=False)
        
        # Start Testing
        logger.info(f"🚀 Running Global Benchmark on {len(df)} Scenarios...")
        
        results = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Benchmarking"):
            question = row['question']
            truth = row['ground_truth']
            lang = row['lang']
            
            start_time = time.time()
            try:
                # Generate Answer
                generated_answer = self.rag_chain.answer(question)
                # Cleanup UI tags
                clean_answer = generated_answer.replace("<div dir='rtl' style='text-align: right;'>", "").replace("</div>", "").replace("<div dir='ltr' style='text-align: left;'>", "")
            except Exception as e:
                clean_answer = "Error"
                logger.error(f"Error on: {question}")
                
            end_time = time.time()
            latency = end_time - start_time
            
            # --- Metrics Calculation ---
            semantic_acc = self.calculate_semantic_score(clean_answer, truth)
            exact_match = self.calculate_exact_match(clean_answer, truth)
            f1_score = self.calculate_f1(clean_answer, truth)
            
            # Hallucination Logic: If semantic accuracy is low, we consider it a hallucination/error
            is_hallucination = 1 if semantic_acc < self.success_threshold else 0
            
            results.append({
                "language": lang,
                "question": question,
                "generated_answer": clean_answer,
                "ground_truth": truth,
                "semantic_accuracy": round(semantic_acc, 4),
                "exact_match": exact_match,
                "f1_score": round(f1_score, 4),
                "hallucination": is_hallucination,
                "latency_seconds": round(latency, 4)
            })
            
        results_df = pd.DataFrame(results)
        
        # --- Aggregated Report ---
        avg_semantic = results_df["semantic_accuracy"].mean()
        avg_em = results_df["exact_match"].mean()
        avg_f1 = results_df["f1_score"].mean()
        avg_hallucination = results_df["hallucination"].mean()
        avg_lat = results_df["latency_seconds"].mean()
        
        print("\n" + "="*60)
        print(f"📊 GLOBAL BENCHMARK REPORT (v4.0)")
        print("="*60)
        print(f"✅ Total Scenarios:         {len(results_df)}")
        print(f"🧠 Semantic Similarity:     {avg_semantic:.2%} (Target: >85%)")
        print(f"🎯 Response Fidelity (EM):  {avg_em:.2%}")
        print(f"🔠 Keyword Overlap (F1):    {avg_f1:.2%}")
        print(f"👻 Hallucination Rate:      {avg_hallucination:.2%} (Target: <10%)")
        print(f"⏱️ Avg Latency:             {avg_lat:.2f}s")
        print("-" * 60)
        
        # --- Performance by Language ---
        print("\n🌍 Performance by Query Language:")
        lang_group = results_df.groupby("language")[["semantic_accuracy", "latency_seconds", "hallucination"]].mean()
        # Rename for display
        lang_group = lang_group.rename(columns={
            "semantic_accuracy": "Accuracy", 
            "latency_seconds": "Latency (s)",
            "hallucination": "Hallucination Rate"
        })
        print(lang_group)
        print("="*60)
        
        # Save Detailed CSV
        output_dir = os.path.join(Config.OUTPUTS_DIR, "benchmark_reports")
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, "global_benchmark_results.csv")
        results_df.to_csv(report_path, index=False, encoding='utf-8-sig')
        logger.info(f"📄 Detailed report saved to: {report_path}")

if __name__ == "__main__":
    try:
        benchmarker = RAGBenchmark()
        benchmarker.initialize_system()
        benchmarker.run_test()
    except KeyboardInterrupt:
        print("\n🛑 Benchmark stopped by user.")