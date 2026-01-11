import time
import pandas as pd
import numpy as np
import os
import sys
import re
import collections
import torch
from typing import List, Dict, Any
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import Project Modules
from config import Config
from utils.logger import setup_logger
from core.model_loader import ModelManager
from core.vector_store import VectorStoreManager
from core.rag_pipeline import ProRAGChain
from data.ingestion import DataIngestor

# Setup Logger
logger = setup_logger("Global_Benchmark")

# --- JUDGE CONFIGURATION ---
# We use Qwen-2.5-14B because it is SOTA in multi-lingual reasoning (AR, ES, RU, DE, FR)
JUDGE_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"

# Prompt in English to ensure the Judge follows instructions perfectly, 
# even when evaluating Arabic/Spanish content.
JUDGE_SYS_PROMPT = """You are an impartial, expert AI evaluator.
Your task is to evaluate the quality of the "Proposed Answer" compared to the "Ground Truth" based on the Question.

Criteria:
1. Accuracy: Is the information correct?
2. Completeness: Is the answer full?
3. Language: Is it in the correct target language?

Score (1-5):
1: Wrong / Irrelevant.
3: Partially correct.
5: Perfect / Accurate.

Question: {question}
Ground Truth: {truth}
Proposed Answer: {prediction}

Output format:
Score: [1-5]
Reason: [Short explanation]
"""

class JudgeModel:
    """
    A dedicated wrapper for the Judge LLM (Qwen-2.5).
    Loaded separately to ensure unbiased evaluation.
    """
    def __init__(self):
        logger.info(f"⚖️ Loading Specialized Judge Model: {JUDGE_MODEL_ID}...")
        
        # ✅ Enforce loading from the specific directory requested
        cache_path = Config.MODELS_DIR
        logger.info(f"📂 Target Model Path: {cache_path}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                JUDGE_MODEL_ID,
                cache_dir=cache_path,
                trust_remote_code=True
            )
            # Load in bfloat16 to fit on A100 alongside ALLaM
            self.model = AutoModelForCausalLM.from_pretrained(
                JUDGE_MODEL_ID,
                torch_dtype=torch.bfloat16,
                device_map="auto", 
                trust_remote_code=True,
                cache_dir=cache_path
            )
            logger.info("✅ Judge Model Loaded Successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to load Judge Model: {e}")
            raise e

    def evaluate(self, question: str, truth: str, prediction: str) -> Dict[str, Any]:
        """
        Runs the evaluation prompt through Qwen.
        """
        prompt = JUDGE_SYS_PROMPT.format(
            question=question,
            truth=truth,
            prediction=prediction
        )
        
        messages = [
            {"role": "system", "content": "You are a strict AI judge."},
            {"role": "user", "content": prompt}
        ]
        
        text_input = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer([text_input], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.1, # Low temp for consistency
                do_sample=False
            )
            
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return self._parse_response(response)

    def _parse_response(self, text: str) -> Dict[str, Any]:
        # Robust parsing using Regex
        score_match = re.search(r"Score\s*[:\-]\s*(\d)", text, re.IGNORECASE)
        reason_match = re.search(r"Reason\s*[:\-]\s*(.+)", text, re.IGNORECASE | re.DOTALL)
        
        score = 3
        reason = "Parse Error"
        
        if score_match:
            score = int(score_match.group(1))
        
        if reason_match:
            reason = reason_match.group(1).strip()
        else:
            # Fallback: take part of the text
            reason = text[:150].replace("\n", " ").strip()
            
        return {"score": score, "reason": reason}


class RAGBenchmark:
    """
    Advanced Multi-Lingual Benchmarking Suite for MOI Chatbot (v5.0 - Expert Judge).
    """
    
    def __init__(self):
        self.rag_chain = None
        self.embed_model = None
        self.judge_model = None 
        self.success_threshold = 0.75
        
    def initialize_system(self):
        logger.info("⚙️ Initializing System for Global Benchmarking...")
        
        # 1. Load Main System Models (ALLaM + Embeddings)
        self.embed_model = ModelManager.get_embedding_model()
        
        # 2. Load Data
        ingestor = DataIngestor()
        docs = ingestor.load_and_process()
        if not docs: raise ValueError("No documents found!")
            
        vector_store = VectorStoreManager.load_or_build(self.embed_model, docs)
        self.rag_chain = ProRAGChain(vector_store, all_documents=docs)
        
        # 3. Load The Judge (New Step)
        # This will download Qwen (~28GB) to your specified folder
        self.judge_model = JudgeModel()
        
        logger.info("✅ Full System & Judge Ready.")

    # --- Metrics Helpers ---
    def normalize_text(self, s: str) -> str:
        def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
        def white_space_fix(text): return ' '.join(text.split())
        def remove_punc(text):
            exclude = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
            return ''.join(ch for ch in text if ch not in exclude)
        return white_space_fix(remove_articles(remove_punc(s.lower())))

    def calculate_exact_match(self, generated: str, reference: str) -> int:
        return int(self.normalize_text(generated) == self.normalize_text(reference))

    def calculate_f1(self, generated: str, reference: str) -> float:
        pred_tokens = self.normalize_text(generated).split()
        truth_tokens = self.normalize_text(reference).split()
        if len(pred_tokens) == 0 or len(truth_tokens) == 0: return int(pred_tokens == truth_tokens)
        common_tokens = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
        num_same = sum(common_tokens.values())
        if num_same == 0: return 0.0
        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(truth_tokens)
        return (2 * precision * recall) / (precision + recall)

    def calculate_semantic_score(self, generated: str, reference: str) -> float:
        if not generated or not reference: return 0.0
        vec_gen = self.embed_model.embed_query(generated)
        vec_ref = self.embed_model.embed_query(reference)
        return float(cosine_similarity(np.array(vec_gen).reshape(1, -1), np.array(vec_ref).reshape(1, -1))[0][0])

    def run_test(self, test_file_path: str = "data/ground_truth.csv"):
        logger.info(f"📝 Loading Test Dataset from {test_file_path}...")
        if not os.path.exists(test_file_path):
            logger.error(f"❌ Missing file: {test_file_path}")
            return
            
        df = pd.read_csv(test_file_path)
        logger.info(f"🚀 Running Benchmark on {len(df)} Scenarios...")
        
        results = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Benchmarking"):
            question = row['question']
            truth = row['ground_truth']
            lang = row['lang']
            
            # 1. System Answer
            start_time = time.time()
            try:
                generated_answer = self.rag_chain.answer(question)
                clean_answer = re.sub(r'<[^>]+>', '', generated_answer).strip()
            except Exception as e:
                clean_answer = "Error"
                logger.error(f"System Error: {e}")
            latency = time.time() - start_time
            
            # 2. Metrics
            semantic_acc = self.calculate_semantic_score(clean_answer, truth)
            exact_match = self.calculate_exact_match(clean_answer, truth)
            f1_score = self.calculate_f1(clean_answer, truth)
            
            # 3. Judge Evaluation (Using Qwen)
            judge_result = self.judge_model.evaluate(question, truth, clean_answer)
            llm_score = judge_result["score"]
            llm_reason = judge_result["reason"]

            # Hallucination Logic
            # Adjusted: If semantic is extremely low OR Judge says 1 (Complete fail)
            is_hallucination = 1 if (semantic_acc < 0.70 or llm_score == 1) else 0
            
            results.append({
                "language": lang,
                "question": question,
                "generated_answer": clean_answer,
                "ground_truth": truth,
                "judge_score": llm_score,
                "judge_reason": llm_reason,
                "semantic_score": round(semantic_acc, 4),
                "exact_match": exact_match,
                "f1_score": round(f1_score, 4),
                "hallucination": is_hallucination,
                "latency": round(latency, 4)
            })
            
        results_df = pd.DataFrame(results)
        
        # Report
        print("\n" + "="*60)
        print(f"📊 GLOBAL BENCHMARK REPORT (v5.0 - Expert Judge Qwen)")
        print("="*60)
        print(f"✅ Total Scenarios:     {len(results_df)}")
        print(f"⚖️ Judge Score:         {results_df['judge_score'].mean():.2f} / 5.0")
        print(f"🧠 Semantic Score:      {results_df['semantic_score'].mean():.2%}")
        print(f"👻 Hallucination Rate:  {results_df['hallucination'].mean():.2%}")
        print("-" * 60)
        
        if "language" in results_df.columns:
            print("\n🌍 Performance by Language:")
            print(results_df.groupby("language")[["judge_score", "semantic_score", "latency"]].mean())
        
        # Save
        out_path = os.path.join(Config.OUTPUTS_DIR, "benchmark_reports", "final_benchmark_qwen_judge.csv")
        results_df.to_csv(out_path, index=False, encoding='utf-8-sig')
        logger.info(f"📄 Report saved to: {out_path}")

if __name__ == "__main__":
    try:
        benchmarker = RAGBenchmark()
        benchmarker.initialize_system()
        benchmarker.run_test()
    except KeyboardInterrupt:
        print("\n🛑 Stopped.")