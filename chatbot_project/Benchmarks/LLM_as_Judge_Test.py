# =========================================================================
# File Name: Benchmarks/LLM_as_Judge_Test.py
# Purpose: Comprehensive Evaluation Suite with Knowledge Graph (KG) Awareness.
# Project: Absher Smart Assistant (MOI ChatBot)
# Logic: Uses a secondary LLM (Qwen-14B) to audit the primary LLM (ALLaM-7B).
# Features:
# - Factuality Audit: Cross-references answers with deterministic JSON data.
# - Polyglot Pricing Logic: Detects variable vs. fixed fees across 8 languages.
# - Resource Management: Automated VRAM clearing to prevent GPU OOM on A100.
# =========================================================================

import pandas as pd
import torch
import os
import sys
import time
import gc
import re
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Path Configuration ---
# Ensures the script can access the project's core modules from the root.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from config import Config
from core.rag_pipeline import RAGPipeline
from utils.logger import setup_logger
from core.model_loader import ModelManager

# Initialize logger for the benchmark engine
logger = setup_logger("Benchmark_Engine")

class MetricsCalculator:
    """
    Calculates objective accuracy metrics. 
    It is specifically designed to handle the complexity of government fees 
    (Fixed vs. Variable) across multiple languages.
    """
    
    @staticmethod
    def get_numerical_accuracy(ground_truth: str, prediction: str) -> float:
        """
        Smart Numerical Guard: Analyzes text to see if the AI correctly 
        identified pricing conditions.
        
        Logic: If a fee depends on age/duration (Variable), the AI shouldn't 
        be penalized for not giving a single number, provided it explains the 
        dependency.
        """
        gt_str = str(ground_truth).lower()
        pred_str = str(prediction).lower()
        
        # Keywords indicating variable fees/conditions in all supported languages.
        var_keywords = [
            "متغير", "حسب", "تعتمد", "تبدأ",  # Arabic
            "variable", "depends", "varies", "starting", "depends on", # English
            "variabel", "abhängig", "je nach", # German
            "dépend", "selon", # French
            "variable", "depende", "según", # Spanish
            "зависит", "варьيруется", # Russian
            "变", "定", "取决于", # Chinese
            "تبدیل", "منحصر" # Urdu
        ]
        
        gt_is_variable = any(k in gt_str for k in var_keywords)
        pred_is_variable = any(k in pred_str for k in var_keywords)

        # 1. Successful identification of variable pricing.
        if gt_is_variable and pred_is_variable:
            return 1.0
            
        # 2. Penalty: If the fact is variable but the AI hallucinated a fixed number.
        if gt_is_variable and not pred_is_variable:
            pred_numbers = re.findall(r'\d+', pred_str)
            return 0.3 if pred_numbers else 1.0 

        # 3. Direct matching for fixed prices (e.g., 300 SAR).
        gt_numbers = re.findall(r'\d+', gt_str)
        if not gt_numbers: 
            return 1.0 # Nothing to verify
        
        pred_numbers = re.findall(r'\d+', pred_str)
        matches = [n for n in gt_numbers if n in pred_numbers]
        
        return len(matches) / len(gt_numbers) if gt_numbers else 1.0

class JudgeModel:
    """
    The 'Auditor' Model (Qwen-14B). 
    Used to grade the generated answers on a scale of 1-10 based on 
    ground truth and Knowledge Graph records.
    """
    
    def __init__(self, model_name=Config.JUDGE_MODEL_NAME):
        logger.info(f"⚖️ Loading Judge Model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=Config.HF_TOKEN)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=Config.TORCH_DTYPE,
            device_map="auto",
            token=Config.HF_TOKEN,
            trust_remote_code=True
        )
        # Inject the Knowledge Graph for the Judge to use as a 'Cheat Sheet'
        self.kg_data = self._load_kg()

    def _load_kg(self):
        """Loads the structured service database for cross-referencing."""
        try:
            path = os.path.join(Config.DATA_DIR, "data_processed", "services_knowledge_graph.json")
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}

    def _get_official_record(self, question: str) -> str:
        """Finds the deterministic fact in the KG that matches the user's query."""
        for sector, services in self.kg_data.items():
            for service_name, details in services.items():
                if service_name in question:
                    return (
                        f"Service: {service_name}\n"
                        f"Official Price: {details['price']}\n"
                        f"Requirements: {details['requirements']}"
                    )
        return "Not available in KG."

    def evaluate(self, question, ground_truth, generated_answer, lang):
        """
        Grades the primary AI response. 
        The prompt forces the Judge to consider the 'Official Database Record' 
        as the ultimate source of truth.
        """
        kg_record = self._get_official_record(question)
        
        prompt = f"""You are a senior auditor for the Saudi Ministry of Interior.
        Evaluate the "AI Answer" against the "Official Fact" and "Database Record".

        [CRITICAL GUIDELINES]
        1. **Variable Pricing:** If the Official Fact/Record says "Variable" or "Depends", and the AI explains that it's not a fixed price, give a HIGH score.
        2. **Database Authority:** If the AI Answer matches the [OFFICIAL DATABASE RECORD], it is CORRECT, even if it differs slightly from the "Official Fact" text.
        3. **Language:** The answer must be in {lang}.

        [INPUTS]
        - Lang: {lang} 
        - Q: {question}
        - [OFFICIAL DATABASE RECORD]: {kg_record}
        - Official Fact (Ground Truth): {ground_truth}
        - AI Answer: {generated_answer}

        Output ONLY:
        Score: [1-10]
        Reason: [Brief Explanation]
        """
        
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=150, do_sample=False)
        
        response = self.tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        
        # Clean & Parse the numerical score from the output string
        clean_response = response.replace('*', '').replace('_', '')
        score_match = re.search(r'Score\D*(\d+\.?\d*)', clean_response, re.IGNORECASE)
        score = float(score_match.group(1)) if score_match else 1.0
        
        return score, response.strip()

class BenchmarkRunner:
    """
    Orchestrates the two-phase benchmark:
    Phase 1: Generate 120+ answers using the RAG Pipeline (ALLaM-7B).
    Phase 2: Grade the answers using the Judge Model (Qwen-14B).
    """
    def __init__(self):
        self.metrics = MetricsCalculator()

    def run_full_benchmark(self, test_file="ground_truth_polyglot.csv"):
        data_path = os.path.join(Config.BENCHMARK_DATA_DIR, test_file)
        if not os.path.exists(data_path):
            logger.error(f"❌ Test file not found: {data_path}")
            return

        df = pd.read_csv(data_path)
        logger.info(f"🚀 Starting Fairness Benchmark: {len(df)} cases.")

        # --- Phase 1: Generation ---
        rag = RAGPipeline()
        gen_results = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="🤖 RAG Generation"):
            start = time.time()
            try:
                ans = rag.run(row['question'])
            except: ans = "ERROR_GENERATING"
            gen_results.append({"answer": ans, "latency": time.time() - start})

        # CRITICAL: Manual VRAM cleanup between models to prevent A100 OOM
        del rag
        gc.collect()
        torch.cuda.empty_cache()

        # --- Phase 2: Multi-Metric Evaluation ---
        judge = JudgeModel()
        
        # Use sentence-transformers for fast semantic similarity
        try:
            from sentence_transformers import SentenceTransformer
            embed_model = SentenceTransformer(Config.EMBEDDING_MODEL_NAME)
            use_st = True
        except:
            embed_model = ModelManager.get_embedding_model()
            use_st = False

        final_data = []

        for i, row in tqdm(df.iterrows(), total=len(df), desc="⚖️ Grading"):
            ans = gen_results[i]['answer']
            gt = str(row['ground_truth'])
            
            # 1. Price Consistency Check
            num_acc = self.metrics.get_numerical_accuracy(gt, ans)
            
            # 2. Semantic Similarity Calculation (BGE-M3)
            if use_st:
                v1 = embed_model.encode([ans])
                v2 = embed_model.encode([gt])
                sem_sim = float(cosine_similarity(v1, v2)[0][0])
            else:
                v1 = embed_model.embed_query(ans)
                v2 = embed_model.embed_query(gt)
                sem_sim = float(cosine_similarity([v1], [v2])[0][0])
            
            # 3. Qualitative Score from LLM Judge
            j_score, j_reason = judge.evaluate(row['question'], gt, ans, row['lang'])

            final_data.append({
                "lang": row['lang'],
                "judge_score": j_score,
                "pricing_accuracy": round(num_acc, 2),
                "semantic_sim": round(sem_sim, 3),
                "latency": round(gen_results[i]['latency'], 2),
                "reason": j_reason
            })

        # Save the finalized report
        report_df = pd.DataFrame(final_data)
        out_path = os.path.join(Config.BENCHMARK_RESULTS_DIR, "Benchmark_by_LLM_judge.csv")
        report_df.to_csv(out_path, index=False, encoding='utf-8-sig')

        # Print Aggregated Summary
        print("\n" + "="*50 + "\n📈 BENCHMARK RESULTS (FINAL)\n" + "="*50)
        print(report_df.groupby('lang')[['judge_score', 'pricing_accuracy', 'semantic_sim']].mean())
        print(f"\n📄 Saved to: {out_path}")

if __name__ == "__main__":
    BenchmarkRunner().run_full_benchmark()