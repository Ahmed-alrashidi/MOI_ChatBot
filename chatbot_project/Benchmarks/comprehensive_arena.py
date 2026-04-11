# =========================================================================
# File Name: Benchmarks/comprehensive_arena.py
# Purpose: Fair Multi-Model Evaluation with Data-Grounded Judging.
# Project: Absher Smart Assistant (MOI ChatBot)
# Version: 6.0 (Data-Driven Judge + Fair Price Matching)
# =========================================================================

import os
import sys
import time
import gc
import re
import json
import warnings
import logging
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from config import Config
from core.rag_pipeline import RAGPipeline
from utils.logger import setup_logger
from utils.text_utils import normalize_arabic

logger = setup_logger("Comprehensive_Arena")

MODELS_TO_TEST = {
    "ALLaM-7B": "ALLaM-AI/ALLaM-7B-Instruct-preview",
    "Qwen-2.5-7B": "Qwen/Qwen2.5-7B-Instruct",
    "Gemma-2-9B": "google/gemma-2-9b-it",
    "Llama-3.1-8B": "meta-llama/Meta-Llama-3.1-8B-Instruct"
}


# ==========================================
# 1. DATA-GROUNDED REFERENCE BUILDER
# ==========================================

class DataGroundedReference:
    """
    Builds the FULL truth reference for each question by combining:
    - Ground Truth answer
    - Knowledge Graph (exact prices + steps)
    - Master CSV (service details)
    
    This ensures the judge evaluates against REAL data, not guesses.
    """
    
    def __init__(self):
        # Load KG
        kg_path = os.path.join(Config.DATA_PROCESSED_DIR, "services_knowledge_graph.json")
        with open(kg_path, 'r', encoding='utf-8') as f:
            self.kg = json.load(f)
        
        # Load Master CSV
        master_path = os.path.join(Config.DATA_MASTER_DIR, "MOI_Master_Knowledge.csv")
        self.master = pd.read_csv(master_path, encoding='utf-8-sig')
        
        # Build flat service index for fast lookup
        self.service_index = {}
        for _, row in self.master.iterrows():
            name = row['Service_Name'].strip()
            self.service_index[normalize_arabic(name)] = {
                "name": name,
                "sector": row['Sector'],
                "fees": row['Service_Fees'],
                "steps": row['Service_Steps'],
                "requirements": row['Requirements'],
                "description": row['Service_Description'],
                "url": row['Official_URL'],
                "audience": row['Target_Audience']
            }
        
        # Build flat KG index
        self.kg_index = {}
        for sector, services in self.kg.items():
            for svc_name, details in services.items():
                self.kg_index[normalize_arabic(svc_name)] = {
                    "name": svc_name,
                    "sector": sector,
                    "price": details.get('price', ''),
                    "steps": details.get('steps', '')
                }
        
        logger.info(f"📚 Data loaded: {len(self.service_index)} services from Master, {len(self.kg_index)} from KG")
    
    def find_service(self, question: str) -> dict:
        """Finds the most relevant service for a question using keyword matching."""
        q_norm = normalize_arabic(question)
        
        best_match = None
        best_score = 0
        
        skip = {'من', 'في', 'عن', 'على', 'هل', 'ما', 'كم', 'كيف', 'هي', 'هو',
                'what', 'how', 'the', 'for', 'and', 'are', 'les', 'des', 'und',
                'اصدار', 'تجديد', 'استعلام', 'خدمه', 'خدمة'}
        
        def strip_al(w):
            return w[2:] if w.startswith('ال') and len(w) > 3 else w
        
        q_words = {strip_al(w) for w in q_norm.split() if len(w) >= 3} - skip
        
        for key, data in self.kg_index.items():
            svc_words = {strip_al(w) for w in key.split() if len(w) >= 3} - skip
            overlap = q_words & svc_words
            if len(overlap) > best_score:
                best_score = len(overlap)
                best_match = data
        
        return best_match
    
    def build_reference(self, question: str, gt_answer: str) -> dict:
        """
        Builds a complete truth reference for judging.
        Returns: {reference_text, service_name, exact_price, exact_steps, source}
        """
        service = self.find_service(question)
        
        if service:
            # Get Master CSV data if available
            master_data = self.service_index.get(normalize_arabic(service['name']), {})
            
            reference = (
                f"[Ground Truth Answer]:\n{gt_answer}\n\n"
                f"[Official KG Data - {service['name']}]:\n"
                f"- القطاع: {service['sector']}\n"
                f"- الرسوم الرسمية: {service['price']}\n"
                f"- الخطوات: {service['steps']}\n"
            )
            
            if master_data:
                reference += (
                    f"\n[Master CSV Data]:\n"
                    f"- المتطلبات: {master_data.get('requirements', 'N/A')}\n"
                    f"- الفئة المستهدفة: {master_data.get('audience', 'N/A')}\n"
                    f"- الرابط الرسمي: {master_data.get('url', 'N/A')}\n"
                )
            
            return {
                "reference_text": reference,
                "service_name": service['name'],
                "exact_price": service['price'],
                "exact_steps": service['steps'],
                "source": "KG+Master"
            }
        
        # Fallback: use GT only
        return {
            "reference_text": f"[Ground Truth Answer]:\n{gt_answer}",
            "service_name": "Unknown",
            "exact_price": "",
            "exact_steps": "",
            "source": "GT_only"
        }
    
    def get_service_prices(self, question: str) -> set:
        """Gets ONLY the relevant prices for this specific question's service."""
        service = self.find_service(question)
        if not service:
            return set()
        
        prices = set()
        price_text = service.get('price', '')
        for match in re.finditer(r'(\d+)', price_text):
            try:
                prices.add(int(match.group(1)))
            except ValueError:
                continue
        return prices


# ==========================================
# 2. FAIR AUTOMATED METRICS
# ==========================================

class FairMetrics:
    """Metrics that evaluate against the CORRECT service, not all services."""
    
    @staticmethod
    def rouge_l(reference: str, hypothesis: str) -> float:
        ref_tokens = normalize_arabic(reference).split()
        hyp_tokens = normalize_arabic(hypothesis).split()
        if not ref_tokens or not hyp_tokens:
            return 0.0
        m, n = len(ref_tokens), len(hyp_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_tokens[i-1] == hyp_tokens[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        lcs = dp[m][n]
        if lcs == 0:
            return 0.0
        p = lcs / n
        r = lcs / m
        return round((2 * p * r) / (p + r), 4) if (p + r) > 0 else 0.0
    
    @staticmethod
    def extract_prices(text: str) -> set:
        prices = set()
        for match in re.finditer(r'(\d{2,})', text):
            val = int(match.group(1))
            if 50 <= val <= 10000:  # Filter noise (years like 5, 10 or random numbers)
                prices.add(val)
        return prices
    
    @staticmethod
    def price_accuracy(answer: str, expected_prices: set) -> dict:
        """
        FAIR price check: only compares against THIS service's prices.
        Score = fraction of expected prices found in answer.
        """
        if not expected_prices:
            return {"score": 1.0, "expected": set(), "found": set(), "reason": "no_price_service"}
        
        found = FairMetrics.extract_prices(answer)
        matches = expected_prices & found
        score = len(matches) / len(expected_prices) if expected_prices else 0.0
        
        return {
            "score": round(score, 2),
            "expected": expected_prices,
            "found": found,
            "reason": f"matched {len(matches)}/{len(expected_prices)}"
        }
    
    @staticmethod
    def attribution_check(answer: str) -> bool:
        markers = ['أبشر', 'absher', 'السجلات الرسمية', 'official records', 'المصدر الرسمي',
                   'based on', 'بناء على', 'وفقا', 'registros oficiales', 'selon les',
                   'nach den offiziellen', 'согласно официальным']
        return any(m in answer.lower() for m in markers)


# ==========================================
# 3. DATA-GROUNDED LLM JUDGE
# ==========================================

class DataGroundedJudge:
    """Judge that receives ALL real data before scoring."""
    
    def __init__(self, model_name=Config.JUDGE_MODEL_NAME):
        logger.info(f"⚖️ Loading Judge: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, token=Config.HF_TOKEN, cache_dir=Config.MODELS_CACHE_DIR
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=Config.TORCH_DTYPE, device_map="auto",
            attn_implementation="sdpa", token=Config.HF_TOKEN, cache_dir=Config.MODELS_CACHE_DIR
        )
    
    def evaluate(self, question: str, full_reference: str, answer: str) -> dict:
        prompt = f"""You are a FAIR and DATA-DRIVEN quality auditor for Saudi government services.

You have been given the OFFICIAL DATA from multiple verified sources.
Judge the AI's answer ONLY based on this verified data — not your own knowledge.

{full_reference}

[SCORING RUBRIC] (0-10 scale):
- [4 PTS] FACTUAL ACCURACY: Prices, fees, and facts MUST match the [Official KG Data] exactly.
  Give FULL 4 points if the answer includes the correct price from KG.
  Give 0 if price is wrong or fabricated.
- [3 PTS] COMPLETENESS: Steps and requirements from the data should be covered.
  Partial credit allowed (1-2 pts for partial coverage).
- [2 PTS] SOURCE ATTRIBUTION: Answer should reference "Absher" or "official records".
- [1 PT] PROFESSIONAL TONE: Formal, helpful government communication style.

[FAIRNESS RULES]:
- Do NOT penalize for answering in a different language than the question.
- Do NOT penalize for extra helpful information if it's factually correct.
- DO penalize for wrong prices, fabricated services, or hallucinated information.

[AI'S ANSWER TO EVALUATE]:
Question: {question}
Answer: {answer}

Return ONLY valid JSON: {{"Score": X, "Reason": "brief explanation"}}"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=256, temperature=0.01)
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True
        ).strip()
        
        try:
            json_str = re.search(r'\{.*\}', response, re.DOTALL).group(0)
            result = json.loads(json_str)
            result["Score"] = max(0.0, min(10.0, float(result.get("Score", 0))))
            return result
        except Exception:
            score_match = re.search(r'(\d+\.?\d*)', response)
            score = float(score_match.group(1)) if score_match else 1.0
            return {"Score": max(0.0, min(10.0, score)), "Reason": f"Extracted: {response[:80]}"}


def hard_vram_reset():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    time.sleep(0.5)


# ==========================================
# 4. DETAILED REPORT GENERATOR
# ==========================================

def generate_report(df: pd.DataFrame, output_dir: str, timestamp: str):
    
    print("\n" + "═" * 60)
    print("🏆  FINAL LEADERBOARD")
    print("═" * 60)
    
    leaderboard = df.groupby('Model').agg({
        'Judge_Score': ['mean', 'std'],
        'ROUGE_L': 'mean',
        'Price_Score': 'mean',
        'Attribution': 'mean',
        'Latency': 'mean'
    }).round(3)
    leaderboard.columns = ['Judge_Avg', 'Judge_Std', 'ROUGE_L', 'Price_Acc', 'Attribution', 'Latency']
    leaderboard = leaderboard.sort_values('Judge_Avg', ascending=False)
    print(leaderboard.to_string())
    
    print("\n" + "═" * 60)
    print("🌍  PER-LANGUAGE SCORES")
    print("═" * 60)
    lang_scores = df.pivot_table(values='Judge_Score', index='Model', columns='Lang', aggfunc='mean').round(1)
    print(lang_scores.to_string())
    
    print("\n" + "═" * 60)
    print("📂  PER-CATEGORY SCORES")
    print("═" * 60)
    cat_scores = df.pivot_table(values='Judge_Score', index='Model', columns='Category', aggfunc='mean').round(1)
    print(cat_scores.to_string())
    
    print("\n" + "═" * 60)
    print("💰  PRICE ACCURACY (only price-related questions)")
    print("═" * 60)
    price_df = df[df['Has_Price'] == True]
    if not price_df.empty:
        price_by_model = price_df.groupby('Model')['Price_Score'].mean().round(3).sort_values(ascending=False)
        print(price_by_model.to_string())
        print(f"\n  Total price questions: {len(price_df)}")
    
    print("\n" + "═" * 60)
    print("⚠️  WORST 5 QUESTIONS")
    print("═" * 60)
    for _, row in df.nsmallest(5, 'Judge_Score').iterrows():
        print(f"  [{row['Model']}] [{row['Lang']}] Score: {row['Judge_Score']}")
        print(f"    Q: {row['Question'][:70]}...")
        print(f"    Service: {row['Matched_Service']}")
        print(f"    Expected Price: {row['Expected_Price']} | Got: {row['Found_Price']}")
        print(f"    Reason: {str(row['Reason'])[:80]}")
        print()
    
    # Save summary
    summary_path = os.path.join(output_dir, f"summary_{timestamp}.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("ABSHER BENCHMARK REPORT v6.0 (Data-Grounded)\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Tests: {len(df)} ({df['Model'].nunique()} models × {len(df)//df['Model'].nunique()} questions)\n\n")
        f.write("LEADERBOARD:\n")
        f.write(leaderboard.to_string())
        f.write("\n\nPER-LANGUAGE:\n")
        f.write(lang_scores.to_string())
        f.write("\n\nPER-CATEGORY:\n")
        f.write(cat_scores.to_string())
        if not price_df.empty:
            f.write("\n\nPRICE ACCURACY:\n")
            f.write(price_by_model.to_string())
    
    logger.info(f"📊 Summary: {summary_path}")


# ==========================================
# 5. MAIN EXECUTION
# ==========================================

def run_comprehensive_arena(quick_test: bool = False):
    timestamp = str(int(time.time()))
    
    # Load ground truth
    data_path = os.path.join(Config.DATA_PROCESSED_DIR, Config.DEFAULT_GROUND_TRUTH)
    df_test = pd.read_csv(data_path)
    if quick_test:
        df_test = df_test.groupby('lang').head(1).reset_index(drop=True)
    
    # Initialize data-grounded reference builder
    ref_builder = DataGroundedReference()
    metrics = FairMetrics()
    
    total = len(df_test) * len(MODELS_TO_TEST)
    logger.info(f"📋 Benchmark: {len(df_test)} questions × {len(MODELS_TO_TEST)} models = {total} tests")
    
    # Pre-build all references ONCE (saves ~1s per question during generation)
    logger.info("📚 Pre-building grounded references...")
    prebuilt = {}
    for _, row in df_test.iterrows():
        q = row['question']
        ref = ref_builder.build_reference(q, row['ground_truth'])
        prices = ref_builder.get_service_prices(q)
        prebuilt[q] = {"ref": ref, "prices": prices}
    logger.info(f"✅ {len(prebuilt)} references pre-built.")
    
    all_results = []
    benchmark_start = time.time()
    
    # ── PHASE 1: GENERATE + MEASURE ──
    print("\n" + "═" * 60)
    print("📝  PHASE 1: GENERATING ANSWERS")
    print("═" * 60)
    
    for name, model_id in MODELS_TO_TEST.items():
        logger.info(f"🚀 Loading: {name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, token=Config.HF_TOKEN,
                cache_dir=Config.MODELS_CACHE_DIR, trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map="auto", torch_dtype=Config.TORCH_DTYPE,
                attn_implementation="sdpa", token=Config.HF_TOKEN,
                cache_dir=Config.MODELS_CACHE_DIR, trust_remote_code=True
            )
            rag_engine = RAGPipeline(llm=model, tokenizer=tokenizer)
            
            for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc=f"  {name}"):
                start = time.time()
                try:
                    ans = rag_engine.run(row['question'])
                except Exception as e:
                    ans = f"Error: {e}"
                latency = round(time.time() - start, 2)
                
                # Use pre-built grounded reference
                pb = prebuilt[row['question']]
                ref = pb['ref']
                expected_prices = pb['prices']
                
                # Fair metrics
                rouge = metrics.rouge_l(row['ground_truth'], ans)
                price = metrics.price_accuracy(ans, expected_prices)
                has_attr = metrics.attribution_check(ans)
                
                all_results.append({
                    "Model": name,
                    "Question": row['question'],
                    "GT": row['ground_truth'],
                    "Answer": ans,
                    "Lang": row['lang'],
                    "Category": row['category'],
                    "Latency": latency,
                    "ROUGE_L": rouge,
                    "Price_Score": price['score'],
                    "Has_Price": len(expected_prices) > 0,
                    "Expected_Price": str(expected_prices),
                    "Found_Price": str(price['found']),
                    "Attribution": 1.0 if has_attr else 0.0,
                    "Matched_Service": ref['service_name'],
                    "Reference": ref['reference_text'],
                })
            
            del rag_engine, model, tokenizer
            hard_vram_reset()
            logger.info(f"✅ {name} done.")
            
        except Exception as e:
            logger.error(f"💥 Failed: {name} — {e}")
    
    # ── PHASE 2: DATA-GROUNDED JUDGING ──
    phase1_time = time.time() - benchmark_start
    logger.info(f"⏱️ Phase 1 complete in {phase1_time/60:.1f} minutes.")
    
    # Save Phase 1 checkpoint (in case judge crashes, you keep the answers)
    checkpoint_df = pd.DataFrame(all_results).drop(columns=['Reference'], errors='ignore')
    checkpoint_path = os.path.join(Config.BENCHMARK_RESULTS_DIR, f"checkpoint_phase1_{timestamp}.csv")
    os.makedirs(Config.BENCHMARK_RESULTS_DIR, exist_ok=True)
    checkpoint_df.to_csv(checkpoint_path, index=False, encoding='utf-8-sig')
    logger.info(f"💾 Phase 1 checkpoint: {checkpoint_path}")
    
    print("\n" + "═" * 60)
    print("⚖️  PHASE 2: DATA-GROUNDED JUDGING")
    print("═" * 60)
    
    judge = DataGroundedJudge()
    
    for item in tqdm(all_results, desc="  Judging"):
        eval_res = judge.evaluate(item['Question'], item['Reference'], item['Answer'])
        item["Judge_Score"] = eval_res.get("Score", 0.0)
        item["Reason"] = eval_res.get("Reason", "")
    
    del judge
    hard_vram_reset()
    
    # ── PHASE 3: REPORT ──
    print("\n" + "═" * 60)
    print("📊  PHASE 3: REPORTS")
    print("═" * 60)
    
    os.makedirs(Config.BENCHMARK_RESULTS_DIR, exist_ok=True)
    
    final_df = pd.DataFrame(all_results)
    
    # Drop the full reference text from CSV to keep file size manageable
    save_df = final_df.drop(columns=['Reference'], errors='ignore')
    csv_path = os.path.join(Config.BENCHMARK_RESULTS_DIR, f"arena_v6_{timestamp}.csv")
    save_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    logger.info(f"💾 Results: {csv_path}")
    
    generate_report(final_df, Config.BENCHMARK_RESULTS_DIR, timestamp)
    
    total_time = time.time() - benchmark_start
    logger.info(f"🏁 Benchmark complete. {len(all_results)} tests in {total_time/60:.1f} minutes.")


if __name__ == "__main__":
    print("\n" + "═" * 50)
    print("  🏆 ABSHER BENCHMARK ARENA v6.0")
    print("  📚 Data-Grounded Fair Evaluation")
    print("═" * 50)
    print(f"  Models: {', '.join(MODELS_TO_TEST.keys())}")
    print(f"  Judge: {Config.JUDGE_MODEL_NAME}")
    print()
    print("  1. Quick Test  (8 questions × 4 models)")
    print("  2. Full Test   (120 questions × 4 models)")
    print()
    choice = input("  Choice (1/2): ").strip()
    run_comprehensive_arena(quick_test=(choice in ("1", "a")))