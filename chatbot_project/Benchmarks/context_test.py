# =========================================================================
# File Name: Benchmarks/context_test.py
# Purpose: Multi-Turn Context Memory Evaluation.
# Version: 2.0 (No deep_translator dependency, uses RAG T-S-T instead)
# =========================================================================
import os, sys, time, pandas as pd
import torch, torch.nn.functional as F
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path: sys.path.append(project_root)

from config import Config
from core.rag_pipeline import RAGPipeline
from core.model_loader import ModelManager
from utils.logger import setup_logger
logger = setup_logger("Context_Test")

SCENARIOS = {
    "Passport_Renew": ["كيف أجدد جواز السفر؟", "كم الرسوم؟", "هل يحتاج فحص طبي؟", "هل يوصل بالبريد؟"],
    "License_Renew": ["كيف أجدد رخصة القيادة؟", "ما هي المتطلبات؟", "كم التكلفة؟", "هل فيه غرامة تأخير؟"],
    "Iqama_Issue": ["كيف أصدر هوية مقيم؟", "من يدفع الرسوم؟", "هل التأمين الطبي مطلوب؟", "ما عقوبة التأخير؟"],
    "Traffic_Violation": ["كيف أعترض على مخالفة مرورية؟", "هل فيه مهلة زمنية؟", "لازم أدفع أول؟", "أقدر أعترض مرة ثانية؟"],
    "Exit_ReEntry": ["كيف أصدر تأشيرة خروج وعودة؟", "كم رسوم المفردة؟", "أقدر أمددها خارج المملكة؟", "شروط الجواز؟"],
}

class ContextJudge:
    def __init__(self):
        self.model = ModelManager.get_embedding_model()

    def similarity(self, text1, text2):
        if hasattr(self.model, 'embed_documents'):
            vecs = torch.tensor(self.model.embed_documents([text1, text2]))
        else:
            vecs = self.model.encode([text1, text2], convert_to_tensor=True)
        if vecs.device.type == 'cuda': vecs = vecs.cpu()
        return F.cosine_similarity(vecs[0].unsqueeze(0), vecs[1].unsqueeze(0)).item()

    def evaluate(self, topic, answer):
        if not answer: return 0.0, False
        score = self.similarity(topic, answer[:500])
        return round(score, 3), score > 0.35

def run_context_benchmark():
    rag = RAGPipeline()
    judge = ContextJudge()
    results = []

    print("\n" + "="*55)
    print("🧠 MULTI-TURN CONTEXT BENCHMARK")
    print("="*55)

    for scenario, questions in SCENARIOS.items():
        print(f"\n  📂 {scenario}")
        history = []
        topic_kw = scenario.replace("_", " ") + " رسوم خطوات متطلبات"

        for i, q in enumerate(questions):
            start = time.time()
            hist_tuples = [tuple(h) for h in history]
            response = rag.run(q, hist_tuples)
            latency = time.time() - start
            history.append((q, response))

            score, passed = judge.evaluate(topic_kw, response)
            status = "✅" if passed else "❌"
            print(f"    Turn {i+1}: {status} (Sim: {score:.2f}) | {latency:.1f}s")
            results.append({"Scenario": scenario, "Turn": i+1, "Question": q,
                           "Score": score, "Passed": 1 if passed else 0, "Latency": round(latency, 2)})

    df = pd.DataFrame(results)
    out = os.path.join(Config.BENCHMARK_RESULTS_DIR, "context_report.csv")
    os.makedirs(Config.BENCHMARK_RESULTS_DIR, exist_ok=True)
    df.to_csv(out, index=False, encoding='utf-8-sig')

    followup = df[df['Turn'] > 1]
    rate = followup['Passed'].mean() * 100 if not followup.empty else 0

    print(f"\n{'='*55}")
    print(f"🧠 CONTEXT RESULTS")
    print(f"  Follow-up Success Rate: {rate:.1f}%")
    print(f"  Avg Similarity: {df['Score'].mean():.3f}")
    print(f"  Report: {out}")
    print("="*55)

if __name__ == "__main__":
    run_context_benchmark()
