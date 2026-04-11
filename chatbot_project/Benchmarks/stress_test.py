# =========================================================================
# File Name: Benchmarks/stress_test.py
# Purpose: Stress Testing & Latency Profiling.
# Version: 2.0 (Fixed paths)
# =========================================================================
import time, os, sys, threading, statistics, random, pandas as pd
from concurrent.futures import ThreadPoolExecutor

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path: sys.path.append(project_root)

from config import Config
from core.rag_pipeline import RAGPipeline
from utils.logger import setup_logger
logger = setup_logger("Stress_Test")

CONCURRENT_USERS = 4
TOTAL_REQUESTS = 20

class StressTester:
    def __init__(self):
        logger.info("⚙️ Initializing RAG for Stress Test...")
        self.rag = RAGPipeline()
        self.latencies, self.errors = [], 0
        self.lock = threading.Lock()
        try:
            df = pd.read_csv(os.path.join(Config.DATA_PROCESSED_DIR, Config.DEFAULT_GROUND_TRUTH), encoding='utf-8-sig')
            self.queries = df['question'].tolist()
        except:
            self.queries = ["كيف أجدد جواز السفر؟", "كم رسوم تجديد الرخصة؟", "How to renew Iqama?"]

    def _single_request(self, rid):
        query = random.choice(self.queries)
        start = time.time()
        try:
            self.rag.run(query)
            dur = time.time() - start
            with self.lock:
                self.latencies.append(dur)
                print(f"  ✅ Req {rid:02d}: {dur:.2f}s | {query[:35]}...")
        except Exception as e:
            with self.lock:
                self.errors += 1
                print(f"  ❌ Req {rid:02d}: {e}")

    def run(self):
        print("\n" + "="*55)
        print(f"🔥 STRESS TEST (Users={CONCURRENT_USERS}, Requests={TOTAL_REQUESTS})")
        print("="*55)
        
        # Warmup
        print("  🔥 Warming up..."); self.rag.run("Warmup"); print("  ✅ Ready\n")
        
        start = time.time()
        with ThreadPoolExecutor(max_workers=CONCURRENT_USERS) as ex:
            futures = [ex.submit(self._single_request, i+1) for i in range(TOTAL_REQUESTS)]
            for f in futures: f.result()
        total = time.time() - start

        if not self.latencies:
            print("❌ No successful requests."); return

        avg = statistics.mean(self.latencies)
        p95 = statistics.quantiles(self.latencies, n=20)[18] if len(self.latencies) >= 20 else max(self.latencies)
        tps = len(self.latencies) / total

        out = os.path.join(Config.BENCHMARK_RESULTS_DIR, "stress_report.csv")
        os.makedirs(Config.BENCHMARK_RESULTS_DIR, exist_ok=True)
        pd.DataFrame({"latency": self.latencies}).to_csv(out, index=False)

        print(f"\n{'='*55}")
        print(f"📊 STRESS TEST RESULTS")
        print(f"  Success: {len(self.latencies)}/{TOTAL_REQUESTS} | Errors: {self.errors}")
        print(f"  TPS: {tps:.2f} q/s | Avg: {avg:.2f}s | P95: {p95:.2f}s | Max: {max(self.latencies):.2f}s")
        print(f"  Report: {out}")
        print("="*55)

if __name__ == "__main__":
    StressTester().run()
