# =========================================================================
# File Name: Benchmarks/stress_test.py
# Purpose: Advanced Stress Testing & Latency Profiling for Production Readiness.
# Project: Absher Smart Assistant (MOI ChatBot)
# Target Hardware: Optimized for NVIDIA A100 (High-Concurrency testing).
# Features:
# - Throughput (TPS): Measures transactions per second.
# - Tail Latency (P95): Identifies the experience of the slowest 5% of users.
# - Cache Neutrality: Uses random query sampling to prevent GPU cache bias.
# - Thread Safety: Uses mutex locks for accurate multi-threaded metric collection.
# =========================================================================

import time
import os
import sys
import threading
import statistics
import random
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# --- Path Resolution ---
# Locates project root to ensure core RAG modules are importable.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from config import Config
from core.rag_pipeline import RAGPipeline
from utils.logger import setup_logger

# Initialize specialized performance logger
logger = setup_logger("Stress_Test")

# --- TEST CONFIGURATION ---
# CONCURRENT_USERS: Simulates how many people are chatting at the EXACT same time.
# On an A100, 4-8 concurrent users is a typical baseline for a 7B model.
CONCURRENT_USERS = 4        
TOTAL_REQUESTS = 20         
DATA_FILE = "ground_truth_polyglot.csv"

class StressTester:
    """
    Simulates high-load scenarios to measure system stability and speed 
    under pressure. It tracks errors and calculates professional SLA metrics.
    """
    def __init__(self):
        logger.info("⚙️ Initializing RAG Pipeline for Stress Test...")
        # Load the reasoning engine
        self.rag = RAGPipeline()
        self.latencies = []
        self.errors = 0
        # Mutex Lock: Ensures that multiple threads don't write to the list simultaneously, 
        # which would corrupt the performance data.
        self.lock = threading.Lock()
        
        # Load real-world queries to simulate realistic traffic variance
        self.queries = self._load_queries()

    def _load_queries(self):
        """
        Loads a diverse set of questions to prevent the GPU from 'cheating' 
        via internal caching of a single repeated query.
        """
        try:
            path = os.path.join(Config.BENCHMARK_DATA_DIR, DATA_FILE)
            if os.path.exists(path):
                df = pd.read_csv(path)
                logger.info(f"📂 Loaded {len(df)} queries for random sampling.")
                return df['question'].tolist()
        except Exception as e:
            logger.warning(f"⚠️ Could not load data file: {e}. Using hardcoded fallbacks.")
        
        return [
            "كيف أجدد جواز السفر؟", "كم رسوم تجديد الرخصة؟", 
            "طريقة نقل الملكية", "شروط العنوان الوطني", 
            "How to renew Iqama?", "Traffic violation objection"
        ]

    def _warmup(self):
        """
        GPU Warm-up Phase: The first request on a GPU is always slow as it 
        loads weights into VRAM. This method discards that outlier.
        """
        print("🔥 Warming up GPU...")
        try:
            self.rag.run("Warmup request")
            print("✅ Warmup Complete.\n")
        except:
            pass

    def _single_request(self, request_id):
        """
        Simulates a single user session.
        Calculates individual latency and updates global results thread-safely.
        """
        # Random sampling avoids sequential bias
        query = random.choice(self.queries)
        
        start = time.time()
        try:
            # Direct pipeline hit bypassing the UI layer
            response = self.rag.run(query)
            duration = time.time() - start
            
            with self.lock:
                self.latencies.append(duration)
                # Diagnostic logging
                print(f"✅ Req {request_id:02d}: {duration:.2f}s | Q: {query[:30]}...")
                
        except Exception as e:
            with self.lock:
                self.errors += 1
                print(f"❌ Req {request_id:02d}: Failed! {e}")

    def run(self):
        """
        Main Orchestrator:
        1. Warms up the hardware.
        2. Spawns threads to simulate concurrent users.
        3. Aggregates statistical metrics (P95, TPS, Median).
        """
        print("\n" + "="*50)
        print(f"🔥 STARTING STRESS TEST on NVIDIA A100")
        print(f"👥 Users: {CONCURRENT_USERS} | 🔄 Total Req: {TOTAL_REQUESTS}")
        print("="*50)
        
        # 1. Prepare Hardware
        self._warmup()
        
        start_global = time.time()
        
        # 2. Parallel Execution: Simulates simultaneous traffic
        with ThreadPoolExecutor(max_workers=CONCURRENT_USERS) as executor:
            futures = [executor.submit(self._single_request, i+1) for i in range(TOTAL_REQUESTS)]
            # Blocking call: waits for all users to finish before reporting
            for f in futures:
                f.result()

        total_time = time.time() - start_global
        
        # --- 3. Statistical Analysis ---
        if not self.latencies:
            print("❌ No successful requests to analyze.")
            return

        # Core Metrics for SLA (Service Level Agreement)
        avg_lat = statistics.mean(self.latencies)
        median_lat = statistics.median(self.latencies)
        max_lat = max(self.latencies)
        
        # P95 (95th Percentile): Standard industry metric. 
        # It shows the maximum wait time for 95% of your users.
        p95 = statistics.quantiles(self.latencies, n=20)[18] 
        
        # Throughput (TPS): How many queries the system handles per second globally.
        rps = len(self.latencies) / total_time 

        print("\n" + "="*50)
        print("📊 STRESS TEST RESULTS (Performance Report)")
        print("="*50)
        print(f"✅ Success Rate:     {len(self.latencies)}/{TOTAL_REQUESTS} ({(len(self.latencies)/TOTAL_REQUESTS)*100:.1f}%)")
        print(f"❌ Failed Requests:  {self.errors}")
        print(f"⏱️  Total Duration:   {total_time:.2f}s")
        print("-" * 30)
        print(f"⚡ Throughput (TPS): {rps:.2f} queries/sec")
        print(f"🐢 Avg Latency:      {avg_lat:.2f}s")
        print(f"🐢 P95 Latency:      {p95:.2f}s  <-- (SLA Metric)")
        print(f"🐢 Max Latency:      {max_lat:.2f}s")
        print("="*50)

        # Persistence: Save raw timing data for visualization (e.g., box plots)
        df_res = pd.DataFrame({"latency": self.latencies})
        out_path = os.path.join(Config.BENCHMARK_RESULTS_DIR, "stress_test_report.csv")
        os.makedirs(Config.BENCHMARK_RESULTS_DIR, exist_ok=True)
        df_res.to_csv(out_path, index=False)
        print(f"📄 Raw latency data saved to: {out_path}")

if __name__ == "__main__":
    tester = StressTester()
    tester.run()