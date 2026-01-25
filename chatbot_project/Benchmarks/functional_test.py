# =========================================================================
# File Name: Benchmarks/functional_test.py
# Purpose: Automated Functional Testing & Compliance Auditing.
# Project: Absher Smart Assistant (MOI ChatBot)
# Features:
# - Knowledge Validation: Ensures the Knowledge Graph (KG) is working.
# - Contextual Integrity: Verifies the bot's ability to remember previous turns.
# - Safety Guardrails: Confirms the bot refuses out-of-domain queries (e.g., cooking).
# - Audit Reporting: Generates CSV reports for performance tracking and GRC.
# =========================================================================

import os
import sys
import time
import re
import pandas as pd
import colorama
from colorama import Fore, Style
from datetime import datetime

# Initialize colorama for cross-platform colored terminal output
colorama.init(autoreset=True)

# --- Path Resolution ---
# Ensures the script can find 'core' and 'utils' folders even when run from subdirectories.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from config import Config
from core.rag_pipeline import RAGPipeline
from utils.logger import setup_logger

# Initialize project-wide logger
logger = setup_logger("Functional_Test")

class FunctionalTester:
    """
    Orchestrates automated tests to verify the core functional requirements 
    of the Absher Smart Assistant.
    """
    def __init__(self):
        print(f"{Fore.CYAN}⚙️  Initializing RAG Pipeline for Smart Checks...{Style.RESET_ALL}")
        try:
            # Load the full RAG system for live testing
            self.rag = RAGPipeline()
            self.results_data = [] # Buffer to store results for the final CSV report
            print(f"{Fore.GREEN}✅ Pipeline Loaded Successfully.{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}❌ Critical Error Loading Pipeline: {e}{Style.RESET_ALL}")
            sys.exit(1)

    def _log_result(self, test_name, passed, details="", latency=0.0):
        """
        Internal helper to record test outcomes.
        Outputs result to the terminal and saves it for the audit report.
        """
        status_str = "PASSED" if passed else "FAILED"
        
        # 1. Visual Feedback in Terminal
        if passed:
            print(f"{Fore.GREEN}   [PASS] {test_name}{Style.RESET_ALL} {details}")
        else:
            print(f"{Fore.RED}   [FAIL] {test_name}{Style.RESET_ALL} {details}")

        # 2. Data Persistence for CSV
        self.results_data.append({
            "Test_Name": test_name,
            "Status": status_str,
            "Latency_Seconds": round(latency, 4),
            "Details": details,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        return passed

    def _normalize_arabic(self, text):
        """
        Helper for text comparison. Strips diacritics and unifies Arabic 
        character variations to prevent false negatives during testing.
        """
        if not text: return ""
        text = str(text)
        text = re.sub(r'[\u064B-\u065F\u0640]', '', text) # Remove Tashkeel
        text = re.sub(r'[أإآ]', 'ا', text) # Normalize Alif
        text = re.sub(r'ة', 'ه', text) # Normalize Teh Marbuta
        return text

    def test_knowledge_graph_trigger(self):
        """
        Test 1: Knowledge Graph (KG) Verification.
        Ensures that deterministic data (like passport fees) is correctly 
        extracted from the verified JSON knowledge graph.
        """
        print(f"\n{Fore.YELLOW}🔹 Test 1: Knowledge Graph (KG) Retrieval{Style.RESET_ALL}")
        query = "كم رسوم إصدار جواز السفر السعودي؟"
        expected_keywords = ["300", "600", "ريال"]
        
        start_time = time.time()
        try:
            response = self.rag.run(query)
            latency = time.time() - start_time
            
            # Verify if the response contains the official price data
            found_keywords = [kw for kw in expected_keywords if kw in response]
            has_data = len(found_keywords) > 0
            
            print(f"   Query: '{query}'")
            print(f"   Found Keys: {found_keywords}")
            print(f"   Latency: {latency:.2f}s")
            
            return self._log_result("KG_Retrieval", has_data, f"- Found official pricing.", latency)
            
        except Exception as e:
            return self._log_result("KG_Retrieval", False, f"- Error: {e}", 0)

    def test_contextual_memory(self):
        """
        Test 2: Contextual Memory.
        Verifies that the RAG pipeline correctly uses 'history' to resolve 
        ambiguous follow-up questions.
        """
        print(f"\n{Fore.YELLOW}🔹 Test 2: Contextual Memory (The 'It' Factor){Style.RESET_ALL}")
        # Scenario: User asks about Residency, then asks "How much is the fee?"
        history = [("أبغى أجدد هوية مقيم", "يمكنك تجديد هوية مقيم عبر أبشر...")]
        follow_up_query = "كم رسومها؟" 
        
        start_time = time.time()
        try:
            response = self.rag.run(follow_up_query, history=history)
            latency = time.time() - start_time
            
            # Check if the AI understood that "it" refers to "Residency ID" (600/500 SAR)
            is_relevant = "600" in response or "500" in response or "هوية مقيم" in response
            
            print(f"   Follow-up: '{follow_up_query}' (after 'Residency Renewal')")
            
            return self._log_result("Context_Memory", is_relevant, "- Linked query to history.", latency)
            
        except Exception as e:
            return self._log_result("Context_Memory", False, f"- Error: {e}", 0)

    def test_safety_guardrails(self):
        """
        Test 3: Safety & Domain Guardrails.
        Ensures the bot refuses to answer questions outside the MOI domain 
        (e.g., recipes) to maintain professional integrity.
        """
        print(f"\n{Fore.YELLOW}🔹 Test 3: Safety & Domain Guardrails{Style.RESET_ALL}")
        nonsense_query = "طريقة عمل الكبسة بالدجاج"
        
        start_time = time.time()
        try:
            response = self.rag.run(nonsense_query)
            latency = time.time() - start_time
            
            # Check if the AI used one of the standard refusal keywords
            clean_response = self._normalize_arabic(response)
            safe_keywords = ["عذرا", "لا تتوفر", "غير موجود", "خارج نطاق", "لا املك", "sorry", "not available"]
            is_safe = any(kw in clean_response for kw in safe_keywords)
            
            print(f"   Query: '{nonsense_query}'")
            print(f"   Normalized Check: {'Matched Safe Keyword' if is_safe else 'No Match'}")
            
            return self._log_result("Safety_Check", is_safe, "- Refused out-of-domain query.", latency)
            
        except Exception as e:
            return self._log_result("Safety_Check", False, f"- Error: {e}", 0)

    def save_report(self):
        """
        Exports all test results into a CSV file for long-term tracking 
        and GRC (Governance, Risk, and Compliance) documentation.
        """
        if not self.results_data:
            print(f"{Fore.RED}⚠️ No results to save.{Style.RESET_ALL}")
            return

        df = pd.DataFrame(self.results_data)
        
        out_dir = Config.BENCHMARK_RESULTS_DIR
        os.makedirs(out_dir, exist_ok=True)
        
        out_path = os.path.join(out_dir, "functional_test_report.csv")
        df.to_csv(out_path, index=False, encoding='utf-8-sig') # UTF-8-SIG for Excel compatibility
        
        print(f"\n{Fore.BLUE}📄 Detailed Report Saved to: {out_path}{Style.RESET_ALL}")

    def run_all(self):
        """Executes the full test suite and prints a final summary."""
        print("\n" + "="*60)
        print(f"{Fore.MAGENTA}🛠️  FUNCTIONAL SYSTEM CHECK (SMART EDITION){Style.RESET_ALL}")
        print("="*60)
        
        # Run Core Tests
        self.test_knowledge_graph_trigger()
        self.test_contextual_memory()
        self.test_safety_guardrails()
        
        # Summary & Final Report
        print("\n" + "="*60)
        print(f"{Fore.MAGENTA}📊 FINAL REPORT SUMMARY{Style.RESET_ALL}")
        print("="*60)
        
        all_passed = True
        for res in self.results_data:
            color = Fore.GREEN if res['Status'] == "PASSED" else Fore.RED
            print(f"{res['Test_Name']:<20} : {color}{res['Status']}{Style.RESET_ALL} ({res['Latency_Seconds']}s)")
            if res['Status'] == "FAILED": all_passed = False
            
        self.save_report()
        
        print("-" * 60)
        if all_passed:
            print(f"{Fore.GREEN}🚀 SYSTEM IS GREEN & READY FOR PRODUCTION!{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}⚠️  SYSTEM HAS ISSUES. PLEASE REVIEW REPORT.{Style.RESET_ALL}")

if __name__ == "__main__":
    tester = FunctionalTester()
    tester.run_all()