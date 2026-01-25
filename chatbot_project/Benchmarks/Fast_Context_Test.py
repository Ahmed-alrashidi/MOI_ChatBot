# =========================================================================
# File Name: Benchmarks/Fast_Context_Test.py
# Purpose: High-Speed Automated Evaluation of Contextual Memory.
# Project: Absher Smart Assistant (MOI ChatBot)
# Efficiency: 10x faster and cheaper than using LLM-as-Judge (GPT-4).
# Logic: Uses Semantic Embedding Similarity to verify if the AI stays on topic.
# =========================================================================

import os
import sys
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from deep_translator import GoogleTranslator
import time

# Ensure the project root is in the path to allow core module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from core.rag_pipeline import RAGPipeline
from core.model_loader import ModelManager

# --- TEST SCENARIOS ---
# These are multi-turn conversations designed to test if the bot 
# remembers the subject from the first question to the last.
SCENARIOS = {
    "Passport_Renew": ["How do I renew my Saudi passport?", "How much is the fee?", "Is a medical test required?", "Can I get it delivered by post?"],
    "License_Renew": ["What are the steps to renew a driving license?", "What are the requirements?", "How much does it cost?", "Is there a fine for delay?"],
    "Iqama_Issue": ["How to issue a new resident ID (Iqama)?", "Who pays the fees?", "Is medical insurance mandatory?", "What is the penalty for delay?"],
    "Traffic_Objection": ["How can I object to a traffic violation?", "Is there a time limit?", "Do I need to pay first?", "Can I object again if rejected?"],
    "National_Address": ["How do I register my National Address?", "Is there a fee?", "Can I register for family?", "How to print the proof?"],
    "Lost_ID": ["I lost my National ID, what to do?", "Is there a fine?", "How to book appointment?", "Time to issue replacement?"],
    "Vehicle_Transfer": ["How to transfer car ownership?", "What is the price?", "Does buyer need license?", "Who pays fees?"],
    "Exit_ReEntry": ["How to issue exit re-entry visa?", "Fee for single trip?", "Can I extend outside KSA?", "Passport validity requirements?"]
}

# Supported languages for global stress testing
LANGUAGES = {
    'ar': 'Arabic', 'en': 'English', 'ur': 'Urdu', 'fr': 'French',
    'es': 'Spanish', 'de': 'German', 'zh-CN': 'Chinese', 'ru': 'Russian'
}

class FastJudge:
    """
    An automated evaluator that uses Embedding Models to calculate 
    semantic similarity between the AI's answer and the intended topic.
    """
    def __init__(self):
        logger_name = "Fast_Judge"
        print(f"⚖️ Initializing {logger_name} (Using BGE-M3 Embeddings)...")
        # Reuse the system's embedding model to save VRAM on the GPU
        self.model = ModelManager.get_embedding_model()
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculates the Cosine Similarity score between two text strings.
        1.0 = Identical meaning, 0.0 = Totally unrelated.
        """
        # Logic to handle both LangChain wrappers and native SentenceTransformers
        if hasattr(self.model, 'embed_documents'):
            raw_embeddings = self.model.embed_documents([text1, text2])
            embeddings = torch.tensor(raw_embeddings)
        else:
            embeddings = self.model.encode([text1, text2], convert_to_tensor=True)
            
        # Ensure tensor is on CPU for the final math calculation
        if embeddings.device.type == 'cuda':
            embeddings = embeddings.cpu()

        # Compute Cosine Similarity between the two vector representations
        return F.cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)).item()

    def evaluate(self, topic_keyword: str, answer: str, lang_code: str):
        """
        Determines if the AI's answer is relevant to the current scenario topic.
        
        Args:
            topic_keyword: The expected subject (e.g., 'Passport Renewal').
            answer: The AI's generated response.
            lang_code: Language of the answer for translation.
            
        Returns:
            tuple: (Similarity Score, Success Boolean)
        """
        try:
            if not answer or not isinstance(answer, str):
                return 0.0, False

            # Translate answer to English for a standardized "Judge" comparison
            if lang_code != 'en':
                answer_en = GoogleTranslator(source='auto', target='en').translate(answer[:500])
            else:
                answer_en = answer
            
            # Calculate how close the answer is to the intended topic
            score = self.calculate_similarity(topic_keyword, answer_en)
            
            # 0.35 Threshold: Based on empirical testing for the BGE-M3 model
            return score, score > 0.35
            
        except Exception as e:
            print(f"Judge Error: {e}")
            return 0.0, False

def run_benchmark():
    """
    Orchestrates the benchmark: Loops through languages and scenarios, 
    executes the RAG pipeline, and judges the results.
    """
    # Initialize the actual RAG system and the Fast Judge
    rag = RAGPipeline()
    judge = FastJudge()
    
    results = []
    
    print("\n🚀 Starting FAST Multi-Turn Context Benchmark...")
    print("--------------------------------------------------")
    
    for lang_code, lang_name in LANGUAGES.items():
        print(f"\n🌍 Testing Language: {lang_name} ({lang_code})")
        # Translator to convert English test questions into the target language
        translator = GoogleTranslator(source='en', target=lang_code)
        
        for scenario_name, questions in SCENARIOS.items():
            print(f"  📂 Scenario: {scenario_name}")
            chat_history = [] 
            
            try:
                # Convert the entire conversation scenario to the target language
                translated_qs = [translator.translate(q) for q in questions]
            except:
                continue
            
            # Construct a 'Gold Standard' keyword for comparison
            topic_keyword = scenario_name.replace("_", " ") + " fee requirements process"
            
            for i, user_q in enumerate(translated_qs):
                # 1. Execute the RAG System with current history
                start_t = time.time()
                history_for_rag = [tuple(h) for h in chat_history] 
                response = rag.run(user_q, history_for_rag)
                latency = time.time() - start_t
                
                # 2. Store the exchange in memory for the next turn
                chat_history.append((user_q, response))
                
                # 3. Automated Judging
                sim_score, passed = judge.evaluate(topic_keyword, response, lang_code)
                
                status = "✅" if passed else "❌"
                print(f"    Turn {i+1}: {status} (Sim: {sim_score:.2f}) | Latency: {latency:.2f}s")
                
                # Save metadata for the final report
                results.append({
                    "Language": lang_name,
                    "Scenario": scenario_name,
                    "Turn": i + 1,
                    "Question": user_q,
                    "Context_Score": sim_score,
                    "Context_Maintained": 1 if passed else 0,
                    "Latency": latency
                })
                
    # --- Final Reporting ---
    df_res = pd.DataFrame(results)
    out_path = os.path.join(Config.BENCHMARK_RESULTS_DIR, "fast_context_benchmark.csv")
    os.makedirs(Config.BENCHMARK_RESULTS_DIR, exist_ok=True)
    df_res.to_csv(out_path, index=False)
    
    print("\n" + "="*50)
    print("🎯 FAST BENCHMARK COMPLETE")
    print(f"📄 Report saved to: {out_path}")
    if not df_res.empty:
        # Calculate success rate for follow-up questions only (Turns 2, 3, 4)
        success_rate = df_res[df_res['Turn'] > 1]['Context_Maintained'].mean() * 100
        print(f"🏆 Context Success Rate (Follow-up turns): {success_rate:.2f}%")
    print("="*50)

if __name__ == "__main__":
    run_benchmark()