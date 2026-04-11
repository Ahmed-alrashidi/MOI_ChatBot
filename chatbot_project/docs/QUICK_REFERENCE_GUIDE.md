# 🚀 ABSHER CHATBOT - QUICK REFERENCE GUIDE

**For Claude Opus 4.6 - Fast Comprehension**

---

## ⚡ 30-SECOND SUMMARY

**What**: Saudi government services AI chatbot (Ministry of Interior — 83 services, 6 sectors)  
**How**: RAG system with hybrid retrieval (FAISS + BM25) + KG enrichment v3.0  
**Languages**: 8 (Arabic, English, Urdu, French, Spanish, German, Russian, Chinese)  
**Hardware**: NVIDIA A100-SXM4-80GB (Ibex cluster)  
**Status**: Production-ready — salted auth, telemetry, 7-test benchmark suite  
**Version**: 5.0 (April 2026) | **Team**: PGD+ at KAUST Academy  

---

## 📁 CRITICAL FILES (Read These First)

| File | Purpose | Priority |
|------|---------|----------|
| `rag_pipeline.py` | **Main RAG v5.0** — Intent guard (4 types), KG v3.0, hallucination control | 🔴 CRITICAL |
| `model_loader.py` | Singleton model manager + VRAM tracking | 🔴 CRITICAL |
| `config.py` | All settings (TEMP=0.2, REP_PENALTY=1.1, system prompt) | 🔴 CRITICAL |
| `main.py` | Command Center — 5 options + 7-benchmark submenu | 🟡 Important |
| `ingestion.py` | CSV → 160 chunks ETL (400 chars, 50 overlap) | 🟡 Important |
| `comprehensive_arena.py` | Data-grounded arena v6.0 (Qwen-32B judge) | 🟢 Reference |

---

## 🔥 KEY INNOVATIONS

### 1. Intent Guard (4 categories — <100ms)
```python
# Greetings, closings, abuse, praise → bypass RAG entirely
if is_social(query):
    return canned_response  # <100ms, no LLM call
else:
    return rag_pipeline(query)  # Full RAG ~3s
```

### 2. KG Enrichment v3.0 (87.5% price accuracy — best model)
```python
# OR-matching + article stripping for better service matching
# 55.9% exact-match on extracted price strings, 87.5% judge-evaluated
context += "\n[المصدر الرسمي]:\n- الرسوم: 300 ريال (verified from KG)"
```

### 3. Hybrid Retrieval (100% hit rate — 120/120)
```python
FAISS (semantic) + BM25 (keyword) → RRF fusion (k=60) → Top-5
```

### 4. Hallucination Control (96.7% attribution)
```python
# Dynamic max_new_tokens + system prompt: "NEVER guess a price"
# TEMPERATURE=0.2, REPETITION_PENALTY=1.1
```

### 5. Data-Grounded Evaluation (v6.0)
```python
# Judge receives KG + Master CSV + GT before scoring
# Fair metrics: ROUGE-L (LCS), per-service price matching, attribution check
```

---

## 🧠 RAG FLOW (v5.0 — 11 Steps)

```
 1. Intent Guard → Social (greeting/closing/abuse/praise)? Return canned response
 2. Lang Detect → ar | en | fr | es | de | ru | zh | ur
 3. Query Rewrite → Resolve pronouns from conversation history
 4. Normalize → Arabic char unification (Alef/Taa), diacritics removal
 5. FAISS Retrieve → Top-5 semantic matches (BGE-M3, 160 vectors)
 6. BM25 Retrieve → Top-5 keyword matches (master_chunks.csv)
 7. RRF Fusion → Merge + rerank → Final Top-5
 8. KG Enrich v3.0 → OR-match services, strip articles, inject prices/steps
 9. Dynamic Token Cap → Adjust max_new_tokens based on context length
10. Generate → ALLaM-7B (bfloat16, temp=0.2, rep_penalty=1.1)
11. TTS → Optional voice output (ar-SA-HamedNeural)
```

---

## 📊 DATA PIPELINE

```
MOI_Master_Knowledge.csv (83 services, 6 sectors, enriched RAG_Content avg 456 chars)
  ↓ schema.py validates (+ duplicate Service_Name detection)
  ↓ ingestion.py chunks (400 chars, 50 overlap)
  ├→ FAISS index (160 vectors @ 1024 dims, cosine similarity)
  └→ BM25 CSV (master_chunks.csv)

Knowledge Graph JSON (6 normalized sectors)
  ├→ Verified prices (OR-matching, article stripping)
  └→ Exact steps (injected into LLM context)

Ground Truth V2 (120 QA pairs × 8 languages)
  └→ Benchmark evaluation dataset
```

---

## ⚙️ CONFIGURATION CHEAT SHEET

### Models
```python
LLM = "ALLaM-AI/ALLaM-7B-Instruct-preview"
Embeddings = "BAAI/bge-m3"
ASR = "openai/whisper-large-v3"
Judge = "Qwen/Qwen2.5-32B-Instruct"
```

### Hyperparameters
```python
RETRIEVAL_K = 5             # Top-K docs per retriever
RRF_K = 60                  # Fusion smoothing constant
MAX_NEW_TOKENS = 1024       # LLM output cap (dynamic adjustment)
TEMPERATURE = 0.2           # Low but not frozen (was 0.05)
REPETITION_PENALTY = 1.1    # Prevents repetitive output
CHUNK_SIZE = 400            # Smaller = more precise (was 800)
CHUNK_OVERLAP = 50          # Context preservation (was 100)
```

### Hardware
```python
DEVICE = "cuda"
DTYPE = torch.bfloat16  # A100 native
TF32 = True             # 3-5x faster matmul
SDPA = True             # Flash Attention fallback
```

---

## 🎯 WHERE TO OPTIMIZE (Priority Order)

### 🔴 High Impact (Before Defense)
1. **Response Streaming** (UX: 3-5s wait → instant progressive text) — 2 hours
2. **Feedback Buttons 👍👎** (Collect user satisfaction in telemetry) — 30 min
3. **Slot-Based Memory** (Replace LLM query rewriter — more reliable) — 1 hour

### 🟡 Medium Impact (Post-Defense)
4. **NLLB-200 Translation** (Fix Urdu: score 0.099 → ~0.35 ROUGE) — 3 hours
5. **Response Caching** (LRU: <100ms for 30-40% of repeated queries) — 2 hours
6. **Auto-KG Extraction** (Parse CSV → auto-generate KG JSON) — 4 hours
7. **Expand GT to 500+** (Edge cases, misspellings, colloquial Arabic) — 8 hours

### 🟢 Nice to Have
8. **Semantic Safety Classifier** (Replace keyword-based → fine-tuned BERT)
9. **Model Quantization** (VRAM: 14GB → 4GB, fit 13B models)
10. **Arabic Dialect Normalization** ("وش" → "ماذا", 50-100 terms)

---

## 🐛 COMMON ISSUES & FIXES

### VRAM OOM
```bash
# Symptom: CUDA Out of Memory
# Fix 1: Reduce batch size
# Fix 2: Unload previous model
ModelManager.unload_all()
gc.collect()
torch.cuda.empty_cache()
```

### FAISS Index Corrupt
```bash
# Symptom: Deserialization error
# Fix: Auto-rebuilds via verify_data_integrity()
# Manual: Delete data/faiss_index/*.faiss
python main.py  # Triggers rebuild
```

### Slow Arabic Queries
```bash
# Symptom: 10s+ latency
# Fix: Check normalization is applied
normalize_arabic(query)  # MUST run before retrieval
```

### Wrong Prices
```bash
# Symptom: LLM hallucinates fees
# Fix: Verify KG enrichment is active
# Check: services_knowledge_graph.json loaded?
```

---

## 🧪 RUNNING BENCHMARKS

### Via Command Center
```bash
python main.py
# → Choose [2] Benchmark Suite
# → 7 options: Arena, Functional, Retrieval, Safety, Stress, Context, Run All
```

### Quick Arena (8 questions × 4 models = 32 tests)
```bash
# Choose: Arena → Quick
# Time: ~15 minutes
# Output: results/arena_v6_*.csv
```

### Full Arena (120 questions × 4 models = 480 tests)
```bash
# Choose: Arena → Full
# Time: ~78 minutes (35 min generation + 43 min judging)
# Output: checkpoint_phase1_*.csv + arena_v6_*.csv + summary_*.txt
```

### Other Benchmarks
```bash
# Functional:  8 scenarios (KG prices, context, safety, attribution)
# Retrieval:   120 queries → 100% hit rate, avg similarity 0.817
# Safety:      10 red-teaming attacks → 100% blocked
# Stress:      20 requests, 4 concurrent users → 0 errors
# Context:     5 multi-turn Arabic scenarios → 100% follow-up success
```

### Results Analysis
```python
import pandas as pd
df = pd.read_csv("results/arena_v6_*.csv")

# Leaderboard
df.groupby('Model')[['Judge_Score', 'ROUGE_L', 'Latency']].mean()

# Per-language breakdown
df.pivot_table(values='ROUGE_L', index='Model', columns='Lang', aggfunc='mean')

# Failure cases
df[df['Judge_Score'] < 5][['Model', 'Lang', 'Question', 'Judge_Reason']]
```

---

## 📈 PERFORMANCE METRICS (Full 480-Test Benchmark — 77.8 min)

### 🏆 Final Leaderboard (Judge + Objective)

| Rank | Model | Judge (0-10) | ROUGE-L | Price Acc | Attribution | Latency |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **Gemma-2-9B** | **7.46** | **0.405** | 85.0% | 92.5% | 5.26s |
| 🥈 | **ALLaM-7B** ⭐ | **7.33** | 0.373 | **87.5%** | **96.7%** | 3.86s |
| 🥉 | **Qwen-2.5-7B** | **7.27** | 0.393 | 83.3% | 92.5% | 4.37s |
| 4 | **Llama-3.1-8B** | **6.30** | 0.358 | 82.3% | 77.5% | **3.52s** |

⭐ Excluding Urdu: **ALLaM and Gemma tie at 7.88**. ALLaM wins on price (87.5%), attribution (96.7%), and speed (27% faster).

### 🌍 Per-Language Judge Scores

| Model | AR | EN | FR | DE | ZH | RU | ES | UR |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALLaM-7B** ⭐ | 7.9 | 7.7 | **8.1** | 7.9 | **8.1** | 7.7 | **7.8** | 3.5 |
| **Gemma-2-9B** | **8.5** | **7.9** | 8.0 | **8.1** | 7.7 | **7.8** | 7.1 | 4.5 |
| **Qwen-2.5-7B** | 8.1 | 7.4 | 7.3 | 7.2 | 7.1 | 7.7 | 7.7 | **5.7** |
| **Llama-3.1-8B** | 7.9 | 7.4 | 7.5 | 7.8 | 7.4 | 6.3 | ⛔ 0.6 | 5.5 |

### 📊 Score Distribution

| Model | Zero (0) | Low (<5) | Mid (5-7.9) | High (8+) |
|:---|:---:|:---:|:---:|:---:|
| **ALLaM-7B** ⭐ | 4 | 12 | 35 | **73 (61%)** |
| **Gemma-2-9B** | 6 | 11 | 30 | **79 (66%)** |
| **Qwen-2.5-7B** | 4 | 9 | 48 | 63 (52%) |
| **Llama-3.1-8B** | **22** | **28** | 27 | 65 (54%) |

### System Metrics
| Metric | Value |
|--------|-------|
| **Retrieval Hit Rate** | 100% (120/120) |
| **Avg Similarity** | 0.817 |
| **Safety Score** | 100% (10/10 attacks blocked) |
| **Stress Test** | 20/20 requests, 0 errors |
| **Intent Guard Hit Rate** | ~30% of queries |
| **VRAM Usage** | 14GB (ALLaM-7B bfloat16) |
| **Latency P50 / P95** | 3.0s / 10.5s (ALLaM) |
| **Total Failures** | 36 zero-scores (Llama: 22, others: 14) |

### Per-Language ROUGE-L (ALLaM-7B)
| AR | EN | FR | ES | DE | RU | ZH | UR |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.448 | 0.412 | 0.408 | 0.386 | 0.437 | 0.381 | 0.414 | **0.099** |

---

## 🔐 SECURITY CHECKLIST

- [x] Salted SHA-256 password hashing (v5.0)
- [x] Backward compatibility with legacy hashes
- [x] Change password option in auth manager
- [x] Minimum 6-char password enforcement
- [ ] Upgrade to bcrypt/Argon2 (recommended)
- [x] Per-user telemetry logs (JSONL)
- [x] Username sanitization (path traversal prevention)
- [ ] IP anonymization (GDPR)
- [x] Auth gate on Gradio
- [ ] HTTPS reverse proxy
- [x] `.gitignore` excludes secrets, users_db.json, *.svg
- [ ] Encrypt logs at rest

---

## 🚀 DEPLOYMENT COMMANDS

### Initial Setup
```bash
# 1. Environment
conda create -n absher python=3.9
conda activate absher
pip install -r requirements.txt --break-system-packages

# 2. Configure
echo "HF_TOKEN=hf_xxxxx" > .env

# 3. Add admin user
python utils/auth_manager.py
# → Add User: admin / password123

# 4. Launch
python main.py
# → Open: http://0.0.0.0:7860
```

### Production (systemd)
```bash
# Create service file
sudo nano /etc/systemd/system/absher.service

[Unit]
Description=Absher Smart Assistant
After=network.target

[Service]
User=rashidah
WorkingDirectory=/ibex/user/rashidah/projects/MOI_ChatBot/chatbot_project
ExecStart=/home/rashidah/miniconda3/envs/absher/bin/python main.py
Restart=always

[Install]
WantedBy=multi-user.target

# Enable & start
sudo systemctl enable absher
sudo systemctl start absher
sudo systemctl status absher
```

---

## 📊 TESTING SCENARIOS

### Test 1: Basic Arabic Query
```bash
Input: "ما رسوم إصدار الجواز؟"
Expected: "بناءً على السجلات الرسمية لمنصة أبشر، رسوم إصدار الجواز السعودي هي:
- 300 ريال (صلاحية 5 سنوات)
- 600 ريال (صلاحية 10 سنوات)"
```

### Test 2: English Query
```bash
Input: "How much does a passport cost?"
Expected: "Based on official Absher records, Saudi passport fees are:
- 300 SAR (5 years validity)
- 600 SAR (10 years validity)"
```

### Test 3: Urdu Query (⚠️ Known Weakness)
```bash
Input: "پاسپورٹ کی فیس کیا ہے؟"
Expected: Urdu response with correct prices
Actual: ALLaM generates broken Urdu (ROUGE 0.099)
Fix: NLLB-200 T-S-T translation layer (planned)
```

### Test 4: Greeting (Intent Guard)
```bash
Input: "السلام عليكم"
Expected: "وعليكم السلام ورحمة الله، كيف يمكنني مساعدتك؟"
Latency: <500ms (no RAG)
```

### Test 5: Multi-Turn Context
```bash
Turn 1: "ما رسوم إصدار الجواز؟"
Response: "300 ريال (5 سنوات)، 600 ريال (10 سنوات)"

Turn 2: "كيف أجدده؟"
Expected: Should understand "أجدده" → "تجديد الجواز" (via query rewriting)
```

---

## 🎓 LEARNING PATH FOR CLAUDE OPUS

### Phase 1: Core Understanding (30 min)
1. Read: `ABSHER_PROJECT_TECHNICAL_REPORT.md` (Sections 1-6)
2. Scan: `rag_pipeline.py` (focus on `run()` method)
3. Check: `config.py` (understand all paths/models)

### Phase 2: Data Flow (20 min)
4. Trace: `ingestion.py` → `vector_store.py` → FAISS
5. Inspect: `MOI_Master_Knowledge.csv` (data structure)
6. Review: `services_knowledge_graph.json` (KG facts)

### Phase 3: Evaluation (15 min)
7. Run: `python main.py` → Option 2 → Quick Arena
8. Analyze: `results/arena_v6_*.csv`
9. Identify: Low-scoring queries (Urdu, Spanish for Llama)

### Phase 4: Optimization (Deep Dive)
10. Profile: Where's the bottleneck? (use `time.time()`)
11. Experiment: Different hyperparameters
12. Benchmark: Measure impact

---

## 💡 QUICK WINS (< 2 hours each)

### Win 1: Add Streaming (2 hours — biggest UX improvement)
```python
# In rag_pipeline.py
from transformers import TextIteratorStreamer
from threading import Thread

streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
thread = Thread(target=model.generate, kwargs={..., "streamer": streamer})
thread.start()
for token in streamer:
    yield partial_response  # Gradio receives incremental updates
```

### Win 2: Feedback Buttons (30 min)
```python
# In app.py
btn_up = gr.Button("👍")
btn_down = gr.Button("👎")
btn_up.click(lambda h: telemetry.log_feedback(h, "positive"), [chatbot])
```

### Win 3: Slot-Based Memory (1 hour — replaces LLM rewriter)
```python
# Rule-based pronoun resolution — no LLM call needed
class ConversationMemory:
    def resolve(self, query):
        if "رسومه" in query and self.last_service:
            return query.replace("رسومه", f"رسوم {self.last_service}")
        return query
```

### Win 4: Cache Hot Queries (1 hour)
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_retrieve(query_hash):
    return vector_db.retrieve(query)  # <100ms for repeated queries
```

---

## 🔬 ADVANCED DEBUGGING

### Enable Full Logging
```python
# config.py
DEBUG_MODE = True  # Enables all debug prints

# Run with verbose
python main.py 2>&1 | tee debug.log
```

### Profile Latency
```python
import time

# In rag_pipeline.py
t1 = time.time()
docs = retrieve(query)
print(f"Retrieval: {time.time()-t1:.2f}s")

t2 = time.time()
response = llm.generate(...)
print(f"Generation: {time.time()-t2:.2f}s")
```

### Inspect Retrieved Docs
```python
# After retrieval in RAGPipeline.run()
for i, doc in enumerate(final_docs):
    print(f"Doc {i}: {doc.page_content[:100]}...")
    print(f"Metadata: {doc.metadata}")
```

---

## 📚 EXTERNAL RESOURCES

- **LangChain Docs**: https://python.langchain.com/docs/
- **FAISS Tutorial**: https://github.com/facebookresearch/faiss/wiki
- **BGE-M3 Paper**: https://arxiv.org/abs/2402.03216
- **ALLaM Info**: https://sdaia.gov.sa/en/SDAIA/about/Pages/ALLaMInitiative.aspx
- **Gradio Docs**: https://gradio.app/docs/

---

**END OF QUICK REFERENCE**

**Total Reading Time**: 15 minutes  
**Total Setup Time**: 30 minutes  
**Time to First Benchmark**: 45 minutes  

---

**REMEMBER**: 
- 🔴 Focus on `rag_pipeline.py` for intelligence
- 🟡 Check `config.py` for all settings
- 🟢 Run benchmarks to validate changes