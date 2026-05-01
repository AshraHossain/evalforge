# EvalForge 🔬

**LLM Evaluation & Reliability Platform** — powered by local Ollama (gemma4:26b)

> Point EvalForge at any LLM app. It auto-generates adversarial test cases, runs them, scores the outputs, detects regressions across runs, and produces a structured reliability report with remediation recommendations. Zero API cost — everything runs locally via Ollama.

---

## Architecture

```
                         EvalForge LangGraph Pipeline
┌─────────────┐    ┌─────────────────────────────────────────────────────────┐
│  FastAPI     │    │                                                         │
│  REST + WS   │──▶│  testgen ──▶ execute ──▶ judge ──▶ memory ──▶ report   │
│  JWT Auth    │    │                                       │                 │
└─────────────┘    │                             hallucination > 30%?        │
                    │                               ↓              ↓         │
┌─────────────┐    │                          hitl_gate      report (auto)   │
│  Redis (ARQ) │    │                               ↓                        │
│  Job Queue   │◀───│                          (human approves)              │
└─────────────┘    └─────────────────────────────────────────────────────────┘
                              ↑ All LLM calls use local Ollama gemma4:26b
┌─────────────┐
│  PostgreSQL  │    Stores: targets, eval_jobs, test_cases, results
│  + Alembic   │    + LangGraph checkpoint state (HITL persistence)
└─────────────┘
```

---

## Quick Start (Phase 1 — CLI, no Docker)

### Prerequisites
```bash
# 1. Python 3.11+
python --version

# 2. Install Ollama
# https://ollama.com/download

# 3. Pull the model
ollama pull gemma4:26b

# 4. Start Ollama server
ollama serve
```

### Install & Run
```bash
# Clone and install
git clone https://github.com/yourusername/evalforge
cd evalforge
pip install -e ".[dev]"

# Copy config
cp .env.example .env

# Run eval against any HTTP endpoint
python run_eval.py \
  --endpoint http://localhost:8001/query \
  --name "My RAG App" \
  --domain finance \
  --num-cases 5
```

**Expected output:**
```
============================================================
  EvalForge — Evaluation Run
  Job ID:   cli-a1b2c3d4
  Target:   My RAG App
  Endpoint: http://localhost:8001/query
  Cases:    5
  Model:    Ollama gemma4:26b (local)
============================================================

📋 Phase 1: Generating test cases...

============================================================
  EVAL COMPLETE  (47.3s)
============================================================
  ✅ Badge:              RELIABLE
  📊 Reliability Score: 84.2/100
  ✓  Pass Rate:         80%
  🧪 Test Cases:        5
  🌀 Hallucinations:    20%
  ⚡ Avg Latency:       523ms
============================================================
```

---

## Full Stack (Phase 3 — API + Worker + DB)

```bash
# Start PostgreSQL + Redis
docker compose up -d

# Run DB migrations
alembic upgrade head

# Start API server
uvicorn api.main:app --reload --port 8000

# Start worker (separate terminal)
python worker/worker.py
```

### API Usage
```bash
# 1. Get token
TOKEN=$(curl -s -X POST http://localhost:8000/auth/token \
  -d "username=admin&password=evalforge123" | jq -r .access_token)

# 2. Register your target
TARGET_ID=$(curl -s -X POST http://localhost:8000/api/v1/targets \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "RAG Stock Analyzer",
    "config": {
      "endpoint": "http://localhost:8001/query",
      "request_template": {"query": "__QUESTION__"},
      "response_path": "$.answer"
    }
  }' | jq -r .id)

# 3. Submit eval job
JOB_ID=$(curl -s -X POST http://localhost:8000/api/v1/jobs \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"target_id\": \"$TARGET_ID\", \"num_test_cases\": 10}" | jq -r .id)

# 4. Stream results (WebSocket)
# ws://localhost:8000/ws/jobs/$JOB_ID

# 5. Poll for report
curl -s http://localhost:8000/api/v1/jobs/$JOB_ID \
  -H "Authorization: Bearer $TOKEN" | jq .report.badge
```

---

## File Structure & Purpose

```
evalforge/
│
├── config.py                  ← Single source of truth for all settings
├── run_eval.py                ← Phase 1 CLI runner (no server needed)
│
├── api/
│   ├── main.py                ← FastAPI app + CORS + lifespan
│   ├── auth.py                ← JWT auth (POST /auth/token)
│   ├── routers/
│   │   ├── targets.py         ← POST/GET /api/v1/targets
│   │   ├── jobs.py            ← POST/GET /api/v1/jobs + HITL approve
│   │   └── websocket.py       ← WS /ws/jobs/{id} — real-time streaming
│   └── schemas/
│       ├── target.py          ← TargetConfig, TargetCreate/Read
│       ├── job.py             ← TestCase, Result, Score, JobCreate/Read
│       └── report.py          ← EvalReport, MemoryContext, RegressionDetail
│
├── agents/                    ← LangGraph nodes (one agent per file)
│   ├── orchestrator.py        ← EvalState, StateGraph, run_eval_job()
│   ├── testgen_agent.py       ← Generates test cases via Ollama
│   ├── execution_agent.py     ← Calls target via HTTP (no LLM)
│   ├── judge_agent.py         ← Routes to scorer_router
│   ├── memory_agent.py        ← Persist + regression detection
│   ├── report_agent.py        ← Builds EvalReport via Ollama
│   └── hitl_node.py           ← Pauses graph for human review
│
├── evaluation/
│   ├── llm_judge.py           ← Ollama gemma4:26b as-judge (Phase 1)
│   ├── lora_judge.py          ← DeBERTa LoRA stub (Phase 4)
│   ├── scorer_router.py       ← Routes LoRA vs LLM based on confidence
│   └── ragas_scorer.py        ← RAGAS metrics for RAG targets
│
├── memory/
│   ├── store.py               ← PostgreSQL read/write
│   ├── regression_detector.py ← Flapping + regression analysis
│   └── trend_analyzer.py      ← Pass-rate trend (improving/degrading)
│
├── worker/
│   ├── tasks.py               ← ARQ job functions (called by worker)
│   └── worker.py              ← ARQ worker process + WorkerSettings
│
└── db/
    ├── models.py              ← SQLAlchemy ORM (Target, EvalJob, etc.)
    ├── session.py             ← Async engine + get_db() dependency
    └── migrations/env.py      ← Alembic migration runtime
```

---

## How Ollama Integration Works

Every LLM call in EvalForge uses `ChatOllama` from `langchain-ollama`:

```python
from langchain_ollama import ChatOllama
from config import settings

llm = ChatOllama(
    model=settings.OLLAMA_MODEL,          # "gemma4:26b"
    base_url=settings.OLLAMA_BASE_URL,    # "http://localhost:11434"
    temperature=0.0,
    format="json",                         # Forces JSON output mode
)
```

**Three places Ollama is called:**

| Agent | Purpose | Temperature |
|---|---|---|
| `testgen_agent.py` | Generate diverse test cases | 0.9 (creative) |
| `llm_judge.py` | Score (question, response) pairs | 0.0 (deterministic) |
| `report_agent.py` | Generate recommendations | 0.3 (balanced) |

RAGAS metrics (for RAG targets) also use Ollama via `LangchainLLMWrapper`.

---

## Running Tests

```bash
# Unit tests (no Ollama or DB required)
pytest tests/unit/ -v

# Integration tests (no Ollama or DB — all mocked)
pytest tests/integration/ -v

# All tests
pytest -v
```

---

## Build Phases

| Phase | Goal | Status |
|---|---|---|
| 1 | Core pipeline (CLI, no streaming) | ✅ Complete |
| 2 | Memory, regression detection, HITL | ✅ Complete |
| 3 | FastAPI, WebSocket streaming, ARQ | ✅ Complete |
| 4 | Fine-tuned LoRA judge | 🔲 Stub ready |
| 5 | Portfolio polish, self-eval CI | 🔲 Planned |

---

## Interview Talking Points

**"Walk me through the architecture."**
LangGraph StateGraph with 5 agents in sequence. TestGen calls Ollama to generate adversarial test cases. Execution Agent hits the target via HTTP (black-box approach — works with any LLM app). Judge Agent scores with local Ollama. Memory Agent persists to PostgreSQL and runs regression detection. Report Agent synthesises scores into a structured reliability report.

**"Why local Ollama instead of OpenAI?"**
Zero cost, zero latency to external APIs, full data privacy. The architecture is provider-agnostic — swap `ChatOllama` for `ChatOpenAI` in one line.

**"How does the HITL gate work?"**
LangGraph's `interrupt_before` feature. If hallucination rate exceeds threshold, the graph raises `NodeInterrupt` at `hitl_gate`, checkpoints full state to PostgreSQL, and waits. `POST /jobs/{id}/approve` resumes the graph from exactly that checkpoint.

**"What's the LoRA judge in Phase 4?"**
Fine-tune DeBERTa-v3 with PEFT/LoRA for multi-label classification (factual, relevant, complete, safe). Routes the 80%+ easy cases to the fast classifier; reserves Ollama for the ambiguous ones. Compare F1 before/after vs zero-shot baseline.
