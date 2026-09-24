# Lokal Hujjat Yordamchisi

Agentic RAG assistant that answers questions about local documents (codexes, regulations, manuals) — fully offline via Ollama.

Supports mixed Uzbek / Russian / English documents.

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running (`ollama serve`)
- ~10 GB free disk space (models + indexes)
- 16 GB RAM minimum, GPU recommended

---

## Quick start

### 1. Clone / open the project folder

```bash
cd local-doc-assistant
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> If PowerShell blocks the activation script, run this once:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

**Windows (cmd):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**macOS / Linux / Git Bash / WSL:**
```bash
python -m venv .venv
source .venv/bin/activate
```

Your prompt should now start with `(.venv)`.

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Pull the required Ollama models

**Windows (PowerShell / cmd):**
```powershell
ollama pull qwen2.5:7b-instruct
ollama pull bge-m3
```

**macOS / Linux / Git Bash:**
```bash
bash scripts/pull_models.sh
```

The reranker (`bge-reranker-v2-m3`) is downloaded automatically from HuggingFace on first use.

### 5. Verify the setup

```bash
python scripts/check_ollama.py
```

You should see:
```
✅ Ollama is running.
✅ All required Ollama models present: {...}
```

### 6. Add your documents

Drop PDFs, DOCX, or TXT files into `data/raw/`.

### 7. Build the index

```bash
python -m src.indexing.build_index
```

### 8. Run the app

```bash
streamlit run src/ui/streamlit_app.py
```

Open the browser tab it prints and start chatting with your documents.

---

## Project structure

```
local-doc-assistant/
├── config.yaml                    # All tunable knobs in one place
├── requirements.txt
├── .env.example                   # For any future secrets
├── README.md
│
├── data/
│   ├── raw/                       # Source PDFs/DOCX go here
│   ├── processed/                 # Extracted text + metadata (JSON)
│   └── indexes/
│       ├── chroma/                # Chroma persistent DB
│       └── bm25/                  # Pickled BM25 index
│
├── logs/
│   ├── audit.jsonl                # Every query + user + sources + retries
│   └── app.log
│
├── src/
│   ├── __init__.py
│   ├── config.py                  # Loads config.yaml, validates with pydantic
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loaders.py             # PDF (pdfplumber), DOCX, TXT
│   │   ├── normalize.py           # Latin/Cyrillic + whitespace
│   │   └── chunker.py             # Article-aware + recursive fallback
│   ├── indexing/
│   │   ├── __init__.py
│   │   ├── build_index.py         # CLI: build/rebuild Chroma + BM25
│   │   └── incremental.py         # Hash-based skip
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vector.py              # Chroma retriever
│   │   ├── bm25.py                # BM25 retriever
│   │   ├── hybrid.py              # RRF fusion
│   │   └── reranker.py            # bge-reranker-v2-m3
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── state.py               # LangGraph state schema (retry counters!)
│   │   ├── nodes.py               # retrieve, grade, rewrite, generate, etc.
│   │   ├── graders.py             # Pydantic-schema graders
│   │   ├── prompts.py             # All prompt templates
│   │   └── graph.py               # StateGraph wiring
│   ├── memory/
│   │   └── contextualize.py       # Multi-turn query rewriting
│   ├── audit/
│   │   └── logger.py              # JSONL audit writer
│   └── ui/
│       └── streamlit_app.py
│
├── eval/
│   ├── golden_set.jsonl           # 30-50 Q/A/expected-source triples
│   ├── run_eval.py                # Runs ragas
│   └── reports/
│
└── scripts/
    ├── check_ollama.py            # Verifies models are pulled
    └── pull_models.sh
```

---

## Architecture

Agentic RAG built on **LangGraph**, combining:

- **Hybrid retrieval** — Chroma vector search + BM25 lexical search, fused with Reciprocal Rank Fusion (RRF)
- **Cross-encoder reranker** — `bge-reranker-v2-m3` (multilingual)
- **Self-correcting agent** — grades retrieved docs, rewrites queries when relevance is low, checks generated answers for hallucinations and topical fit
- **Fully local** — all models run via Ollama; no data leaves the machine
- **Access control** — Chroma metadata filters for per-user/department document scoping
- **Audit log** — every query, source, and retry recorded to `logs/audit.jsonl`

Inspired by [Corrective RAG](https://arxiv.org/abs/2401.15884) and [Self-RAG](https://arxiv.org/abs/2310.11511).

---

## Configuration

All settings live in `config.yaml`. Key knobs:

- `models.*` — swap generator, grader, embedder, or reranker
- `chunking.chunk_size` / `chunk_overlap` — chunk granularity
- `retrieval.top_k_*` — how many candidates each stage keeps
- `agent.max_query_rewrites` / `max_generation_retries` — loop caps

---

## Evaluation

Build a golden set at `eval/golden_set.jsonl`, then:

```bash
python eval/run_eval.py
```

Reports `faithfulness`, `answer_relevancy`, `context_precision`, and `context_recall` via [ragas](https://github.com/explodinggradients/ragas).

---

## Troubleshooting

**"Ollama is not running"** — start it with `ollama serve` in a separate terminal (or make sure the Ollama app is running on Windows/Mac).

**PowerShell blocks `Activate.ps1`** — see step 2 above (`Set-ExecutionPolicy`).

**Slow reranker on first use** — the model (~1.2 GB) downloads from HuggingFace. Subsequent runs use the cache.

**Out-of-memory during generation** — lower `retrieval.top_k_after_rerank` in `config.yaml`, or switch `models.generator` to a smaller model like `qwen2.5:3b-instruct`.