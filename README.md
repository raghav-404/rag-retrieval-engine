# rag-retrieval-engine

A local-first Retrieval-Augmented Generation system for document question answering, built with FastAPI, FAISS, sentence-transformers, Ollama, and a small Streamlit demo UI.

It focuses on the practical parts of a small RAG system: document ingestion, chunking, embedding, hybrid retrieval, grounding, local model orchestration, and lightweight evaluation.

## Overview

Many RAG demos are dense-only retrieval wrapped in a chat UI. This repo is more deliberate about the retrieval stack and the system tradeoffs:

- combines dense retrieval with BM25-style sparse scoring
- rewrites the user query before retrieval to improve recall
- separates ingestion from serving so the data pipeline is explicit
- keeps the serving layer simple while preserving a real API boundary
- runs entirely locally with Ollama, which makes model behavior easier to test and reason about
- includes a small test suite covering chunking, ingestion, retrieval shape, and API health

## What It Does

Given a folder of `.txt` documents, the system:

1. cleans and chunks the documents
2. embeds each chunk with `all-MiniLM-L6-v2`
3. stores vectors in FAISS and chunk metadata in JSON
4. rewrites the user query for better retrieval
5. combines dense similarity with BM25-style sparse scoring
6. builds a grounded context window from the top results
7. asks a local Ollama model to answer using only that context
8. computes a lightweight answer grounding score and retries with the original query when needed

## Stack

- Python 3.12+
- FastAPI
- Streamlit
- FAISS
- sentence-transformers
- Ollama
- pytest

## Architecture

```text
docs/*.txt
  -> ingestion.py
  -> faiss.index + metadata.json
  -> retrieval.py
  -> FastAPI /ask
  -> Streamlit demo UI
```

Core modules:

- `src/rag_retrieval_engine/config.py`: local settings for paths, model names, top-k, and Ollama URL
- `src/rag_retrieval_engine/ingestion.py`: document loading, chunking, embedding, and index building
- `src/rag_retrieval_engine/retrieval.py`: artifact loading, hybrid retrieval, context assembly, and answer generation
- `src/rag_retrieval_engine/api/app.py`: FastAPI app with `/health` and `/ask`
- `src/rag_retrieval_engine/ui/streamlit_app.py`: lightweight demo client

Retrieval pipeline:

```text
question
  -> query rewrite
  -> embedding lookup in FAISS
  -> BM25-style lexical scoring
  -> score fusion
  -> top-k context assembly
  -> Ollama answer generation
  -> lightweight grounding check
```

Thin root entrypoints are kept so local commands stay simple:

- `python ingest.py`
- `uvicorn backend:app --reload`
- `streamlit run app.py`

## Applied AI Notes

- Dense retrieval handles semantic similarity, while the BM25-style component helps recover lexical matches that embeddings can miss.
- Query rewriting is intentionally small and cheap, but it improves the chances of retrieving the right chunks for underspecified questions.
- The answer layer is grounded against retrieved context rather than treating the LLM as the source of truth.
- The evaluation step is simple, but it reflects the right instinct: retrieval quality should influence generation behavior.
- The whole stack is local-first, which is useful for debugging, iteration speed, privacy-sensitive use cases, and cost control.

## Design Choices

- The project emphasizes retrieval-system behavior, not just prompt orchestration.
- The code is modular without introducing extra infrastructure.
- FastAPI is treated as the real interface; Streamlit is just a demo surface.
- The app does not assume cloud infrastructure, vector databases, or extra services unless they are actually needed.
- Error handling focuses on realistic failure modes like missing retrieval artifacts or bad requests.

## Local Setup

### 1. Install dependencies

Using `uv`:

```bash
uv sync
```

Or using the existing virtualenv approach:

```bash
pip install -r requirements.txt
```

### 2. Start Ollama

Pull the model:

```bash
ollama pull qwen2.5:7b
```

Run Ollama locally so the API is available at `http://localhost:11434`.

### 3. Add documents

Place `.txt` files in `docs/`.

Sample files included in this repo:

- `docs/sample_company_overview.txt`
- `docs/sample_faq.txt`
- `docs/sample_product_specs.txt`
- `docs/sample_support_policy.txt`

### 4. Build the retrieval artifacts

```bash
python ingest.py
```

This creates:

- `faiss.index`
- `metadata.json`

### 5. Run the API

```bash
uvicorn backend:app --reload
```

### 6. Run the demo UI

```bash
streamlit run app.py
```

## How To Test

Automated tests:

```bash
python -m pytest --basetemp .pytest-tmp -p no:cacheprovider
```

Manual API smoke test:

```bash
curl http://127.0.0.1:8000/health
```

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"question\":\"What is the support response window?\"}"
```

## Example API Response

```json
{
  "answer": "Support requests are typically handled within the documented response window.",
  "sources": ["sample_support_policy.txt"],
  "rewritten_query": "support response window policy",
  "eval_score": 0.312
}
```

## Tradeoffs

This repo is intentionally strong on retrieval clarity and local reproducibility, not breadth.

What it does well:

- clear end-to-end RAG flow
- understandable retrieval and serving boundaries
- fully local model + retrieval stack
- easy experimentation with chunking, retrieval, and prompting behavior

What is still prototype-level:

- `.txt`-only ingestion
- heuristic reranking and lightweight grounding score
- no auth, persistence layer, or background jobs
- no production observability or deployment story

## What I Would Build Next

- support Markdown and PDF ingestion
- cache embeddings and artifact metadata more explicitly
- add retrieval diagnostics such as score breakdowns and hit inspection
- benchmark chunking and retrieval settings on a small evaluation set
- improve groundedness evaluation beyond token overlap
- optionally swap FAISS + local files for a more scalable artifact layer if the use case required it

## Summary

A small, local-first hybrid RAG system with explicit ingestion, retrieval, and serving boundaries. The implementation stays simple, but still captures the parts of the workflow that matter most in practice: retrieval quality, grounded generation, and repeatable local testing.
