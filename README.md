# Hybrid RAG Retrieval Engine

A document question-answering service with hybrid retrieval and source citations. It makes each retrieval step visible, compares retrieval methods on labeled questions, and returns an answer with chunk citations and timing.

## Architecture

```text
docs/*.txt → token-aware chunks → normalized MiniLM embeddings → FAISS
                         └───────────────────────────────→ BM25
question → independent FAISS and BM25 searches → weighted RRF (c=60)
         → optional CrossEncoder rerank → labeled context
         → ChatPromptTemplate → ChatGroq → answer, sources, metrics, request ID
```

LangChain supplies `Document`, the token-aware splitter, Hugging Face embeddings, FAISS and BM25 retrievers, `EnsembleRetriever`, prompt templates, and `ChatGroq`. The service keeps retrieval, optional reranking, context construction, and generation as explicit steps. FastAPI loads the models and indexes once at startup. Streamlit is an optional demo UI.

## Features and stack

- Python 3.12, `uv`, FastAPI, Pydantic, LangChain, Sentence Transformers, FAISS, BM25, Groq, pytest, and optional Streamlit.
- Stable chunk IDs and a manifest that checks source hashes, artifact hashes, chunk settings, embedding model, and vector dimensions.
- Dense, sparse, and hybrid retrieval modes. Dense and BM25 search the complete corpus independently; RRF combines ranks rather than incompatible raw scores.
- Optional query rewrite after a documented poor-retrieval check and optional CrossEncoder reranking. Both are off by default.
- Structured `/ask` responses with citations, timings, and request IDs. Provider, timeout, retrieval, and invalid-request failures have distinct HTTP responses.

## Setup and run

```bash
uv sync --locked --extra dev
cp .env.example .env
```

Edit `.env` with your Groq API key and a chat model available to your Groq account. Load it into the shell before starting the API:

```bash
set -a
source .env
set +a
uv run python ingest.py
uv run uvicorn backend:app --reload
```

Add or change `.txt` files in `docs/`, then rerun ingestion. The generated FAISS index, chunk metadata, and manifest stay local and are ignored by Git. The first ingestion downloads the public MiniLM embedding model. API startup needs the index and Groq settings; it fails with an explanatory error if they are missing or incompatible.

In another terminal, the optional UI runs with `uv run streamlit run app.py`.

## API

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What are the support hours?","mode":"hybrid"}'
```

`POST /retrieve` accepts the same question and mode plus an optional `k` for inspecting retrieved chunks. `/ask` returns this shape; answer text and timing depend on the run:

```json
{
  "answer": "...",
  "sources": [{"source": "sample_support_policy.txt", "chunk_id": "sample_support_policy_txt_chunk_0001", "rank": 1}],
  "metrics": {"retrieval_ms": 0.0, "generation_ms": 0.0, "total_ms": 0.0, "retrieval_mode": "hybrid", "reranker_enabled": false, "query_rewritten": false},
  "request_id": "..."
}
```

## Tests and evaluation

```bash
uv run pytest -q
uv run python evaluate.py
```

The default evaluation calls only retrieval, so it needs no Groq key or credits. Six labeled questions live in `evaluation/questions.jsonl`; the measured report is in `evaluation/results/retrieval_baseline.json`. Use `RAG_RERANKER_ENABLED=true uv run python evaluate.py --output evaluation/results/reranked.json` to add the reranked mode without replacing the baseline. `--with-answers` separately calls Groq and checks expected keywords; keyword matching is not a faithfulness measure.

| Mode | Recall@1 | Recall@3 | Recall@5 | MRR | Average retrieval |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense | 0.8333 | 1.0000 | 1.0000 | 1.0000 | 4.54 ms |
| BM25 | 0.5000 | 0.9167 | 1.0000 | 0.8056 | 0.09 ms |
| Hybrid | 0.8333 | 1.0000 | 1.0000 | 1.0000 | 3.78 ms |

These values come from one local run on four sample chunks and six questions. Recall@5 is automatically 1.0 when every chunk is returned, so this is a reproducible smoke benchmark rather than evidence of general performance. Latency varies by machine and cache state. Reranking and answer generation were not included in this baseline.

## Limits and next steps

Ingestion supports UTF-8 `.txt` files. The labels are small and tied to the sample documents. Source citations identify retrieved chunks; they do not prove every generated claim is supported. The optional rewrite trigger is a simple lexical heuristic. Groq requires network access, and automated tests mock it.

The next useful improvement is a larger, independently labeled evaluation set with harder negatives, followed by error analysis and chunking experiments. Markdown or PDF loading could be added when a real document set calls for it.
