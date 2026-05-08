# RAG Retrieval

A local Retrieval-Augmented Generation (RAG) app built with FastAPI, Streamlit, FAISS, sentence-transformers, and Ollama.

It ingests plain-text documents, builds a vector index, retrieves relevant chunks with hybrid search, and answers questions with source-aware responses.

## Features

- Local-first stack with no hosted LLM dependency
- Hybrid retrieval using FAISS dense search plus BM25-style sparse scoring
- Query rewriting to improve retrieval for vague questions
- Lightweight reranking before answer generation
- FastAPI backend and Streamlit chat UI
- Source tracking and a simple faithfulness-style evaluation score

## Tech Stack

- Python 3.12+
- FastAPI
- Streamlit
- FAISS
- sentence-transformers (`all-MiniLM-L6-v2`)
- Ollama (`qwen2.5:7b`)

## Project Structure

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit frontend for asking questions and viewing sources |
| `backend.py` | FastAPI API with `POST /ask` and `GET /health` |
| `ingest.py` | Reads `.txt` files from `docs/` and builds the retrieval index |
| `rag.py` | Core retrieval, reranking, prompting, and evaluation pipeline |
| `utils.py` | Text cleaning, chunking, and embedding helpers |
| `docs/` | Sample source documents for indexing |

## How It Works

```text
Question
  -> Query rewrite
  -> Dense retrieval (FAISS)
  -> Sparse scoring (BM25-style)
  -> Heuristic rerank
  -> Context assembly
  -> Ollama answer generation
  -> Eval score and source return
```

## Getting Started

### 1. Install dependencies

Using `pip`:

```bash
pip install -r requirements.txt
```

Or using `uv`:

```bash
uv sync
```

### 2. Start Ollama and pull the model

```bash
ollama pull qwen2.5:7b
```

Make sure the Ollama server is running locally at `http://localhost:11434`.

### 3. Add or replace documents

Put `.txt` files inside `docs/`.

The repository includes sample files for demo purposes:

- `docs/sample_company_overview.txt`
- `docs/sample_faq.txt`
- `docs/sample_product_specs.txt`
- `docs/sample_support_policy.txt`

### 4. Build the index

```bash
python ingest.py
```

This generates:

- `faiss.index`
- `metadata.json`

These files are local build artifacts and are ignored by Git.

### 5. Run the backend

```bash
uvicorn backend:app --reload
```

### 6. Run the frontend

```bash
streamlit run app.py
```

Then open `http://localhost:8501`.

## API Example

Request:

```http
POST /ask
Content-Type: application/json
```

```json
{
  "question": "What is the support response window?"
}
```

Response:

```json
{
  "answer": "Support requests are typically handled within the documented response window.",
  "sources": ["sample_support_policy.txt"],
  "rewritten_query": "support response window policy",
  "eval_score": 0.312
}
```

## Future Improvements

- Add tests for ingestion and retrieval behavior
- Support PDF or Markdown document ingestion
- Add configurable chunking and retrieval parameters
- Improve evaluation with stronger groundedness checks
