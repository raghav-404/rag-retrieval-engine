"""
Core RAG pipeline:
  query → rewrite → hybrid search (FAISS + BM25) → rerank → context → LLM → eval loop
"""

import json
import math
import requests
import numpy as np
import faiss
from collections import Counter
from utils import embed_query

INDEX_PATH = "faiss.index"
META_PATH = "metadata.json"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"
TOP_K = 10
RERANK_TOP = 4
EVAL_THRESHOLD = 0.08   # retry with original query if score is below this


# ── Load index & corpus once at import time ──────────────────────────────────

index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

corpus = [m["text"] for m in metadata]


# ── BM25 helpers ──────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return text.lower().split()

def _build_idf(corpus: list[str]) -> dict:
    N = len(corpus)
    df: Counter = Counter()
    for doc in corpus:
        for t in set(_tokenize(doc)):
            df[t] += 1
    return {t: math.log((N - n + 0.5) / (n + 0.5) + 1) for t, n in df.items()}

_idf = _build_idf(corpus)
_avg_dl = sum(len(_tokenize(d)) for d in corpus) / max(len(corpus), 1)


def _bm25(query_tokens: list[str], doc: str) -> float:
    k1, b = 1.5, 0.75
    tokens = _tokenize(doc)
    dl = len(tokens)
    tf = Counter(tokens)
    score = 0.0
    for t in query_tokens:
        f = tf.get(t, 0)
        if f == 0:
            continue
        score += _idf.get(t, 0) * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / _avg_dl))
    return score


# ── Pipeline steps ────────────────────────────────────────────────────────────

def _ollama(prompt: str, timeout: int = 60) -> str:
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        return ""


def rewrite_query(query: str) -> str:
    """Ask the LLM to reformulate the query for better retrieval."""
    prompt = (
        "Rewrite the following search query to improve document retrieval. "
        "Return ONLY the rewritten query, no explanation.\n"
        f"Query: {query}"
    )
    rewritten = _ollama(prompt, timeout=30)
    return rewritten if rewritten else query


def hybrid_search(query: str, top_k: int = TOP_K) -> list[dict]:
    """Dense (FAISS) + sparse (BM25) search, scores combined 60/40."""
    vec = embed_query(query).astype(np.float32).reshape(1, -1)
    dense_scores, indices = index.search(vec, top_k)

    query_tokens = _tokenize(query)
    results = []
    for dense_score, idx in zip(dense_scores[0], indices[0]):
        if idx == -1:
            continue
        bm = _bm25(query_tokens, corpus[idx])
        bm_norm = bm / (bm + 1)                           # squash to [0, 1)
        combined = 0.6 * float(dense_score) + 0.4 * bm_norm
        results.append({"meta": metadata[idx], "score": combined, "idx": int(idx)})

    return sorted(results, key=lambda x: x["score"], reverse=True)


def rerank(results: list[dict], query: str, top_n: int = RERANK_TOP) -> list[dict]:
    """Heuristic rerank: boost chunks with higher query-token overlap."""
    q_tokens = set(_tokenize(query))
    for r in results:
        doc_tokens = set(_tokenize(r["meta"]["text"]))
        overlap_ratio = len(q_tokens & doc_tokens) / max(len(q_tokens), 1)
        r["score"] += 0.15 * overlap_ratio
    return sorted(results, key=lambda x: x["score"], reverse=True)[:top_n]


def build_context(results: list[dict]) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"[{i}] (source: {r['meta']['source']})\n{r['meta']['text']}")
    return "\n\n---\n\n".join(parts)


def _generate_answer(query: str, context: str) -> str:
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context provided.
If the context does not contain enough information, say so clearly.

Context:
{context}

Question: {query}
Answer:"""
    return _ollama(prompt, timeout=120)


def _eval_score(response: str, context: str) -> float:
    """Heuristic faithfulness: fraction of response tokens found in context."""
    r_tokens = set(_tokenize(response))
    c_tokens = set(_tokenize(context))
    if not r_tokens:
        return 0.0
    return len(r_tokens & c_tokens) / len(r_tokens)


# ── Public entry point ────────────────────────────────────────────────────────

def ask(query: str) -> dict:
    rewritten = rewrite_query(query)

    results = hybrid_search(rewritten)
    results = rerank(results, rewritten)
    context = build_context(results)

    answer = _generate_answer(query, context)
    score = _eval_score(answer, context)

    # Evaluation loop: if quality is poor, retry with the original query
    if score < EVAL_THRESHOLD:
        results2 = hybrid_search(query)
        results2 = rerank(results2, query)
        context2 = build_context(results2)
        answer2 = _generate_answer(query, context2)
        score2 = _eval_score(answer2, context2)
        if score2 > score:
            answer, context, results, score = answer2, context2, results2, score2

    sources = sorted({r["meta"]["source"] for r in results})

    return {
        "answer": answer,
        "sources": sources,
        "rewritten_query": rewritten,
        "eval_score": round(score, 3),
    }
