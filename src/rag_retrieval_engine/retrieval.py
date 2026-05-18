from __future__ import annotations

import json
import math
from collections import Counter

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

from .config import AppConfig

_embedder: SentenceTransformer | None = None


class ArtifactsNotReadyError(FileNotFoundError):
    pass


def artifacts_ready(config: AppConfig) -> bool:
    return config.index_path.exists() and config.metadata_path.exists()


def embed_query(query: str, config: AppConfig) -> np.ndarray:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(config.embed_model_name)
    return _embedder.encode([query], normalize_embeddings=True)[0]


def load_artifacts(config: AppConfig) -> tuple[faiss.Index, list[dict[str, str]], list[str], dict[str, float], float]:
    missing = [str(path) for path in (config.index_path, config.metadata_path) if not path.exists()]
    if missing:
        raise ArtifactsNotReadyError("Run `python ingest.py` first. Missing: " + ", ".join(missing))
    metadata = json.loads(config.metadata_path.read_text(encoding="utf-8"))
    corpus = [item["text"] for item in metadata]
    idf = {
        token: math.log((len(corpus) - freq + 0.5) / (freq + 0.5) + 1)
        for token, freq in Counter(token for doc in corpus for token in set(doc.lower().split())).items()
    }
    avg_len = sum(len(doc.split()) for doc in corpus) / max(len(corpus), 1)
    return faiss.read_index(str(config.index_path)), metadata, corpus, idf, avg_len


def search(query: str, artifacts: tuple, config: AppConfig, *, embed_fn=embed_query) -> list[dict[str, object]]:
    index, metadata, corpus, idf, avg_len = artifacts
    scores, indices = index.search(embed_fn(query, config).astype(np.float32).reshape(1, -1), config.top_k)
    q_tokens, results = query.lower().split(), []
    for dense, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        tokens, tf, score = corpus[idx].lower().split(), Counter(corpus[idx].lower().split()), 0.0
        for token in q_tokens:
            freq = tf.get(token, 0)
            if freq:
                score += idf.get(token, 0.0) * (freq * 2.5) / (freq + 1.5 * (0.25 + 0.75 * len(tokens) / max(avg_len, 1)))
        overlap = len(set(q_tokens) & set(tokens)) / max(len(set(q_tokens)), 1)
        results.append({
            "meta": metadata[idx],
            "score": 0.6 * float(dense) + 0.4 * (score / (score + 1)) + 0.15 * overlap,
            "idx": int(idx),
        })
    return sorted(results, key=lambda item: item["score"], reverse=True)[: config.rerank_top_n]


def hybrid_search(query: str, artifacts: tuple, config: AppConfig, *, embed_fn=embed_query) -> list[dict[str, object]]:
    return search(query, artifacts, config, embed_fn=embed_fn)


def build_context(results: list[dict[str, object]]) -> str:
    return "\n\n---\n\n".join(
        f"[{i}] (source: {r['meta']['source']})\n{r['meta']['text']}" for i, r in enumerate(results, 1)
    )


def ollama(prompt: str, config: AppConfig, timeout: int, http=requests) -> str:
    try:
        response = http.post(
            config.ollama_url,
            json={"model": config.model_name, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception:
        return ""


def answer_once(query: str, search_query: str, artifacts: tuple, config: AppConfig, *, embed_fn=embed_query, http=requests):
    results = search(search_query, artifacts, config, embed_fn=embed_fn)
    context = build_context(results)
    answer = ollama(
        f"You are a helpful assistant. Answer using ONLY this context.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:",
        config,
        120,
        http,
    )
    score = 0.0 if not answer else len(set(answer.lower().split()) & set(context.lower().split())) / len(set(answer.lower().split()))
    return results, answer, score


def ask(query: str, config: AppConfig, *, embed_fn=embed_query, http=requests) -> dict[str, object]:
    artifacts = load_artifacts(config)
    rewritten = ollama(
        "Rewrite the following search query to improve document retrieval. Return ONLY the rewritten query.\n"
        f"Query: {query}",
        config,
        30,
        http,
    ) or query
    results, answer, score = answer_once(query, rewritten, artifacts, config, embed_fn=embed_fn, http=http)
    if score < config.eval_threshold:
        fallback = answer_once(query, query, artifacts, config, embed_fn=embed_fn, http=http)
        if fallback[2] > score:
            results, answer, score = fallback
    return {
        "answer": answer,
        "sources": sorted({result["meta"]["source"] for result in results}),
        "rewritten_query": rewritten,
        "eval_score": round(score, 3),
    }
