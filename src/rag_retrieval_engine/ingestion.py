from __future__ import annotations

import json
import re

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import AppConfig

_embedder: SentenceTransformer | None = None


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid chunk settings.")
    words, step = text.split(), chunk_size - overlap
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), step) if words[i : i + chunk_size]]


def embed_texts(texts: list[str], config: AppConfig) -> np.ndarray:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(config.embed_model_name)
    return _embedder.encode(texts, normalize_embeddings=True, show_progress_bar=True)


def build_index(config: AppConfig, *, chunk_size: int = 300, overlap: int = 50, embed_fn=embed_texts) -> dict[str, object]:
    docs = [
        {"source": path.name, "text": clean_text(path.read_text(encoding="utf-8"))}
        for path in sorted(config.docs_dir.glob("*.txt"))
    ]
    if not config.docs_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {config.docs_dir}")
    if not docs:
        raise FileNotFoundError(f"No .txt files found in '{config.docs_dir}'")

    metadata = [
        {"source": doc["source"], "text": chunk}
        for doc in docs
        for chunk in chunk_text(doc["text"], chunk_size=chunk_size, overlap=overlap)
    ]
    vectors = embed_fn([item["text"] for item in metadata], config).astype(np.float32)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(config.index_path))
    config.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"document_count": len(docs), "chunk_count": len(metadata)}
