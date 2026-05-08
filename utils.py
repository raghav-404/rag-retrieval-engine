from sentence_transformers import SentenceTransformer
import numpy as np
import re

EMBED_MODEL = "all-MiniLM-L6-v2"
_embedder = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Sliding window word-level chunking."""
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def embed_texts(texts: list[str]) -> np.ndarray:
    return get_embedder().encode(texts, normalize_embeddings=True, show_progress_bar=True)


def embed_query(query: str) -> np.ndarray:
    return get_embedder().encode([query], normalize_embeddings=True)[0]


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()
