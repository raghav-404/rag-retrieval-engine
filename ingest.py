"""
Run once to index your documents:
    python ingest.py

Reads all .txt files from docs/, chunks them, embeds them, and saves:
    faiss.index   — the vector index
    metadata.json — chunk text + source filename
"""

import os
import json
import faiss
import numpy as np
from utils import chunk_text, embed_texts, clean_text

DOCS_DIR = "docs"
INDEX_PATH = "faiss.index"
META_PATH = "metadata.json"


def load_documents(docs_dir: str) -> list[dict]:
    docs = []
    for fname in sorted(os.listdir(docs_dir)):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(docs_dir, fname), "r", encoding="utf-8") as f:
            docs.append({"source": fname, "text": clean_text(f.read())})
    return docs


def build_index():
    docs = load_documents(DOCS_DIR)
    if not docs:
        raise FileNotFoundError(f"No .txt files found in '{DOCS_DIR}/'")

    chunks, metadata = [], []
    for doc in docs:
        for chunk in chunk_text(doc["text"]):
            chunks.append(chunk)
            metadata.append({"source": doc["source"], "text": chunk})

    print(f"Embedding {len(chunks)} chunks from {len(docs)} documents...")
    embeddings = embed_texts(chunks).astype(np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # cosine sim (vectors are normalized)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Done. Saved {INDEX_PATH} + {META_PATH}")


if __name__ == "__main__":
    build_index()
