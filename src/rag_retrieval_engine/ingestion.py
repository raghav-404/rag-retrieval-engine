from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from .config import AppConfig
from .embeddings import create_embeddings

ARTIFACT_VERSION = 1


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chunk_documents(config: AppConfig, tokenizer) -> tuple[list[Document], dict[str, str]]:
    """Read every .txt file and assign stable, source-based chunk IDs."""
    config.validate()
    if not config.docs_dir.is_dir():
        raise FileNotFoundError(f"Documents directory not found: {config.docs_dir}")
    paths = sorted(config.docs_dir.glob("*.txt"))
    if not paths:
        raise FileNotFoundError(f"No .txt files found in {config.docs_dir}")

    def token_count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        length_function=token_count,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Document] = []
    source_hashes: dict[str, str] = {}
    used_prefixes: set[str] = set()
    for path in paths:
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            raise ValueError(f"Document must be UTF-8: {path.name}") from exc
        source_hashes[path.name] = hashlib.sha256(raw).hexdigest()
        prefix = re.sub(r"[^a-z0-9]+", "_", path.name.lower()).strip("_")
        if prefix in used_prefixes:
            raise ValueError(f"Source names produce duplicate chunk IDs: {path.name}")
        used_prefixes.add(prefix)
        if not text:
            continue
        source = Document(page_content=text, metadata={"source": path.name})
        for position, chunk in enumerate(splitter.split_documents([source]), start=1):
            chunk_id = f"{prefix}_chunk_{position:04d}"
            chunk.id = chunk_id
            chunk.metadata.update(
                chunk_id=chunk_id,
                position=position,
                token_count=token_count(chunk.page_content),
            )
            chunks.append(chunk)
    if not chunks:
        raise ValueError("All documents are empty or whitespace-only; nothing to index.")
    return chunks, source_hashes


def build_index(config: AppConfig, *, embeddings: Embeddings | None = None, tokenizer=None) -> dict[str, int]:
    """Embed LangChain Documents and write inspectable retrieval artifacts."""
    embeddings = embeddings or create_embeddings(config)
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(config.embed_model_name)
    chunks, source_hashes = chunk_documents(config, tokenizer)
    store = FAISS.from_documents(
        chunks,
        embeddings,
        ids=[chunk.metadata["chunk_id"] for chunk in chunks],
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )
    for path in (config.index_path, config.metadata_path, config.manifest_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(store.index, str(config.index_path))
    records = [
        {"page_content": chunk.page_content, "metadata": chunk.metadata}
        for chunk in chunks
    ]
    config.metadata_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": ARTIFACT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": config.embed_model_name,
        "embedding_dimension": store.index.d,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "chunk_count": len(chunks),
        "source_hashes": source_hashes,
        "artifact_hashes": {
            "index": file_hash(config.index_path),
            "metadata": file_hash(config.metadata_path),
        },
    }
    config.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"document_count": len(source_hashes), "chunk_count": len(chunks)}
