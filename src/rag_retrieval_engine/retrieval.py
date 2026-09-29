from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import faiss
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

from .config import AppConfig
from .embeddings import create_embeddings
from .ingestion import ARTIFACT_VERSION, file_hash

RetrievalMode = Literal["dense", "sparse", "hybrid"]


def bm25_tokens(text: str) -> list[str]:
    """Match words despite case and surrounding punctuation."""
    return re.findall(r"\w+", text.casefold())


class ArtifactsNotReadyError(FileNotFoundError):
    """Ingestion has not produced every required artifact."""


class ArtifactValidationError(ValueError):
    """Saved artifacts are stale, corrupted, or incompatible."""


def artifacts_ready(config: AppConfig) -> bool:
    return all(path.is_file() for path in (config.index_path, config.metadata_path, config.manifest_path))


@dataclass
class RetrievalEngine:
    """Keep all three LangChain retrievers callable and visible."""

    dense_retriever: BaseRetriever
    sparse_retriever: BM25Retriever
    hybrid_retriever: EnsembleRetriever
    documents: list[Document]
    manifest: dict
    final_k: int

    def search(self, question: str, mode: RetrievalMode = "hybrid", *, k: int | None = None) -> list[Document]:
        if not question.strip():
            raise ValueError("Question cannot be empty.")
        if mode not in ("dense", "sparse", "hybrid"):
            raise ValueError(f"Unknown retrieval mode: {mode}")
        limit = self.final_k if k is None else k
        if limit <= 0:
            raise ValueError("k must be positive.")
        retriever = {
            "dense": self.dense_retriever,
            "sparse": self.sparse_retriever,
            "hybrid": self.hybrid_retriever,
        }[mode]
        return retriever.invoke(question)[:limit]


def load_artifacts(config: AppConfig, *, embeddings: Embeddings | None = None) -> RetrievalEngine:
    """Validate local files, then restore FAISS and build BM25 over every chunk."""
    config.validate()
    missing = [str(path) for path in (config.index_path, config.metadata_path, config.manifest_path) if not path.is_file()]
    if missing:
        raise ArtifactsNotReadyError("Run `python ingest.py` first. Missing: " + ", ".join(missing))
    try:
        manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError("Manifest is unreadable; run ingestion again.") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != ARTIFACT_VERSION:
        raise ArtifactValidationError("Artifact schema version is incompatible; run ingestion again.")
    try:
        created_at = datetime.fromisoformat(manifest["created_at"])
        if created_at.tzinfo is None:
            raise ValueError("Timestamp needs a timezone.")
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactValidationError("Manifest creation timestamp is invalid; run ingestion again.") from exc
    if manifest.get("embedding_model") != config.embed_model_name:
        raise ArtifactValidationError("Embedding model differs from the manifest; run ingestion again.")
    if (manifest.get("chunk_size"), manifest.get("chunk_overlap")) != (config.chunk_size, config.chunk_overlap):
        raise ArtifactValidationError("Chunk settings differ from the manifest; run ingestion again.")

    hashes = manifest.get("artifact_hashes")
    try:
        artifacts_match = isinstance(hashes, dict) and all(
            hashes.get(name) == file_hash(path)
            for name, path in (("index", config.index_path), ("metadata", config.metadata_path))
        )
    except OSError as exc:
        raise ArtifactValidationError("Index or chunk metadata is unreadable; run ingestion again.") from exc
    if not artifacts_match:
        raise ArtifactValidationError("Index or chunk metadata is corrupted; run ingestion again.")
    source_hashes = manifest.get("source_hashes")
    current_sources = {path.name: path for path in config.docs_dir.glob("*.txt")}
    try:
        sources_match = (
            isinstance(source_hashes, dict)
            and set(source_hashes) == set(current_sources)
            and all(source_hashes[name] == file_hash(path) for name, path in current_sources.items())
        )
    except OSError as exc:
        raise ArtifactValidationError("Source documents are unreadable; run ingestion again.") from exc
    if not sources_match:
        raise ArtifactValidationError("Source documents changed since ingestion; run ingestion again.")

    try:
        records = json.loads(config.metadata_path.read_text(encoding="utf-8"))
        documents = [
            Document(
                id=item["metadata"]["chunk_id"],
                page_content=item["page_content"],
                metadata=item["metadata"],
            )
            for item in records
        ]
        ids = [doc.metadata["chunk_id"] for doc in documents]
        valid = (
            isinstance(records, list)
            and len(documents) > 0
            and len(documents) == manifest.get("chunk_count")
            and len(ids) == len(set(ids))
            and all(
                isinstance(doc.page_content, str)
                and doc.page_content.strip()
                and doc.metadata.get("source") in source_hashes
                and isinstance(doc.metadata.get("position"), int)
                and isinstance(doc.metadata.get("token_count"), int)
                and doc.metadata["token_count"] > 0
                for doc in documents
            )
        )
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ArtifactValidationError("Chunk metadata is malformed; run ingestion again.") from exc
    if not valid:
        raise ArtifactValidationError("Chunk metadata does not match the manifest; run ingestion again.")

    try:
        index = faiss.read_index(str(config.index_path))
    except (OSError, RuntimeError) as exc:
        raise ArtifactValidationError("FAISS index is unreadable; run ingestion again.") from exc
    if (
        index.d != manifest.get("embedding_dimension")
        or index.ntotal != len(documents)
        or index.metric_type != faiss.METRIC_INNER_PRODUCT
    ):
        raise ArtifactValidationError("FAISS index dimensions, count, or metric do not match the manifest.")
    embeddings = embeddings or create_embeddings(config)
    if len(embeddings.embed_query("dimension check")) != index.d:
        raise ArtifactValidationError("Embedding model dimension differs from the FAISS index.")

    # The verified JSON replaces LangChain's pickle docstore when restoring FAISS.
    store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore({doc.metadata["chunk_id"]: doc for doc in documents}),
        index_to_docstore_id={position: doc.metadata["chunk_id"] for position, doc in enumerate(documents)},
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )
    dense = store.as_retriever(search_kwargs={"k": config.dense_candidates})
    sparse = BM25Retriever.from_documents(documents, k=config.sparse_candidates, preprocess_func=bm25_tokens)
    hybrid = EnsembleRetriever(
        retrievers=[dense, sparse],
        weights=[config.dense_weight, config.sparse_weight],
        c=config.rrf_constant,
        id_key="chunk_id",
    )
    return RetrievalEngine(dense, sparse, hybrid, documents, manifest, config.final_k)
