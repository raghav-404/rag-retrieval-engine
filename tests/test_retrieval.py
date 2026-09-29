import json
from dataclasses import replace

import pytest
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from rag_retrieval_engine.ingestion import build_index
from rag_retrieval_engine.retrieval import (
    ArtifactValidationError,
    ArtifactsNotReadyError,
    load_artifacts,
)


def indexed_engine(config, tokenizer, embeddings):
    config.docs_dir.joinpath("a.txt").write_text("semantic unrelated", encoding="utf-8")
    config.docs_dir.joinpath("b.txt").write_text("zebra voucher details", encoding="utf-8")
    config.docs_dir.joinpath("c.txt").write_text("other details", encoding="utf-8")
    config = replace(config, chunk_size=20, chunk_overlap=4, dense_candidates=1, sparse_candidates=3, final_k=3)
    embeddings.vectors.update({
        "semantic unrelated": [1.0, 0.0],
        "zebra voucher details": [0.0, 1.0],
        "other details": [-1.0, 0.0],
        "zebra": [1.0, 0.0],
    })
    build_index(config, embeddings=embeddings, tokenizer=tokenizer)
    return config, load_artifacts(config, embeddings=embeddings)


def test_dense_sparse_and_hybrid_modes_are_independent(config, tokenizer, embeddings) -> None:
    config, engine = indexed_engine(config, tokenizer, embeddings)

    dense = engine.search("zebra", "dense")
    sparse = engine.search("zebra", "sparse")
    hybrid = engine.search("zebra", "hybrid")

    assert [doc.metadata["source"] for doc in dense] == ["a.txt"]
    assert sparse[0].metadata["source"] == "b.txt"
    assert engine.search("zebra?", "sparse")[0].metadata["source"] == "b.txt"
    assert "b.txt" in [doc.metadata["source"] for doc in hybrid]
    assert len({doc.metadata["chunk_id"] for doc in hybrid}) == len(hybrid)
    assert len(engine.documents) == 3  # BM25 was built over the complete corpus.
    assert len(engine.sparse_retriever.docs) == 3


def test_rrf_uses_rank_and_deduplicates_by_chunk_id() -> None:
    a = Document(page_content="A", metadata={"chunk_id": "a"})
    b = Document(page_content="B", metadata={"chunk_id": "b"})
    ensemble = EnsembleRetriever(
        retrievers=[RunnableLambda(lambda _: [a, b]), RunnableLambda(lambda _: [b])],
        weights=[0.5, 0.5],
        c=60,
        id_key="chunk_id",
    )

    assert [doc.metadata["chunk_id"] for doc in ensemble.invoke("question")] == ["b", "a"]


def test_rrf_keeps_identical_text_from_different_chunks() -> None:
    a = Document(page_content="same text", metadata={"chunk_id": "a"})
    b = Document(page_content="same text", metadata={"chunk_id": "b"})
    ensemble = EnsembleRetriever(
        retrievers=[RunnableLambda(lambda _: [a]), RunnableLambda(lambda _: [b])],
        weights=[0.5, 0.5],
        c=60,
        id_key="chunk_id",
    )

    assert {doc.metadata["chunk_id"] for doc in ensemble.invoke("question")} == {"a", "b"}


def test_missing_artifacts_have_actionable_error(config, embeddings) -> None:
    with pytest.raises(ArtifactsNotReadyError, match="Run `python ingest.py` first"):
        load_artifacts(config, embeddings=embeddings)


def test_manifest_model_mismatch_is_rejected(config, tokenizer, embeddings) -> None:
    config, _ = indexed_engine(config, tokenizer, embeddings)
    manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))
    manifest["embedding_model"] = "different-model"
    config.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="Embedding model differs"):
        load_artifacts(config, embeddings=embeddings)


def test_corrupted_index_is_rejected(config, tokenizer, embeddings) -> None:
    config, _ = indexed_engine(config, tokenizer, embeddings)
    config.index_path.write_bytes(b"not a FAISS index")

    with pytest.raises(ArtifactValidationError, match="corrupted"):
        load_artifacts(config, embeddings=embeddings)


def test_embedding_dimension_mismatch_is_rejected(config, tokenizer, embeddings) -> None:
    config, _ = indexed_engine(config, tokenizer, embeddings)
    embeddings.default = [1.0, 0.0, 0.0]

    with pytest.raises(ArtifactValidationError, match="dimension differs"):
        load_artifacts(config, embeddings=embeddings)


def test_changed_source_requires_reingestion(config, tokenizer, embeddings) -> None:
    config, _ = indexed_engine(config, tokenizer, embeddings)
    config.docs_dir.joinpath("a.txt").write_text("changed", encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="Source documents changed"):
        load_artifacts(config, embeddings=embeddings)
