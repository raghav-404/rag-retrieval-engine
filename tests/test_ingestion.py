import json

import faiss
import pytest

from rag_retrieval_engine.ingestion import build_index, chunk_documents


def test_build_index_writes_manifest_and_chunks(config, tokenizer, embeddings) -> None:
    config.docs_dir.joinpath("sample.txt").write_text("alpha beta gamma delta epsilon zeta", encoding="utf-8")

    summary = build_index(config, embeddings=embeddings, tokenizer=tokenizer)

    manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))
    records = json.loads(config.metadata_path.read_text(encoding="utf-8"))
    assert summary == {"document_count": 1, "chunk_count": len(records)}
    assert manifest["embedding_model"] == "test-embedder"
    assert manifest["embedding_dimension"] == 2
    assert manifest["chunk_size"] == 5
    assert manifest["chunk_overlap"] == 2
    assert manifest["chunk_count"] == len(records)
    assert len(manifest["source_hashes"]["sample.txt"]) == 64
    index = faiss.read_index(str(config.index_path))
    assert index.ntotal == len(records)
    assert index.metric_type == faiss.METRIC_INNER_PRODUCT
    assert records[0]["metadata"]["chunk_id"] == "sample_txt_chunk_0001"


def test_empty_files_are_skipped_and_all_empty_corpus_fails(config, tokenizer) -> None:
    config.docs_dir.joinpath("empty.txt").write_text("  \n ", encoding="utf-8")
    with pytest.raises(ValueError, match="All documents are empty"):
        chunk_documents(config, tokenizer)

    config.docs_dir.joinpath("useful.txt").write_text("useful information", encoding="utf-8")
    chunks, hashes = chunk_documents(config, tokenizer)
    assert len(chunks) == 1
    assert set(hashes) == {"empty.txt", "useful.txt"}


def test_non_utf8_file_has_clear_error(config, tokenizer) -> None:
    config.docs_dir.joinpath("bad.txt").write_bytes(b"\xff")
    with pytest.raises(ValueError, match="must be UTF-8"):
        chunk_documents(config, tokenizer)
