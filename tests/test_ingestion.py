import json

import faiss
import numpy as np

from rag_retrieval_engine.config import AppConfig
from rag_retrieval_engine.ingestion import build_index


def test_build_index_writes_artifacts(tmp_path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "sample.txt").write_text("alpha beta gamma delta epsilon", encoding="utf-8")

    config = AppConfig(
        project_root=tmp_path,
        docs_dir=docs_dir,
        index_path=tmp_path / "faiss.index",
        metadata_path=tmp_path / "metadata.json",
        ollama_url="http://localhost:11434/api/generate",
        model_name="qwen2.5:7b",
        embed_model_name="test-embedder",
        top_k=5,
        rerank_top_n=2,
        eval_threshold=0.08,
        api_url="http://localhost:8000/ask",
    )

    def fake_embed_texts(texts, _config):
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)

    summary = build_index(config, chunk_size=3, overlap=1, embed_fn=fake_embed_texts)

    assert summary["document_count"] == 1
    assert summary["chunk_count"] == 3
    assert config.index_path.exists()
    assert config.metadata_path.exists()

    index = faiss.read_index(str(config.index_path))
    metadata = json.loads(config.metadata_path.read_text(encoding="utf-8"))

    assert index.ntotal == 3
    assert metadata[0]["source"] == "sample.txt"
