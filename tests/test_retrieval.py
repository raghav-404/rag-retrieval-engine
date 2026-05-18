import json

import faiss
import numpy as np

from rag_retrieval_engine.config import AppConfig
from rag_retrieval_engine.retrieval import hybrid_search, load_artifacts


def test_hybrid_search_returns_expected_result_shape(tmp_path) -> None:
    index = faiss.IndexFlatIP(2)
    index.add(np.array([[1.0, 0.0], [0.6, 0.8]], dtype=np.float32))
    faiss.write_index(index, str(tmp_path / "faiss.index"))

    metadata = [
        {"source": "a.txt", "text": "alpha beta gamma"},
        {"source": "b.txt", "text": "delta epsilon zeta"},
    ]
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    config = AppConfig(
        project_root=tmp_path,
        docs_dir=tmp_path / "docs",
        index_path=tmp_path / "faiss.index",
        metadata_path=tmp_path / "metadata.json",
        ollama_url="http://localhost:11434/api/generate",
        model_name="qwen2.5:7b",
        embed_model_name="test-embedder",
        top_k=2,
        rerank_top_n=2,
        eval_threshold=0.08,
        api_url="http://localhost:8000/ask",
    )

    def fake_embed_query(query, _config):
        assert query == "alpha"
        return np.array([1.0, 0.0], dtype=np.float32)

    artifacts = load_artifacts(config)
    results = hybrid_search("alpha", artifacts, config, embed_fn=fake_embed_query)

    assert len(results) == 2
    assert set(results[0].keys()) == {"meta", "score", "idx"}
    assert set(results[0]["meta"].keys()) == {"source", "text"}
    assert isinstance(results[0]["score"], float)
    assert isinstance(results[0]["idx"], int)
