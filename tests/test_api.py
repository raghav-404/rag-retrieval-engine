from fastapi.testclient import TestClient

from rag_retrieval_engine.api.app import create_app
from rag_retrieval_engine.config import AppConfig


def test_health_endpoint_reports_ok(tmp_path) -> None:
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
    app = create_app(config)

    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "artifacts_ready": False}
