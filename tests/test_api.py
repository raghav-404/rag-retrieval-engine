import importlib

from fastapi.testclient import TestClient

from rag_retrieval_engine.api.app import create_app
from rag_retrieval_engine.ingestion import build_index
from rag_retrieval_engine.retrieval import load_artifacts


def test_health_endpoint_reports_artifact_status(config) -> None:
    response = TestClient(create_app(config)).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "artifacts_ready": False}


def test_retrieve_endpoint_returns_cited_chunks_without_model_download(config, tokenizer, embeddings, monkeypatch) -> None:
    config.docs_dir.joinpath("policy.txt").write_text("alpha beta gamma", encoding="utf-8")
    build_index(config, embeddings=embeddings, tokenizer=tokenizer)
    engine = load_artifacts(config, embeddings=embeddings)
    api_module = importlib.import_module("rag_retrieval_engine.api.app")
    monkeypatch.setattr(api_module, "load_artifacts", lambda _: engine)

    response = TestClient(create_app(config)).post("/retrieve", json={"question": "alpha", "mode": "sparse"})

    assert response.status_code == 200
    assert response.json()["results"][0]["chunk_id"] == "policy_txt_chunk_0001"
