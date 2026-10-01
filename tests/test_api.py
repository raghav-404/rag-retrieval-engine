import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from rag_retrieval_engine.api.app import create_app
from rag_retrieval_engine.retrieval import ArtifactsNotReadyError
from rag_retrieval_engine.service import RAGService


def make_client(config, stub_engine_class, fake_model_class, responses):
    document = Document(
        page_content="The refund period is 30 days.",
        metadata={"source": "refund.txt", "chunk_id": "refund_txt_chunk_0001"},
    )
    engine = stub_engine_class([document])
    model = fake_model_class(responses)
    calls = []

    def factory(_config):
        calls.append(1)
        return RAGService(config, engine, model)

    return TestClient(create_app(config, service_factory=factory)), calls, model, engine


def test_api_success_and_startup_caching(config, stub_engine_class, fake_model_class) -> None:
    client, factory_calls, model, engine = make_client(
        config, stub_engine_class, fake_model_class, ["Refunds take 30 days [1].", "Refunds take 30 days [1]."]
    )
    with client:
        first = client.post("/ask", json={"question": "What is the refund period?"})
        second = client.post("/ask", json={"question": "What is the refund period?"})
        retrieved = client.post("/retrieve", json={"question": "refund", "mode": "sparse"})
        health = client.get("/health")

    assert first.status_code == second.status_code == retrieved.status_code == health.status_code == 200
    assert first.json()["answer"] == "Refunds take 30 days [1]."
    assert first.json()["sources"] == [{"source": "refund.txt", "chunk_id": "refund_txt_chunk_0001", "rank": 1}]
    assert first.json()["metrics"]["retrieval_mode"] == "hybrid"
    assert first.json()["metrics"]["total_ms"] >= 0
    assert first.json()["request_id"]
    assert retrieved.json()["results"][0]["chunk_id"] == "refund_txt_chunk_0001"
    assert factory_calls == [1]
    assert len(model.calls) == 2
    assert len(engine.calls) == 3


@pytest.mark.parametrize(
    ("response", "status"),
    [(TimeoutError("late"), 504), (RuntimeError("provider down"), 502), ("  ", 502)],
)
def test_api_maps_provider_failures(config, stub_engine_class, fake_model_class, response, status) -> None:
    client, _, _, _ = make_client(config, stub_engine_class, fake_model_class, [response])
    with client:
        result = client.post("/ask", json={"question": "What is the refund period?"})
    assert result.status_code == status
    assert result.json()["detail"]["request_id"]
    assert "Groq" in result.json()["detail"]["message"]


def test_api_rejects_invalid_questions(config, stub_engine_class, fake_model_class) -> None:
    client, _, _, _ = make_client(config, stub_engine_class, fake_model_class, [])
    with client:
        blank = client.post("/ask", json={"question": "   "})
        missing = client.post("/ask", json={})
        bad_mode = client.post("/ask", json={"question": "valid", "mode": "other"})
    assert blank.status_code == 400
    assert missing.status_code == bad_mode.status_code == 422


def test_api_maps_retrieval_failure(config, stub_engine_class, fake_model_class) -> None:
    client, _, _, engine = make_client(config, stub_engine_class, fake_model_class, [])

    def fail_search(*args, **kwargs):
        raise RuntimeError("index failed")

    engine.search = fail_search
    with client:
        result = client.post("/ask", json={"question": "What is the refund period?"})
    assert result.status_code == 503
    assert result.json()["detail"]["request_id"]


def test_missing_index_fails_at_startup_before_model_download(config) -> None:
    with pytest.raises(ArtifactsNotReadyError, match="Run `python ingest.py` first"):
        with TestClient(create_app(config)):
            pass
