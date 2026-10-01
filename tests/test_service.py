import asyncio
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from rag_retrieval_engine.service import (
    EmptyModelResponseError,
    ProviderError,
    ProviderTimeoutError,
    RAGService,
    build_context,
    create_chat_model,
)


def document(chunk_id: str, content: str) -> Document:
    return Document(page_content=content, metadata={"source": "policy.txt", "chunk_id": chunk_id})


def test_service_builds_labelled_context_and_citations(config, stub_engine_class, fake_model_class) -> None:
    chunk = document("policy_txt_chunk_0001", "The refund period is 30 days.")
    model = fake_model_class(["The refund period is 30 days [1]."])
    service = RAGService(config, stub_engine_class([chunk]), model)

    result = asyncio.run(service.ask("What is the refund period?", request_id="test-id"))

    assert result.request_id == "test-id"
    assert result.sources[0].chunk_id == chunk.metadata["chunk_id"]
    assert result.sources[0].rank == 1
    assert result.metrics.retrieval_ms >= 0
    assert result.metrics.generation_ms >= 0
    assert "source=policy.txt chunk_id=policy_txt_chunk_0001" in model.calls[0][0].content
    assert model.calls[0][1].content == "What is the refund period?"
    assert build_context([chunk]).startswith("[1] source=policy.txt")


def test_no_context_does_not_call_groq(config, stub_engine_class, fake_model_class) -> None:
    model = fake_model_class([])
    service = RAGService(config, stub_engine_class([]), model)

    result = asyncio.run(service.ask("unknown question"))

    assert "not contain enough information" in result.answer
    assert result.sources == []
    assert result.metrics.generation_ms == 0
    assert model.calls == []


@pytest.mark.parametrize(
    ("response", "error"),
    [(TimeoutError("late"), ProviderTimeoutError), (RuntimeError("down"), ProviderError), (" ", EmptyModelResponseError)],
)
def test_generation_failures_are_explicit(config, stub_engine_class, fake_model_class, response, error) -> None:
    service = RAGService(config, stub_engine_class([document("a", "some context")]), fake_model_class([response]))
    with pytest.raises(error):
        asyncio.run(service.ask("some question"))


def test_query_rewrite_failure_falls_back_to_original(config, stub_engine_class, fake_model_class) -> None:
    config = replace(config, rewrite_enabled=True)
    engine = stub_engine_class([document("a", "Refund rules are written here.")])
    model = fake_model_class([RuntimeError("rewrite unavailable"), "Answer from the original context."])

    result = asyncio.run(RAGService(config, engine, model).ask("galaxy"))

    assert result.metrics.query_rewritten is False
    assert [call[0] for call in engine.calls] == ["galaxy"]
    assert result.answer == "Answer from the original context."


def test_rewritten_results_are_fused_and_deduplicated(config, stub_engine_class, fake_model_class) -> None:
    config = replace(config, rewrite_enabled=True)
    first = document("a", "First useful chunk")
    second = document("b", "Second useful chunk")
    engine = stub_engine_class([], results_by_query={"galaxy": [first], "better query": [second, first]})
    model = fake_model_class(["better query", "Answer [1] [2]."])

    result = asyncio.run(RAGService(config, engine, model).ask("galaxy"))

    assert result.metrics.query_rewritten is True
    assert {source.chunk_id for source in result.sources} == {"a", "b"}
    assert len(result.sources) == 2


def test_optional_reranker_changes_context_order(config, stub_engine_class, fake_model_class) -> None:
    first = document("a", "First chunk")
    second = document("b", "Second chunk")

    class ReverseReranker:
        def rerank(self, question, documents, k):
            return list(reversed(documents))[:k]

    service = RAGService(config, stub_engine_class([first, second]), fake_model_class(["Answer [1]."]), ReverseReranker())
    result = asyncio.run(service.ask("question"))

    assert result.metrics.reranker_enabled is True
    assert [source.chunk_id for source in result.sources] == ["b", "a"]


def test_chat_model_uses_environment_and_factual_settings(config, monkeypatch) -> None:
    options = {}

    class FakeChatGroq:
        def __init__(self, **kwargs):
            options.update(kwargs)

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "langchain_groq", SimpleNamespace(ChatGroq=FakeChatGroq))
    create_chat_model(replace(config, groq_model="test-groq-model", groq_timeout=12))

    assert options == {"model": "test-groq-model", "temperature": 0, "timeout": 12, "max_retries": 0}


def test_chat_model_requires_key_and_model(config, monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        create_chat_model(config)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    with pytest.raises(RuntimeError, match="GROQ_MODEL"):
        create_chat_model(config)
