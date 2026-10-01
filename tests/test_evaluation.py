import asyncio

import pytest
from langchain_core.documents import Document

from rag_retrieval_engine.evaluation import EvaluationCase, answer_metrics, load_cases, retrieval_metrics
from rag_retrieval_engine.service import RAGMetrics, RAGResult


def make_document(chunk_id):
    return Document(page_content=chunk_id, metadata={"source": f"{chunk_id}.txt", "chunk_id": chunk_id})


def test_retrieval_metrics_use_real_recall_and_mrr_formulas() -> None:
    a, b, c = (make_document(name) for name in "abc")
    cases = [
        EvaluationCase(question="one", relevant_chunk_ids=["a"]),
        EvaluationCase(question="two", relevant_chunk_ids=["c"]),
    ]
    rankings = {
        ("one", "dense"): [a, b, c], ("two", "dense"): [b, a, c],
        ("one", "sparse"): [b, a, c], ("two", "sparse"): [c, b, a],
        ("one", "hybrid"): [a, b, c], ("two", "hybrid"): [c, b, a],
    }

    class Engine:
        documents = [a, b, c]

        def search(self, question, mode, *, k):
            return rankings[(question, mode)][:k]

    results = retrieval_metrics(Engine(), cases)

    assert results["dense"]["recall_at_1"] == 0.5
    assert results["dense"]["recall_at_3"] == 1.0
    assert results["dense"]["mrr"] == 0.6667
    assert results["sparse"]["mrr"] == 0.75
    assert results["hybrid"]["recall_at_1"] == 1.0
    assert results["hybrid"]["error_count"] == 0
    assert "hybrid_reranked" not in results


def test_errors_count_as_zero_recall() -> None:
    a = make_document("a")
    cases = [EvaluationCase(question="one", relevant_chunk_ids=["a"])]

    class BrokenEngine:
        documents = [a]

        def search(self, question, mode, *, k):
            raise RuntimeError("retrieval failed")

    results = retrieval_metrics(BrokenEngine(), cases)
    assert results["dense"]["recall_at_5"] == 0
    assert results["dense"]["error_count"] == 1
    assert results["dense"]["error_rate"] == 1


def test_recall_counts_each_relevant_chunk_and_reranker_is_optional() -> None:
    a, b, c = (make_document(name) for name in "abc")
    cases = [EvaluationCase(question="one", relevant_chunk_ids=["a", "b"])]

    class Hybrid:
        def invoke(self, question):
            return [c, a, b]

    class Engine:
        documents = [a, b, c]
        hybrid_retriever = Hybrid()

        def search(self, question, mode, *, k):
            return [a, c][:k]

    class Reranker:
        def rerank(self, question, documents, k):
            return [a, b, c][:k]

    results = retrieval_metrics(Engine(), cases, reranker=Reranker())
    assert results["dense"]["recall_at_1"] == 0.5
    assert results["dense"]["recall_at_5"] == 0.5
    assert results["hybrid_reranked"]["recall_at_3"] == 1.0


def test_unknown_evaluation_labels_fail_before_scoring() -> None:
    class Engine:
        documents = []

    with pytest.raises(ValueError, match="absent from the index"):
        retrieval_metrics(Engine(), [EvaluationCase(question="one", relevant_chunk_ids=["missing"])])


def test_jsonl_loader_and_optional_answer_metrics(tmp_path) -> None:
    path = tmp_path / "questions.jsonl"
    path.write_text('{"question":"When?","relevant_chunk_ids":["a"],"expected_keywords":["30 days"]}\n', encoding="utf-8")
    cases = load_cases(path)

    class Service:
        async def ask(self, question):
            return RAGResult(
                answer="The period is 30 days.", sources=[], request_id="test",
                metrics=RAGMetrics(
                    retrieval_ms=1, generation_ms=8, total_ms=9,
                    retrieval_mode="hybrid", reranker_enabled=False, query_rewritten=False,
                ),
            )

    metrics = asyncio.run(answer_metrics(Service(), cases))
    assert metrics["keyword_match_rate"] == 1
    assert metrics["average_generation_ms"] == 8
    assert metrics["error_count"] == 0
