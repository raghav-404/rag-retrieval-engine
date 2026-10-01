from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from pydantic import BaseModel, Field

from .reranker import CrossEncoderReranker
from .retrieval import RetrievalEngine


class EvaluationCase(BaseModel):
    question: str = Field(min_length=1)
    relevant_chunk_ids: list[str] = Field(min_length=1)
    expected_keywords: list[str] = Field(default_factory=list)


def load_cases(path: Path) -> list[EvaluationCase]:
    cases = [EvaluationCase.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cases:
        raise ValueError("Evaluation dataset has no cases.")
    return cases


def validate_labels(cases: list[EvaluationCase], engine: RetrievalEngine) -> None:
    known_ids = {doc.metadata["chunk_id"] for doc in engine.documents}
    unknown = {chunk_id for case in cases for chunk_id in case.relevant_chunk_ids if chunk_id not in known_ids}
    if unknown:
        raise ValueError(f"Evaluation labels are absent from the index: {', '.join(sorted(unknown))}")


def retrieval_metrics(
    engine: RetrievalEngine,
    cases: list[EvaluationCase],
    *,
    reranker: CrossEncoderReranker | None = None,
    rerank_candidates: int = 20,
) -> dict[str, dict]:
    """Measure retrieval only. This function never creates or calls Groq."""
    validate_labels(cases, engine)
    modes = ["dense", "sparse", "hybrid"]
    if reranker:
        modes.append("hybrid_reranked")
    results: dict[str, dict] = {}
    for mode in modes:
        recall = {1: 0.0, 3: 0.0, 5: 0.0}
        reciprocal_ranks = 0.0
        latencies: list[float] = []
        errors = 0
        case_results = []
        for case in cases:
            started = perf_counter()
            try:
                if mode == "hybrid_reranked":
                    candidates = engine.hybrid_retriever.invoke(case.question)[:rerank_candidates]
                    documents = reranker.rerank(case.question, candidates, 5)
                else:
                    documents = engine.search(case.question, mode, k=5)
                ids = [doc.metadata["chunk_id"] for doc in documents]
                relevant = set(case.relevant_chunk_ids)
                for k in recall:
                    recall[k] += len(relevant.intersection(ids[:k])) / len(relevant)
                first_rank = next((rank for rank, chunk_id in enumerate(ids, 1) if chunk_id in relevant), None)
                reciprocal_ranks += 1 / first_rank if first_rank else 0.0
                case_results.append({"question": case.question, "retrieved_chunk_ids": ids})
            except Exception as exc:
                errors += 1
                case_results.append({"question": case.question, "error_type": type(exc).__name__})
            latencies.append((perf_counter() - started) * 1000)
        count = len(cases)
        results[mode] = {
            "recall_at_1": round(recall[1] / count, 4),
            "recall_at_3": round(recall[3] / count, 4),
            "recall_at_5": round(recall[5] / count, 4),
            "mrr": round(reciprocal_ranks / count, 4),
            "average_retrieval_ms": round(sum(latencies) / count, 2),
            "error_count": errors,
            "error_rate": round(errors / count, 4),
            "cases": case_results,
        }
    return results


async def answer_metrics(service, cases: list[EvaluationCase]) -> dict:
    """Optional answer check: keyword presence and latency, never a faithfulness score."""
    latencies: list[float] = []
    errors = 0
    matched = 0
    scored = 0
    case_results = []
    for case in cases:
        try:
            result = await service.ask(case.question)
            latencies.append(result.metrics.generation_ms)
            keywords_match = all(word.casefold() in result.answer.casefold() for word in case.expected_keywords)
            if case.expected_keywords:
                scored += 1
                matched += int(keywords_match)
            case_results.append({"question": case.question, "keyword_match": keywords_match, "generation_ms": result.metrics.generation_ms})
        except Exception as exc:
            errors += 1
            case_results.append({"question": case.question, "error_type": type(exc).__name__})
    return {
        "average_generation_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "keyword_match_rate": round(matched / scored, 4) if scored else None,
        "error_count": errors,
        "error_rate": round(errors / len(cases), 4),
        "cases": case_results,
    }
