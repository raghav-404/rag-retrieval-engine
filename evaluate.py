"""Run reproducible retrieval evaluation without using Groq by default."""

import argparse
import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from rag_retrieval_engine.config import AppConfig
from rag_retrieval_engine.embeddings import create_embeddings
from rag_retrieval_engine.evaluation import answer_metrics, load_cases, retrieval_metrics
from rag_retrieval_engine.reranker import CrossEncoderReranker
from rag_retrieval_engine.retrieval import load_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval against labelled questions.")
    parser.add_argument("--dataset", type=Path, default=Path("evaluation/questions.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/retrieval_baseline.json"))
    parser.add_argument("--with-answers", action="store_true", help="Also call Groq and check answer keywords.")
    args = parser.parse_args()

    config = AppConfig.from_env()
    embeddings = create_embeddings(config)
    engine = load_artifacts(config, embeddings=embeddings)
    cases = load_cases(args.dataset)
    reranker = CrossEncoderReranker(config.reranker_model) if config.reranker_enabled else None
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "embedding_model": engine.manifest["embedding_model"],
        "chunk_count": engine.manifest["chunk_count"],
        "question_count": len(cases),
        "retrieval": retrieval_metrics(engine, cases, reranker=reranker, rerank_candidates=config.rerank_candidates),
    }
    if args.with_answers:
        from rag_retrieval_engine.service import RAGService, create_chat_model

        service = RAGService(config, engine, create_chat_model(config), reranker)
        report["answers"] = asyncio.run(answer_metrics(service, cases))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Evaluated {len(cases)} questions; results saved to {args.output}")


if __name__ == "__main__":
    main()
