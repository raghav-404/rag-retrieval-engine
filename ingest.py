"""Thin ingestion entrypoint kept for `python ingest.py`."""

from pathlib import Path
import sys

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_retrieval_engine.config import AppConfig
from rag_retrieval_engine.ingestion import build_index


if __name__ == "__main__":
    config = AppConfig.from_env()
    summary = build_index(config)
    print(f"Indexed {summary['chunk_count']} chunks from {summary['document_count']} documents.")
