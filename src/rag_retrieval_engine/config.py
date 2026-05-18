from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    docs_dir: Path
    index_path: Path
    metadata_path: Path
    ollama_url: str
    model_name: str
    embed_model_name: str
    top_k: int = 10
    rerank_top_n: int = 4
    eval_threshold: float = 0.08
    api_url: str = "http://localhost:8000/ask"

    @classmethod
    def from_env(cls) -> "AppConfig":
        root = Path(os.getenv("RAG_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
        return cls(
            project_root=root,
            docs_dir=Path(os.getenv("RAG_DOCS_DIR", root / "docs")),
            index_path=Path(os.getenv("RAG_INDEX_PATH", root / "faiss.index")),
            metadata_path=Path(os.getenv("RAG_METADATA_PATH", root / "metadata.json")),
            ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate"),
            model_name=os.getenv("RAG_MODEL_NAME", "qwen2.5:7b"),
            embed_model_name=os.getenv("RAG_EMBED_MODEL", "all-MiniLM-L6-v2"),
            top_k=int(os.getenv("RAG_TOP_K", 10)),
            rerank_top_n=int(os.getenv("RAG_RERANK_TOP_N", 4)),
            eval_threshold=float(os.getenv("RAG_EVAL_THRESHOLD", 0.08)),
            api_url=os.getenv("RAG_API_URL", "http://localhost:8000/ask"),
        )
