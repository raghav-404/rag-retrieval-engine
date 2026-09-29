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
    manifest_path: Path
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 200
    chunk_overlap: int = 40
    dense_candidates: int = 20
    sparse_candidates: int = 20
    final_k: int = 5
    rrf_constant: int = 60
    dense_weight: float = 0.5
    sparse_weight: float = 0.5
    api_url: str = "http://localhost:8000/ask"

    def validate(self) -> None:
        if self.chunk_size <= 0 or not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("Chunk size must be positive and overlap must be smaller.")
        if min(self.dense_candidates, self.sparse_candidates, self.final_k) <= 0:
            raise ValueError("Retriever candidate counts and final_k must be positive.")
        if self.rrf_constant < 0 or min(self.dense_weight, self.sparse_weight) <= 0:
            raise ValueError("RRF constant must be nonnegative and weights must be positive.")

    @classmethod
    def from_env(cls) -> AppConfig:
        root = Path(os.getenv("RAG_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
        return cls(
            project_root=root,
            docs_dir=Path(os.getenv("RAG_DOCS_DIR", root / "docs")),
            index_path=Path(os.getenv("RAG_INDEX_PATH", root / "faiss.index")),
            metadata_path=Path(os.getenv("RAG_METADATA_PATH", root / "metadata.json")),
            manifest_path=Path(os.getenv("RAG_MANIFEST_PATH", root / "manifest.json")),
            embed_model_name=os.getenv("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", 200)),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", 40)),
            dense_candidates=int(os.getenv("RAG_DENSE_CANDIDATES", 20)),
            sparse_candidates=int(os.getenv("RAG_SPARSE_CANDIDATES", 20)),
            final_k=int(os.getenv("RAG_FINAL_K", 5)),
            rrf_constant=int(os.getenv("RAG_RRF_CONSTANT", 60)),
            dense_weight=float(os.getenv("RAG_DENSE_WEIGHT", 0.5)),
            sparse_weight=float(os.getenv("RAG_SPARSE_WEIGHT", 0.5)),
            api_url=os.getenv("RAG_API_URL", "http://localhost:8000/ask"),
        )
