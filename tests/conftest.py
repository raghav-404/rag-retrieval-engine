from pathlib import Path
import sys

import pytest

from langchain_core.embeddings import Embeddings

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_retrieval_engine.config import AppConfig


class TinyTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[str]:
        return [piece for word in text.split() for piece in ([word] * (3 if word == "extraordinary" else 1))]


class StaticEmbeddings(Embeddings):
    def __init__(self, vectors: dict[str, list[float]] | None = None, default: list[float] | None = None):
        self.vectors = vectors or {}
        self.default = default or [1.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors.get(text, self.default) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.vectors.get(text, self.default)


@pytest.fixture
def tokenizer() -> TinyTokenizer:
    return TinyTokenizer()


@pytest.fixture
def embeddings() -> StaticEmbeddings:
    return StaticEmbeddings()


@pytest.fixture
def config(tmp_path: Path) -> AppConfig:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    return AppConfig(
        project_root=tmp_path,
        docs_dir=docs_dir,
        index_path=tmp_path / "faiss.index",
        metadata_path=tmp_path / "metadata.json",
        manifest_path=tmp_path / "manifest.json",
        embed_model_name="test-embedder",
        chunk_size=5,
        chunk_overlap=2,
        dense_candidates=2,
        sparse_candidates=2,
        final_k=2,
    )
