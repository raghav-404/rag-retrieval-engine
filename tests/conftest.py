from pathlib import Path
import sys

import pytest

from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

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


class FakeChatModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return AIMessage(content=result)


class StubEngine:
    def __init__(self, documents: list[Document], *, results_by_query=None):
        self.documents = documents
        self.results_by_query = results_by_query or {}
        self.calls = []

    def search(self, question, mode="hybrid", *, k=None):
        self.calls.append((question, mode, k))
        documents = self.results_by_query.get(question, self.documents)
        return documents[:k] if k else documents


@pytest.fixture
def fake_model_class():
    return FakeChatModel


@pytest.fixture
def stub_engine_class():
    return StubEngine


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
