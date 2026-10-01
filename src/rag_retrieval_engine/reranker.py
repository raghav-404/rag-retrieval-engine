from __future__ import annotations

from langchain_core.documents import Document


class CrossEncoderReranker:
    """Score each query and candidate together, then keep the best chunks."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(self, question: str, documents: list[Document], k: int) -> list[Document]:
        if not documents:
            return []
        pairs = [(question, document.page_content) for document in documents]
        scores = self.model.predict(pairs)
        order = sorted(range(len(documents)), key=lambda index: float(scores[index]), reverse=True)
        return [documents[index] for index in order[:k]]
