from langchain_core.documents import Document

from rag_retrieval_engine.reranker import CrossEncoderReranker


def test_cross_encoder_orders_candidates_by_score() -> None:
    documents = [Document(page_content="first"), Document(page_content="second")]

    class FakeModel:
        def predict(self, pairs):
            assert pairs == [("question", "first"), ("question", "second")]
            return [0.2, 0.9]

    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker.model = FakeModel()

    assert reranker.rerank("question", documents, 1) == [documents[1]]
    assert reranker.rerank("question", [], 1) == []
