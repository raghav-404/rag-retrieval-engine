from rag_retrieval_engine.ingestion import chunk_text


def test_chunk_text_uses_overlap_window() -> None:
    text = " ".join(f"word{i}" for i in range(12))

    chunks = chunk_text(text, chunk_size=5, overlap=2)

    assert chunks == [
        "word0 word1 word2 word3 word4",
        "word3 word4 word5 word6 word7",
        "word6 word7 word8 word9 word10",
        "word9 word10 word11",
    ]
