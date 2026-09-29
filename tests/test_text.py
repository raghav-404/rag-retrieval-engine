from dataclasses import replace

from rag_retrieval_engine.ingestion import chunk_documents


def test_chunks_use_tokenizer_counts_and_overlap(config, tokenizer) -> None:
    config.docs_dir.joinpath("policy.txt").write_text(
        "extraordinary alpha beta gamma delta epsilon zeta eta", encoding="utf-8"
    )
    config = replace(config, chunk_size=5, chunk_overlap=2)

    chunks, _ = chunk_documents(config, tokenizer)

    assert len(chunks) > 1  # Eight words occupy ten tokenizer tokens.
    assert all(0 < doc.metadata["token_count"] <= 5 for doc in chunks)
    first_words = set(chunks[0].page_content.split())
    second_words = set(chunks[1].page_content.split())
    assert first_words & second_words


def test_chunk_ids_and_metadata_are_stable(config, tokenizer) -> None:
    config.docs_dir.joinpath("policy.txt").write_text("alpha beta gamma delta epsilon zeta", encoding="utf-8")

    first, _ = chunk_documents(config, tokenizer)
    second, _ = chunk_documents(config, tokenizer)

    assert [doc.metadata for doc in first] == [doc.metadata for doc in second]
    assert first[0].metadata == {
        "source": "policy.txt",
        "chunk_id": "policy_txt_chunk_0001",
        "position": 1,
        "token_count": len(tokenizer.encode(first[0].page_content)),
    }
    assert [doc.id for doc in first] == [doc.metadata["chunk_id"] for doc in first]
