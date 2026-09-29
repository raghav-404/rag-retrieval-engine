from langchain_huggingface import HuggingFaceEmbeddings
import torch

from .config import AppConfig


def create_embeddings(config: AppConfig) -> HuggingFaceEmbeddings:
    """Use the same normalized model for document and query vectors."""
    # FAISS and Torch can crash together on macOS when Torch uses multiple CPU threads.
    torch.set_num_threads(1)
    return HuggingFaceEmbeddings(
        model=config.embed_model_name,
        encode_kwargs={"normalize_embeddings": True},
        query_encode_kwargs={"normalize_embeddings": True},
    )
