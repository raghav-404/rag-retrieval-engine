from __future__ import annotations

import asyncio
import logging
import os
from time import perf_counter
from uuid import uuid4

from groq import APITimeoutError
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from .config import AppConfig
from .embeddings import create_embeddings
from .reranker import CrossEncoderReranker
from .retrieval import (
    ArtifactsNotReadyError,
    RetrievalEngine,
    RetrievalMode,
    artifacts_ready,
    bm25_tokens,
    load_artifacts,
)

logger = logging.getLogger(__name__)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer only from the context below. If it does not contain the answer, say the "
            "information is insufficient. Do not invent facts. Keep the answer focused and "
            "refer to source labels such as [1] when helpful.\n\nContext:\n{context}",
        ),
        ("human", "{question}"),
    ]
)
REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "Rewrite the question as one concise document search query. Return only the query."),
        ("human", "{question}"),
    ]
)


class SourceRef(BaseModel):
    source: str
    chunk_id: str
    rank: int


class RAGMetrics(BaseModel):
    retrieval_ms: float
    generation_ms: float
    total_ms: float
    retrieval_mode: RetrievalMode
    reranker_enabled: bool
    query_rewritten: bool


class RAGResult(BaseModel):
    answer: str
    sources: list[SourceRef]
    metrics: RAGMetrics
    request_id: str


class RetrievalError(RuntimeError):
    pass


class ProviderError(RuntimeError):
    pass


class ProviderTimeoutError(ProviderError):
    pass


class EmptyModelResponseError(ProviderError):
    pass


def build_context(documents: list[Document]) -> str:
    return "\n\n".join(
        f"[{rank}] source={doc.metadata['source']} chunk_id={doc.metadata['chunk_id']}\n{doc.page_content}"
        for rank, doc in enumerate(documents, start=1)
    )


def fuse_query_results(original: list[Document], rewritten: list[Document], c: int = 60) -> list[Document]:
    """RRF keeps chunks found by either query and rewards chunks found by both."""
    scores: dict[str, float] = {}
    by_id: dict[str, Document] = {}
    for ranked_list in (original, rewritten):
        for rank, document in enumerate(ranked_list, start=1):
            chunk_id = document.metadata["chunk_id"]
            by_id.setdefault(chunk_id, document)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (c + rank)
    return sorted(by_id.values(), key=lambda doc: scores[doc.metadata["chunk_id"]], reverse=True)


class RAGService:
    def __init__(self, config: AppConfig, engine: RetrievalEngine, model, reranker: CrossEncoderReranker | None = None):
        self.config = config
        self.engine = engine
        self.model = model
        self.reranker = reranker

    def _poor_retrieval(self, question: str, documents: list[Document]) -> bool:
        """Rewrite only when no chunk contains a query term of at least five characters."""
        if not documents:
            return True
        terms = {term for term in bm25_tokens(question) if len(term) >= 5}
        if not terms:
            return False
        return not any(terms.intersection(bm25_tokens(doc.page_content)) for doc in documents)

    async def _rewrite(self, question: str, request_id: str) -> str | None:
        try:
            response = await self.model.ainvoke(REWRITE_PROMPT.format_messages(question=question))
            rewritten = response.content.strip() if isinstance(response.content, str) else ""
            return rewritten.splitlines()[0][:300] if rewritten else None
        except Exception as exc:
            logger.warning("request_id=%s event=rewrite_failed error_type=%s", request_id, type(exc).__name__)
            return None

    async def _retrieve(self, question: str, mode: RetrievalMode, request_id: str) -> tuple[list[Document], bool]:
        candidate_count = self.config.rerank_candidates if self.reranker and mode == "hybrid" else self.config.final_k
        try:
            documents = await asyncio.to_thread(self.engine.search, question, mode, k=candidate_count)
        except Exception as exc:
            raise RetrievalError("Document retrieval failed.") from exc
        rewritten_used = False
        if self.config.rewrite_enabled and self._poor_retrieval(question, documents):
            rewritten = await self._rewrite(question, request_id)
            if rewritten and rewritten.casefold() != question.casefold():
                try:
                    second = await asyncio.to_thread(self.engine.search, rewritten, mode, k=candidate_count)
                    documents = fuse_query_results(documents, second, self.config.rrf_constant)[:candidate_count]
                    rewritten_used = True
                except Exception as exc:
                    logger.warning("request_id=%s event=rewrite_search_failed error_type=%s", request_id, type(exc).__name__)
        if self.reranker and mode == "hybrid":
            try:
                documents = await asyncio.to_thread(self.reranker.rerank, question, documents, self.config.final_k)
            except Exception as exc:
                raise RetrievalError("Reranking failed.") from exc
        return [doc for doc in documents[: self.config.final_k] if doc.page_content.strip()], rewritten_used

    async def _generate(self, question: str, context: str) -> str:
        messages = ANSWER_PROMPT.format_messages(question=question, context=context)
        try:
            response = await self.model.ainvoke(messages)
        except (TimeoutError, APITimeoutError) as exc:
            raise ProviderTimeoutError("Groq request timed out.") from exc
        except Exception as exc:
            raise ProviderError("Groq request failed.") from exc
        content = getattr(response, "content", None)
        answer = content.strip() if isinstance(content, str) else ""
        if not answer:
            raise EmptyModelResponseError("Groq returned an empty response.")
        return answer

    async def ask(self, question: str, mode: RetrievalMode = "hybrid", *, request_id: str | None = None) -> RAGResult:
        if not question.strip():
            raise ValueError("Question cannot be empty.")
        request_id = request_id or uuid4().hex
        started = perf_counter()
        documents, rewritten = await self._retrieve(question.strip(), mode, request_id)
        retrieval_ms = (perf_counter() - started) * 1000
        reranked = self.reranker is not None and mode == "hybrid"
        sources = [
            SourceRef(source=doc.metadata["source"], chunk_id=doc.metadata["chunk_id"], rank=rank)
            for rank, doc in enumerate(documents, start=1)
        ]
        generation_ms = 0.0
        if documents:
            context = build_context(documents)
            generation_started = perf_counter()
            try:
                answer = await self._generate(question.strip(), context)
            finally:
                generation_ms = (perf_counter() - generation_started) * 1000
        else:
            answer = "The indexed documents do not contain enough information to answer this question."
        total_ms = (perf_counter() - started) * 1000
        logger.info(
            "request_id=%s event=answer_complete mode=%s sources=%d total_ms=%.1f",
            request_id, mode, len(sources), total_ms,
        )
        return RAGResult(
            answer=answer,
            sources=sources,
            metrics=RAGMetrics(
                retrieval_ms=round(retrieval_ms, 1),
                generation_ms=round(generation_ms, 1),
                total_ms=round(total_ms, 1),
                retrieval_mode=mode,
                reranker_enabled=reranked,
                query_rewritten=rewritten,
            ),
            request_id=request_id,
        )


def create_chat_model(config: AppConfig):
    """Read Groq settings only at startup; no credentials enter the code or logs."""
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is required to start answer generation.")
    if not config.groq_model:
        raise RuntimeError("GROQ_MODEL is required to start answer generation.")
    from langchain_groq import ChatGroq

    return ChatGroq(model=config.groq_model, temperature=0, timeout=config.groq_timeout, max_retries=0)


def build_service(config: AppConfig) -> RAGService:
    """FastAPI lifespan calls this once, then reuses the returned service."""
    if not artifacts_ready(config):
        raise ArtifactsNotReadyError("Retrieval artifacts are missing. Run `python ingest.py` first.")
    embeddings = create_embeddings(config)
    engine = load_artifacts(config, embeddings=embeddings)
    reranker = CrossEncoderReranker(config.reranker_model) if config.reranker_enabled else None
    model = create_chat_model(config)
    return RAGService(config, engine, model, reranker)
