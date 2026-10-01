from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Callable
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..config import AppConfig
from ..retrieval import RetrievalMode, artifacts_ready
from ..service import (
    ProviderError,
    ProviderTimeoutError,
    RAGResult,
    RAGService,
    RetrievalError,
    build_service,
)

logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    mode: RetrievalMode = "hybrid"


class RetrieveRequest(AskRequest):
    k: int = Field(default=5, ge=1, le=100)


def create_app(
    config: AppConfig | None = None,
    *,
    service_factory: Callable[[AppConfig], RAGService] = build_service,
) -> FastAPI:
    config = config or AppConfig.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.rag_service = service_factory(config)
        try:
            yield
        finally:
            del app.state.rag_service

    app = FastAPI(title="RAG Retrieval Engine API", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "artifacts_ready": artifacts_ready(config)}

    @app.post("/retrieve")
    async def retrieve_route(body: RetrieveRequest) -> dict[str, object]:
        if not body.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty.")
        request_id = uuid4().hex
        service: RAGService = app.state.rag_service
        try:
            documents = await asyncio.to_thread(service.engine.search, body.question, body.mode, k=body.k)
        except Exception as exc:
            logger.error("request_id=%s event=retrieval_failed error_type=%s", request_id, type(exc).__name__)
            raise HTTPException(
                status_code=503,
                detail={"message": "Document retrieval failed.", "request_id": request_id},
            ) from exc
        return {
            "request_id": request_id,
            "mode": body.mode,
            "results": [
                {"chunk_id": doc.metadata["chunk_id"], "source": doc.metadata["source"], "text": doc.page_content}
                for doc in documents
            ],
        }

    @app.post("/ask", response_model=RAGResult)
    async def ask_route(body: AskRequest) -> RAGResult:
        if not body.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty.")
        request_id = uuid4().hex
        service: RAGService = app.state.rag_service
        try:
            return await service.ask(body.question, body.mode, request_id=request_id)
        except ProviderTimeoutError as exc:
            logger.warning("request_id=%s event=provider_timeout", request_id)
            raise HTTPException(status_code=504, detail={"message": str(exc), "request_id": request_id}) from exc
        except ProviderError as exc:
            logger.warning("request_id=%s event=provider_failed error_type=%s", request_id, type(exc).__name__)
            raise HTTPException(status_code=502, detail={"message": str(exc), "request_id": request_id}) from exc
        except RetrievalError as exc:
            logger.error("request_id=%s event=retrieval_failed", request_id)
            raise HTTPException(status_code=503, detail={"message": str(exc), "request_id": request_id}) from exc

    return app


app = create_app()
