from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..config import AppConfig
from ..retrieval import ArtifactValidationError, ArtifactsNotReadyError, RetrievalMode, artifacts_ready, load_artifacts


class RetrieveRequest(BaseModel):
    question: str = Field(min_length=1)
    mode: RetrievalMode = "hybrid"
    k: int = Field(default=5, ge=1)


def create_app(config: AppConfig | None = None) -> FastAPI:
    config = config or AppConfig.from_env()
    app = FastAPI(title="RAG Retrieval Engine API", version="1.0.0")

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "artifacts_ready": artifacts_ready(config)}

    @app.post("/retrieve")
    def retrieve_route(body: RetrieveRequest) -> dict[str, object]:
        if not body.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty.")
        try:
            engine = load_artifacts(config)
            documents = engine.search(body.question, body.mode, k=body.k)
        except (ArtifactsNotReadyError, ArtifactValidationError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "mode": body.mode,
            "results": [
                {"chunk_id": doc.metadata["chunk_id"], "source": doc.metadata["source"], "text": doc.page_content}
                for doc in documents
            ],
        }

    @app.post("/ask")
    def ask_route() -> None:
        raise HTTPException(status_code=503, detail="Answer generation will be added in Phase 2.")

    return app


app = create_app()
