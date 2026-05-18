from fastapi import FastAPI, HTTPException

from ..config import AppConfig
from ..retrieval import ArtifactsNotReadyError, artifacts_ready, ask


def create_app(config: AppConfig | None = None) -> FastAPI:
    config = config or AppConfig.from_env()
    app = FastAPI(title="RAG Retrieval Engine API", version="1.0.0")

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "artifacts_ready": artifacts_ready(config)}

    @app.post("/ask")
    def ask_route(body: dict[str, str]) -> dict[str, object]:
        question = body.get("question", "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty.")
        try:
            return ask(question, config)
        except ArtifactsNotReadyError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return app


app = create_app()
