"""Health check endpoint."""

from fastapi import APIRouter
from pydantic import BaseModel

from app.providers.embedding.vietnamese_embedding import is_model_loaded
from app.providers.vector_store.chroma import get_vector_store

router = APIRouter()


class ComponentStatus(BaseModel):
    vector_store: str
    embedding_model: str
    llm: str


class HealthResponse(BaseModel):
    status: str
    components: ComponentStatus


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Kiểm tra trạng thái hệ thống",
)
async def health_check() -> HealthResponse:
    """
    Lightweight health check.
    Does NOT make a real LLM call — only verifies configuration and vector store connectivity.
    """
    from app.core.config import get_settings

    settings = get_settings()

    # Vector store
    vector_store_status = "down"
    try:
        vs = get_vector_store()
        vector_store_status = "up" if vs.is_available() else "down"
    except Exception:
        vector_store_status = "down"

    # Embedding model
    embedding_status = "loaded" if is_model_loaded() else "not_loaded"

    # LLM — only check configuration, do not call the API
    llm_status = "configured" if settings.GEMINI_API_KEY else "not_configured"

    overall = (
        "healthy"
        if vector_store_status == "up"
        else "degraded"
    )

    return HealthResponse(
        status=overall,
        components=ComponentStatus(
            vector_store=vector_store_status,
            embedding_model=embedding_status,
            llm=llm_status,
        ),
    )
