"""FastAPI application entry point for the Agriculture RAG backend."""

# Use the OS system certificate store (fixes SSL errors behind corporate proxies
# that install a self-signed CA into the Windows/macOS/Linux trust store).
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.dependencies import request_id_middleware
from app.api.routes import documents, health, query
from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    settings = get_settings()
    setup_logging(level="DEBUG" if settings.APP_ENV == "development" else "INFO")

    from app.core.logging import get_logger
    logger = get_logger(__name__)
    logger.info("Agriculture RAG backend starting up (env=%s)", settings.APP_ENV)

    # Pre-load embedding model so the first request is not slow
    try:
        from app.providers.embedding.vietnamese_embedding import get_embedding_model
        get_embedding_model()
    except Exception as exc:
        logger.warning("Could not pre-load embedding model: %s", exc)

    # Ensure vector store is initialized
    try:
        from app.providers.vector_store.chroma import get_vector_store
        get_vector_store()
    except Exception as exc:
        logger.warning("Could not pre-initialize vector store: %s", exc)

    # Ensure SQLite DB / tables exist
    try:
        from app.models.document import get_engine
        get_engine()
    except Exception as exc:
        logger.warning("Could not initialize SQLite: %s", exc)

    logger.info("Agriculture RAG backend ready")
    yield
    logger.info("Agriculture RAG backend shutting down")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_TITLE,
        version=settings.APP_VERSION,
        description=(
            "Backend RAG hỏi đáp khuyến nông. "
            "Upload tài liệu PDF tiếng Việt và đặt câu hỏi bằng tiếng Việt."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request ID middleware
    app.add_middleware(BaseHTTPMiddleware, dispatch=request_id_middleware)

    # Exception handlers
    register_exception_handlers(app)

    # Routers
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(query.router)

    return app


app = create_app()
