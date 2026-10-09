"""Vietnamese embedding provider using AITeamVN/Vietnamese_Embedding via sentence-transformers."""

import time
from typing import List

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_embedding_model = None


def get_embedding_model():
    """Return a singleton embedding model (loaded once on first call)."""
    global _embedding_model
    if _embedding_model is None:
        settings = get_settings()
        logger.info(
            "Loading embedding model: %s on device: %s",
            settings.EMBEDDING_MODEL,
            settings.EMBEDDING_DEVICE,
        )
        from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore

        _embedding_model = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": settings.EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
            show_progress=True,
        )
        logger.info("Embedding model loaded successfully")
    return _embedding_model


def embed_documents(texts: List[str]) -> List[List[float]]:
    """Embed a batch of documents and return their vectors."""
    model = get_embedding_model()
    start = time.time()
    vectors = model.embed_documents(texts)
    elapsed = time.time() - start
    logger.info("Embedded %d texts in %.2fs", len(texts), elapsed)
    return vectors


def embed_query(text: str) -> List[float]:
    """Embed a single query and return its vector."""
    model = get_embedding_model()
    return model.embed_query(text)


def is_model_loaded() -> bool:
    return _embedding_model is not None
