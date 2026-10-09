"""Retrieval service: embed query and fetch relevant chunks from vector store."""

import time
from typing import List, Optional

from app.core.config import get_settings
from app.core.logging import get_logger
from app.providers.embedding.vietnamese_embedding import embed_query
from app.providers.vector_store.chroma import RetrievalResult, get_vector_store

logger = get_logger(__name__)


class RetrievalService:
    """Handles similarity search against the vector store."""

    def retrieve(
        self,
        question: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        document_id_filter: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """
        Embed the question and retrieve the top-K most relevant chunks.

        Args:
            question: The user's question in Vietnamese.
            top_k: Number of results to return (defaults to settings.DEFAULT_TOP_K).
            min_score: Minimum similarity score threshold (defaults to settings.MIN_RELEVANCE_SCORE).
            document_id_filter: Optionally restrict retrieval to a single document.

        Returns:
            List of RetrievalResult sorted by score descending.
        """
        settings = get_settings()
        k = min(top_k or settings.DEFAULT_TOP_K, settings.MAX_TOP_K)
        threshold = min_score if min_score is not None else settings.MIN_RELEVANCE_SCORE

        logger.info(
            "Retrieval start: question_len=%d top_k=%d min_score=%.2f",
            len(question),
            k,
            threshold,
        )

        start_embed = time.time()
        query_vector = embed_query(question)
        elapsed_embed = time.time() - start_embed
        logger.info("Query embedded in %.2fs", elapsed_embed)

        where = None
        if document_id_filter:
            where = {"document_id": document_id_filter}

        start_search = time.time()
        vector_store = get_vector_store()
        results = vector_store.query(
            query_embedding=query_vector,
            top_k=k,
            min_score=threshold,
            where=where,
        )
        elapsed_search = time.time() - start_search
        logger.info(
            "Vector search returned %d results in %.2fs",
            len(results),
            elapsed_search,
        )

        return results
