"""Vietnamese CrossEncoder reranker provider.

Uses sentence_transformers.CrossEncoder to rerank a candidate list of
retrieval results for a given query.

Default model  : BAAI/bge-reranker-v2-m3   (multilingual, Vietnamese-capable)
Alternative     : AITeamVN/Vietnamese_Reranker (same team as embedding model)

Score range depends on the model — scores are raw logits from the
cross-encoder head; higher always means more relevant. Do NOT compare
these scores with cosine similarity scores from the vector store.
"""

import time
from typing import List, Optional

from app.core.logging import get_logger

logger = get_logger(__name__)

_reranker = None


def get_reranker():
    """Return the singleton CrossEncoder (lazy-loaded on first call)."""
    global _reranker
    if _reranker is None:
        from app.core.config import get_settings
        settings = get_settings()
        logger.info("Loading reranker model: %s (device=%s)", settings.RERANKER_MODEL, settings.EMBEDDING_DEVICE)
        start = time.time()
        try:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder(
                settings.RERANKER_MODEL,
                device=settings.EMBEDDING_DEVICE,
                max_length=512,
            )
            logger.info(
                "Reranker '%s' loaded in %.2fs",
                settings.RERANKER_MODEL,
                time.time() - start,
            )
        except Exception as exc:
            logger.error("Failed to load reranker '%s': %s", settings.RERANKER_MODEL, exc)
            raise
    return _reranker


def is_reranker_loaded() -> bool:
    """Return True if the reranker model has been loaded into memory."""
    return _reranker is not None


def rerank(
    question: str,
    candidates: list,  # List[RetrievalResult] — avoid circular import
    top_k: int,
) -> list:
    """
    Rerank *candidates* using the cross-encoder model.

    Args:
        question   : The original user question.
        candidates : Candidate RetrievalResult list (may be pre-sorted).
        top_k      : Maximum results to return after reranking.

    Returns:
        Up to top_k RetrievalResult objects sorted by cross-encoder score (desc).
        The `score` field of each result is **overwritten** with the reranker logit.
    """
    if not candidates:
        return candidates

    model = get_reranker()
    pairs = [(question, r.content) for r in candidates]

    start = time.time()
    raw_scores = model.predict(pairs)
    elapsed = time.time() - start
    logger.info("Reranker scored %d candidates in %.2fs", len(candidates), elapsed)

    ranked = sorted(
        zip(raw_scores, candidates),
        key=lambda x: float(x[0]),
        reverse=True,
    )
    out = []
    for score, result in ranked[:top_k]:
        result.score = float(score)
        out.append(result)
    return out
