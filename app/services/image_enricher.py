"""Image enrichment layer for RAG responses.

This module is the sole integration point between the existing RAG pipeline
and the new image feature.  It is designed so that:

  * ``RAGService.answer()`` is never touched.
  * Any failure inside this layer (DB error, missing file, etc.) is caught
    and the original response is returned with ``images=[]``.
  * The enrichment is skipped entirely when ``IMAGE_EXTRACTION_ENABLED=false``
    or when no sources carry a page number.

Entry point: :func:`enrich_response`.
"""

from __future__ import annotations

from typing import List

from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.query import EnrichedQueryResponse, ImageReference, QueryResponse
from app.services.image_service import get_images_for_pages

logger = get_logger(__name__)


def enrich_response(rag_response: QueryResponse) -> EnrichedQueryResponse:
    """Attach relevant illustration images to *rag_response*.

    The function inspects the ``sources`` list, finds the pages referenced by
    the top-scored source chunks, and looks up images extracted from those
    pages during ingestion.  Images are returned in descending order of the
    source score of their page so that the most relevant illustration appears
    first.

    If image retrieval fails for any reason the function returns an
    :class:`EnrichedQueryResponse` with ``images=[]``, leaving all other
    fields from *rag_response* intact.

    Args:
        rag_response: The :class:`QueryResponse` produced by
                      :class:`~app.services.rag_service.RAGService`.

    Returns:
        An :class:`EnrichedQueryResponse` that is a superset of *rag_response*.
    """
    settings = get_settings()

    # Fast-path: feature disabled or no sources to work from.
    if not settings.IMAGE_EXTRACTION_ENABLED or not rag_response.sources:
        return _wrap(rag_response, [])

    try:
        images = _collect_images(rag_response, settings.MAX_IMAGES_PER_RESPONSE)
    except Exception as exc:
        logger.warning("Image enrichment failed, returning empty images: %s", exc)
        images = []

    return _wrap(rag_response, images)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _collect_images(
    rag_response: QueryResponse,
    max_images: int,
) -> List[ImageReference]:
    """Core logic: gather images for the pages referenced in *rag_response*."""

    # Build a mapping: (document_id, page_number) → max source score for that page.
    # This lets us rank pages by relevance and pick the best images first.
    page_scores: dict[tuple[str, int], float] = {}
    for source in rag_response.sources:
        if source.page is None:
            continue
        key = (source.document_id, source.page)
        page_scores[key] = max(page_scores.get(key, 0.0), source.score)

    if not page_scores:
        return []

    # Sort pages by their maximum source score, highest first.
    sorted_pages = sorted(page_scores.items(), key=lambda kv: kv[1], reverse=True)
    ranked_page_keys = [k for k, _ in sorted_pages]

    # Fetch all images for these pages in one DB round-trip.
    db_rows = get_images_for_pages(ranked_page_keys)

    if not db_rows:
        return []

    # Re-order results so images from the highest-scored page come first,
    # then convert to API schema objects.
    page_rank = {key: rank for rank, key in enumerate(ranked_page_keys)}
    db_rows.sort(
        key=lambda r: (page_rank.get((r.document_id, r.page_number), 9999), r.image_index)
    )

    images: List[ImageReference] = []
    for row in db_rows:
        if len(images) >= max_images:
            break
        images.append(
            ImageReference(
                document_id=row.document_id,
                file_name=row.file_name,
                page=row.page_number,
                image_index=row.image_index,
                url=(
                    f"/api/v1/images/{row.document_id}"
                    f"/{row.page_number}/{row.image_index}"
                ),
                width=row.width,
                height=row.height,
                mime_type=row.mime_type,
            )
        )

    logger.info(
        "Image enrichment: %d image(s) attached for question (sources=%d)",
        len(images),
        len(rag_response.sources),
    )
    return images


def _wrap(rag_response: QueryResponse, images: List[ImageReference]) -> EnrichedQueryResponse:
    """Convert *rag_response* to an :class:`EnrichedQueryResponse`."""
    return EnrichedQueryResponse(
        question=rag_response.question,
        answer=rag_response.answer,
        sources=rag_response.sources,
        model=rag_response.model,
        images=images,
    )
