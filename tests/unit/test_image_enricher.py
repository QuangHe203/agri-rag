"""Unit tests for the image enrichment layer.

All DB and filesystem calls are mocked so these tests run offline and never
touch real files or databases.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.prompts.agriculture_qa import FALLBACK_ANSWER
from app.schemas.query import (
    EnrichedQueryResponse,
    ImageReference,
    QueryResponse,
    SourceReference,
)
from app.services.image_enricher import _collect_images, _wrap, enrich_response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_source(
    document_id: str = "doc-1",
    file_name: str = "lua.pdf",
    page: int | None = 5,
    score: float = 0.85,
) -> SourceReference:
    return SourceReference(
        document_id=document_id,
        file_name=file_name,
        page=page,
        chunk_id="chunk-x",
        score=score,
        excerpt="Bệnh đạo ôn do nấm gây ra.",
    )


def _make_rag_response(sources: list[SourceReference]) -> QueryResponse:
    return QueryResponse(
        question="Phòng bệnh đạo ôn?",
        answer="Cần phun thuốc...",
        sources=sources,
        model="mock-llm",
    )


def _make_image_row(
    document_id: str = "doc-1",
    file_name: str = "lua.pdf",
    page_number: int = 5,
    image_index: int = 0,
    file_path: str = "/data/images/doc-1/page_0005_img_0000.png",
    width: int = 640,
    height: int = 480,
    mime_type: str = "image/png",
) -> MagicMock:
    row = MagicMock()
    row.document_id = document_id
    row.file_name = file_name
    row.page_number = page_number
    row.image_index = image_index
    row.file_path = file_path
    row.width = width
    row.height = height
    row.mime_type = mime_type
    return row


# ---------------------------------------------------------------------------
# _wrap helper
# ---------------------------------------------------------------------------

def test_wrap_produces_enriched_response():
    base = _make_rag_response([_make_source()])
    result = _wrap(base, [])
    assert isinstance(result, EnrichedQueryResponse)
    assert result.question == base.question
    assert result.answer == base.answer
    assert result.sources == base.sources
    assert result.model == base.model
    assert result.images == []


# ---------------------------------------------------------------------------
# enrich_response — feature disabled
# ---------------------------------------------------------------------------

def test_enrich_returns_empty_images_when_disabled():
    base = _make_rag_response([_make_source()])
    with patch("app.services.image_enricher.get_settings") as mock_settings:
        mock_settings.return_value.IMAGE_EXTRACTION_ENABLED = False
        mock_settings.return_value.MAX_IMAGES_PER_RESPONSE = 3
        result = enrich_response(base)
    assert isinstance(result, EnrichedQueryResponse)
    assert result.images == []
    # core fields intact
    assert result.answer == base.answer
    assert result.sources == base.sources


# ---------------------------------------------------------------------------
# enrich_response — no sources → no images
# ---------------------------------------------------------------------------

def test_enrich_returns_empty_images_when_no_sources():
    base = _make_rag_response([])
    with patch("app.services.image_enricher.get_settings") as mock_settings:
        mock_settings.return_value.IMAGE_EXTRACTION_ENABLED = True
        mock_settings.return_value.MAX_IMAGES_PER_RESPONSE = 3
        result = enrich_response(base)
    assert result.images == []


# ---------------------------------------------------------------------------
# enrich_response — sources without page numbers → no images
# ---------------------------------------------------------------------------

def test_enrich_returns_empty_images_when_sources_have_no_page():
    base = _make_rag_response([_make_source(page=None)])
    with (
        patch("app.services.image_enricher.get_settings") as mock_settings,
        patch("app.services.image_enricher.get_images_for_pages", return_value=[]) as mock_get,
    ):
        mock_settings.return_value.IMAGE_EXTRACTION_ENABLED = True
        mock_settings.return_value.MAX_IMAGES_PER_RESPONSE = 3
        result = enrich_response(base)
    # get_images_for_pages should not have been called
    mock_get.assert_not_called()
    assert result.images == []


# ---------------------------------------------------------------------------
# enrich_response — happy path: images found
# ---------------------------------------------------------------------------

def test_enrich_attaches_images_for_source_pages():
    source = _make_source(document_id="doc-1", page=5, score=0.9)
    base = _make_rag_response([source])
    row = _make_image_row(document_id="doc-1", page_number=5, image_index=0)

    with (
        patch("app.services.image_enricher.get_settings") as mock_settings,
        patch("app.services.image_enricher.get_images_for_pages", return_value=[row]),
    ):
        mock_settings.return_value.IMAGE_EXTRACTION_ENABLED = True
        mock_settings.return_value.MAX_IMAGES_PER_RESPONSE = 3
        result = enrich_response(base)

    assert len(result.images) == 1
    img = result.images[0]
    assert isinstance(img, ImageReference)
    assert img.document_id == "doc-1"
    assert img.page == 5
    assert img.image_index == 0
    assert img.url == "/api/v1/images/doc-1/5/0"
    assert img.width == 640
    assert img.height == 480


# ---------------------------------------------------------------------------
# enrich_response — max_images cap respected
# ---------------------------------------------------------------------------

def test_enrich_respects_max_images_cap():
    source = _make_source(page=3)
    base = _make_rag_response([source])
    rows = [_make_image_row(image_index=i) for i in range(5)]  # 5 available

    with (
        patch("app.services.image_enricher.get_settings") as mock_settings,
        patch("app.services.image_enricher.get_images_for_pages", return_value=rows),
    ):
        mock_settings.return_value.IMAGE_EXTRACTION_ENABLED = True
        mock_settings.return_value.MAX_IMAGES_PER_RESPONSE = 2
        result = enrich_response(base)

    assert len(result.images) == 2


# ---------------------------------------------------------------------------
# enrich_response — images ranked by source score (highest-scored page first)
# ---------------------------------------------------------------------------

def test_enrich_orders_images_by_source_score():
    low_score_source = _make_source(document_id="doc-1", page=2, score=0.5)
    high_score_source = _make_source(document_id="doc-1", page=7, score=0.95)
    base = _make_rag_response([low_score_source, high_score_source])

    row_page2 = _make_image_row(page_number=2, image_index=0)
    row_page7 = _make_image_row(page_number=7, image_index=0)

    # DB returns in arbitrary order (page 2 first)
    with (
        patch("app.services.image_enricher.get_settings") as mock_settings,
        patch(
            "app.services.image_enricher.get_images_for_pages",
            return_value=[row_page2, row_page7],
        ),
    ):
        mock_settings.return_value.IMAGE_EXTRACTION_ENABLED = True
        mock_settings.return_value.MAX_IMAGES_PER_RESPONSE = 3
        result = enrich_response(base)

    # Page 7 (higher score) should come first
    assert result.images[0].page == 7
    assert result.images[1].page == 2


# ---------------------------------------------------------------------------
# enrich_response — DB / enrichment failure → graceful fallback
# ---------------------------------------------------------------------------

def test_enrich_falls_back_gracefully_on_db_error():
    base = _make_rag_response([_make_source()])

    with (
        patch("app.services.image_enricher.get_settings") as mock_settings,
        patch(
            "app.services.image_enricher.get_images_for_pages",
            side_effect=RuntimeError("DB exploded"),
        ),
    ):
        mock_settings.return_value.IMAGE_EXTRACTION_ENABLED = True
        mock_settings.return_value.MAX_IMAGES_PER_RESPONSE = 3
        result = enrich_response(base)

    # Answer / sources must be unchanged
    assert result.answer == base.answer
    assert result.sources == base.sources
    # Images empty, not an exception
    assert result.images == []


# ---------------------------------------------------------------------------
# enrich_response — fallback response (no sources) also enriched correctly
# ---------------------------------------------------------------------------

def test_enrich_fallback_rag_response_returns_empty_images():
    """A RAG fallback (no sources) enriched → images is always []."""
    base = QueryResponse(
        question="Câu hỏi ngoài phạm vi",
        answer=FALLBACK_ANSWER,
        sources=[],
        model="mock-llm",
    )
    with patch("app.services.image_enricher.get_settings") as mock_settings:
        mock_settings.return_value.IMAGE_EXTRACTION_ENABLED = True
        mock_settings.return_value.MAX_IMAGES_PER_RESPONSE = 3
        result = enrich_response(base)

    assert result.answer == FALLBACK_ANSWER
    assert result.images == []
