"""Unit tests for the chunking logic in ingestion_service."""

import pytest

from app.providers.pdf.opendataloader_parser import ParsedPage
from app.services.ingestion_service import _chunk_pages


def _make_pages(texts: list[str]) -> list[ParsedPage]:
    return [ParsedPage(page_number=i + 1, content=t) for i, t in enumerate(texts)]


def test_no_empty_chunks():
    pages = _make_pages(["Kỹ thuật trồng lúa nước " * 100])
    chunks = _chunk_pages(
        pages=pages,
        chunk_size=200,
        chunk_overlap=30,
        document_id="doc-1",
        file_name="test.pdf",
        uploaded_at="2026-01-01T00:00:00Z",
    )
    assert all(c.content.strip() for c in chunks), "All chunks must be non-empty"


def test_chunks_have_required_metadata():
    pages = _make_pages(["Phòng trừ sâu bệnh " * 50])
    chunks = _chunk_pages(
        pages=pages,
        chunk_size=200,
        chunk_overlap=30,
        document_id="doc-abc",
        file_name="sample.pdf",
        uploaded_at="2026-01-01T00:00:00Z",
    )
    for chunk in chunks:
        assert chunk.metadata["document_id"] == "doc-abc"
        assert chunk.metadata["file_name"] == "sample.pdf"
        assert "page" in chunk.metadata
        assert "chunk_index" in chunk.metadata


def test_chunks_inherit_page_number():
    pages = _make_pages(["Trang một " * 50, "Trang hai " * 50])
    chunks = _chunk_pages(
        pages=pages,
        chunk_size=200,
        chunk_overlap=20,
        document_id="doc-2",
        file_name="multi.pdf",
        uploaded_at="2026-01-01T00:00:00Z",
    )
    page_numbers = {c.metadata["page"] for c in chunks}
    assert 1 in page_numbers
    assert 2 in page_numbers


def test_empty_page_produces_no_chunks():
    pages = _make_pages(["   ", "\n\n\n", ""])
    chunks = _chunk_pages(
        pages=pages,
        chunk_size=200,
        chunk_overlap=20,
        document_id="doc-3",
        file_name="empty.pdf",
        uploaded_at="2026-01-01T00:00:00Z",
    )
    assert len(chunks) == 0


def test_chunks_have_checksum():
    pages = _make_pages(["Nội dung kiểm thử " * 30])
    chunks = _chunk_pages(
        pages=pages,
        chunk_size=200,
        chunk_overlap=20,
        document_id="doc-4",
        file_name="test.pdf",
        uploaded_at="2026-01-01T00:00:00Z",
    )
    for chunk in chunks:
        assert "content_checksum" in chunk.metadata
        assert len(chunk.metadata["content_checksum"]) == 32  # MD5 hex length


def test_chunk_count_bounded_by_content():
    short_content = "Nội dung ngắn."
    pages = _make_pages([short_content])
    chunks = _chunk_pages(
        pages=pages,
        chunk_size=800,
        chunk_overlap=120,
        document_id="doc-5",
        file_name="short.pdf",
        uploaded_at="2026-01-01T00:00:00Z",
    )
    # A short text should produce exactly 1 chunk
    assert len(chunks) == 1
