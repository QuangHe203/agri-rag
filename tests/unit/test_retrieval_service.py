"""Unit tests for the RetrievalService using mocked dependencies."""

from unittest.mock import MagicMock, patch

import pytest

from app.providers.vector_store.chroma import RetrievalResult
from app.services.retrieval_service import RetrievalService


def _make_result(chunk_id: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        content="Nội dung kiểm thử",
        metadata={"document_id": "doc-1", "file_name": "test.pdf", "page": 1},
        score=score,
    )


@patch("app.services.retrieval_service.get_vector_store")
@patch("app.services.retrieval_service.embed_query")
def test_retrieval_returns_sorted_results(mock_embed, mock_vs_factory):
    mock_embed.return_value = [0.1] * 1024
    mock_vs = MagicMock()
    mock_vs.query.return_value = [
        _make_result("chunk-3", 0.9),
        _make_result("chunk-1", 0.6),
        _make_result("chunk-2", 0.75),
    ]
    mock_vs_factory.return_value = mock_vs

    service = RetrievalService()
    results = service.retrieve(question="Làm sao để trồng lúa?", top_k=3)

    # Should be sorted by score descending
    assert results[0].score == 0.9
    assert results[1].score == 0.75
    assert results[2].score == 0.6


@patch("app.services.retrieval_service.get_vector_store")
@patch("app.services.retrieval_service.embed_query")
def test_retrieval_respects_top_k_limit(mock_embed, mock_vs_factory):
    mock_embed.return_value = [0.1] * 1024
    mock_vs = MagicMock()
    # Return 3 results even if 10 were requested (vector store handles filtering)
    mock_vs.query.return_value = [_make_result(f"chunk-{i}", 0.8) for i in range(3)]
    mock_vs_factory.return_value = mock_vs

    service = RetrievalService()
    results = service.retrieve(question="Câu hỏi", top_k=3)

    assert len(results) <= 3


@patch("app.services.retrieval_service.get_vector_store")
@patch("app.services.retrieval_service.embed_query")
def test_retrieval_empty_when_no_results(mock_embed, mock_vs_factory):
    mock_embed.return_value = [0.1] * 1024
    mock_vs = MagicMock()
    mock_vs.query.return_value = []
    mock_vs_factory.return_value = mock_vs

    service = RetrievalService()
    results = service.retrieve(question="Câu hỏi không có trong tài liệu")

    assert results == []
