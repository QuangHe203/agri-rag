"""Unit tests for reranker provider and retrieval service reranking logic."""

from unittest.mock import MagicMock, patch

import pytest

from app.providers.vector_store.chroma import RetrievalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(chunk_id: str, content: str, score: float = 0.5) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        content=content,
        metadata={"document_id": "d1", "page": 1},
        score=score,
    )


# ---------------------------------------------------------------------------
# rerank() function
# ---------------------------------------------------------------------------

class TestRerankFunction:
    def test_empty_returns_empty(self):
        from app.providers.reranker.vietnamese_reranker import rerank
        with patch("app.providers.reranker.vietnamese_reranker.get_reranker") as mock_get:
            mock_model = MagicMock()
            mock_model.predict.return_value = []
            mock_get.return_value = mock_model
            result = rerank("question", [], top_k=5)
            assert result == []

    def test_scores_overwrite_original(self):
        from app.providers.reranker.vietnamese_reranker import rerank
        candidates = [
            _make_result("c1", "phòng bệnh lúa", score=0.9),
            _make_result("c2", "chăn nuôi bò", score=0.3),
        ]
        with patch("app.providers.reranker.vietnamese_reranker.get_reranker") as mock_get:
            mock_model = MagicMock()
            # Cross-encoder says c2 is actually more relevant
            mock_model.predict.return_value = [1.5, 3.7]
            mock_get.return_value = mock_model

            result = rerank("câu hỏi", candidates, top_k=2)

        assert len(result) == 2
        # c2 has higher reranker score → should come first
        assert result[0].chunk_id == "c2"
        assert result[1].chunk_id == "c1"
        # Scores overwritten with cross-encoder values
        assert result[0].score == pytest.approx(3.7)
        assert result[1].score == pytest.approx(1.5)

    def test_top_k_limits_output(self):
        from app.providers.reranker.vietnamese_reranker import rerank
        candidates = [_make_result(f"c{i}", f"nội dung {i}") for i in range(5)]
        with patch("app.providers.reranker.vietnamese_reranker.get_reranker") as mock_get:
            mock_model = MagicMock()
            mock_model.predict.return_value = [5.0, 4.0, 3.0, 2.0, 1.0]
            mock_get.return_value = mock_model

            result = rerank("question", candidates, top_k=3)

        assert len(result) == 3

    def test_model_receives_correct_pairs(self):
        from app.providers.reranker.vietnamese_reranker import rerank
        question = "lúa nước phòng bệnh"
        candidates = [_make_result("c1", "đạo ôn"), _make_result("c2", "rầy nâu")]

        with patch("app.providers.reranker.vietnamese_reranker.get_reranker") as mock_get:
            mock_model = MagicMock()
            mock_model.predict.return_value = [0.8, 0.6]
            mock_get.return_value = mock_model

            rerank(question, candidates, top_k=2)

            called_pairs = mock_model.predict.call_args[0][0]
            assert called_pairs == [(question, "đạo ôn"), (question, "rầy nâu")]

    def test_is_reranker_loaded_false_initially(self):
        import importlib
        import app.providers.reranker.vietnamese_reranker as mod
        # Save and reset singleton
        original = mod._reranker
        mod._reranker = None
        try:
            assert mod.is_reranker_loaded() is False
        finally:
            mod._reranker = original

    def test_is_reranker_loaded_true_after_mock(self):
        import app.providers.reranker.vietnamese_reranker as mod
        original = mod._reranker
        mod._reranker = MagicMock()
        try:
            assert mod.is_reranker_loaded() is True
        finally:
            mod._reranker = original


# ---------------------------------------------------------------------------
# RetrievalService._rerank integration path
# ---------------------------------------------------------------------------

class TestRetrievalServiceReranking:
    def test_rerank_called_when_enabled(self):
        """RetrievalService._rerank should call the reranker with correct args."""
        from app.services.retrieval_service import RetrievalService

        service = RetrievalService()
        candidates = [_make_result("c1", "lúa nước"), _make_result("c2", "ngô"), _make_result("c3", "cà phê")]

        settings_mock = MagicMock()
        settings_mock.RERANK_CANDIDATE_MULTIPLIER = 2

        with patch("app.services.retrieval_service.rerank", create=True) as mock_rerank:
            # Provide the mock via import patch inside _rerank
            with patch(
                "app.providers.reranker.vietnamese_reranker.rerank",
                return_value=candidates[:2],
            ) as patched:
                result = service._rerank("question", candidates, top_k=2, settings=settings_mock)
                # Should feed top 2*2=4 candidates (capped by len=3)
                patched.assert_called_once()
                call_args = patched.call_args
                assert call_args[1]["top_k"] == 2 or call_args[0][2] == 2
