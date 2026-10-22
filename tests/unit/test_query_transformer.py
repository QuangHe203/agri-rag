"""Unit tests for query transformation (Query Rewrite / Multi-query / HyDE)."""

from unittest.mock import MagicMock, patch

import pytest

from app.providers.vector_store.chroma import RetrievalResult
from app.services.query_transformer import QueryTransformer, _dedupe_keep_order
from app.services.retrieval_service import RetrievalService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _settings(
    rewrite=False, multi=False, hyde=False, count=3, mult=4, hybrid=False
):
    s = MagicMock()
    s.QUERY_REWRITE_ENABLED = rewrite
    s.MULTI_QUERY_ENABLED = multi
    s.HYDE_ENABLED = hyde
    s.MULTI_QUERY_COUNT = count
    s.QUERY_TRANSFORM_CANDIDATE_MULTIPLIER = mult
    s.HYBRID_SEARCH_ENABLED = hybrid
    # Attributes read by RetrievalService.retrieve()
    s.DEFAULT_TOP_K = 5
    s.MAX_TOP_K = 10
    s.MIN_RELEVANCE_SCORE = 0.0
    s.RERANK_ENABLED = False
    return s


def _result(chunk_id, score):
    return RetrievalResult(
        chunk_id=chunk_id,
        content=f"nội dung {chunk_id}",
        metadata={"document_id": "d1", "file_name": "f.pdf", "page": 1},
        score=score,
    )


# ---------------------------------------------------------------------------
# build_plan
# ---------------------------------------------------------------------------

class TestBuildPlan:
    def test_noop_when_all_disabled(self):
        mock_llm = MagicMock()
        transformer = QueryTransformer(llm_provider=mock_llm)
        with patch("app.services.query_transformer.get_settings", return_value=_settings()):
            plan = transformer.build_plan("trồng lúa thế nào?")

        assert plan.transformed is False
        assert plan.queries == ["trồng lúa thế nào?"]
        assert plan.dense_texts == ["trồng lúa thế nào?"]
        # No LLM call on the fast path
        mock_llm.generate.assert_not_called()

    def test_rewrite_replaces_base_question(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "kỹ thuật trồng lúa nước"
        transformer = QueryTransformer(llm_provider=mock_llm)
        with patch("app.services.query_transformer.get_settings", return_value=_settings(rewrite=True)):
            plan = transformer.build_plan("trồng lúa thế nào?")

        assert plan.transformed is True
        assert plan.queries == ["kỹ thuật trồng lúa nước"]
        assert plan.dense_texts == ["kỹ thuật trồng lúa nước"]

    def test_multi_query_adds_variants(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = (
            "cách trồng lúa\nkỹ thuật canh tác lúa nước\nhướng dẫn gieo sạ lúa"
        )
        transformer = QueryTransformer(llm_provider=mock_llm)
        with patch("app.services.query_transformer.get_settings", return_value=_settings(multi=True, count=3)):
            plan = transformer.build_plan("trồng lúa thế nào?")

        assert plan.transformed is True
        # base question + 3 variants
        assert plan.queries[0] == "trồng lúa thế nào?"
        assert len(plan.queries) == 4
        assert "cách trồng lúa" in plan.queries
        assert plan.dense_texts == plan.queries

    def test_hyde_sets_dense_texts(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Lúa nước cần đất phù sa, bón phân NPK và tưới tiêu hợp lý."
        transformer = QueryTransformer(llm_provider=mock_llm)
        with patch("app.services.query_transformer.get_settings", return_value=_settings(hyde=True)):
            plan = transformer.build_plan("trồng lúa thế nào?")

        assert plan.transformed is True
        assert plan.queries == ["trồng lúa thế nào?"]
        assert plan.dense_texts == ["Lúa nước cần đất phù sa, bón phân NPK và tưới tiêu hợp lý."]

    def test_rewrite_failure_falls_back_to_original(self):
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = RuntimeError("LLM down")
        transformer = QueryTransformer(llm_provider=mock_llm)
        with patch("app.services.query_transformer.get_settings", return_value=_settings(rewrite=True)):
            plan = transformer.build_plan("trồng lúa thế nào?")

        # Graceful degradation: no transformation applied
        assert plan.transformed is False
        assert plan.queries == ["trồng lúa thế nào?"]

    def test_multi_query_respects_count_limit(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "a\nb\nc\nd\ne"
        transformer = QueryTransformer(llm_provider=mock_llm)
        with patch("app.services.query_transformer.get_settings", return_value=_settings(multi=True, count=2)):
            plan = transformer.build_plan("câu gốc")

        # base + at most 2 variants
        assert len(plan.queries) == 3


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

class TestParsing:
    def test_parse_lines_strips_markers(self):
        text = "1. cách trồng lúa\n2) kỹ thuật lúa\n- bón phân\n* tưới nước\n• thu hoạch"
        result = QueryTransformer._parse_lines(text)
        assert result == [
            "cách trồng lúa",
            "kỹ thuật lúa",
            "bón phân",
            "tưới nước",
            "thu hoạch",
        ]

    def test_parse_lines_handles_empty(self):
        assert QueryTransformer._parse_lines(None) == []
        assert QueryTransformer._parse_lines("\n\n  \n") == []

    def test_dedupe_keep_order_case_insensitive(self):
        assert _dedupe_keep_order(["Lúa", "lúa ", "ngô", "NGÔ"]) == ["Lúa", "ngô"]


# ---------------------------------------------------------------------------
# RetrievalService multi-query fusion path
# ---------------------------------------------------------------------------

class TestMultiQueryRetrieval:
    @patch("app.services.retrieval_service.get_vector_store")
    @patch("app.services.retrieval_service.embed_query")
    def test_transformed_plan_uses_fusion_path(self, mock_embed, mock_vs_factory):
        mock_embed.return_value = [0.1] * 1024
        mock_vs = MagicMock()
        # Two dense legs return overlapping + distinct chunks
        mock_vs.query.side_effect = [
            [_result("c1", 0.9), _result("c2", 0.6)],
            [_result("c2", 0.8), _result("c3", 0.5)],
        ]
        mock_vs_factory.return_value = mock_vs

        # Transformer yields a 2-query plan (multi-query), no HyDE, no hybrid
        mock_transformer = MagicMock()
        from app.services.query_transformer import QueryPlan
        mock_transformer.build_plan.return_value = QueryPlan(
            original="q",
            queries=["q", "q2"],
            dense_texts=["q", "q2"],
            transformed=True,
        )

        service = RetrievalService(query_transformer=mock_transformer)
        with patch(
            "app.services.retrieval_service.get_settings",
            return_value=_settings(multi=True),
        ):
            results = service.retrieve(question="q", top_k=5)

        # Two dense legs were queried
        assert mock_vs.query.call_count == 2
        # All three unique chunks fused, sorted by RRF score
        ids = {r.chunk_id for r in results}
        assert ids == {"c1", "c2", "c3"}
        # c2 appears in both lists → highest RRF score → ranked first
        assert results[0].chunk_id == "c2"

    @patch("app.services.retrieval_service.get_vector_store")
    @patch("app.services.retrieval_service.embed_query")
    def test_noop_plan_uses_vector_path(self, mock_embed, mock_vs_factory):
        mock_embed.return_value = [0.1] * 1024
        mock_vs = MagicMock()
        mock_vs.query.return_value = [_result("c1", 0.9)]
        mock_vs_factory.return_value = mock_vs

        mock_transformer = MagicMock()
        from app.services.query_transformer import QueryPlan
        mock_transformer.build_plan.return_value = QueryPlan(
            original="q", queries=["q"], dense_texts=["q"], transformed=False
        )

        service = RetrievalService(query_transformer=mock_transformer)
        with patch(
            "app.services.retrieval_service.get_settings",
            return_value=_settings(),
        ):
            results = service.retrieve(question="q", top_k=5)

        # Single vector query (baseline path)
        assert mock_vs.query.call_count == 1
        assert results[0].chunk_id == "c1"
