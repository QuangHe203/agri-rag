"""Unit tests for BM25 sparse retrieval index."""

import tempfile
import os
import pytest

from app.providers.retrieval.bm25 import BM25Index, BM25Entry, _tokenize


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class TestTokenizer:
    def test_basic_ascii(self):
        tokens = _tokenize("lúa nước phòng bệnh")
        assert "phòng" in tokens or "ph" in tokens or len(tokens) > 0

    def test_lowercase(self):
        tokens = _tokenize("Lúa NƯỚC")
        for t in tokens:
            assert t == t.lower()

    def test_drops_single_char(self):
        tokens = _tokenize("a b c de")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "de" in tokens

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_vietnamese_diacritics(self):
        tokens = _tokenize("đạo ôn trên lúa")
        assert len(tokens) >= 1


# ---------------------------------------------------------------------------
# BM25Index — basic operations
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_index(tmp_path):
    """Return a fresh BM25Index backed by a temp file."""
    return BM25Index(persist_path=str(tmp_path / "bm25.pkl"))


def _make_entry(chunk_id: str, doc_id: str, content: str) -> BM25Entry:
    return BM25Entry(chunk_id=chunk_id, document_id=doc_id, content=content, metadata={})


class TestBM25IndexOperations:
    def test_empty_search_returns_empty(self, tmp_index):
        results = tmp_index.search("lúa nước", top_k=5)
        assert results == []

    def test_add_and_search(self, tmp_index):
        tmp_index.add_chunks([
            _make_entry("c1", "d1", "kỹ thuật trồng lúa nước"),
            _make_entry("c2", "d1", "phòng bệnh đạo ôn trên cây lúa"),
            _make_entry("c3", "d2", "chăn nuôi bò sữa"),
        ])
        results = tmp_index.search("lúa nước", top_k=5)
        assert len(results) > 0
        chunk_ids = [cid for cid, _, _ in results]
        assert "c1" in chunk_ids or "c2" in chunk_ids

    def test_no_overlap_returns_empty(self, tmp_index):
        tmp_index.add_chunks([_make_entry("c1", "d1", "kỹ thuật trồng lúa")])
        results = tmp_index.search("xyz1234notaword", top_k=5)
        assert results == []

    def test_top_k_limits_results(self, tmp_index):
        entries = [_make_entry(f"c{i}", "d1", f"lúa nước kỹ thuật trồng {i}") for i in range(10)]
        tmp_index.add_chunks(entries)
        results = tmp_index.search("lúa nước", top_k=3)
        assert len(results) <= 3

    def test_count(self, tmp_index):
        assert tmp_index.count() == 0
        tmp_index.add_chunks([_make_entry("c1", "d1", "nội dung"), _make_entry("c2", "d1", "khác")])
        assert tmp_index.count() == 2

    def test_skip_duplicate_chunk_ids(self, tmp_index):
        tmp_index.add_chunks([_make_entry("c1", "d1", "lúa")])
        tmp_index.add_chunks([_make_entry("c1", "d1", "lúa")])  # duplicate
        assert tmp_index.count() == 1

    def test_delete_by_ids(self, tmp_index):
        tmp_index.add_chunks([
            _make_entry("c1", "d1", "lúa nước"),
            _make_entry("c2", "d1", "cây ăn quả"),
        ])
        removed = tmp_index.delete_by_ids(["c1"])
        assert removed == 1
        assert tmp_index.count() == 1
        results = tmp_index.search("lúa", top_k=5)
        assert all(cid != "c1" for cid, _, _ in results)

    def test_delete_by_document_id(self, tmp_index):
        tmp_index.add_chunks([
            _make_entry("c1", "doc_a", "lúa nước kỹ thuật"),
            _make_entry("c2", "doc_a", "bệnh đạo ôn"),
            _make_entry("c3", "doc_b", "chăn nuôi"),
        ])
        removed = tmp_index.delete_by_document_id("doc_a")
        assert removed == 2
        assert tmp_index.count() == 1

    def test_delete_nonexistent_noop(self, tmp_index):
        tmp_index.add_chunks([_make_entry("c1", "d1", "lúa")])
        removed = tmp_index.delete_by_ids(["nonexistent"])
        assert removed == 0
        assert tmp_index.count() == 1


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestBM25IndexPersistence:
    def test_persist_and_reload(self, tmp_path):
        path = str(tmp_path / "bm25.pkl")
        idx1 = BM25Index(persist_path=path)
        # BM25 IDF = log((N - df + 0.5) / (df + 0.5)).
        # With N=2, df=1 the IDF is log(1)=0, giving score 0.
        # Use ≥3 docs so queries with df=1 get positive IDF.
        idx1.add_chunks([
            _make_entry("c1", "d1", "kỹ thuật trồng lúa nước"),
            _make_entry("c2", "d2", "chăn nuôi gia súc bò dê"),
            _make_entry("c3", "d3", "phòng bệnh cây ăn quả"),
        ])

        # Reload from disk
        idx2 = BM25Index(persist_path=path)
        assert idx2.count() == 3
        results = idx2.search("lúa nước kỹ thuật", top_k=5)
        assert any(cid == "c1" for cid, _, _ in results)

    def test_reload_missing_file_starts_empty(self, tmp_path):
        idx = BM25Index(persist_path=str(tmp_path / "nofile.pkl"))
        assert idx.count() == 0
        assert idx.search("anything", top_k=5) == []

    def test_reload_after_delete(self, tmp_path):
        path = str(tmp_path / "bm25.pkl")
        idx1 = BM25Index(persist_path=path)
        idx1.add_chunks([_make_entry("c1", "d1", "lúa"), _make_entry("c2", "d2", "ngô")])
        idx1.delete_by_ids(["c1"])

        idx2 = BM25Index(persist_path=path)
        assert idx2.count() == 1


# ---------------------------------------------------------------------------
# RRF helper (imported from retrieval_service)
# ---------------------------------------------------------------------------

class TestRRF:
    def test_single_list(self):
        from app.services.retrieval_service import _reciprocal_rank_fusion
        ranked = [("a", 0.9), ("b", 0.7), ("c", 0.5)]
        scores = _reciprocal_rank_fusion([ranked])
        # Rank 1 should score higher than rank 3
        assert scores["a"] > scores["b"] > scores["c"]

    def test_two_lists_boost_overlap(self):
        from app.services.retrieval_service import _reciprocal_rank_fusion
        v_ranked = [("a", 0.9), ("b", 0.8)]
        b_ranked = [("b", 5.0), ("c", 3.0)]
        scores = _reciprocal_rank_fusion([v_ranked, b_ranked])
        # "b" appears in both lists → should outscore "a" (only in vector)
        assert scores["b"] > scores["a"]

    def test_empty_lists(self):
        from app.services.retrieval_service import _reciprocal_rank_fusion
        assert _reciprocal_rank_fusion([]) == {}
        assert _reciprocal_rank_fusion([[]]) == {}
