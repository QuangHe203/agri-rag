"""Retrieval service: embed query and fetch relevant chunks.

Supports three modes controlled by environment settings:

  1. Vector-only (default)
       Dense cosine similarity search via ChromaDB.

  2. Hybrid: BM25 + Vector  (HYBRID_SEARCH_ENABLED=true)
       Two-leg retrieval fused with Reciprocal Rank Fusion (RRF):
         score_rrf(d) = Σ 1 / (k + rank_i(d)),  k = 60

  3. + Reranking  (RERANK_ENABLED=true)
       CrossEncoder applied on top of either mode.
       Score scale changes from cosine/RRF to cross-encoder logit.

  4. + Query transformation  (QUERY_REWRITE/MULTI_QUERY/HYDE)
       An LLM expands the question into one or more search queries (and/or
       HyDE passages) before retrieval.  Each query/leg is retrieved
       independently and fused with the same RRF used for hybrid search.
"""

import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.metrics import observe_stage_seconds
from app.providers.embedding.vietnamese_embedding import embed_query
from app.providers.vector_store.chroma import RetrievalResult, get_vector_store
from app.services.query_transformer import QueryPlan, QueryTransformer, get_query_transformer

logger = get_logger(__name__)

_RRF_K = 60  # Standard RRF constant (Cormack et al., 2009)


# ---------------------------------------------------------------------------
# RRF helper
# ---------------------------------------------------------------------------

def _reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, float]]],
) -> Dict[str, float]:
    """Compute Reciprocal Rank Fusion scores across multiple ranked lists.

    Args:
        ranked_lists: Each element is a list of (chunk_id, score) already
                      sorted by score descending.
    Returns:
        Dict mapping chunk_id → RRF score.
    """
    rrf: Dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, (chunk_id, _) in enumerate(ranked, start=1):
            rrf[chunk_id] += 1.0 / (_RRF_K + rank)
    return dict(rrf)


# ---------------------------------------------------------------------------
# Retrieval service
# ---------------------------------------------------------------------------

class RetrievalService:
    """Orchestrates query transformation, dense/BM25 search, and reranking."""

    def __init__(self, query_transformer: Optional[QueryTransformer] = None):
        # Lazy singleton; LLM inside is only loaded if a transformation is enabled.
        self._transformer = query_transformer or get_query_transformer()

    def retrieve(
        self,
        question: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        document_id_filter: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve the most relevant chunks for *question*.

        Args:
            question           : User question in Vietnamese.
            top_k              : Max results to return (capped at MAX_TOP_K).
            min_score          : Override MIN_RELEVANCE_SCORE (vector-only mode).
                                 Ignored in hybrid mode; not applicable after reranking.
            document_id_filter : Restrict search to a specific document.

        Returns:
            List[RetrievalResult] sorted by relevance (score desc).
        """
        settings = get_settings()
        k = min(top_k or settings.DEFAULT_TOP_K, settings.MAX_TOP_K)
        threshold = min_score if min_score is not None else settings.MIN_RELEVANCE_SCORE

        # ── Step 0: optional query transformation (rewrite / multi-query / HyDE)
        _qt_start = time.time()
        plan = self._transformer.build_plan(question)
        if plan.transformed:
            observe_stage_seconds("query_transform", time.time() - _qt_start)

        logger.info(
            "Retrieval start: question_len=%d top_k=%d min_score=%.2f "
            "hybrid=%s rerank=%s transformed=%s",
            len(question), k, threshold,
            settings.HYBRID_SEARCH_ENABLED,
            settings.RERANK_ENABLED,
            plan.transformed,
        )

        # ── Step 1: candidate retrieval ──────────────────────────────────────
        if plan.transformed:
            candidates = self._multi_query_retrieve(plan, k, document_id_filter, settings)
        elif settings.HYBRID_SEARCH_ENABLED:
            candidates = self._hybrid_retrieve(question, k, document_id_filter, settings)
        else:
            candidates = self._vector_retrieve(question, k, threshold, document_id_filter)

        if not candidates:
            return []

        # ── Step 2: optional reranking ────────────────────────────────────────
        if settings.RERANK_ENABLED:
            candidates = self._rerank(question, candidates, k, settings)
            # Apply cross-encoder threshold (logit scale, separate from cosine threshold)
            candidates = [r for r in candidates if r.score >= settings.RERANK_MIN_SCORE]

        return candidates[:k]

    # ------------------------------------------------------------------
    # Leg 1: dense vector retrieval
    # ------------------------------------------------------------------

    def _vector_retrieve(
        self,
        question: str,
        top_k: int,
        min_score: float,
        document_id_filter: Optional[str],
    ) -> List[RetrievalResult]:
        start = time.time()
        query_vector = embed_query(question)
        elapsed = time.time() - start
        observe_stage_seconds("embedding", elapsed)
        logger.info("Query embedded in %.2fs", elapsed)

        where = {"document_id": document_id_filter} if document_id_filter else None
        start = time.time()
        results = get_vector_store().query(
            query_embedding=query_vector,
            top_k=top_k,
            min_score=min_score,
            where=where,
        )
        elapsed = time.time() - start
        observe_stage_seconds("vector_search", elapsed)
        logger.info("Vector search: %d results in %.2fs", len(results), elapsed)
        # Guarantee stable sort even if provider doesn't sort (e.g., mocks in tests)
        return sorted(results, key=lambda r: r.score, reverse=True)

    # ------------------------------------------------------------------
    # Leg 1+2: hybrid BM25 + vector with RRF fusion
    # ------------------------------------------------------------------

    def _hybrid_retrieve(
        self,
        question: str,
        top_k: int,
        document_id_filter: Optional[str],
        settings,
    ) -> List[RetrievalResult]:
        candidate_k = top_k * settings.HYBRID_CANDIDATE_MULTIPLIER

        # -- Dense leg --
        start = time.time()
        query_vector = embed_query(question)
        elapsed = time.time() - start
        observe_stage_seconds("embedding", elapsed)
        logger.info("Query embedded in %.2fs", elapsed)

        where = {"document_id": document_id_filter} if document_id_filter else None
        start = time.time()
        vector_results = get_vector_store().query(
            query_embedding=query_vector,
            top_k=candidate_k,
            min_score=0.0,   # no threshold — RRF re-decides relevance
            where=where,
        )
        elapsed = time.time() - start
        observe_stage_seconds("vector_search", elapsed)
        logger.info("Vector leg: %d candidates in %.2fs", len(vector_results), elapsed)

        # -- Sparse leg (BM25) --
        from app.providers.retrieval.bm25 import get_bm25_index
        start = time.time()
        bm25_hits = get_bm25_index().search(question, top_k=candidate_k)
        elapsed = time.time() - start
        observe_stage_seconds("bm25_search", elapsed)
        logger.info("BM25 leg: %d candidates in %.2fs", len(bm25_hits), elapsed)

        # -- RRF fusion --
        vector_ranked: List[Tuple[str, float]] = [(r.chunk_id, r.score) for r in vector_results]
        bm25_ranked: List[Tuple[str, float]] = [(cid, sc) for cid, sc, _ in bm25_hits]

        rrf_scores = _reciprocal_rank_fusion([vector_ranked, bm25_ranked])

        # Build result map: prefer vector results (already have embeddings & scores)
        result_map: Dict[str, RetrievalResult] = {r.chunk_id: r for r in vector_results}

        # Promote BM25-only hits (not found by vector search)
        for chunk_id, _score, entry in bm25_hits:
            if chunk_id not in result_map:
                result_map[chunk_id] = RetrievalResult(
                    chunk_id=chunk_id,
                    content=entry.content,
                    metadata=entry.metadata,
                    score=0.0,
                )

        # Overwrite scores with RRF values and sort
        for chunk_id, rrf_score in rrf_scores.items():
            if chunk_id in result_map:
                result_map[chunk_id].score = rrf_score

        merged = sorted(result_map.values(), key=lambda r: r.score, reverse=True)
        logger.info(
            "Hybrid RRF: %d unique candidates (vector=%d bm25=%d)",
            len(merged), len(vector_results), len(bm25_hits),
        )
        return merged

    # ------------------------------------------------------------------
    # Query transformation: multi-query / HyDE retrieval with RRF fusion
    # ------------------------------------------------------------------

    def _multi_query_retrieve(
        self,
        plan: QueryPlan,
        top_k: int,
        document_id_filter: Optional[str],
        settings,
    ) -> List[RetrievalResult]:
        """Retrieve for each query variant / HyDE passage and fuse with RRF.

        One dense leg is run per ``plan.dense_texts`` entry (the query text, or
        a HyDE passage when HyDE is enabled).  When hybrid search is also on,
        one BM25 leg is run per ``plan.queries`` entry.  All resulting ranked
        lists are combined via Reciprocal Rank Fusion.
        """
        candidate_k = top_k * settings.QUERY_TRANSFORM_CANDIDATE_MULTIPLIER
        where = {"document_id": document_id_filter} if document_id_filter else None

        ranked_lists: List[List[Tuple[str, float]]] = []
        result_map: Dict[str, RetrievalResult] = {}

        # -- Dense legs (one per query variant or HyDE passage) --
        for text in plan.dense_texts:
            start = time.time()
            query_vector = embed_query(text)
            vector_results = get_vector_store().query(
                query_embedding=query_vector,
                top_k=candidate_k,
                min_score=0.0,  # no threshold — RRF re-decides relevance
                where=where,
            )
            elapsed = time.time() - start
            observe_stage_seconds("vector_search", elapsed)
            logger.info(
                "Dense leg: %d candidates in %.2fs", len(vector_results), elapsed
            )
            ranked_lists.append([(r.chunk_id, r.score) for r in vector_results])
            for r in vector_results:
                result_map.setdefault(r.chunk_id, r)

        # -- Sparse legs (BM25), only when hybrid search is enabled --
        if settings.HYBRID_SEARCH_ENABLED:
            from app.providers.retrieval.bm25 import get_bm25_index

            bm25 = get_bm25_index()
            for q in plan.queries:
                start = time.time()
                bm25_hits = bm25.search(q, top_k=candidate_k)
                elapsed = time.time() - start
                observe_stage_seconds("bm25_search", elapsed)
                logger.info(
                    "BM25 leg: %d candidates in %.2fs", len(bm25_hits), elapsed
                )
                ranked_lists.append([(cid, sc) for cid, sc, _ in bm25_hits])
                for chunk_id, _score, entry in bm25_hits:
                    if chunk_id not in result_map:
                        result_map[chunk_id] = RetrievalResult(
                            chunk_id=chunk_id,
                            content=entry.content,
                            metadata=entry.metadata,
                            score=0.0,
                        )

        if not result_map:
            return []

        # -- RRF fusion across all legs --
        rrf_scores = _reciprocal_rank_fusion(ranked_lists)
        for chunk_id, rrf_score in rrf_scores.items():
            if chunk_id in result_map:
                result_map[chunk_id].score = rrf_score

        merged = sorted(result_map.values(), key=lambda r: r.score, reverse=True)
        logger.info(
            "Multi-query RRF: %d queries, %d dense texts, %d legs → %d candidates",
            len(plan.queries), len(plan.dense_texts), len(ranked_lists), len(merged),
        )
        return merged

    # ------------------------------------------------------------------
    # Step 2: cross-encoder reranking
    # ------------------------------------------------------------------

    def _rerank(
        self,
        question: str,
        candidates: List[RetrievalResult],
        top_k: int,
        settings,
    ) -> List[RetrievalResult]:
        from app.providers.reranker.vietnamese_reranker import rerank
        # Feed reranker up to RERANK_CANDIDATE_MULTIPLIER * top_k candidates
        rerank_input = candidates[: top_k * settings.RERANK_CANDIDATE_MULTIPLIER]
        start = time.time()
        reranked = rerank(question, rerank_input, top_k=top_k)
        observe_stage_seconds("rerank", time.time() - start)
        logger.info("Reranker: %d → %d results", len(rerank_input), len(reranked))
        return reranked
