"""Query transformation service: Query Rewrite / Multi-query / HyDE.

This runs *before* retrieval and uses the LLM to expand a single user question
into one or more search queries, optionally generating HyDE (Hypothetical
Document Embeddings) passages used for the dense retrieval leg.

All techniques are opt-in via settings and default OFF, so the baseline
pipeline is unchanged.  The service is defensive: any LLM failure degrades
gracefully back to the original question rather than breaking retrieval.

Produced :class:`QueryPlan`:

    queries      Lexical search queries (for BM25 / general retrieval). Always
                 contains at least the original (or rewritten) question.
    dense_texts  Texts to embed for the dense vector leg. Equals ``queries``
                 unless HyDE is enabled, in which case it holds the generated
                 hypothetical passages (one per query).
    transformed  True when any transformation actually changed the plan, i.e.
                 the retrieval layer should use the multi-query fusion path.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from app.core.config import get_settings
from app.core.logging import get_logger
from app.prompts.query_transform import (
    HYDE_SYSTEM_PROMPT,
    MULTI_QUERY_SYSTEM_PROMPT,
    REWRITE_SYSTEM_PROMPT,
    build_hyde_prompt,
    build_multi_query_prompt,
    build_rewrite_prompt,
)
from app.providers.llm.base import LLMProvider

logger = get_logger(__name__)


@dataclass
class QueryPlan:
    """The set of queries to execute for a single user question."""

    original: str
    queries: List[str] = field(default_factory=list)
    dense_texts: List[str] = field(default_factory=list)
    transformed: bool = False


def _dedupe_keep_order(items: List[str]) -> List[str]:
    """Deduplicate strings case-insensitively while preserving order."""
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item.strip())
    return out


class QueryTransformer:
    """Expand a user question into search queries using the LLM.

    The LLM provider is shared with the RAG pipeline (lazy-loaded singleton),
    so enabling transformations adds extra LLM calls but no new dependencies.
    """

    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        self._llm: Optional[LLMProvider] = llm_provider  # lazy-loaded

    def _ensure_llm(self) -> LLMProvider:
        if self._llm is None:
            # Imported lazily to mirror RAGService and avoid import cycles.
            from app.services.rag_service import _get_llm_provider

            self._llm = _get_llm_provider()
        return self._llm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_plan(self, question: str) -> QueryPlan:
        """Build the retrieval query plan for *question* based on settings."""
        settings = get_settings()
        plan = QueryPlan(original=question, queries=[question], dense_texts=[question])

        rewrite = settings.QUERY_REWRITE_ENABLED
        multi = settings.MULTI_QUERY_ENABLED
        hyde = settings.HYDE_ENABLED

        if not (rewrite or multi or hyde):
            return plan  # fast path: no transformation

        # 1. Query rewrite — replace the base question with a cleaner search query.
        base_question = question
        if rewrite:
            rewritten = self._rewrite(question)
            if rewritten:
                base_question = rewritten
                plan.transformed = True

        # 2. Multi-query — generate alternative phrasings of the base question.
        queries = [base_question]
        if multi:
            variants = self._multi_query(base_question, settings.MULTI_QUERY_COUNT)
            if variants:
                queries.extend(variants)
                plan.transformed = True
        queries = _dedupe_keep_order(queries)
        plan.queries = queries

        # 3. HyDE — embed hypothetical answer passages on the dense leg.
        if hyde:
            dense_texts = []
            for q in queries:
                doc = self._hyde(q)
                dense_texts.append(doc if doc else q)
            plan.dense_texts = _dedupe_keep_order(dense_texts)
            plan.transformed = True
        else:
            plan.dense_texts = list(queries)

        logger.info(
            "Query plan: rewrite=%s multi=%s hyde=%s → %d queries, %d dense texts",
            rewrite, multi, hyde, len(plan.queries), len(plan.dense_texts),
        )
        return plan

    # ------------------------------------------------------------------
    # Individual techniques (each resilient to LLM failure)
    # ------------------------------------------------------------------

    def _rewrite(self, question: str) -> Optional[str]:
        try:
            text = self._ensure_llm().generate(
                system_prompt=REWRITE_SYSTEM_PROMPT,
                user_prompt=build_rewrite_prompt(question),
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Query rewrite failed (%s) — using original question.", exc)
            return None
        rewritten = (text or "").strip().splitlines()
        rewritten = rewritten[0].strip() if rewritten else ""
        if not rewritten:
            return None
        logger.info("Query rewritten: %r → %r", question, rewritten)
        return rewritten

    def _multi_query(self, question: str, count: int) -> List[str]:
        try:
            text = self._ensure_llm().generate(
                system_prompt=MULTI_QUERY_SYSTEM_PROMPT,
                user_prompt=build_multi_query_prompt(question, count),
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Multi-query generation failed (%s) — skipping.", exc)
            return []
        variants = self._parse_lines(text)
        logger.info("Multi-query generated %d variants", len(variants))
        return variants[:count]

    def _hyde(self, question: str) -> Optional[str]:
        try:
            text = self._ensure_llm().generate(
                system_prompt=HYDE_SYSTEM_PROMPT,
                user_prompt=build_hyde_prompt(question),
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("HyDE generation failed (%s) — using query text.", exc)
            return None
        doc = (text or "").strip()
        return doc or None

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_lines(text: Optional[str]) -> List[str]:
        """Parse an LLM list response into clean lines.

        Strips common list markers (numbering, bullets) the model may add
        despite instructions.
        """
        import re

        lines: List[str] = []
        for raw in (text or "").splitlines():
            line = raw.strip()
            if not line:
                continue
            # Remove leading "1.", "1)", "-", "*", "•" markers
            line = re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", line).strip()
            if line:
                lines.append(line)
        return _dedupe_keep_order(lines)


_transformer_instance: Optional[QueryTransformer] = None


def get_query_transformer() -> QueryTransformer:
    """Return a singleton QueryTransformer."""
    global _transformer_instance
    if _transformer_instance is None:
        _transformer_instance = QueryTransformer()
    return _transformer_instance
