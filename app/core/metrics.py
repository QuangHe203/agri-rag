"""Prometheus metrics for observability.

Defines the counters/histograms for the RAG pipeline plus small helpers to
record them.  All instrumentation is best-effort: recording a metric must
*never* break a request, so callers should keep the observation code outside
the critical path or wrap it defensively.

Exposed at ``GET /metrics`` (see ``app/api/routes/metrics.py``) in the standard
Prometheus text format, ready to be scraped by Prometheus / visualised in
Grafana.

Metric catalogue
----------------
rag_requests_total{result}              Counter  answered | fallback | error
rag_request_duration_seconds            Histogram  end-to-end query latency
rag_stage_duration_seconds{stage}       Histogram  per-stage latency
rag_retrieved_chunks                    Histogram  #chunks returned per query
rag_llm_calls_total{provider,status}    Counter  success | rate_limited | error
rag_llm_call_duration_seconds{provider} Histogram  single generate() latency
rag_llm_tokens_total{provider,type}     Counter  prompt | completion | total
"""

from contextlib import contextmanager
from time import perf_counter
from typing import Iterator

from prometheus_client import Counter, Histogram

# Latency buckets tuned for a CPU-bound RAG pipeline (seconds). The pipeline is
# slow (local embedding + rerank + remote LLM) so buckets extend past 20s.
_LATENCY_BUCKETS = (
    0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, float("inf"),
)

_CHUNK_BUCKETS = (0, 1, 2, 3, 5, 8, 10, 15, 20, float("inf"))


RAG_REQUESTS = Counter(
    "rag_requests_total",
    "Total RAG query requests grouped by outcome.",
    ["result"],  # answered | fallback | error
)

RAG_REQUEST_DURATION = Histogram(
    "rag_request_duration_seconds",
    "End-to-end latency of a RAG query (retrieval + generation).",
    buckets=_LATENCY_BUCKETS,
)

STAGE_DURATION = Histogram(
    "rag_stage_duration_seconds",
    "Latency of individual RAG pipeline stages.",
    # embedding | vector_search | bm25_search | rerank | retrieval | query_transform
    ["stage"],
    buckets=_LATENCY_BUCKETS,
)

RETRIEVED_CHUNKS = Histogram(
    "rag_retrieved_chunks",
    "Number of chunks returned by retrieval for a query.",
    buckets=_CHUNK_BUCKETS,
)

LLM_CALLS = Counter(
    "rag_llm_calls_total",
    "LLM generation calls grouped by provider and status.",
    ["provider", "status"],  # status: success | rate_limited | error
)

LLM_CALL_DURATION = Histogram(
    "rag_llm_call_duration_seconds",
    "Latency of a single LLM generate() call.",
    ["provider"],
    buckets=_LATENCY_BUCKETS,
)

LLM_TOKENS = Counter(
    "rag_llm_tokens_total",
    "LLM token usage.",
    ["provider", "type"],  # type: prompt | completion | total
)


@contextmanager
def observe_stage(stage: str) -> Iterator[None]:
    """Record elapsed wall-clock time into ``rag_stage_duration_seconds``.

    Usage::

        with observe_stage("retrieval"):
            results = retrieve(...)
    """
    start = perf_counter()
    try:
        yield
    finally:
        STAGE_DURATION.labels(stage=stage).observe(perf_counter() - start)


def observe_stage_seconds(stage: str, seconds: float) -> None:
    """Record an already-measured duration for *stage* (seconds)."""
    STAGE_DURATION.labels(stage=stage).observe(seconds)


def record_tokens(
    provider: str,
    prompt: int = 0,
    completion: int = 0,
    total: int = 0,
) -> None:
    """Increment token-usage counters, skipping zero/unknown values."""
    if prompt:
        LLM_TOKENS.labels(provider=provider, type="prompt").inc(prompt)
    if completion:
        LLM_TOKENS.labels(provider=provider, type="completion").inc(completion)
    if total:
        LLM_TOKENS.labels(provider=provider, type="total").inc(total)
