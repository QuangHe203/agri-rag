"""Unit tests for Prometheus metrics instrumentation."""

from unittest.mock import MagicMock

from prometheus_client import generate_latest

from app.core import metrics
from app.providers.vector_store.chroma import RetrievalResult
from app.services.rag_service import RAGService


def _counter_value(counter, **labels) -> float:
    """Read the current value of a labelled Counter child."""
    return counter.labels(**labels)._value.get()


def _make_rag_service(results, llm_response="Trả lời"):
    mock_retrieval = MagicMock()
    mock_retrieval.retrieve.return_value = results
    mock_llm = MagicMock()
    mock_llm.generate.return_value = llm_response
    mock_llm.model_name = "mock-llm"
    return RAGService(retrieval_service=mock_retrieval, llm_provider=mock_llm)


def _result():
    return RetrievalResult(
        chunk_id="c1",
        content="Nội dung ngữ cảnh.",
        metadata={"document_id": "d1", "file_name": "a.pdf", "page": 1},
        score=0.9,
    )


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def test_observe_stage_records_duration():
    before = metrics.STAGE_DURATION.labels(stage="unit_test_stage")._sum.get()
    with metrics.observe_stage("unit_test_stage"):
        pass
    after = metrics.STAGE_DURATION.labels(stage="unit_test_stage")._sum.get()
    assert after >= before


def test_record_tokens_increments_only_nonzero():
    prompt_before = _counter_value(metrics.LLM_TOKENS, provider="unit", type="prompt")
    metrics.record_tokens("unit", prompt=10, completion=0, total=10)
    prompt_after = _counter_value(metrics.LLM_TOKENS, provider="unit", type="prompt")
    total_after = _counter_value(metrics.LLM_TOKENS, provider="unit", type="total")
    completion_after = _counter_value(metrics.LLM_TOKENS, provider="unit", type="completion")
    assert prompt_after == prompt_before + 10
    assert total_after == 10
    assert completion_after == 0  # zero values are skipped


# ---------------------------------------------------------------------------
# RAGService instrumentation
# ---------------------------------------------------------------------------

def test_answered_request_increments_counter():
    before = _counter_value(metrics.RAG_REQUESTS, result="answered")
    rag = _make_rag_service([_result()])
    rag.answer(question="Câu hỏi?", top_k=5)
    after = _counter_value(metrics.RAG_REQUESTS, result="answered")
    assert after == before + 1


def test_fallback_request_increments_counter():
    before = _counter_value(metrics.RAG_REQUESTS, result="fallback")
    rag = _make_rag_service([])
    rag.answer(question="Ngoài phạm vi", top_k=5)
    after = _counter_value(metrics.RAG_REQUESTS, result="fallback")
    assert after == before + 1


def test_error_request_increments_counter():
    before = _counter_value(metrics.RAG_REQUESTS, result="error")
    mock_retrieval = MagicMock()
    mock_retrieval.retrieve.side_effect = RuntimeError("boom")
    rag = RAGService(retrieval_service=mock_retrieval, llm_provider=MagicMock())
    try:
        rag.answer(question="Câu hỏi?", top_k=5)
    except RuntimeError:
        pass
    after = _counter_value(metrics.RAG_REQUESTS, result="error")
    assert after == before + 1


def test_request_duration_observed():
    before = metrics.RAG_REQUEST_DURATION._sum.get()
    rag = _make_rag_service([_result()])
    rag.answer(question="Câu hỏi?", top_k=5)
    after = metrics.RAG_REQUEST_DURATION._sum.get()
    assert after >= before


# ---------------------------------------------------------------------------
# Exposition format
# ---------------------------------------------------------------------------

def test_generate_latest_contains_rag_metrics():
    rag = _make_rag_service([_result()])
    rag.answer(question="Câu hỏi?", top_k=5)
    text = generate_latest().decode("utf-8")
    assert "rag_requests_total" in text
    assert "rag_request_duration_seconds" in text
    assert "rag_retrieved_chunks" in text
