"""Unit tests for the RAGService using mocked LLM and retrieval."""

from unittest.mock import MagicMock

import pytest

from app.providers.vector_store.chroma import RetrievalResult
from app.services.rag_service import RAGService
from app.prompts.agriculture_qa import FALLBACK_ANSWER


def _make_result(chunk_id: str = "chunk-1", score: float = 0.8) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        content="Bệnh đạo ôn do nấm Pyricularia oryzae gây ra.",
        metadata={"document_id": "doc-1", "file_name": "lua.pdf", "page": 5},
        score=score,
    )


def _make_rag_service(retrieval_results, llm_response="Câu trả lời giả lập"):
    mock_retrieval = MagicMock()
    mock_retrieval.retrieve.return_value = retrieval_results

    mock_llm = MagicMock()
    mock_llm.generate.return_value = llm_response
    mock_llm.model_name = "mock-llm"

    return RAGService(retrieval_service=mock_retrieval, llm_provider=mock_llm)


def test_answer_with_context_calls_llm():
    rag = _make_rag_service(
        retrieval_results=[_make_result()],
        llm_response="Phòng bệnh đạo ôn bằng cách...",
    )
    response = rag.answer(question="Phòng bệnh đạo ôn như thế nào?", top_k=5)

    assert response.answer == "Phòng bệnh đạo ôn bằng cách..."
    assert len(response.sources) == 1
    assert response.sources[0].file_name == "lua.pdf"
    assert response.sources[0].page == 5


def test_answer_without_context_uses_fallback():
    rag = _make_rag_service(retrieval_results=[])
    response = rag.answer(question="Câu hỏi không có trong tài liệu", top_k=5)

    assert response.answer == FALLBACK_ANSWER
    assert response.sources == []


def test_fallback_does_not_call_llm():
    mock_retrieval = MagicMock()
    mock_retrieval.retrieve.return_value = []
    mock_llm = MagicMock()
    mock_llm.model_name = "mock-llm"

    rag = RAGService(retrieval_service=mock_retrieval, llm_provider=mock_llm)
    rag.answer(question="Câu hỏi ngoài phạm vi", top_k=5)

    # LLM should NOT have been called
    mock_llm.generate.assert_not_called()


def test_sources_sorted_by_score():
    results = [
        _make_result("chunk-b", score=0.6),
        _make_result("chunk-a", score=0.9),
        _make_result("chunk-c", score=0.75),
    ]
    rag = _make_rag_service(retrieval_results=results)
    response = rag.answer(question="Câu hỏi", top_k=5)

    scores = [s.score for s in response.sources]
    assert scores == sorted(scores, reverse=True)


def test_response_contains_model_name():
    rag = _make_rag_service(retrieval_results=[_make_result()])
    response = rag.answer(question="Câu hỏi", top_k=5)
    assert response.model == "mock-llm"


def test_prompt_contains_context_and_question():
    """Ensure the prompt passed to LLM contains both context and question."""
    mock_retrieval = MagicMock()
    mock_retrieval.retrieve.return_value = [_make_result()]

    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Trả lời"
    mock_llm.model_name = "mock-llm"

    rag = RAGService(retrieval_service=mock_retrieval, llm_provider=mock_llm)
    rag.answer(question="Hỏi về lúa?", top_k=5)

    _, kwargs = mock_llm.generate.call_args
    user_prompt = kwargs.get("user_prompt", "")
    assert "Hỏi về lúa?" in user_prompt
    assert "NGỮ CẢNH" in user_prompt
