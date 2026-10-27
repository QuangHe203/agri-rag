"""RAG service: orchestrates retrieval + LLM generation."""

from time import perf_counter
from typing import List, Optional

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.metrics import (
    RAG_REQUEST_DURATION,
    RAG_REQUESTS,
    RETRIEVED_CHUNKS,
    observe_stage,
)
from app.prompts.agriculture_qa import (
    FALLBACK_ANSWER,
    SYSTEM_PROMPT,
    build_context_block,
    build_user_prompt,
)
from app.providers.llm.base import LLMProvider, extract_text_content
from app.providers.llm.gemini import get_gemini_provider
from app.providers.vector_store.chroma import RetrievalResult
from app.schemas.query import QueryResponse, SourceReference
from app.services.retrieval_service import RetrievalService

logger = get_logger(__name__)


def _get_llm_provider() -> LLMProvider:
    settings = get_settings()
    if settings.LLM_PROVIDER == "gemini":
        return get_gemini_provider()
    raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")


class RAGService:
    """Retrieval-Augmented Generation service for agriculture Q&A."""

    def __init__(
        self,
        retrieval_service: Optional[RetrievalService] = None,
        llm_provider: Optional[LLMProvider] = None,
    ):
        self._retrieval = retrieval_service or RetrievalService()
        self._llm: Optional[LLMProvider] = llm_provider  # lazy-loaded

    def _ensure_llm(self) -> LLMProvider:
        if self._llm is None:
            self._llm = _get_llm_provider()
        return self._llm

    def answer(self, question: str, top_k: int) -> QueryResponse:
        """
        Full RAG pipeline:
          1. Retrieve relevant chunks.
          2. If none meet the threshold, return fallback (no LLM call).
          3. Build context and call LLM.
          4. Return answer + sources.
        """
        start = perf_counter()
        try:
            # 1. Retrieve
            with observe_stage("retrieval"):
                results: List[RetrievalResult] = self._retrieval.retrieve(
                    question=question, top_k=top_k
                )
            RETRIEVED_CHUNKS.observe(len(results))

            # 2. Fallback when no relevant context found
            if not results:
                logger.info("No relevant context found for question. Returning fallback.")
                RAG_REQUESTS.labels(result="fallback").inc()
                settings = get_settings()
                provider_name = settings.GEMINI_MODEL if settings.LLM_PROVIDER == "gemini" else settings.LLM_PROVIDER
                return QueryResponse(
                    question=question,
                    answer=FALLBACK_ANSWER,
                    sources=[],
                    model=provider_name,
                )

            # 3. Build context and prompt
            context = build_context_block(results)
            user_prompt = build_user_prompt(context=context, question=question)

            # 4. Call LLM
            llm = self._ensure_llm()
            answer_text = extract_text_content(
                llm.generate(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
            )

            # 5. Build source references, sorted by score descending
            sources = [
                SourceReference(
                    document_id=r.metadata.get("document_id", ""),
                    file_name=r.metadata.get("file_name", ""),
                    page=r.metadata.get("page"),
                    chunk_id=r.chunk_id,
                    score=round(r.score, 4),
                    excerpt=r.content[:300],
                )
                for r in sorted(results, key=lambda r: r.score, reverse=True)
            ]

            RAG_REQUESTS.labels(result="answered").inc()
            return QueryResponse(
                question=question,
                answer=answer_text,
                sources=sources,
                model=llm.model_name,
            )
        except Exception:
            RAG_REQUESTS.labels(result="error").inc()
            raise
        finally:
            RAG_REQUEST_DURATION.observe(perf_counter() - start)
