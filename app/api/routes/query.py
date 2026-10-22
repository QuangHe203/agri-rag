"""Query (ask) endpoint."""

from fastapi import APIRouter

from app.core.logging import get_logger
from app.schemas.query import EnrichedQueryResponse, QueryRequest
from app.services.image_enricher import enrich_response
from app.services.rag_service import RAGService

router = APIRouter(prefix="/api/v1", tags=["Query"])
logger = get_logger(__name__)

_rag_service: RAGService | None = None


def _get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


@router.post(
    "/query",
    response_model=EnrichedQueryResponse,
    summary="Đặt câu hỏi về khuyến nông",
)
async def query(request: QueryRequest) -> EnrichedQueryResponse:
    """
    Nhận câu hỏi tiếng Việt, truy xuất tài liệu liên quan và sinh câu trả lời.

    - `question`: câu hỏi tiếng Việt (bắt buộc).
    - `top_k`: số chunk cần truy xuất (1–10, mặc định 5).

    The response is a superset of the previous ``QueryResponse``: all existing
    fields are unchanged.  The new ``images`` field (default ``[]``) carries
    illustration images extracted from the source PDF pages most relevant to
    the answer.  Image retrieval failures never affect the answer or sources.
    """
    logger.info(
        "Query received: question_len=%d top_k=%d",
        len(request.question),
        request.top_k,
    )
    rag = _get_rag_service()
    rag_response = rag.answer(question=request.question, top_k=request.top_k)
    return enrich_response(rag_response)
