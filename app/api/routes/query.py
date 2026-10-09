"""Query (ask) endpoint."""

from fastapi import APIRouter

from app.core.logging import get_logger
from app.schemas.query import QueryRequest, QueryResponse
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
    response_model=QueryResponse,
    summary="Đặt câu hỏi về khuyến nông",
)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Nhận câu hỏi tiếng Việt, truy xuất tài liệu liên quan và sinh câu trả lời.

    - `question`: câu hỏi tiếng Việt (bắt buộc).
    - `top_k`: số chunk cần truy xuất (1–10, mặc định 5).
    """
    logger.info(
        "Query received: question_len=%d top_k=%d",
        len(request.question),
        request.top_k,
    )
    rag = _get_rag_service()
    return rag.answer(question=request.question, top_k=request.top_k)
