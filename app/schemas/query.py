"""Pydantic schemas for the Query API."""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from app.core.config import get_settings


class QueryRequest(BaseModel):
    question: str = Field(..., description="Câu hỏi tiếng Việt về khuyến nông")
    top_k: int = Field(default=5, ge=1, le=10, description="Số chunk truy xuất (1-10)")

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Câu hỏi không được để trống hoặc chỉ chứa khoảng trắng.")
        settings = get_settings()
        if len(v) > settings.MAX_QUESTION_LENGTH:
            raise ValueError(
                f"Câu hỏi không được vượt quá {settings.MAX_QUESTION_LENGTH} ký tự."
            )
        return v.strip()


class SourceReference(BaseModel):
    document_id: str
    file_name: str
    page: Optional[int] = None
    chunk_id: str
    score: float
    excerpt: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceReference]
    model: str


# ---------------------------------------------------------------------------
# Image enrichment types (added as independent layer — QueryResponse unchanged)
# ---------------------------------------------------------------------------

class ImageReference(BaseModel):
    """A single illustration image extracted from a source PDF page."""

    document_id: str
    file_name: str = Field(description="Source PDF file name")
    page: int = Field(description="1-based page number inside the source PDF")
    image_index: int = Field(description="0-based image index within the page")
    url: str = Field(
        description="API path to fetch the image: GET /api/v1/images/{document_id}/{page}/{image_index}"
    )
    width: Optional[int] = None
    height: Optional[int] = None
    mime_type: str = "image/png"


class EnrichedQueryResponse(QueryResponse):
    """QueryResponse extended with optional illustration images.

    This schema is a strict superset of :class:`QueryResponse`: all existing
    fields are unchanged and ``images`` defaults to an empty list so that
    clients not yet aware of images receive a valid response.
    """

    images: List[ImageReference] = Field(
        default_factory=list,
        description="Illustration images from the source pages most relevant to the answer. "
        "Empty when no images are available or when image extraction is disabled.",
    )
