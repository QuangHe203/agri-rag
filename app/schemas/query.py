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
