"""Pydantic schemas for the Document API."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class DocumentUploadItem(BaseModel):
    document_id: str
    file_name: str
    status: str


class DocumentUploadResponse(BaseModel):
    documents: list[DocumentUploadItem]


class DocumentListItem(BaseModel):
    document_id: str
    file_name: str
    status: str
    page_count: Optional[int] = None
    chunk_count: Optional[int] = None
    uploaded_at: datetime


class DocumentListResponse(BaseModel):
    items: list[DocumentListItem]


class DocumentDetailResponse(BaseModel):
    document_id: str
    file_name: str
    status: str
    page_count: Optional[int] = None
    chunk_count: Optional[int] = None
    uploaded_at: datetime
    indexed_at: Optional[datetime] = None
    error_message: Optional[str] = None
