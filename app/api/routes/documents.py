"""Document upload, list, detail, and delete API routes."""

import hashlib
import os
import re
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile, status

from app.core.config import get_settings
from app.core.exceptions import (
    DocumentNotFoundError,
    DuplicateDocumentError,
    FileTooLargeError,
    InvalidFileTypeError,
)
from app.core.logging import get_logger
from app.models.document import get_db
from app.providers.vector_store.chroma import get_vector_store
from app.schemas.document import (
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentUploadItem,
    DocumentUploadResponse,
)
from app.services.document_service import DocumentService
from app.services.ingestion_service import compute_sha256, ingest_document

router = APIRouter(prefix="/api/v1/documents", tags=["Documents"])
logger = get_logger(__name__)

_SAFE_FILENAME_RE = re.compile(r"[^\w\-. ]", re.UNICODE)


def _sanitize_filename(name: str) -> str:
    """Remove path-traversal characters and normalize the filename."""
    # Strip directory separators and null bytes
    name = os.path.basename(name.replace("\\", "/").replace("\x00", ""))
    # Remove characters that are not word chars, dash, dot, or space
    name = _SAFE_FILENAME_RE.sub("_", name)
    return name or "document.pdf"


def _validate_pdf_upload(file: UploadFile, max_bytes: int) -> None:
    """Validate file extension and MIME type."""
    file_name = file.filename or ""
    if not file_name.lower().endswith(".pdf"):
        raise InvalidFileTypeError("Chỉ chấp nhận file PDF (.pdf).")
    if file.content_type not in ("application/pdf", "application/octet-stream", None):
        # Be lenient with content_type as browsers differ; rely on extension + first bytes
        if file.content_type and "pdf" not in file.content_type.lower():
            raise InvalidFileTypeError(
                f"MIME type không hợp lệ: {file.content_type}. Chỉ chấp nhận application/pdf."
            )


async def _read_and_validate_bytes(file: UploadFile, max_bytes: int) -> bytes:
    content = await file.read()
    if not content:
        raise InvalidFileTypeError("File rỗng, vui lòng tải lên file PDF có nội dung.")
    if len(content) > max_bytes:
        raise FileTooLargeError(
            f"File vượt quá giới hạn {max_bytes // (1024 * 1024)} MB."
        )
    # Check PDF magic bytes
    if not content.startswith(b"%PDF"):
        raise InvalidFileTypeError("File không phải là PDF hợp lệ.")
    return content


@router.post(
    "",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload một hoặc nhiều file PDF",
)
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
):
    settings = get_settings()
    max_bytes = settings.max_upload_size_bytes
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    db = get_db()
    doc_service = DocumentService(db=db)
    result_items: list[DocumentUploadItem] = []

    try:
        for file in files:
            _validate_pdf_upload(file, max_bytes)
            content = await _read_and_validate_bytes(file, max_bytes)

            file_hash = hashlib.sha256(content).hexdigest()
            safe_name = _sanitize_filename(file.filename or "document.pdf")

            # Check for duplicate
            existing = doc_service.get_by_hash(file_hash)
            if existing:
                logger.info("Duplicate detected: hash=%s existing_id=%s", file_hash, existing.document_id)
                result_items.append(
                    DocumentUploadItem(
                        document_id=existing.document_id,
                        file_name=existing.file_name,
                        status="duplicate",
                    )
                )
                continue

            document_id = str(uuid.uuid4())
            dest_path = upload_dir / f"{document_id}_{safe_name}"
            dest_path.write_bytes(content)

            doc = doc_service.create(
                document_id=document_id,
                file_name=safe_name,
                file_hash=file_hash,
                file_size=len(content),
                mime_type=file.content_type or "application/pdf",
            )

            # Schedule ingestion as a background task
            background_tasks.add_task(
                ingest_document,
                document_id=document_id,
                file_path=str(dest_path),
                file_name=safe_name,
            )

            logger.info(
                "Upload accepted: document_id=%s file=%s size=%d bytes",
                document_id,
                safe_name,
                len(content),
            )
            result_items.append(
                DocumentUploadItem(
                    document_id=document_id,
                    file_name=safe_name,
                    status="processing",
                )
            )
    finally:
        doc_service.close()

    return DocumentUploadResponse(documents=result_items)


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="Danh sách tài liệu",
)
async def list_documents():
    db = get_db()
    doc_service = DocumentService(db=db)
    try:
        docs = doc_service.get_all()
        return DocumentListResponse(
            items=[
                {
                    "document_id": d.document_id,
                    "file_name": d.file_name,
                    "status": d.status,
                    "page_count": d.page_count,
                    "chunk_count": d.chunk_count,
                    "uploaded_at": d.uploaded_at,
                }
                for d in docs
            ]
        )
    finally:
        doc_service.close()


@router.get(
    "/{document_id}",
    response_model=DocumentDetailResponse,
    summary="Chi tiết tài liệu",
)
async def get_document(document_id: str):
    db = get_db()
    doc_service = DocumentService(db=db)
    try:
        doc = doc_service.get_by_id(document_id)
        return DocumentDetailResponse(
            document_id=doc.document_id,
            file_name=doc.file_name,
            status=doc.status,
            page_count=doc.page_count,
            chunk_count=doc.chunk_count,
            uploaded_at=doc.uploaded_at,
            indexed_at=doc.indexed_at,
            error_message=doc.error_message,
        )
    finally:
        doc_service.close()


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Xóa tài liệu và toàn bộ vector liên quan",
)
async def delete_document(document_id: str):
    db = get_db()
    doc_service = DocumentService(db=db)
    settings = get_settings()

    try:
        doc = doc_service.get_by_id(document_id)  # raises 404 if not found

        # Delete vectors from ChromaDB
        vector_store = get_vector_store()
        vector_store.delete_by_document_id(document_id)

        # Delete from BM25 index (no-op when hybrid search is disabled)
        settings2 = get_settings()
        if settings2.HYBRID_SEARCH_ENABLED:
            try:
                from app.providers.retrieval.bm25 import get_bm25_index
                get_bm25_index().delete_by_document_id(document_id)
            except Exception as bm25_exc:
                logger.warning("BM25 delete failed for %s: %s", document_id, bm25_exc)

        # Delete extracted images (DB rows + disk files) — non-blocking
        try:
            from app.services.image_service import delete_images_for_document
            delete_images_for_document(
                document_id=document_id,
                image_store_path=settings.IMAGE_STORE_PATH,
            )
        except Exception as img_exc:
            logger.warning("Image cleanup failed for %s: %s", document_id, img_exc)

        # Delete uploaded file
        upload_dir = Path(settings.UPLOAD_DIR)
        for fpath in upload_dir.glob(f"{document_id}_*"):
            fpath.unlink(missing_ok=True)
            logger.info("Deleted file: %s", fpath)

        # Delete DB record
        doc_service.delete(document_id)
        logger.info("Document fully deleted: id=%s", document_id)
    finally:
        doc_service.close()
