"""Document metadata CRUD operations backed by SQLite."""

from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy.orm import Session

from app.core.exceptions import DocumentNotFoundError
from app.core.logging import get_logger
from app.models.document import DocumentModel, get_db

logger = get_logger(__name__)


class DocumentService:
    """Service for managing document metadata in SQLite."""

    def __init__(self, db: Optional[Session] = None):
        self._db = db

    @property
    def db(self) -> Session:
        if self._db is None:
            self._db = get_db()
        return self._db

    def close(self):
        if self._db is not None:
            self._db.close()
            self._db = None

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_all(self) -> List[DocumentModel]:
        return self.db.query(DocumentModel).order_by(DocumentModel.uploaded_at.desc()).all()

    def get_by_id(self, document_id: str) -> DocumentModel:
        doc = self.db.query(DocumentModel).filter(DocumentModel.document_id == document_id).first()
        if doc is None:
            raise DocumentNotFoundError(f"Tài liệu '{document_id}' không tồn tại.")
        return doc

    def get_by_hash(self, file_hash: str) -> Optional[DocumentModel]:
        return (
            self.db.query(DocumentModel)
            .filter(DocumentModel.file_hash == file_hash)
            .first()
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def create(
        self,
        document_id: str,
        file_name: str,
        file_hash: str,
        file_size: int,
        mime_type: str = "application/pdf",
    ) -> DocumentModel:
        doc = DocumentModel(
            document_id=document_id,
            file_name=file_name,
            file_hash=file_hash,
            file_size=file_size,
            mime_type=mime_type,
            status="processing",
            uploaded_at=datetime.now(timezone.utc),
        )
        self.db.add(doc)
        self.db.commit()
        self.db.refresh(doc)
        logger.info("Document created: id=%s name=%s", document_id, file_name)
        return doc

    def update_status(
        self,
        document_id: str,
        status: str,
        page_count: Optional[int] = None,
        chunk_count: Optional[int] = None,
        error_message: Optional[str] = None,
        indexed_at: Optional[datetime] = None,
    ) -> DocumentModel:
        doc = self.get_by_id(document_id)
        doc.status = status
        if page_count is not None:
            doc.page_count = page_count
        if chunk_count is not None:
            doc.chunk_count = chunk_count
        if error_message is not None:
            doc.error_message = error_message
        if indexed_at is not None:
            doc.indexed_at = indexed_at
        self.db.commit()
        self.db.refresh(doc)
        logger.info("Document status updated: id=%s status=%s", document_id, status)
        return doc

    def delete(self, document_id: str) -> None:
        doc = self.get_by_id(document_id)
        self.db.delete(doc)
        self.db.commit()
        logger.info("Document deleted from DB: id=%s", document_id)
