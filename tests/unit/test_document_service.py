"""Unit tests for DocumentService using an in-memory SQLite database."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.document import Base, DocumentModel
from app.services.document_service import DocumentService
from app.core.exceptions import DocumentNotFoundError


@pytest.fixture
def db_session():
    """Provide an in-memory SQLite session for each test."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture
def doc_service(db_session):
    return DocumentService(db=db_session)


def test_create_document(doc_service):
    doc = doc_service.create(
        document_id="doc-1",
        file_name="test.pdf",
        file_hash="abc123",
        file_size=1024,
    )
    assert doc.document_id == "doc-1"
    assert doc.status == "processing"


def test_get_by_id(doc_service):
    doc_service.create("doc-2", "a.pdf", "hash2", 2048)
    retrieved = doc_service.get_by_id("doc-2")
    assert retrieved.file_name == "a.pdf"


def test_get_by_id_not_found(doc_service):
    with pytest.raises(DocumentNotFoundError):
        doc_service.get_by_id("nonexistent-id")


def test_get_by_hash(doc_service):
    doc_service.create("doc-3", "b.pdf", "unique-hash", 512)
    found = doc_service.get_by_hash("unique-hash")
    assert found is not None
    assert found.document_id == "doc-3"


def test_get_by_hash_missing(doc_service):
    result = doc_service.get_by_hash("no-such-hash")
    assert result is None


def test_update_status(doc_service):
    doc_service.create("doc-4", "c.pdf", "hash4", 100)
    updated = doc_service.update_status("doc-4", status="indexed", chunk_count=42, page_count=10)
    assert updated.status == "indexed"
    assert updated.chunk_count == 42
    assert updated.page_count == 10


def test_update_status_failed(doc_service):
    doc_service.create("doc-5", "d.pdf", "hash5", 100)
    updated = doc_service.update_status("doc-5", status="failed", error_message="Parse error")
    assert updated.status == "failed"
    assert updated.error_message == "Parse error"


def test_delete_document(doc_service):
    doc_service.create("doc-6", "e.pdf", "hash6", 100)
    doc_service.delete("doc-6")
    with pytest.raises(DocumentNotFoundError):
        doc_service.get_by_id("doc-6")


def test_get_all_returns_all(doc_service):
    doc_service.create("doc-7", "f.pdf", "hash7", 100)
    doc_service.create("doc-8", "g.pdf", "hash8", 200)
    all_docs = doc_service.get_all()
    ids = [d.document_id for d in all_docs]
    assert "doc-7" in ids
    assert "doc-8" in ids


def test_duplicate_hash_detected(doc_service):
    doc_service.create("doc-9", "h.pdf", "same-hash", 100)
    duplicate = doc_service.get_by_hash("same-hash")
    assert duplicate is not None
    assert duplicate.document_id == "doc-9"
