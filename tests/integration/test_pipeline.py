"""
Integration tests for the full RAG pipeline.

These tests use a real temporary ChromaDB + SQLite instance but mock the
embedding model and LLM so they do NOT call external APIs.
"""

import io
import os
import tempfile
import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# ---------------------------------------------------------------------------
# Fixtures: patch heavy providers before app import
# ---------------------------------------------------------------------------

FAKE_EMBEDDING = [0.01] * 1024
FAKE_LLM_RESPONSE = "Câu trả lời giả lập từ LLM."


@pytest.fixture(scope="module", autouse=True)
def patch_providers(tmp_path_factory):
    """Patch embedding model and LLM for the entire module."""
    tmp = tmp_path_factory.mktemp("rag_integration")
    vector_db = str(tmp / "vector_db")
    upload_dir = str(tmp / "uploads")
    sqlite_db = str(tmp / "documents.db")
    os.makedirs(vector_db, exist_ok=True)
    os.makedirs(upload_dir, exist_ok=True)

    env_overrides = {
        "VECTOR_DB_PATH": vector_db,
        "UPLOAD_DIR": upload_dir,
        "SQLITE_DB_PATH": sqlite_db,
        "GEMINI_API_KEY": "fake-key-for-tests",
        "COLLECTION_NAME": f"test_{uuid.uuid4().hex[:8]}",
    }

    with patch.dict(os.environ, env_overrides):
        with patch(
            "app.providers.embedding.vietnamese_embedding.get_embedding_model"
        ) as mock_model_factory:
            mock_model = MagicMock()
            mock_model.embed_documents.return_value = [FAKE_EMBEDDING]
            mock_model.embed_query.return_value = FAKE_EMBEDDING
            mock_model_factory.return_value = mock_model

            with patch(
                "app.providers.llm.gemini.GeminiProvider.generate",
                return_value=FAKE_LLM_RESPONSE,
            ):
                # Reset singletons so they pick up patched env
                import app.providers.embedding.vietnamese_embedding as emb_mod
                import app.providers.vector_store.chroma as chroma_mod
                import app.providers.llm.gemini as gemini_mod
                import app.models.document as db_mod

                emb_mod._embedding_model = None
                chroma_mod._chroma_instance = None
                gemini_mod._gemini_instance = None
                db_mod._engine = None
                db_mod._SessionLocal = None

                yield


@pytest.fixture(scope="module")
def client(patch_providers):
    # Import after patching
    from app.main import create_app
    app = create_app()
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_pdf_bytes() -> bytes:
    """Return a minimal valid PDF binary."""
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << >> >>\nendobj\n"
        b"4 0 obj\n<< /Length 44 >>\nstream\n"
        b"BT /F1 12 Tf 100 700 Td (Ky thuat trong lua) Tj ET\n"
        b"endstream\nendobj\n"
        b"xref\n0 5\n0000000000 65535 f \n"
        b"trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n0\n%%EOF\n"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_components(self, client):
        data = resp = client.get("/health").json()
        assert "status" in data
        assert "components" in data
        assert "vector_store" in data["components"]
        assert "embedding_model" in data["components"]
        assert "llm" in data["components"]


class TestDocumentListEmpty:
    def test_list_documents_empty(self, client):
        resp = client.get("/api/v1/documents")
        assert resp.status_code == 200
        assert "items" in resp.json()


class TestDocumentUpload:
    def test_upload_invalid_file_type(self, client):
        resp = client.post(
            "/api/v1/documents",
            files=[("files", ("not_a_pdf.txt", b"Hello", "text/plain"))],
        )
        assert resp.status_code == 415

    def test_upload_empty_file(self, client):
        resp = client.post(
            "/api/v1/documents",
            files=[("files", ("empty.pdf", b"", "application/pdf"))],
        )
        assert resp.status_code == 415

    def test_upload_fake_pdf_wrong_magic(self, client):
        resp = client.post(
            "/api/v1/documents",
            files=[("files", ("fake.pdf", b"Not a real pdf", "application/pdf"))],
        )
        assert resp.status_code == 415

    def test_upload_valid_pdf_accepted(self, client):
        pdf_bytes = _minimal_pdf_bytes()
        resp = client.post(
            "/api/v1/documents",
            files=[("files", ("test_upload.pdf", pdf_bytes, "application/pdf"))],
        )
        assert resp.status_code == 202
        body = resp.json()
        assert "documents" in body
        assert len(body["documents"]) == 1
        doc = body["documents"][0]
        assert doc["status"] in ("processing", "duplicate")
        assert doc["file_name"] == "test_upload.pdf"


class TestDocumentDetail:
    def test_get_nonexistent_document_404(self, client):
        resp = client.get("/api/v1/documents/nonexistent-id")
        assert resp.status_code == 404

    def test_get_existing_document(self, client):
        pdf_bytes = _minimal_pdf_bytes()
        upload_resp = client.post(
            "/api/v1/documents",
            files=[("files", ("detail_test.pdf", pdf_bytes, "application/pdf"))],
        )
        doc_id = upload_resp.json()["documents"][0]["document_id"]
        resp = client.get(f"/api/v1/documents/{doc_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["document_id"] == doc_id
        assert data["file_name"] == "detail_test.pdf"


class TestDocumentDelete:
    def test_delete_nonexistent_returns_404(self, client):
        resp = client.delete("/api/v1/documents/no-such-id")
        assert resp.status_code == 404

    def test_delete_existing_returns_204(self, client):
        pdf_bytes = _minimal_pdf_bytes()
        upload_resp = client.post(
            "/api/v1/documents",
            files=[("files", ("delete_me.pdf", pdf_bytes, "application/pdf"))],
        )
        doc_id = upload_resp.json()["documents"][0]["document_id"]
        resp = client.delete(f"/api/v1/documents/{doc_id}")
        assert resp.status_code == 204

    def test_deleted_document_not_found(self, client):
        pdf_bytes = _minimal_pdf_bytes()
        upload_resp = client.post(
            "/api/v1/documents",
            files=[("files", ("delete_check.pdf", pdf_bytes, "application/pdf"))],
        )
        doc_id = upload_resp.json()["documents"][0]["document_id"]
        client.delete(f"/api/v1/documents/{doc_id}")
        resp = client.get(f"/api/v1/documents/{doc_id}")
        assert resp.status_code == 404


class TestQueryEndpoint:
    def test_query_blank_question_rejected(self, client):
        resp = client.post("/api/v1/query", json={"question": "   "})
        assert resp.status_code == 422

    def test_query_missing_question_rejected(self, client):
        resp = client.post("/api/v1/query", json={})
        assert resp.status_code == 422

    def test_query_top_k_out_of_range(self, client):
        resp = client.post("/api/v1/query", json={"question": "Câu hỏi", "top_k": 15})
        assert resp.status_code == 422

    def test_query_no_context_returns_fallback(self, client):
        """When no indexed docs exist (or vector store returns nothing), fallback is used."""
        resp = client.post(
            "/api/v1/query",
            json={"question": "Câu hỏi không có trong tài liệu nào", "top_k": 5},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "answer" in body
        assert "sources" in body
        assert isinstance(body["sources"], list)
