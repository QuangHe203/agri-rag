"""Ingestion pipeline: PDF parsing, chunking, embedding, and indexing into vector store."""

import hashlib
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.document import get_db
from app.providers.embedding.vietnamese_embedding import embed_documents
from app.providers.pdf.opendataloader_parser import ParsedPage, PDFParseResult, parse_pdf
from app.providers.vector_store.chroma import ChunkDocument, ChromaVectorStore, get_vector_store
from app.services.document_service import DocumentService

logger = get_logger(__name__)


def compute_sha256(file_path: str | Path) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _chunk_pages(
    pages: List[ParsedPage],
    chunk_size: int,
    chunk_overlap: int,
    document_id: str,
    file_name: str,
    uploaded_at: str,
) -> List[ChunkDocument]:
    """Split page content into overlapping chunks with full metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", "。", ".", " ", ""],
        keep_separator=True,
    )

    chunks: List[ChunkDocument] = []
    chunk_index = 0

    for page in pages:
        if not page.content.strip():
            continue
        sub_chunks = splitter.split_text(page.content)
        for sub in sub_chunks:
            text = sub.strip()
            if not text:
                continue
            # Compute checksum to detect duplicate chunks
            checksum = hashlib.md5(text.encode("utf-8")).hexdigest()
            chunk_id = f"chunk-{document_id}-{chunk_index}"
            chunks.append(
                ChunkDocument(
                    chunk_id=chunk_id,
                    content=text,
                    metadata={
                        "document_id": document_id,
                        "file_name": file_name,
                        "page": page.page_number,
                        "chunk_index": chunk_index,
                        "content_checksum": checksum,
                        "uploaded_at": uploaded_at,
                    },
                )
            )
            chunk_index += 1

    logger.info("Chunked into %d pieces from %d pages", len(chunks), len(pages))
    return chunks


def ingest_document(document_id: str, file_path: str, file_name: str) -> None:
    """
    Full ingestion pipeline for a single PDF document.

    This is designed to run as a FastAPI background task.
    It updates the document status in SQLite and rolls back vectors on failure.
    """
    db = get_db()
    doc_service = DocumentService(db=db)
    settings = get_settings()
    vector_store: ChromaVectorStore = get_vector_store()
    ingested_chunk_ids: List[str] = []

    try:
        # 1. Parse PDF
        logger.info("Ingestion started: document_id=%s", document_id)
        doc_service.update_status(document_id, status="processing")
        parse_result: PDFParseResult = parse_pdf(file_path, file_name)

        if not parse_result.success:
            error_msg = parse_result.error or "Không thể trích xuất nội dung từ PDF."
            logger.error("Parse failed for %s: %s", document_id, error_msg)
            doc_service.update_status(document_id, status="failed", error_message=error_msg)
            return

        logger.info("Parsed %d pages for document_id=%s", parse_result.page_count, document_id)
        doc_service.update_status(document_id, status="processing", page_count=parse_result.page_count)

        # 2. Chunk
        uploaded_at = datetime.now(timezone.utc).isoformat()
        chunks = _chunk_pages(
            pages=parse_result.pages,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            document_id=document_id,
            file_name=file_name,
            uploaded_at=uploaded_at,
        )

        if not chunks:
            doc_service.update_status(
                document_id,
                status="failed",
                error_message="PDF không có nội dung sau khi chunking.",
            )
            return

        # 3. Embed
        logger.info("Embedding %d chunks for document_id=%s", len(chunks), document_id)
        texts = [c.content for c in chunks]
        start_embed = time.time()
        embeddings = embed_documents(texts)
        elapsed_embed = time.time() - start_embed
        logger.info("Embedding completed in %.2fs", elapsed_embed)

        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        ingested_chunk_ids = [c.chunk_id for c in chunks]

        # 4. Index into vector store
        vector_store.add_chunks(chunks)

        # 5. Mark as indexed
        doc_service.update_status(
            document_id,
            status="indexed",
            page_count=parse_result.page_count,
            chunk_count=len(chunks),
            indexed_at=datetime.now(timezone.utc),
        )
        logger.info(
            "Ingestion complete: document_id=%s pages=%d chunks=%d",
            document_id,
            parse_result.page_count,
            len(chunks),
        )

    except Exception as exc:
        logger.exception("Ingestion failed for document_id=%s", document_id)
        # Rollback: remove any vectors already written
        if ingested_chunk_ids:
            try:
                vector_store._collection.delete(ids=ingested_chunk_ids)
                logger.info(
                    "Rolled back %d vectors for document_id=%s",
                    len(ingested_chunk_ids),
                    document_id,
                )
            except Exception as rollback_exc:
                logger.error("Rollback failed: %s", rollback_exc)
        try:
            doc_service.update_status(
                document_id,
                status="failed",
                error_message=str(exc)[:500],
            )
        except Exception:
            logger.exception("Could not update status to failed for %s", document_id)
    finally:
        doc_service.close()
