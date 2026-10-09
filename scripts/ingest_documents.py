"""
Batch ingestion script.

Usage:
    python scripts/ingest_documents.py path/to/folder_or_file.pdf [path2.pdf ...]

This script ingests one or more PDF files directly without going through the
HTTP API. It is useful for seeding the knowledge base before starting the server.
"""

import argparse
import sys
from pathlib import Path

# Make sure the project root is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uuid

from app.core.logging import setup_logging
from app.services.document_service import DocumentService
from app.services.ingestion_service import compute_sha256, ingest_document
from app.models.document import get_db


def main():
    parser = argparse.ArgumentParser(description="Ingest PDF documents into the RAG knowledge base.")
    parser.add_argument(
        "paths",
        nargs="+",
        help="Paths to PDF files or directories containing PDFs.",
    )
    args = parser.parse_args()

    setup_logging("INFO")

    pdf_files: list[Path] = []
    for raw_path in args.paths:
        p = Path(raw_path)
        if p.is_dir():
            pdf_files.extend(sorted(p.glob("**/*.pdf")))
        elif p.is_file() and p.suffix.lower() == ".pdf":
            pdf_files.append(p)
        else:
            print(f"[SKIP] Not a PDF or directory: {raw_path}")

    if not pdf_files:
        print("No PDF files found. Exiting.")
        sys.exit(0)

    print(f"Found {len(pdf_files)} PDF file(s) to ingest.\n")

    db = get_db()
    doc_service = DocumentService(db=db)

    success = 0
    skipped = 0
    failed = 0

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        file_hash = compute_sha256(pdf_path)
        existing = doc_service.get_by_hash(file_hash)
        if existing:
            print(f"  -> Duplicate (already indexed as {existing.document_id}). Skipping.")
            skipped += 1
            continue

        document_id = str(uuid.uuid4())
        doc_service.create(
            document_id=document_id,
            file_name=pdf_path.name,
            file_hash=file_hash,
            file_size=pdf_path.stat().st_size,
        )

        try:
            ingest_document(
                document_id=document_id,
                file_path=str(pdf_path),
                file_name=pdf_path.name,
            )
            # Re-read status
            doc = doc_service.get_by_id(document_id)
            if doc.status == "indexed":
                print(f"  -> Indexed: {doc.chunk_count} chunks from {doc.page_count} pages.")
                success += 1
            else:
                print(f"  -> Failed: {doc.error_message}")
                failed += 1
        except Exception as exc:
            print(f"  -> Error: {exc}")
            failed += 1

    doc_service.close()
    print(f"\nDone. Success: {success}  Skipped: {skipped}  Failed: {failed}")


if __name__ == "__main__":
    main()
