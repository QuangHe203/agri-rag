"""Image service: persistence and lookup of page-level images.

Provides three public operations needed by the rest of the system:

  * ``store_images``              – bulk-insert extracted image records.
  * ``get_images_for_pages``      – return all stored images for a set of
                                    (document_id, page_number) pairs, ordered
                                    by page then image_index.
  * ``delete_images_for_document`` – remove DB rows and disk files for a
                                    document (called from the delete endpoint).

All methods are synchronous (matching the existing ORM pattern in this codebase).
Every public call is guarded so that any unexpected DB error is logged rather
than propagated to the caller, keeping image operations non-blocking.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.models.document import get_db
from app.models.image import PageImageModel
from app.providers.image.pdf_image_extractor import ExtractedImage

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Write operations
# ---------------------------------------------------------------------------

def store_images(
    extracted: Sequence[ExtractedImage],
    db: Optional[Session] = None,
) -> None:
    """Persist a batch of :class:`ExtractedImage` records to the database.

    Uses upsert-by-coordinates logic: if a row for the same
    (document_id, page_number, image_index) already exists, it is replaced.
    A new DB session is opened and closed internally when *db* is ``None``.
    """
    if not extracted:
        return

    own_session = db is None
    if own_session:
        db = get_db()

    try:
        for img in extracted:
            # Delete any stale row for the same coordinate triple.
            db.query(PageImageModel).filter_by(
                document_id=img.document_id,
                page_number=img.page_number,
                image_index=img.image_index,
            ).delete(synchronize_session=False)

            record = PageImageModel(
                document_id=img.document_id,
                file_name=img.file_name,
                page_number=img.page_number,
                image_index=img.image_index,
                file_path=img.file_path,
                width=img.width,
                height=img.height,
                mime_type=img.mime_type,
            )
            db.add(record)
        db.commit()
        logger.info("Stored %d image records for document_id=%s", len(extracted), extracted[0].document_id)
    except Exception as exc:
        logger.error("Failed to store image records: %s", exc)
        try:
            db.rollback()
        except Exception:
            pass
    finally:
        if own_session:
            db.close()


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------

def get_images_for_pages(
    pages: Sequence[Tuple[str, int]],
    db: Optional[Session] = None,
) -> List[PageImageModel]:
    """Return stored images for the given (document_id, page_number) pairs.

    Results are sorted by page_number ascending, then image_index ascending so
    that the enricher always returns images in a stable, predictable order.

    Args:
        pages:  Sequence of ``(document_id, page_number)`` tuples.
        db:     Optional existing session; a new one is opened if ``None``.

    Returns:
        Possibly-empty list of :class:`PageImageModel` ORM objects.
        Returns ``[]`` on any DB error.
    """
    if not pages:
        return []

    own_session = db is None
    if own_session:
        db = get_db()

    try:
        from sqlalchemy import or_, and_, tuple_  # noqa: F401

        # Build a filter for each (document_id, page_number) pair.
        conditions = [
            and_(
                PageImageModel.document_id == doc_id,
                PageImageModel.page_number == page_num,
            )
            for doc_id, page_num in pages
        ]
        rows = (
            db.query(PageImageModel)
            .filter(or_(*conditions))
            .order_by(PageImageModel.page_number, PageImageModel.image_index)
            .all()
        )
        return rows
    except Exception as exc:
        logger.warning("Image DB lookup failed: %s", exc)
        return []
    finally:
        if own_session:
            db.close()


def get_image_by_coordinates(
    document_id: str,
    page_number: int,
    image_index: int,
    db: Optional[Session] = None,
) -> Optional[PageImageModel]:
    """Return the single image identified by its coordinate triple, or ``None``."""
    own_session = db is None
    if own_session:
        db = get_db()
    try:
        return (
            db.query(PageImageModel)
            .filter_by(
                document_id=document_id,
                page_number=page_number,
                image_index=image_index,
            )
            .first()
        )
    except Exception as exc:
        logger.warning("Image coordinate lookup failed: %s", exc)
        return None
    finally:
        if own_session:
            db.close()


# ---------------------------------------------------------------------------
# Delete / cleanup operations
# ---------------------------------------------------------------------------

def delete_images_for_document(
    document_id: str,
    image_store_path: str,
    db: Optional[Session] = None,
) -> None:
    """Remove all DB rows and disk files associated with *document_id*.

    Disk deletion failures are logged but do not raise so that the overall
    document-delete flow is never interrupted.
    """
    own_session = db is None
    if own_session:
        db = get_db()

    try:
        deleted = (
            db.query(PageImageModel)
            .filter_by(document_id=document_id)
            .delete(synchronize_session=False)
        )
        db.commit()
        logger.info("Deleted %d image records for document_id=%s", deleted, document_id)
    except Exception as exc:
        logger.error("Failed to delete image records for %s: %s", document_id, exc)
        try:
            db.rollback()
        except Exception:
            pass
    finally:
        if own_session:
            db.close()

    # Remove the per-document image directory from disk.
    image_dir = Path(image_store_path) / document_id
    try:
        if image_dir.is_dir():
            shutil.rmtree(image_dir)
            logger.info("Removed image directory: %s", image_dir)
    except Exception as exc:
        logger.warning("Could not remove image directory %s: %s", image_dir, exc)
