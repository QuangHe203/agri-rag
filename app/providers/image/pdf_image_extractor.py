"""Extract images from PDF pages and persist them as PNG files.

Uses pypdf (already a project dependency) for page iteration and
Pillow (PIL) for format normalisation and size filtering.

Every extraction step is wrapped in try/except so that a broken or
unsupported image inside a PDF never aborts the overall ingestion.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from app.core.logging import get_logger

logger = get_logger(__name__)

# --- tuneable constants -------------------------------------------------------
_MIN_IMAGE_SIDE: int = 80      # pixels; skip images smaller than this in either axis
_MAX_IMAGES_PER_PAGE: int = 5  # cap per page to avoid flooding from icon-heavy PDFs
# ------------------------------------------------------------------------------


@dataclass
class ExtractedImage:
    """Metadata returned by :func:`extract_images` for one saved image."""

    document_id: str
    file_name: str       # original PDF file name (for display)
    page_number: int     # 1-based page number
    image_index: int     # 0-based index within that page
    file_path: str       # absolute path to the saved PNG file
    width: Optional[int]
    height: Optional[int]
    mime_type: str = "image/png"


def extract_images(
    file_path: str | Path,
    document_id: str,
    file_name: str,
    image_store_path: str | Path,
) -> List[ExtractedImage]:
    """Extract all qualifying images from *file_path* and save them to disk.

    Images smaller than ``_MIN_IMAGE_SIDE`` pixels in either dimension are
    silently dropped (they are typically decorative bullets / icons).

    Args:
        file_path:        Path to the source PDF file.
        document_id:      UUID of the document record (used for directory name).
        file_name:        Original PDF file name stored in the document record.
        image_store_path: Root directory under which ``{document_id}/`` is created.

    Returns:
        A (possibly empty) list of :class:`ExtractedImage` objects.
        Never raises – any failure is logged and an empty list is returned.
    """
    file_path = Path(file_path)
    output_dir = Path(image_store_path) / document_id

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logger.warning("Cannot create image output dir %s: %s", output_dir, exc)
        return []

    try:
        from pypdf import PdfReader  # type: ignore
        reader = PdfReader(str(file_path))
    except Exception as exc:
        logger.warning("Cannot open PDF for image extraction (%s): %s", file_path, exc)
        return []

    if not _pillow_available():
        logger.info(
            "Pillow not available – skipping image extraction for %s", file_name
        )
        return []

    results: List[ExtractedImage] = []

    for page_num, page in enumerate(reader.pages, start=1):
        try:
            page_images = list(page.images)
        except Exception as exc:
            logger.debug("Cannot enumerate images on page %d: %s", page_num, exc)
            continue

        page_saved = 0
        for img_obj in page_images:
            if page_saved >= _MAX_IMAGES_PER_PAGE:
                break
            extracted = _save_one_image(
                img_obj=img_obj,
                document_id=document_id,
                file_name=file_name,
                page_number=page_num,
                image_index=page_saved,
                output_dir=output_dir,
            )
            if extracted is not None:
                results.append(extracted)
                page_saved += 1

    logger.info(
        "Image extraction complete: document_id=%s file=%s images=%d",
        document_id,
        file_name,
        len(results),
    )
    return results


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _pillow_available() -> bool:
    try:
        import PIL  # noqa: F401
        return True
    except ImportError:
        return False


def _save_one_image(
    img_obj,
    document_id: str,
    file_name: str,
    page_number: int,
    image_index: int,
    output_dir: Path,
) -> Optional[ExtractedImage]:
    """Convert a pypdf ImageFile to a PNG on disk.

    Returns ``None`` if the image should be skipped (too small, conversion
    failed, etc.).
    """
    from PIL import Image, UnidentifiedImageError  # type: ignore

    pil_img: Optional[Image.Image] = None

    # Preferred path: pypdf already decoded the image into a PIL Image.
    try:
        pil_img = img_obj.image  # PIL.Image.Image, requires Pillow
    except Exception:
        pass

    # Fallback: open the raw bytes ourselves.
    if pil_img is None:
        try:
            raw_data: bytes = img_obj.data
            if not raw_data:
                return None
            pil_img = Image.open(io.BytesIO(raw_data))
        except (UnidentifiedImageError, Exception):
            return None

    if pil_img is None:
        return None

    try:
        width, height = pil_img.size
    except Exception:
        return None

    # Filter tiny/decorative images.
    if width < _MIN_IMAGE_SIDE or height < _MIN_IMAGE_SIDE:
        return None

    # Normalise colour space so PIL can save as PNG without complaints.
    try:
        if pil_img.mode not in ("RGB", "RGBA", "L", "P"):
            pil_img = pil_img.convert("RGB")
    except Exception:
        pass

    dest_path = output_dir / f"page_{page_number:04d}_img_{image_index:04d}.png"
    try:
        pil_img.save(str(dest_path), format="PNG")
    except Exception as exc:
        logger.debug(
            "Failed to save image page=%d idx=%d: %s", page_number, image_index, exc
        )
        return None

    return ExtractedImage(
        document_id=document_id,
        file_name=file_name,
        page_number=page_number,
        image_index=image_index,
        file_path=str(dest_path),
        width=width,
        height=height,
        mime_type="image/png",
    )
