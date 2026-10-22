"""Image serving endpoint.

Serves PNG images that were extracted from source PDF pages during ingestion.

Route: GET /api/v1/images/{document_id}/{page_number}/{image_index}
"""

import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.image_service import get_image_by_coordinates

router = APIRouter(prefix="/api/v1/images", tags=["Images"])
logger = get_logger(__name__)


@router.get(
    "/{document_id}/{page_number}/{image_index}",
    summary="Lấy ảnh minh họa từ tài liệu nguồn",
    response_class=FileResponse,
    responses={
        200: {"content": {"image/png": {}}, "description": "PNG image"},
        404: {"description": "Image not found"},
    },
)
def get_image(document_id: str, page_number: int, image_index: int) -> FileResponse:
    """Return the PNG image identified by its coordinate triple.

    - ``document_id``: UUID of the source document.
    - ``page_number``: 1-based page number.
    - ``image_index``: 0-based image index within the page.
    """
    row = get_image_by_coordinates(document_id, page_number, image_index)
    if row is None:
        raise HTTPException(status_code=404, detail="Image not found.")

    if not os.path.isfile(row.file_path):
        logger.warning(
            "Image record exists but file missing: document_id=%s page=%d idx=%d path=%s",
            document_id,
            page_number,
            image_index,
            row.file_path,
        )
        raise HTTPException(status_code=404, detail="Image file not found on disk.")

    return FileResponse(
        path=row.file_path,
        media_type=row.mime_type,
        filename=os.path.basename(row.file_path),
    )
