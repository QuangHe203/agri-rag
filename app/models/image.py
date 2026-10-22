"""SQLAlchemy ORM model for PDF page images extracted during ingestion."""

from sqlalchemy import Column, Integer, String

from app.models.document import Base


class PageImageModel(Base):
    """Stores metadata for each image extracted from an ingested PDF page.

    The (document_id, page_number, image_index) triple uniquely identifies
    an image.  ``image_index`` is 0-based within the page.
    The actual PNG file lives at ``file_path`` on disk.
    """

    __tablename__ = "page_images"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(String, nullable=False, index=True)
    file_name = Column(String, nullable=False)          # PDF file name, for display
    page_number = Column(Integer, nullable=False)
    image_index = Column(Integer, nullable=False)       # 0-based index within the page
    file_path = Column(String, nullable=False)          # absolute path to stored PNG
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    mime_type = Column(String, nullable=False, default="image/png")
