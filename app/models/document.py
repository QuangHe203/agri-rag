"""SQLAlchemy ORM models for document metadata stored in SQLite."""

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings


class Base(DeclarativeBase):
    pass


class DocumentModel(Base):
    __tablename__ = "documents"

    document_id = Column(String, primary_key=True, index=True)
    file_name = Column(String, nullable=False)
    file_hash = Column(String, nullable=False, index=True)  # SHA-256
    file_size = Column(Integer, nullable=False)
    mime_type = Column(String, nullable=False, default="application/pdf")
    status = Column(String, nullable=False, default="uploaded")
    # uploaded / processing / indexed / failed / duplicate
    page_count = Column(Integer, nullable=True)
    chunk_count = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    uploaded_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    indexed_at = Column(DateTime, nullable=True)


def _get_db_url() -> str:
    settings = get_settings()
    import os
    os.makedirs(os.path.dirname(settings.SQLITE_DB_PATH), exist_ok=True)
    return f"sqlite:///{settings.SQLITE_DB_PATH}"


def create_db_engine():
    url = _get_db_url()
    engine = create_engine(url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    return engine


_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_db_engine()
    return _engine


def get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return _SessionLocal


def get_db() -> Session:
    factory = get_session_factory()
    return factory()
