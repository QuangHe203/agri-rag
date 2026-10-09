"""PDF parser using OpenDataLoader PDF (with LangChain integration)."""

import unicodedata
from pathlib import Path
from typing import Optional

from app.core.logging import get_logger

logger = get_logger(__name__)


def normalize_vietnamese(text: str) -> str:
    """Normalize Unicode to NFC and clean whitespace, preserving Vietnamese diacritics."""
    text = unicodedata.normalize("NFC", text)
    # Collapse excessive whitespace (but keep newlines for structure)
    lines = text.splitlines()
    lines = [" ".join(line.split()) for line in lines]
    return "\n".join(lines).strip()


class ParsedPage:
    """Represents a single parsed page from a PDF."""

    def __init__(self, page_number: int, content: str):
        self.page_number = page_number
        self.content = content

    def __repr__(self) -> str:
        return f"ParsedPage(page={self.page_number}, chars={len(self.content)})"


class PDFParseResult:
    """Result of parsing a PDF file."""

    def __init__(self, file_name: str, pages: list[ParsedPage], error: Optional[str] = None):
        self.file_name = file_name
        self.pages = pages[:30]
        self.error = error

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def success(self) -> bool:
        return self.error is None and len(self.pages) > 0


def parse_pdf(file_path: str | Path, file_name: str) -> PDFParseResult:
    """
    Parse a PDF file using OpenDataLoader PDF via LangChain integration.

    Falls back to PyPDF2 if the OpenDataLoader is unavailable.
    """
    file_path = Path(file_path)
    logger.info("Starting PDF parse: %s", file_name)

    try:
        return _parse_with_opendataloader(file_path, file_name)
    except ImportError:
        logger.warning(
            "langchain_opendataloader_pdf not available, falling back to pypdf"
        )
        return _parse_with_pypdf(file_path, file_name)
    except Exception as exc:
        logger.error("PDF parse failed for %s: %s", file_name, str(exc))
        return PDFParseResult(file_name=file_name, pages=[], error=str(exc))


def _parse_with_opendataloader(file_path: Path, file_name: str) -> PDFParseResult:
    from langchain_opendataloader_pdf import OpenDataLoaderPDFLoader  # type: ignore

    loader = OpenDataLoaderPDFLoader(
        file_path=[str(file_path)],
        format="markdown",
    )
    documents = loader.load()

    pages: list[ParsedPage] = []
    for doc in documents:
        raw_content = doc.page_content or ""
        content = normalize_vietnamese(raw_content)
        if not content:
            continue
        # OpenDataLoader exposes page in metadata as "page" (0-indexed or 1-indexed)
        page_num = doc.metadata.get("page", doc.metadata.get("page_number", 1))
        if isinstance(page_num, int) and page_num == 0:
            page_num = 1
        pages.append(ParsedPage(page_number=int(page_num), content=content))

    logger.info("Parsed %d pages from %s using OpenDataLoader", len(pages), file_name)
    return PDFParseResult(file_name=file_name, pages=pages)


def _parse_with_pypdf(file_path: Path, file_name: str) -> PDFParseResult:
    from pypdf import PdfReader  # type: ignore

    reader = PdfReader(str(file_path))
    pages: list[ParsedPage] = []

    for i, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        content = normalize_vietnamese(raw_text)
        if not content:
            continue
        pages.append(ParsedPage(page_number=i, content=content))

    logger.info("Parsed %d pages from %s using pypdf", len(pages), file_name)
    return PDFParseResult(file_name=file_name, pages=pages)
