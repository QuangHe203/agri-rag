"""Unit tests for the PDF parser module."""

import unicodedata

import pytest

from app.providers.pdf.opendataloader_parser import (
    PDFParseResult,
    ParsedPage,
    normalize_vietnamese,
)


def test_normalize_removes_extra_whitespace():
    raw = "  Xin  chào   thế   giới  "
    result = normalize_vietnamese(raw)
    assert "  " not in result
    assert result == "Xin chào thế giới"


def test_normalize_preserves_vietnamese_diacritics():
    text = "Kỹ thuật trồng lúa nước"
    result = normalize_vietnamese(text)
    assert result == "Kỹ thuật trồng lúa nước"


def test_normalize_does_not_strip_diacritics():
    text = "đạo ôn, bệnh khô vằn"
    result = normalize_vietnamese(text)
    assert "đ" in result
    assert "ô" in result


def test_normalize_nfc():
    # NFD form (decomposed) should be normalized to NFC
    nfd = unicodedata.normalize("NFD", "ầ")
    result = normalize_vietnamese(nfd)
    assert unicodedata.is_normalized("NFC", result)


def test_normalize_strips_leading_trailing():
    text = "\n\n  Nội dung PDF  \n\n"
    result = normalize_vietnamese(text)
    assert not result.startswith("\n")
    assert not result.endswith("\n")


def test_parse_result_success():
    pages = [ParsedPage(page_number=1, content="Nội dung trang 1")]
    result = PDFParseResult(file_name="test.pdf", pages=pages)
    assert result.success is True
    assert result.page_count == 1


def test_parse_result_failure_no_pages():
    result = PDFParseResult(file_name="test.pdf", pages=[], error="parse error")
    assert result.success is False


def test_parse_result_failure_with_error():
    result = PDFParseResult(file_name="test.pdf", pages=[], error="Java not found")
    assert result.success is False
    assert result.error == "Java not found"


def test_parsed_page_repr():
    page = ParsedPage(page_number=5, content="abc")
    assert "5" in repr(page)
