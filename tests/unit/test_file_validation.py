"""Unit tests for file validation helpers in the documents route."""

import pytest

from app.api.routes.documents import _sanitize_filename


def test_sanitize_normal_filename():
    assert _sanitize_filename("ky-thuat-trong-lua.pdf") == "ky-thuat-trong-lua.pdf"


def test_sanitize_strips_path_traversal():
    assert "/" not in _sanitize_filename("../../../etc/passwd.pdf")
    assert "\\" not in _sanitize_filename("..\\evil.pdf")


def test_sanitize_removes_null_bytes():
    name = "file\x00name.pdf"
    result = _sanitize_filename(name)
    assert "\x00" not in result


def test_sanitize_falls_back_to_default():
    result = _sanitize_filename("")
    assert result == "document.pdf"


def test_sanitize_replaces_special_chars():
    result = _sanitize_filename("file<>|?*.pdf")
    # Special characters should be replaced or removed
    assert "<" not in result
    assert ">" not in result
