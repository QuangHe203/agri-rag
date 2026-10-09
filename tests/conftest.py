"""Shared pytest configuration and fixtures."""

import os

import pytest


def pytest_configure(config):
    """Ensure tests don't accidentally load a real .env with secrets."""
    # Override critical env vars so tests never use real API keys
    os.environ.setdefault("GEMINI_API_KEY", "test-fake-key")
    os.environ.setdefault("APP_ENV", "test")
