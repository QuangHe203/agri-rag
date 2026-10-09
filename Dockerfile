# ─────────────────────────────────────────────────────────────────────────────
# Multi-stage Dockerfile for Agriculture RAG Backend
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

# Install Java 11 (required by OpenDataLoader PDF)
RUN apt-get update && apt-get install -y --no-install-recommends \
    default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Dependency stage ──────────────────────────────────────────────────────────
FROM base AS deps

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM deps AS runtime

# Create non-root user
RUN useradd -m -u 1000 raguser

# Copy source
COPY --chown=raguser:raguser app/ ./app/
COPY --chown=raguser:raguser scripts/ ./scripts/

# Create data directories
RUN mkdir -p /app/data/uploads /app/data/vector_db \
    && chown -R raguser:raguser /app/data

USER raguser

# Environment defaults (override via docker run -e or docker-compose)
ENV APP_ENV=production \
    UPLOAD_DIR=/app/data/uploads \
    VECTOR_DB_PATH=/app/data/vector_db \
    SQLITE_DB_PATH=/app/data/documents.db \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
