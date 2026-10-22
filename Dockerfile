
# syntax=docker/dockerfile:1.7

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_INPUT=1

WORKDIR /app

# Install Java only once in a reusable base layer
RUN apt-get update \
    && apt-get install -y --no-install-recommends default-jre-headless \
    && rm -rf /var/lib/apt/lists/*


# ── Dependency stage ───────────────────────────────────────────
FROM base AS deps

COPY requirements.txt .

# Cache pip downloads between Docker builds.
# Install CPU-only PyTorch first to avoid unnecessary CUDA packages.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip \
    && python -m pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.9.0" \
    && python -m pip install -r requirements.txt


# ── Runtime stage ──────────────────────────────────────────────
FROM deps AS runtime

RUN useradd --create-home --uid 1000 raguser \
    && mkdir -p /app/data/uploads /app/data/vector_db \
    && chown -R raguser:raguser /app/data

COPY --chown=raguser:raguser app/ ./app/
COPY --chown=raguser:raguser scripts/ ./scripts/

ENV APP_ENV=production \
    UPLOAD_DIR=/app/data/uploads \
    VECTOR_DB_PATH=/app/data/vector_db \
    SQLITE_DB_PATH=/app/data/documents.db

USER raguser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
