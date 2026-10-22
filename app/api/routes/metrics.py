"""Prometheus metrics endpoint.

Exposes the process + RAG pipeline metrics in the Prometheus text exposition
format so they can be scraped by a Prometheus server and visualised in Grafana.
"""

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

router = APIRouter()


@router.get(
    "/metrics",
    tags=["Observability"],
    summary="Prometheus metrics",
    include_in_schema=False,
)
async def metrics() -> Response:
    """Return all registered metrics in the Prometheus text format."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
