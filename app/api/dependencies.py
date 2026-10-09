"""FastAPI dependency injection helpers."""

import uuid

from fastapi import Request
from fastapi.responses import JSONResponse

from app.core.logging import request_id_var


async def request_id_middleware(request: Request, call_next):
    """Attach a unique request_id to every request."""
    rid = str(uuid.uuid4())
    request_id_var.set(rid)
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response
