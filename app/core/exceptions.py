from fastapi import Request
from fastapi.responses import JSONResponse

from app.core.logging import get_request_id


class AppError(Exception):
    """Base application error."""

    status_code: int = 500
    error_code: str = "INTERNAL_SERVER_ERROR"

    def __init__(self, message: str, error_code: str | None = None):
        self.message = message
        if error_code:
            self.error_code = error_code
        super().__init__(message)


class InvalidFileTypeError(AppError):
    status_code = 415
    error_code = "INVALID_FILE_TYPE"


class FileTooLargeError(AppError):
    status_code = 413
    error_code = "FILE_TOO_LARGE"


class DuplicateDocumentError(AppError):
    status_code = 409
    error_code = "DUPLICATE_DOCUMENT"


class DocumentNotFoundError(AppError):
    status_code = 404
    error_code = "DOCUMENT_NOT_FOUND"


class DocumentParseFailedError(AppError):
    status_code = 422
    error_code = "DOCUMENT_PARSE_FAILED"


class EmbeddingFailedError(AppError):
    status_code = 500
    error_code = "EMBEDDING_FAILED"


class VectorStoreUnavailableError(AppError):
    status_code = 503
    error_code = "VECTOR_STORE_UNAVAILABLE"


class NoRelevantContextError(AppError):
    status_code = 200
    error_code = "NO_RELEVANT_CONTEXT"


class LLMConfigurationError(AppError):
    status_code = 500
    error_code = "LLM_CONFIGURATION_ERROR"


class LLMRateLimitedError(AppError):
    status_code = 429
    error_code = "LLM_RATE_LIMITED"


class LLMRequestFailedError(AppError):
    status_code = 502
    error_code = "LLM_REQUEST_FAILED"


def _error_response(request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.error_code,
                "message": exc.message,
                "request_id": get_request_id(),
            }
        },
    )


def register_exception_handlers(app) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        return _error_response(request, exc)

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception):
        from app.core.logging import get_logger
        logger = get_logger(__name__)
        logger.exception("Unhandled exception")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "Lỗi nội bộ của hệ thống.",
                    "request_id": get_request_id(),
                }
            },
        )
