from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any, cast

from fastapi import FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from sqlalchemy import exc as sa_exc
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.logging import get_request_id
from app.storage.base import StorageError, StoredFileNotFoundError

logger = logging.getLogger(__name__)

# Starlette's add_exception_handler stub wants every handler typed
# (Request, Exception) -> Response — contravariant in the exception type —
# but FastAPI dispatches by the exact/subclass type at runtime, so our
# narrower handler signatures are correct and safe; this cast just tells
# mypy that.
ExceptionHandler = Callable[[Request, Exception], Awaitable[Response] | Response]

PROBLEM_JSON = "application/problem+json"


class AppError(Exception):
    status_code: int = 500
    code: str = "INTERNAL_ERROR"
    title: str = "Internal error"

    def __init__(self, detail: str | None = None, *, extra: dict[str, Any] | None = None) -> None:
        self.detail = detail or self.title
        self.extra = extra or {}
        super().__init__(self.detail)


class InvalidImageError(AppError):
    status_code = status.HTTP_400_BAD_REQUEST
    code = "INVALID_IMAGE"
    title = "Invalid image"


class ImageTooSmallError(AppError):
    status_code = status.HTTP_400_BAD_REQUEST
    code = "IMAGE_TOO_SMALL"
    title = "Image too small"


class ImageTooLargeDimensionsError(AppError):
    status_code = status.HTTP_400_BAD_REQUEST
    code = "IMAGE_TOO_LARGE_DIMENSIONS"
    title = "Image dimensions too large"


class MissingClientIdError(AppError):
    status_code = status.HTTP_400_BAD_REQUEST
    code = "MISSING_CLIENT_ID"
    title = "Missing X-Client-Id header"


class InvalidClientIdError(AppError):
    status_code = status.HTTP_400_BAD_REQUEST
    code = "INVALID_CLIENT_ID"
    title = "Invalid X-Client-Id header"


class InvalidSignatureError(AppError):
    status_code = status.HTTP_403_FORBIDDEN
    code = "INVALID_SIGNATURE"
    title = "Invalid or expired signature"


class SampleImmutableError(AppError):
    status_code = status.HTTP_403_FORBIDDEN
    code = "SAMPLE_IMMUTABLE"
    title = "Sample analyses cannot be modified"


class AnalysisNotFoundError(AppError):
    status_code = status.HTTP_404_NOT_FOUND
    code = "ANALYSIS_NOT_FOUND"
    title = "Analysis not found"


class PayloadTooLargeError(AppError):
    status_code = status.HTTP_413_CONTENT_TOO_LARGE
    code = "IMAGE_TOO_LARGE"
    title = "Upload too large"


class UnsupportedMediaTypeError(AppError):
    status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    code = "UNSUPPORTED_MEDIA_TYPE"
    title = "Unsupported media type"


class InferenceError(AppError):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    code = "INFERENCE_FAILED"
    title = "Inference failed"


class ModelUnavailableError(AppError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "MODEL_UNAVAILABLE"
    title = "Model unavailable"


class DatabaseUnavailableError(AppError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "DATABASE_UNAVAILABLE"
    title = "Database unavailable"


def _problem_response(
    request: Request,
    *,
    status_code: int,
    code: str,
    title: str,
    detail: str,
    extra: dict[str, Any] | None = None,
) -> JSONResponse:
    body: dict[str, Any] = {
        "type": f"/errors/{code.lower().replace('_', '-')}",
        "title": title,
        "status": status_code,
        "detail": detail,
        "instance": request.url.path,
        "code": code,
        "request_id": get_request_id(),
    }
    if extra:
        body.update(extra)
    return JSONResponse(status_code=status_code, content=body, media_type=PROBLEM_JSON)


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    if exc.status_code >= 500:
        logger.error("app_error", extra={"code": exc.code}, exc_info=exc)
    else:
        logger.warning("app_error", extra={"code": exc.code})
    return _problem_response(
        request,
        status_code=exc.status_code,
        code=exc.code,
        title=exc.title,
        detail=exc.detail,
        extra=exc.extra,
    )


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = [
        {"field": ".".join(str(part) for part in err["loc"]), "message": err["msg"]}
        for err in exc.errors()
    ]
    logger.info("validation_error", extra={"errors": errors})
    return _problem_response(
        request,
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        code="VALIDATION_ERROR",
        title="Validation error",
        detail="One or more fields failed validation.",
        extra={"errors": errors},
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    logger.info("http_exception", extra={"status_code": exc.status_code})
    title = str(exc.detail) if exc.detail else "HTTP error"
    return _problem_response(
        request,
        status_code=exc.status_code,
        code="NOT_FOUND" if exc.status_code == 404 else "HTTP_ERROR",
        title=title,
        detail=title,
    )


async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    response = _problem_response(
        request,
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        code="RATE_LIMITED",
        title="Too many requests",
        detail=f"Rate limit exceeded: {exc.detail}",
    )
    # slowapi's own handler uses this private helper to add Retry-After /
    # X-RateLimit-* headers from the limit that was hit; there's no public
    # equivalent yet, and this is the pattern slowapi's own docs use.
    limiter = request.app.state.limiter
    return limiter._inject_headers(response, request.state.view_rate_limit)


async def database_unavailable_handler(
    request: Request, exc: sa_exc.SQLAlchemyError
) -> JSONResponse:
    """SQL Server unreachable / connection dropped / pool exhausted -> 503.

    The driver message (server name, SQL text, parameters) goes to the log only;
    the client gets a fixed sentence (spec 11: never leak internals).
    """
    logger.error("database_unavailable", extra={"error_type": type(exc).__name__}, exc_info=exc)
    return _problem_response(
        request,
        status_code=DatabaseUnavailableError.status_code,
        code=DatabaseUnavailableError.code,
        title=DatabaseUnavailableError.title,
        detail="The database is temporarily unavailable. Please try again shortly.",
    )


async def database_error_handler(request: Request, exc: sa_exc.SQLAlchemyError) -> JSONResponse:
    """Any other database error (constraint violation, bad SQL, ...) is OUR bug -> 500."""
    logger.error("database_error", extra={"error_type": type(exc).__name__}, exc_info=exc)
    return _problem_response(
        request,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        code="INTERNAL_ERROR",
        title="Internal error",
        detail="An unexpected error occurred.",
    )


async def stored_file_not_found_handler(
    request: Request, exc: StoredFileNotFoundError
) -> JSONResponse:
    """A file the database promised is missing on disk. The service normally
    converts this itself (with context for the log); this is the safety net."""
    logger.error("stored_file_missing", exc_info=exc)
    return _problem_response(
        request,
        status_code=AnalysisNotFoundError.status_code,
        code=AnalysisNotFoundError.code,
        title=AnalysisNotFoundError.title,
        detail="Image not found.",
    )


async def storage_error_handler(request: Request, exc: StorageError) -> JSONResponse:
    """Any other storage failure (path guard tripped, disk error) -> 500, no paths leaked."""
    logger.error("storage_error", extra={"error_type": type(exc).__name__}, exc_info=exc)
    return _problem_response(
        request,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        code="INTERNAL_ERROR",
        title="Internal error",
        detail="An unexpected error occurred.",
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("unhandled_exception", exc_info=exc)
    return _problem_response(
        request,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        code="INTERNAL_ERROR",
        title="Internal error",
        detail="An unexpected error occurred.",
    )


class CatchAllErrorsMiddleware:
    """Turns an unexpected exception into the usual problem+json 500 *inside* the stack.

    Starlette runs the ``Exception`` handler in its outermost layer, outside every
    middleware of ours. A 500 produced there has no ``X-Request-ID`` header, a null
    ``request_id`` in the body (the request-id context is already reset), no CORS
    headers (so the browser hides the body from the frontend) and no security
    headers. Catching here, innermost, gives 500s the same treatment as every other
    error. The ``Exception`` handler stays registered as the last-resort safety net.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        response_started = False

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, receive, tracking_send)
        except Exception as exc:
            if response_started:
                raise  # too late to change the answer; let the server drop the connection
            response = await unhandled_exception_handler(Request(scope), exc)
            await response(scope, receive, send)


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, cast(ExceptionHandler, app_error_handler))
    app.add_exception_handler(
        RequestValidationError, cast(ExceptionHandler, validation_error_handler)
    )
    app.add_exception_handler(
        StarletteHTTPException, cast(ExceptionHandler, http_exception_handler)
    )
    app.add_exception_handler(RateLimitExceeded, cast(ExceptionHandler, rate_limit_handler))
    # Handlers are looked up along the exception's MRO, so the most specific wins:
    # connection-level failures (and pool exhaustion) -> 503, everything else -> 500.
    for unavailable in (sa_exc.OperationalError, sa_exc.InterfaceError, sa_exc.TimeoutError):
        app.add_exception_handler(unavailable, cast(ExceptionHandler, database_unavailable_handler))
    app.add_exception_handler(
        sa_exc.SQLAlchemyError, cast(ExceptionHandler, database_error_handler)
    )
    app.add_exception_handler(
        StoredFileNotFoundError, cast(ExceptionHandler, stored_file_not_found_handler)
    )
    app.add_exception_handler(StorageError, cast(ExceptionHandler, storage_error_handler))
    app.add_exception_handler(Exception, unhandled_exception_handler)
