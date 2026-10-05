"""OpenAPI documentation of the error responses (RFC 9457 problem+json).

Every route lists the errors it can really produce, so the generated TypeScript
client (and /docs) show them. The body is always ``ProblemDetail``; the media
type is rewritten to ``application/problem+json`` in ``app.api.openapi``.
"""

from __future__ import annotations

from typing import Any

from app.schemas.common import ProblemDetail

_DESCRIPTIONS: dict[int, str] = {
    400: "Bad request: missing/invalid X-Client-Id, or the image is invalid or too small.",
    403: "Forbidden: invalid or expired image signature, or the analysis is an immutable sample.",
    404: "Not found (also returned for another client's analysis — existence is never revealed).",
    413: "The upload exceeds the size limit.",
    415: "The image format is not supported.",
    422: "Validation error: a field is missing, malformed or out of range.",
    429: "Rate limit exceeded; see the Retry-After header.",
    500: "Unexpected server error or the model failed during inference.",
    503: "The model or the database is currently unavailable.",
}

ProblemResponses = dict[int | str, dict[str, Any]]


def problem_responses(*codes: int) -> ProblemResponses:
    """``responses=`` mapping for a route: one ProblemDetail entry per status code."""
    return {code: {"model": ProblemDetail, "description": _DESCRIPTIONS[code]} for code in codes}
