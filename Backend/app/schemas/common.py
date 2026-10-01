from __future__ import annotations

from pydantic import BaseModel


class ProblemDetail(BaseModel):
    """RFC 9457 application/problem+json body shape (for OpenAPI docs)."""

    type: str
    title: str
    status: int
    detail: str
    instance: str
    code: str
    request_id: str | None = None
    errors: list[dict[str, str]] | None = None
