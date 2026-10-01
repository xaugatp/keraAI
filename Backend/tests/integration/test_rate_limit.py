from __future__ import annotations

from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient

from app.core.rate_limit import limiter


def test_rate_limit_exceeded_returns_429_problem_json(app: FastAPI) -> None:
    # No route is rate-limited yet in Phase 1 (health must stay unthrottled,
    # and predict doesn't exist until Phase 5) — this proves the
    # limiter/SlowAPIMiddleware/handler wiring actually works end-to-end
    # before anything in the real app depends on it.
    #
    # `response: Response` is required here, not decorative: on a successful
    # hit slowapi injects X-RateLimit-* headers by mutating this exact
    # parameter via kwargs — an endpoint that just returns a plain dict with
    # no `response` param makes slowapi crash trying to header-inject `None`.
    @app.get("/__test__/limited")
    @limiter.limit("1/minute")
    async def _limited(request: Request, response: Response) -> dict[str, bool]:
        return {"ok": True}

    with TestClient(app) as client:
        first = client.get("/__test__/limited")
        second = client.get("/__test__/limited")

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.headers["content-type"] == "application/problem+json"
    body = second.json()
    assert body["code"] == "RATE_LIMITED"
    assert body["status"] == 429
    assert "retry-after" in {key.lower() for key in second.headers}
