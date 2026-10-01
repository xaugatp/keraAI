from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_request_id_is_generated_when_absent(client: TestClient) -> None:
    response = client.get("/health")
    assert "x-request-id" in response.headers
    assert len(response.headers["x-request-id"]) > 0


def test_request_id_is_echoed_when_provided(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-ID": "my-correlation-id"})
    assert response.headers["x-request-id"] == "my-correlation-id"


def test_security_headers_present(client: TestClient) -> None:
    response = client.get("/health")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["x-frame-options"] == "DENY"


def test_cors_allows_configured_origin(client: TestClient) -> None:
    response = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_cors_rejects_other_origins(client: TestClient) -> None:
    response = client.get("/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in response.headers


def test_gzip_compresses_large_json_response(app: FastAPI) -> None:
    # Nothing in Phase 1 returns a >1KB body (minimum_size=1024) to exercise
    # this against — a dedicated oversized route confirms the wiring works
    # before Phase 5's analysis list/detail endpoints exist to prove it.
    @app.get("/__test__/large")
    async def _large() -> dict[str, list[int]]:
        return {"items": list(range(2000))}

    with TestClient(app) as client:
        response = client.get("/__test__/large", headers={"Accept-Encoding": "gzip"})

    assert response.status_code == 200
    assert response.headers.get("content-encoding") == "gzip"
