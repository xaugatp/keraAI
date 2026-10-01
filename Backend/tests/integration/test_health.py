from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_health_is_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_ready_with_no_checks_registered(client: TestClient) -> None:
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "components": {}}


def test_health_ready_reports_unavailable_component(app: FastAPI) -> None:
    async def failing_check() -> tuple[bool, str | None]:
        return False, "SQL Server unreachable"

    # readiness_checks must be set AFTER the TestClient starts the app, since
    # lifespan startup resets it to {} — setting it before would just get
    # overwritten.
    with TestClient(app) as client:
        app.state.readiness_checks = {"database": failing_check}
        response = client.get("/health/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["components"]["database"] == {
        "status": "unavailable",
        "detail": "SQL Server unreachable",
    }


def test_health_ready_check_that_raises_is_reported_not_crashed(app: FastAPI) -> None:
    async def broken_check() -> tuple[bool, str | None]:
        raise RuntimeError("boom")

    with TestClient(app) as client:
        app.state.readiness_checks = {"database": broken_check}
        response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["components"]["database"]["status"] == "unavailable"


def test_health_ready_all_ok_is_200(app: FastAPI) -> None:
    async def ok_check() -> tuple[bool, str | None]:
        return True, None

    with TestClient(app) as client:
        app.state.readiness_checks = {"database": ok_check}
        response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
