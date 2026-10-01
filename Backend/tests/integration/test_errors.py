from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.errors import AnalysisNotFoundError, InferenceError

PROBLEM_JSON = "application/problem+json"


def test_app_error_returns_problem_json(app: FastAPI) -> None:
    @app.get("/__test__/app-error")
    async def _raise_app_error() -> None:
        raise AnalysisNotFoundError()

    with TestClient(app) as client:
        response = client.get("/__test__/app-error")

    assert response.status_code == 404
    assert response.headers["content-type"] == PROBLEM_JSON
    body = response.json()
    assert body["code"] == "ANALYSIS_NOT_FOUND"
    assert body["status"] == 404
    assert body["instance"] == "/__test__/app-error"
    assert body["request_id"] == response.headers["x-request-id"]


def test_validation_error_returns_field_errors(app: FastAPI) -> None:
    @app.get("/__test__/validate")
    async def _validate(count: int) -> dict[str, int]:
        return {"count": count}

    with TestClient(app) as client:
        response = client.get("/__test__/validate", params={"count": "not-a-number"})

    assert response.status_code == 422
    assert response.headers["content-type"] == PROBLEM_JSON
    body = response.json()
    assert body["code"] == "VALIDATION_ERROR"
    assert body["errors"]
    assert body["errors"][0]["field"].endswith("count")


def test_unhandled_exception_returns_500_without_leaking_details(app: FastAPI) -> None:
    @app.get("/__test__/boom")
    async def _boom() -> None:
        raise RuntimeError("super secret internal detail")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/__test__/boom")

    assert response.status_code == 500
    assert response.headers["content-type"] == PROBLEM_JSON
    body = response.json()
    assert body["code"] == "INTERNAL_ERROR"
    assert "super secret internal detail" not in response.text


def test_5xx_app_error_returns_problem_json(app: FastAPI) -> None:
    @app.get("/__test__/server-app-error")
    async def _raise_inference_error() -> None:
        raise InferenceError()

    with TestClient(app) as client:
        response = client.get("/__test__/server-app-error")

    assert response.status_code == 500
    assert response.headers["content-type"] == PROBLEM_JSON
    assert response.json()["code"] == "INFERENCE_FAILED"


def test_404_on_unknown_route_is_problem_json(client: TestClient) -> None:
    response = client.get("/this-route-does-not-exist")
    assert response.status_code == 404
    assert response.headers["content-type"] == PROBLEM_JSON
    assert response.json()["code"] == "NOT_FOUND"
