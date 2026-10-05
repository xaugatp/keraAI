"""Every error response — whatever produced it — is problem+json with code and request_id."""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.integration.api.conftest import MakeClient
from tests.integration.api.helpers import (
    ANALYSES_URL,
    PREFIX,
    TREE_URL,
    assert_problem,
    headers_for,
    post_image,
)

CID = {"X-Client-Id": str(uuid.uuid4())}
SOME_ID = uuid.uuid4()

# (method, url, kwargs, expected status, expected code)
CASES: list[tuple[str, str, dict[str, Any], int, str]] = [
    ("GET", "/no/such/route", {}, 404, "NOT_FOUND"),
    ("GET", f"{PREFIX}/no/such/route", {}, 404, "NOT_FOUND"),
    ("GET", f"{PREFIX}/analyses/{SOME_ID}/nope", {"headers": CID}, 404, "NOT_FOUND"),
    ("DELETE", f"{PREFIX}/models", {}, 405, "HTTP_ERROR"),
    ("GET", f"{PREFIX}/tree/predict", {"headers": CID}, 405, "HTTP_ERROR"),
    ("POST", ANALYSES_URL, {"headers": CID}, 405, "HTTP_ERROR"),
    ("GET", ANALYSES_URL, {}, 400, "MISSING_CLIENT_ID"),
    ("GET", ANALYSES_URL, {"headers": {"X-Client-Id": "x"}}, 400, "INVALID_CLIENT_ID"),
    ("GET", ANALYSES_URL, {"headers": CID, "params": {"page": 0}}, 422, "VALIDATION_ERROR"),
    ("GET", f"{ANALYSES_URL}/not-a-uuid", {"headers": CID}, 422, "VALIDATION_ERROR"),
    ("GET", f"{ANALYSES_URL}/{SOME_ID}", {"headers": CID}, 404, "ANALYSIS_NOT_FOUND"),
    ("DELETE", f"{ANALYSES_URL}/{SOME_ID}", {"headers": CID}, 404, "ANALYSIS_NOT_FOUND"),
    ("GET", f"{ANALYSES_URL}/{SOME_ID}/image", {}, 403, "INVALID_SIGNATURE"),
    ("POST", TREE_URL, {"headers": CID}, 422, "VALIDATION_ERROR"),  # no body at all
]


@pytest.mark.parametrize(("method", "url", "kwargs", "status", "code"), CASES)
def test_error_responses_are_problem_json_with_code_and_request_id(
    api: TestClient, method: str, url: str, kwargs: dict[str, Any], status: int, code: str
) -> None:
    response = api.request(method, url, **kwargs)

    body = assert_problem(response, status, code)
    assert set(body) >= {"type", "title", "status", "detail", "instance", "code", "request_id"}
    assert body["instance"] == url
    assert body["type"] == f"/errors/{code.lower().replace('_', '-')}"


def test_incoming_request_id_is_echoed_in_the_error_body(api: TestClient) -> None:
    response = api.get(ANALYSES_URL, headers={"X-Request-ID": "trace-me-123"})
    assert response.headers["x-request-id"] == "trace-me-123"
    assert response.json()["request_id"] == "trace-me-123"


def test_unexpected_exception_is_a_problem_json_500_with_the_full_header_treatment(
    make_client: MakeClient,
) -> None:
    client = make_client()

    @client.app.get("/__test__/boom")  # type: ignore[attr-defined]
    async def boom() -> None:
        raise RuntimeError("super secret internal detail at C:\\secret\\path")

    response = client.get("/__test__/boom", headers={"Origin": "http://localhost:3000"})

    body = assert_problem(response, 500, "INTERNAL_ERROR")
    assert "secret" not in response.text and "RuntimeError" not in response.text
    assert body["detail"] == "An unexpected error occurred."
    # The catch-all runs INSIDE the middleware stack, so a 500 gets everything a 4xx gets:
    assert response.headers["x-content-type-options"] == "nosniff"
    # ... including CORS, without which the browser would hide this body from the frontend.
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_cors_headers_are_present_on_predict_errors_and_location_is_exposed(
    api: TestClient, client_id: uuid.UUID
) -> None:
    ok = post_image(api, TREE_URL, client_id, extra_headers={"Origin": "http://localhost:3000"})
    assert ok.status_code == 201
    assert "Location" in ok.headers["access-control-expose-headers"]

    bad = post_image(
        api,
        TREE_URL,
        client_id,
        content=b"nope",
        extra_headers={"Origin": "http://localhost:3000"},
    )
    assert bad.status_code == 400
    assert bad.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_preflight_allows_the_client_id_header(api: TestClient) -> None:
    response = api.options(
        TREE_URL,
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "x-client-id,content-type",
        },
    )
    assert response.status_code == 200
    assert "X-Client-Id" in response.headers["access-control-allow-headers"]


def test_every_documented_route_answers_problem_json_to_a_bad_request(
    api: TestClient,
) -> None:
    """Sweep: each real operation in the OpenAPI document, hit with a malformed request."""
    schema = api.get("/openapi.json").json()
    checked = 0
    for path, item in schema["paths"].items():
        if not path.startswith(PREFIX):
            continue
        for method in item:
            url = path.replace("{analysis_id}", "not-a-uuid")
            response = api.request(method.upper(), url, headers=headers_for(uuid.uuid4()))
            if response.status_code < 400:
                continue  # e.g. GET /models: nothing can go wrong with it
            assert response.headers["content-type"] == "application/problem+json", (method, path)
            assert response.json()["code"], (method, path)
            checked += 1
    assert checked >= 5


def test_storage_exceptions_map_to_problem_json_without_leaking_paths(
    make_client: MakeClient,
) -> None:
    from app.storage.base import InvalidStoragePathError, StoredFileNotFoundError

    client = make_client()

    @client.app.get("/__test__/stored-file-missing")  # type: ignore[attr-defined]
    async def missing() -> None:
        raise StoredFileNotFoundError(r"No stored file at 'C:\secret\kera-data\x.jpg'")

    @client.app.get("/__test__/bad-storage-path")  # type: ignore[attr-defined]
    async def bad_path() -> None:
        raise InvalidStoragePathError(r"Storage path escapes DATA_ROOT: '..\..\secret'")

    gone = client.get("/__test__/stored-file-missing")
    assert_problem(gone, 404, "ANALYSIS_NOT_FOUND")
    assert "secret" not in gone.text

    broken = client.get("/__test__/bad-storage-path")
    assert_problem(broken, 500, "INTERNAL_ERROR")
    assert "secret" not in broken.text and "DATA_ROOT" not in broken.text


def test_catch_all_cannot_change_a_response_that_already_started() -> None:
    import asyncio

    from app.core.errors import CatchAllErrorsMiddleware

    sent: list[str] = []

    async def app(scope: Any, receive: Any, send: Any) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        raise RuntimeError("blew up while streaming the body")

    async def send(message: dict[str, Any]) -> None:
        sent.append(message["type"])

    async def receive() -> dict[str, Any]:
        return {"type": "http.request"}

    middleware = CatchAllErrorsMiddleware(app)
    with pytest.raises(RuntimeError, match="streaming"):
        asyncio.run(middleware({"type": "http", "path": "/x", "headers": []}, receive, send))
    assert sent == ["http.response.start"]  # no second, conflicting response was attempted


def test_catch_all_ignores_non_http_scopes() -> None:
    import asyncio

    from app.core.errors import CatchAllErrorsMiddleware

    calls: list[str] = []

    async def app(scope: Any, receive: Any, send: Any) -> None:
        calls.append(scope["type"])

    asyncio.run(CatchAllErrorsMiddleware(app)({"type": "lifespan"}, None, None))  # type: ignore[arg-type]
    assert calls == ["lifespan"]
