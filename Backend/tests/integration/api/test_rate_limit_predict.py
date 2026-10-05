"""RATE_LIMIT_PREDICT through the whole middleware stack."""

from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from tests.integration.api.conftest import MakeClient
from tests.integration.api.helpers import (
    ANALYSES_URL,
    LEAF_URL,
    TREE_URL,
    assert_problem,
    headers_for,
    post_image,
)


def test_predict_is_rate_limited_per_client_ip(
    make_client: MakeClient, client_id: uuid.UUID
) -> None:
    client = make_client(RATE_LIMIT_PREDICT="2/minute")

    first = post_image(client, TREE_URL, client_id)
    second = post_image(client, TREE_URL, client_id)
    third = post_image(client, TREE_URL, client_id)

    assert first.status_code == second.status_code == 201
    # Successful responses advertise the budget.
    assert first.headers["x-ratelimit-limit"] == "2"
    assert first.headers["x-ratelimit-remaining"] == "1"
    assert second.headers["x-ratelimit-remaining"] == "0"

    body = assert_problem(third, 429, "RATE_LIMITED")
    assert body["title"] == "Too many requests"
    assert int(third.headers["retry-after"]) >= 1
    assert third.headers["x-request-id"] == body["request_id"]
    assert third.headers["x-content-type-options"] == "nosniff"  # security headers still applied


def test_the_budget_is_shared_by_both_predict_routes(
    make_client: MakeClient, client_id: uuid.UUID
) -> None:
    client = make_client(RATE_LIMIT_PREDICT="2/minute")

    assert post_image(client, TREE_URL, client_id).status_code == 201
    assert post_image(client, LEAF_URL, client_id).status_code == 201
    assert_problem(post_image(client, TREE_URL, client_id), 429, "RATE_LIMITED")
    assert_problem(post_image(client, LEAF_URL, client_id), 429, "RATE_LIMITED")


def test_the_limit_is_per_ip_not_per_client_id(make_client: MakeClient) -> None:
    client = make_client(RATE_LIMIT_PREDICT="2/minute")

    # Rotating the (self-chosen) X-Client-Id must not buy a fresh budget.
    assert post_image(client, TREE_URL, uuid.uuid4()).status_code == 201
    assert post_image(client, TREE_URL, uuid.uuid4()).status_code == 201
    assert_problem(post_image(client, TREE_URL, uuid.uuid4()), 429, "RATE_LIMITED")


def test_cloudflare_header_selects_the_budget_when_trusted(
    make_client: MakeClient, client_id: uuid.UUID
) -> None:
    client = make_client(RATE_LIMIT_PREDICT="1/minute", TRUST_CLOUDFLARE_HEADERS="true")

    def post(ip: str) -> int:
        return post_image(
            client, TREE_URL, client_id, extra_headers={"CF-Connecting-IP": ip}
        ).status_code

    assert post("203.0.113.7") == 201
    assert post("203.0.113.8") == 201  # a different visitor behind the tunnel
    assert post("203.0.113.7") == 429


def test_rejected_requests_do_not_consume_the_budget(
    make_client: MakeClient, client_id: uuid.UUID
) -> None:
    client = make_client(RATE_LIMIT_PREDICT="1/minute")

    for _ in range(3):  # invalid uploads fail validation before reaching the limiter
        assert post_image(client, TREE_URL, client_id, data={"latitude": "1"}).status_code == 422
    assert post_image(client, TREE_URL, client_id).status_code == 201


def test_other_routes_are_not_affected_by_the_predict_limit(
    make_client: MakeClient, client_id: uuid.UUID
) -> None:
    client = make_client(RATE_LIMIT_PREDICT="1/minute")
    post_image(client, TREE_URL, client_id)
    assert_problem(post_image(client, TREE_URL, client_id), 429, "RATE_LIMITED")

    assert client.get(ANALYSES_URL, headers=headers_for(client_id)).status_code == 200
    assert client.get("/health").status_code == 200


def test_default_limit_allows_normal_use(api: TestClient, client_id: uuid.UUID) -> None:
    # RATE_LIMIT_PREDICT defaults to 10/minute.
    statuses = [post_image(api, TREE_URL, client_id).status_code for _ in range(10)]
    assert statuses == [201] * 10
    assert_problem(post_image(api, TREE_URL, client_id), 429, "RATE_LIMITED")


def test_rate_limit_headers_are_not_duplicated(
    make_client: MakeClient, client_id: uuid.UUID
) -> None:
    client = make_client(RATE_LIMIT_PREDICT="5/minute")

    response = post_image(client, TREE_URL, client_id)

    # The decorator and SlowAPIMiddleware both know about the limit; each header must
    # still appear once, or clients would read "5, 5".
    for name in ("x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-reset", "retry-after"):
        assert len(response.headers.get_list(name)) <= 1, name
    assert response.headers.get_list("x-ratelimit-limit") == ["5"]
