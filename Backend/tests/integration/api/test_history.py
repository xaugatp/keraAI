"""GET /analyses and GET /analyses/{id}: who sees what, filters, paging."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.api.helpers import (
    ANALYSES_URL,
    TREE_URL,
    assert_problem,
    headers_for,
    post_image,
    seed_analysis,
)

BASE = datetime(2026, 10, 1, 8, 0, 0)  # naive UTC, like the DB


class World:
    """Rows for client A, client B and the public samples, newest first within each owner."""

    def __init__(
        self,
        factory: sessionmaker[Session],
        data_root: Path,
        a: uuid.UUID,
        b: uuid.UUID,
    ) -> None:
        def seed(offset: int, **kw: object) -> uuid.UUID:
            return seed_analysis(
                factory,
                data_root,
                created_at=BASE + timedelta(minutes=offset),
                **kw,  # type: ignore[arg-type]
            )

        self.a1 = seed(1, client_id=a)
        self.a2 = seed(2, client_id=a, model_key="leaf_segmentation", with_result=True)
        self.a3 = seed(3, client_id=a, title="mine")
        self.b1 = seed(4, client_id=b)
        self.b2 = seed(5, client_id=b)
        self.s1 = seed(6, client_id=None, is_sample=True, title="Healthy tree")
        self.s2 = seed(
            7, client_id=None, is_sample=True, title="Leaf spot", model_key="leaf_segmentation"
        )
        # Rows that must never be listed:
        self.deleted = seed(8, client_id=a, deleted_at=BASE + timedelta(hours=1))
        self.failed = seed(9, client_id=a, status="failed")
        self.failed_sample = seed(10, client_id=None, is_sample=True, status="failed")


@pytest.fixture
def world(
    api: TestClient,
    session_factory: sessionmaker[Session],
    data_root: Path,
    client_id: uuid.UUID,
    other_client_id: uuid.UUID,
) -> World:
    return World(session_factory, data_root, client_id, other_client_id)


def _ids(response_json: dict[str, object]) -> list[str]:
    items = response_json["items"]
    assert isinstance(items, list)
    return [item["id"] for item in items]


def test_default_scope_is_my_rows_plus_samples_newest_first(
    api: TestClient, client_id: uuid.UUID, world: World
) -> None:
    response = api.get(ANALYSES_URL, headers=headers_for(client_id))

    assert response.status_code == 200
    body = response.json()
    assert _ids(body) == [
        str(world.s2),
        str(world.s1),
        str(world.a3),
        str(world.a2),
        str(world.a1),
    ]
    assert (body["page"], body["page_size"], body["total"], body["total_pages"]) == (1, 20, 5, 1)


def test_another_clients_rows_deleted_rows_and_failed_rows_are_never_listed(
    api: TestClient, client_id: uuid.UUID, world: World
) -> None:
    ids = _ids(api.get(ANALYSES_URL, headers=headers_for(client_id)).json())
    for hidden in (world.b1, world.b2, world.deleted, world.failed, world.failed_sample):
        assert str(hidden) not in ids


def test_each_client_sees_only_their_own_rows(
    api: TestClient, other_client_id: uuid.UUID, world: World
) -> None:
    ids = _ids(api.get(ANALYSES_URL, headers=headers_for(other_client_id)).json())
    assert ids == [str(world.s2), str(world.s1), str(world.b2), str(world.b1)]


def test_a_client_with_no_history_sees_only_the_samples(api: TestClient, world: World) -> None:
    ids = _ids(api.get(ANALYSES_URL, headers=headers_for(uuid.uuid4())).json())
    assert ids == [str(world.s2), str(world.s1)]


@pytest.mark.parametrize(
    ("scope", "expected"),
    [
        ("mine", ["a3", "a2", "a1"]),
        ("samples", ["s2", "s1"]),
        ("all_visible", ["s2", "s1", "a3", "a2", "a1"]),
    ],
)
def test_scope_filter(
    api: TestClient, client_id: uuid.UUID, world: World, scope: str, expected: list[str]
) -> None:
    response = api.get(ANALYSES_URL, params={"scope": scope}, headers=headers_for(client_id))
    assert _ids(response.json()) == [str(getattr(world, name)) for name in expected]


def test_model_key_filter(api: TestClient, client_id: uuid.UUID, world: World) -> None:
    leaf = api.get(
        ANALYSES_URL, params={"model_key": "leaf_segmentation"}, headers=headers_for(client_id)
    ).json()
    assert _ids(leaf) == [str(world.s2), str(world.a2)]

    tree = api.get(
        ANALYSES_URL,
        params={"model_key": "tree_classification", "scope": "mine"},
        headers=headers_for(client_id),
    ).json()
    assert _ids(tree) == [str(world.a3), str(world.a1)]

    planned = api.get(
        ANALYSES_URL, params={"model_key": "leaf_disease"}, headers=headers_for(client_id)
    ).json()
    assert planned["items"] == [] and planned["total"] == 0


def test_pagination(api: TestClient, client_id: uuid.UUID, world: World) -> None:
    headers = headers_for(client_id)
    page1 = api.get(ANALYSES_URL, params={"page_size": 2}, headers=headers).json()
    page2 = api.get(ANALYSES_URL, params={"page_size": 2, "page": 2}, headers=headers).json()
    page3 = api.get(ANALYSES_URL, params={"page_size": 2, "page": 3}, headers=headers).json()
    beyond = api.get(ANALYSES_URL, params={"page_size": 2, "page": 4}, headers=headers).json()

    assert [len(p["items"]) for p in (page1, page2, page3, beyond)] == [2, 2, 1, 0]
    assert all(p["total"] == 5 and p["total_pages"] == 3 for p in (page1, page2, page3, beyond))
    assert (page2["page"], page2["page_size"]) == (2, 2)
    paged = _ids(page1) + _ids(page2) + _ids(page3)
    assert paged == _ids(api.get(ANALYSES_URL, headers=headers).json())  # no gaps, no repeats
    assert len(set(paged)) == 5


def test_empty_history_page(api: TestClient) -> None:
    body = api.get(ANALYSES_URL, headers=headers_for(uuid.uuid4())).json()
    assert body == {"items": [], "page": 1, "page_size": 20, "total": 0, "total_pages": 0}


@pytest.mark.parametrize(
    "params",
    [
        {"page": "0"},
        {"page": "-1"},
        {"page": "abc"},
        {"page_size": "0"},
        {"page_size": "101"},
        {"scope": "everything"},
        {"model_key": "banana_split"},
    ],
)
def test_invalid_query_is_422(
    api: TestClient, client_id: uuid.UUID, params: dict[str, str]
) -> None:
    assert_problem(
        api.get(ANALYSES_URL, params=params, headers=headers_for(client_id)),
        422,
        "VALIDATION_ERROR",
    )


def test_page_size_100_is_allowed(api: TestClient, client_id: uuid.UUID) -> None:
    body = api.get(ANALYSES_URL, params={"page_size": 100}, headers=headers_for(client_id)).json()
    assert body["page_size"] == 100


def test_summary_shape_and_image_links(api: TestClient, client_id: uuid.UUID, world: World) -> None:
    items = {
        item["id"]: item
        for item in api.get(ANALYSES_URL, headers=headers_for(client_id)).json()["items"]
    }

    own = items[str(world.a3)]
    assert set(own) == {
        "id",
        "model_key",
        "source",
        "is_sample",
        "title",
        "display_label",
        "confidence",
        "is_uncertain",
        "thumbnail_url",
        "latitude",
        "longitude",
        "created_at",
    }
    assert own["title"] == "mine" and own["is_sample"] is False and own["source"] == "upload"
    assert own["display_label"] == "Banana tree" and own["confidence"] == 0.947
    assert own["created_at"] == "2026-10-01T08:03:00Z"
    query = parse_qs(urlparse(own["thumbnail_url"]).query)
    assert {"variant", "exp", "sig"} <= set(query)  # private image: signed link

    sample = items[str(world.s1)]
    assert sample["is_sample"] is True and sample["source"] == "sample"
    # Samples are public: a plain, stable URL (cacheable) with no signature.
    assert sample["thumbnail_url"] == (f"/api/v1/analyses/{world.s1}/image?variant=thumbnail")
    assert api.get(sample["thumbnail_url"]).status_code == 200  # works with no credentials at all
    assert api.get(own["thumbnail_url"]).status_code == 200


def test_history_requires_a_client_id(api: TestClient) -> None:
    assert_problem(api.get(ANALYSES_URL), 400, "MISSING_CLIENT_ID")
    assert_problem(api.get(ANALYSES_URL, headers={"X-Client-Id": "nope"}), 400, "INVALID_CLIENT_ID")


def test_new_predictions_appear_in_history_immediately(
    api: TestClient, client_id: uuid.UUID
) -> None:
    created = post_image(api, TREE_URL, client_id).json()
    listing = api.get(ANALYSES_URL, headers=headers_for(client_id)).json()
    assert _ids(listing) == [created["id"]]
    assert listing["items"][0]["display_label"] == "Banana tree"


# --- GET /analyses/{id} --------------------------------------------------------------------


def test_detail_of_my_own_analysis(api: TestClient, client_id: uuid.UUID, world: World) -> None:
    response = api.get(f"{ANALYSES_URL}/{world.a2}", headers=headers_for(client_id))

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == str(world.a2)
    assert body["model_key"] == "leaf_segmentation"
    assert body["image"]["result_url"] is not None  # this row has a result image
    assert body["created_at"] == "2026-10-01T08:02:00Z"
    assert set(body) == {
        "id",
        "model_key",
        "model_version",
        "status",
        "source",
        "is_sample",
        "title",
        "description",
        "prediction",
        "details",
        "location",
        "image",
        "timings_ms",
        "created_at",
    }


def test_detail_matches_what_predict_returned(api: TestClient, client_id: uuid.UUID) -> None:
    created = post_image(api, TREE_URL, client_id, data={"latitude": "1", "longitude": "2"}).json()
    fetched = api.get(f"{ANALYSES_URL}/{created['id']}", headers=headers_for(client_id)).json()

    # Signed URLs are minted per response (their exp/sig may differ); everything else is identical.
    for body in (created, fetched):
        for key in ("original_url", "thumbnail_url"):
            body["image"][key] = body["image"][key].split("&exp=")[0]
    assert fetched == created


def test_samples_are_readable_by_any_client(
    api: TestClient, other_client_id: uuid.UUID, world: World
) -> None:
    response = api.get(f"{ANALYSES_URL}/{world.s1}", headers=headers_for(other_client_id))
    assert response.status_code == 200
    body = response.json()
    assert body["is_sample"] is True and body["title"] == "Healthy tree"
    assert body["image"]["original_url"] == f"/api/v1/analyses/{world.s1}/image?variant=original"


@pytest.mark.parametrize("which", ["b1", "deleted_for_a", "missing"])
def test_foreign_deleted_and_missing_ids_are_all_the_same_404(
    api: TestClient, client_id: uuid.UUID, world: World, which: str
) -> None:
    target = {"b1": world.b1, "deleted_for_a": world.deleted, "missing": uuid.uuid4()}[which]
    response = api.get(f"{ANALYSES_URL}/{target}", headers=headers_for(client_id))
    body = assert_problem(response, 404, "ANALYSIS_NOT_FOUND")
    # Identical wording in every case: nothing reveals that another user's row exists.
    assert body["detail"] == "Analysis not found"


def test_failed_sample_is_not_visible_to_anyone(
    api: TestClient, client_id: uuid.UUID, world: World
) -> None:
    response = api.get(f"{ANALYSES_URL}/{world.failed_sample}", headers=headers_for(client_id))
    assert_problem(response, 404, "ANALYSIS_NOT_FOUND")


def test_detail_id_must_be_a_uuid(api: TestClient, client_id: uuid.UUID) -> None:
    response = api.get(f"{ANALYSES_URL}/not-a-uuid", headers=headers_for(client_id))
    assert_problem(response, 422, "VALIDATION_ERROR")


def test_detail_requires_a_client_id(api: TestClient, world: World) -> None:
    assert_problem(api.get(f"{ANALYSES_URL}/{world.a1}"), 400, "MISSING_CLIENT_ID")
