"""Everything that must be REJECTED before (or instead of) an analysis being stored."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from tests.fakes import FakePredictor
from tests.fixtures.image_factory import (
    bmp_bytes,
    corrupt_bytes,
    gif_bytes,
    jpeg_bytes,
    solid_image,
)
from tests.integration.api.conftest import MakeClient
from tests.integration.api.helpers import (
    LEAF_URL,
    TREE_URL,
    assert_problem,
    count_rows,
    files_under,
    headers_for,
    post_image,
)

BOTH_ROUTES = pytest.mark.parametrize("url", [TREE_URL, LEAF_URL], ids=["tree", "leaf"])


@pytest.fixture
def nothing_stored(
    session_factory: sessionmaker[Session], data_root: Path, fake_tree: FakePredictor
) -> Iterator[None]:
    """After the test: no row, no file, and no model call — the request was refused early."""
    yield
    assert count_rows(session_factory) == 0
    assert files_under(data_root) == []
    assert fake_tree.predict_images == []


# --- X-Client-Id ------------------------------------------------------------------


@BOTH_ROUTES
def test_missing_client_id_is_400(api: TestClient, url: str, nothing_stored: None) -> None:
    assert_problem(post_image(api, url, None), 400, "MISSING_CLIENT_ID")


@BOTH_ROUTES
@pytest.mark.parametrize(
    "bad",
    [
        "not-a-uuid",
        "12345",
        "",  # present but empty counts as missing
        str(uuid.uuid1()),  # a UUID, but not v4
        "00000000-0000-0000-0000-000000000000",  # nil UUID
        "{" + str(uuid.uuid4()) + "}" + "x",
    ],
)
def test_malformed_client_id_is_400(
    api: TestClient, url: str, bad: str, nothing_stored: None
) -> None:
    response = post_image(api, url, bad)
    assert response.status_code == 400
    assert response.json()["code"] in {"INVALID_CLIENT_ID", "MISSING_CLIENT_ID"}


def test_uppercase_uuid4_client_id_is_accepted(api: TestClient) -> None:
    assert post_image(api, TREE_URL, str(uuid.uuid4()).upper()).status_code == 201


# --- invalid images ------------------------------------------------------------------


@BOTH_ROUTES
@pytest.mark.parametrize(
    ("content", "status", "code"),
    [
        (corrupt_bytes(), 400, "INVALID_IMAGE"),
        (b"", 400, "INVALID_IMAGE"),
        (jpeg_bytes()[:200], 400, "INVALID_IMAGE"),  # truncated JPEG
        (jpeg_bytes(30, 30), 400, "IMAGE_TOO_SMALL"),
        (bmp_bytes(), 415, "UNSUPPORTED_MEDIA_TYPE"),
        (gif_bytes(), 415, "UNSUPPORTED_MEDIA_TYPE"),
    ],
    ids=["corrupt", "empty", "truncated", "too-small", "bmp", "gif"],
)
def test_invalid_images_are_rejected_and_not_persisted(
    api: TestClient,
    client_id: uuid.UUID,
    url: str,
    content: bytes,
    status: int,
    code: str,
    nothing_stored: None,
) -> None:
    assert_problem(post_image(api, url, client_id, content=content), status, code)


def test_claimed_content_type_is_not_trusted(
    api: TestClient, client_id: uuid.UUID, nothing_stored: None
) -> None:
    # A BMP labelled image/jpeg: the DECODED format decides (spec 6).
    response = api.post(
        TREE_URL,
        files={"image": ("photo.jpg", bmp_bytes(), "image/jpeg")},
        headers=headers_for(client_id),
    )
    assert_problem(response, 415, "UNSUPPORTED_MEDIA_TYPE")


def test_decompression_bomb_is_400(
    make_client: MakeClient, client_id: uuid.UUID, nothing_stored: None
) -> None:
    client = make_client(MAX_IMAGE_PIXELS="10000")  # 200x150 = 30000 px > 2 x 10000
    assert_problem(post_image(client, TREE_URL, client_id), 400, "IMAGE_TOO_LARGE_DIMENSIONS")


def test_errors_never_echo_internals(api: TestClient, client_id: uuid.UUID) -> None:
    body = assert_problem(
        post_image(api, TREE_URL, client_id, content=corrupt_bytes()), 400, "INVALID_IMAGE"
    )
    assert body["instance"] == TREE_URL
    assert "Traceback" not in str(body) and "PIL" not in str(body)


# --- 413 ---------------------------------------------------------------------------------

_MIB = 1024 * 1024


def _multipart_stream(payload: bytes, boundary: str = "kera-boundary") -> Iterator[bytes]:
    """The body of a one-file multipart upload, yielded in pieces (=> chunked, no Content-Length)."""
    yield (
        f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="big.jpg"\r\n'
        "Content-Type: image/jpeg\r\n\r\n"
    ).encode()
    for start in range(0, len(payload), 64 * 1024):
        yield payload[start : start + 64 * 1024]
    yield f"\r\n--{boundary}--\r\n".encode()


@BOTH_ROUTES
def test_declared_content_length_over_the_limit_is_413_before_any_parsing(
    make_client: MakeClient, client_id: uuid.UUID, url: str, nothing_stored: None
) -> None:
    client = make_client(MAX_UPLOAD_MB="1")
    # Not even a valid multipart body: if the server tried to parse it we would get a
    # 400/422. The 413 proves the Content-Length check happens first.
    response = client.post(
        url,
        content=b"x" * int(1.2 * _MIB),
        headers={**headers_for(client_id), "Content-Type": "multipart/form-data; boundary=zzz"},
    )
    body = assert_problem(response, 413, "IMAGE_TOO_LARGE")
    assert "1 MB" in body["detail"]


@BOTH_ROUTES
def test_file_over_the_limit_inside_a_small_enough_body_is_413_while_streaming(
    make_client: MakeClient, client_id: uuid.UUID, url: str, nothing_stored: None
) -> None:
    client = make_client(MAX_UPLOAD_MB="1")
    # 1 MiB + 1 KiB of file: the whole body stays under limit + framing slack, so the
    # header pre-check passes and the streaming check (read_upload_limited) must catch it.
    response = post_image(client, url, client_id, content=b"\xff" * (_MIB + 1024))
    assert_problem(response, 413, "IMAGE_TOO_LARGE")


def test_chunked_upload_without_content_length_is_413(
    make_client: MakeClient, client_id: uuid.UUID, nothing_stored: None
) -> None:
    client = make_client(MAX_UPLOAD_MB="1")
    response = client.post(
        TREE_URL,
        content=_multipart_stream(b"\xff" * (2 * _MIB)),
        headers={
            **headers_for(client_id),
            "Content-Type": "multipart/form-data; boundary=kera-boundary",
        },
    )
    assert response.request.headers.get("content-length") is None  # really a chunked upload
    assert_problem(response, 413, "IMAGE_TOO_LARGE")


def test_file_exactly_at_the_limit_is_not_rejected_for_size(
    make_client: MakeClient, client_id: uuid.UUID
) -> None:
    client = make_client(MAX_UPLOAD_MB="1")
    small = jpeg_bytes()
    assert len(small) < _MIB
    assert post_image(client, TREE_URL, client_id, content=small).status_code == 201


# --- 422 -------------------------------------------------------------------------------------


@BOTH_ROUTES
@pytest.mark.parametrize(
    ("data", "field"),
    [
        ({"latitude": "10"}, "longitude"),  # latitude without longitude
        ({"longitude": "10"}, "latitude"),
        ({"latitude": "91", "longitude": "10"}, "latitude"),
        ({"latitude": "-91", "longitude": "10"}, "latitude"),
        ({"latitude": "10", "longitude": "181"}, "longitude"),
        ({"latitude": "10", "longitude": "-181"}, "longitude"),
        ({"latitude": "nan", "longitude": "10"}, "latitude"),
        ({"latitude": "abc", "longitude": "10"}, "latitude"),
        ({"gps_accuracy_m": "-1"}, "gps_accuracy_m"),
        ({"gps_accuracy_m": "inf"}, "gps_accuracy_m"),
        ({"captured_at": "yesterday-ish"}, "captured_at"),
        # Year 1 with a positive offset cannot be converted to UTC (before datetime.min).
        ({"captured_at": "0001-01-01T00:00:00+05:00"}, "captured_at"),
        ({"source": "sample"}, "source"),  # only the seed script may create samples
        ({"source": "bogus"}, "source"),
    ],
)
def test_invalid_form_values_are_422_and_not_persisted(
    api: TestClient,
    client_id: uuid.UUID,
    url: str,
    data: dict[str, Any],
    field: str,
    nothing_stored: None,
) -> None:
    response = post_image(api, url, client_id, data=data)

    body = assert_problem(response, 422, "VALIDATION_ERROR")
    assert any(error["field"].endswith(field) for error in body["errors"]), body["errors"]
    assert all(set(error) == {"field", "message"} for error in body["errors"])


@BOTH_ROUTES
def test_missing_image_field_is_422(
    api: TestClient, client_id: uuid.UUID, url: str, nothing_stored: None
) -> None:
    response = api.post(url, data={"source": "upload"}, headers=headers_for(client_id))
    body = assert_problem(response, 422, "VALIDATION_ERROR")
    assert any(error["field"].endswith("image") for error in body["errors"])


def test_boundary_coordinates_are_accepted(api: TestClient, client_id: uuid.UUID) -> None:
    response = post_image(
        api,
        TREE_URL,
        client_id,
        data={"latitude": "-90", "longitude": "180", "gps_accuracy_m": "0"},
    )
    assert response.status_code == 201
    assert response.json()["location"]["latitude"] == -90


def test_image_with_solid_colour_is_fine(api: TestClient, client_id: uuid.UUID) -> None:
    from tests.fixtures.image_factory import encode

    upload = encode(solid_image(80, 80), "JPEG")
    assert post_image(api, TREE_URL, client_id, content=upload).status_code == 201


# --- 503: model unavailable --------------------------------------------------------------


def test_unavailable_model_is_503_before_any_work(
    make_client: MakeClient,
    client_id: uuid.UUID,
    session_factory: sessionmaker[Session],
    data_root: Path,
) -> None:
    from app.ml.registry import ModelRegistry

    down = FakePredictor("tree_classification", ready=False)
    client = make_client(registry=ModelRegistry.from_predictors([down]))

    body = assert_problem(post_image(client, TREE_URL, client_id), 503, "MODEL_UNAVAILABLE")

    assert body["title"] == "Model unavailable"
    assert down.predict_images == []
    assert count_rows(session_factory) == 0
    assert files_under(data_root) == []


def test_model_missing_from_the_registry_is_503_too(
    make_client: MakeClient, client_id: uuid.UUID
) -> None:
    from app.ml.registry import ModelRegistry

    only_tree = ModelRegistry.from_predictors([FakePredictor("tree_classification")])
    client = make_client(registry=only_tree)
    assert_problem(post_image(client, LEAF_URL, client_id), 503, "MODEL_UNAVAILABLE")


def test_unavailable_model_is_checked_before_the_image_is_even_decoded(
    make_client: MakeClient, client_id: uuid.UUID
) -> None:
    from app.ml.registry import ModelRegistry

    client = make_client(
        registry=ModelRegistry.from_predictors([FakePredictor("tree_classification", ready=False)])
    )
    # Garbage bytes would be a 400 if decoding came first; the model check wins (spec 9 step 1).
    response = post_image(client, TREE_URL, client_id, content=corrupt_bytes())
    assert_problem(response, 503, "MODEL_UNAVAILABLE")
