"""POST /tree/predict and /leaf-segmentation/predict: the happy paths."""

from __future__ import annotations

import io
import uuid
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy.orm import Session, sessionmaker

from app.core import signing
from app.core.config import Settings
from app.db.models.analysis import Analysis
from tests.fakes import FakePredictor
from tests.fixtures.image_factory import exif_bytes, jpeg_bytes, png_bytes, webp_bytes
from tests.integration.api.conftest import MakeClient
from tests.integration.api.helpers import (
    LEAF_URL,
    PREFIX,
    TREE_URL,
    fetch_row,
    files_under,
    post_image,
)

TREE = "tree_classification"
LEAF = "leaf_segmentation"


def _row(factory: sessionmaker[Session], analysis_id: str) -> Analysis:
    row = fetch_row(factory, analysis_id)
    assert row is not None
    return row


# --- tree ---------------------------------------------------------------------


def test_tree_predict_returns_201_with_location_and_full_detail(
    api: TestClient, client_id: uuid.UUID
) -> None:
    response = post_image(api, TREE_URL, client_id)

    assert response.status_code == 201, response.text
    body = response.json()
    assert response.headers["location"] == f"{PREFIX}/analyses/{body['id']}"
    assert uuid.UUID(body["id"]).version == 4
    assert body["model_key"] == TREE
    assert body["model_version"] == "tree_fake_v1"
    assert body["status"] == "completed"
    assert body["source"] == "upload"
    assert body["is_sample"] is False
    assert body["title"] is None and body["description"] is None
    assert body["prediction"] == {
        "label": "banana_tree",
        "display_label": "Banana tree",
        "confidence": 0.947,
        "is_uncertain": False,
    }
    assert body["details"]["kind"] == TREE
    assert body["details"]["verdict"] == "banana_tree"
    assert [p["class_key"] for p in body["details"]["probabilities"]] == [
        "banana_tree",
        "non_banana",
    ]
    assert body["location"] is None  # the client sent no GPS data
    assert body["image"]["width"] == 200 and body["image"]["height"] == 150
    assert body["image"]["result_url"] is None  # classification has no overlay
    timings = body["timings_ms"]
    assert set(timings) == {"preprocess", "inference", "postprocess", "total"}
    assert timings["total"] >= timings["preprocess"] + timings["inference"]
    assert body["created_at"].endswith("Z")  # aware UTC on the wire


def test_tree_predict_writes_the_row_and_files_under_the_same_id(
    api: TestClient,
    client_id: uuid.UUID,
    session_factory: sessionmaker[Session],
    data_root: Path,
) -> None:
    body = post_image(api, TREE_URL, client_id, filename="IMG_0042.JPG").json()

    row = _row(session_factory, body["id"])
    assert row.client_id == client_id
    assert row.model_key == TREE and row.status == "completed" and row.source == "upload"
    assert row.predicted_label == "banana_tree" and row.confidence == 0.947
    assert row.original_filename == "IMG_0042.JPG"
    assert row.details is not None and row.details["kind"] == TREE
    assert row.deleted_at is None
    assert row.image_width == 200 and row.image_height == 150
    assert len(row.image_sha256) == 64

    # original.jpg + thumb.webp, in a folder named after the analysis id (D-02).
    on_disk = files_under(data_root)
    assert sorted(path.name for path in on_disk) == ["original.jpg", "thumb.webp"]
    folder = data_root.joinpath(*row.original_image_path.split("/")).parent
    assert folder.name == body["id"]
    assert {path.parent for path in on_disk} == {folder}
    assert row.original_image_path.startswith(f"images/{TREE}/")
    assert row.thumbnail_path == f"{row.original_image_path.rsplit('/', 1)[0]}/thumb.webp"
    assert row.result_image_path is None


def test_stored_original_is_a_jpeg_without_exif_and_is_upright(
    api: TestClient,
    client_id: uuid.UUID,
    session_factory: sessionmaker[Session],
    data_root: Path,
) -> None:
    # 200x150 pixels tagged "rotate 90 degrees" + camera make/model + GPS EXIF.
    upload = jpeg_bytes(200, 150, exif=exif_bytes(orientation=6))

    body = post_image(api, TREE_URL, client_id, content=upload).json()

    # Rotated into upright orientation: 150 wide x 200 high.
    assert (body["image"]["width"], body["image"]["height"]) == (150, 200)
    row = _row(session_factory, body["id"])
    stored = data_root.joinpath(*row.original_image_path.split("/")).read_bytes()
    with Image.open(io.BytesIO(stored)) as image:
        assert image.format == "JPEG"
        assert image.size == (150, 200)
        assert len(image.getexif()) == 0  # EXIF (incl. GPS) is gone (P-06)
    assert b"KeraTestMake" not in stored and b"KeraTestModel" not in stored


@pytest.mark.parametrize("make_upload", [png_bytes, webp_bytes])
def test_png_and_webp_uploads_are_accepted_and_stored_as_jpeg(
    api: TestClient, client_id: uuid.UUID, data_root: Path, make_upload: object
) -> None:
    assert callable(make_upload)
    response = post_image(api, TREE_URL, client_id, content=make_upload())
    assert response.status_code == 201
    stored = next(p for p in files_under(data_root) if p.name == "original.jpg")
    with Image.open(stored) as image:
        assert image.format == "JPEG"


def test_location_metadata_is_stored_and_returned_in_utc(
    api: TestClient, client_id: uuid.UUID, session_factory: sessionmaker[Session]
) -> None:
    response = post_image(
        api,
        TREE_URL,
        client_id,
        data={
            "latitude": "27.71724",
            "longitude": "85.32402",
            "gps_accuracy_m": "4.2",
            # Nepal time (UTC+05:45) -> must come back as UTC.
            "captured_at": "2026-10-01T06:57:15+05:45",
            "source": "camera",
        },
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["source"] == "camera"
    assert body["location"] == {
        "latitude": 27.71724,
        "longitude": 85.32402,
        "accuracy_m": 4.2,
        "captured_at": "2026-10-01T01:12:15Z",
    }
    row = _row(session_factory, body["id"])
    assert row.latitude == pytest.approx(27.71724)
    assert row.captured_at == datetime(2026, 10, 1, 1, 12, 15)  # naive UTC in the DB (5.2)
    assert row.captured_at is not None and row.captured_at.tzinfo is None


def test_naive_captured_at_is_read_as_utc(api: TestClient, client_id: uuid.UUID) -> None:
    body = post_image(api, TREE_URL, client_id, data={"captured_at": "2026-10-01T01:12:15"}).json()
    assert body["location"]["captured_at"] == "2026-10-01T01:12:15Z"
    assert body["location"]["latitude"] is None  # a capture time alone is still a "location" block


def test_empty_optional_form_fields_are_treated_as_absent(
    api: TestClient, client_id: uuid.UUID
) -> None:
    # Browsers submit empty inputs as "" rather than omitting them.
    response = post_image(
        api,
        TREE_URL,
        client_id,
        data={"latitude": "", "longitude": "", "gps_accuracy_m": "", "captured_at": ""},
    )
    assert response.status_code == 201
    assert response.json()["location"] is None


def test_the_service_is_called_with_the_normalised_rgb_image(
    api: TestClient, client_id: uuid.UUID, fake_tree: FakePredictor
) -> None:
    post_image(api, TREE_URL, client_id)
    assert len(fake_tree.predict_images) == 1  # an injected registry is not warmed up
    assert fake_tree.predict_images[0].mode == "RGB"
    assert fake_tree.predict_images[0].size == (200, 150)


# --- signed image URLs ----------------------------------------------------------


def test_returned_image_urls_are_signed_and_resolve_without_a_client_id(
    api: TestClient, client_id: uuid.UUID, settings: Settings
) -> None:
    body = post_image(api, TREE_URL, client_id).json()
    original_url = body["image"]["original_url"]
    thumbnail_url = body["image"]["thumbnail_url"]

    parsed = urlparse(original_url)
    assert parsed.path == f"{PREFIX}/analyses/{body['id']}/image"
    query = parse_qs(parsed.query)
    assert query["variant"] == ["original"]
    exp = int(query["exp"][0])
    signing.verify(body["id"], "original", exp, query["sig"][0], settings)  # does not raise

    original = api.get(original_url)  # NOTE: no X-Client-Id header
    assert original.status_code == 200
    assert original.headers["content-type"] == "image/jpeg"
    assert original.headers["cache-control"] == "private, max-age=3600"
    assert Image.open(io.BytesIO(original.content)).format == "JPEG"

    thumbnail = api.get(thumbnail_url)
    assert thumbnail.status_code == 200
    assert thumbnail.headers["content-type"] == "image/webp"
    assert Image.open(io.BytesIO(thumbnail.content)).format == "WEBP"


# --- leaf segmentation ---------------------------------------------------------


def test_leaf_segmentation_predict_stores_result_png_and_returns_its_url(
    api: TestClient,
    client_id: uuid.UUID,
    session_factory: sessionmaker[Session],
    data_root: Path,
) -> None:
    response = post_image(api, LEAF_URL, client_id)

    assert response.status_code == 201, response.text
    body = response.json()
    assert response.headers["location"] == f"{PREFIX}/analyses/{body['id']}"
    assert body["model_key"] == LEAF and body["model_version"] == "leaf_seg_fake_v1"
    assert body["prediction"]["label"] == "affected"
    assert body["prediction"]["confidence"] is None  # segmentation has no single honest number
    assert body["details"]["kind"] == LEAF  # discriminated-union branch picked by `kind`
    assert body["details"]["lesion_count"] == 3
    assert body["details"]["thresholds"]["affected"] == 0.85

    row = _row(session_factory, body["id"])
    assert row.result_image_path is not None
    assert sorted(p.name for p in files_under(data_root)) == [
        "original.jpg",
        "result.png",
        "thumb.webp",
    ]
    assert len({p.parent for p in files_under(data_root)}) == 1  # one folder = one analysis id

    result = api.get(body["image"]["result_url"])
    assert result.status_code == 200
    assert result.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(result.content)).size == (40, 30)


def test_both_models_share_one_pipeline_and_one_history(
    api: TestClient, client_id: uuid.UUID
) -> None:
    tree = post_image(api, TREE_URL, client_id).json()
    leaf = post_image(api, LEAF_URL, client_id).json()

    listing = api.get(f"{PREFIX}/analyses", headers={"X-Client-Id": str(client_id)}).json()
    assert {item["id"] for item in listing["items"]} == {tree["id"], leaf["id"]}


# --- logging / privacy ----------------------------------------------------------------


def test_no_log_line_of_a_request_contains_the_coordinates(
    make_client: MakeClient, client_id: uuid.UUID
) -> None:
    import logging

    class Collect(logging.Handler):
        def __init__(self) -> None:
            super().__init__(level=logging.DEBUG)
            self.lines: list[str] = []

        def emit(self, record: logging.LogRecord) -> None:
            fields = {k: v for k, v in record.__dict__.items() if k not in _STD_ATTRS}
            self.lines.append(f"{record.getMessage()} {fields}")

    _STD_ATTRS = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime", "request_id"}
    collector = Collect()
    app_logger = logging.getLogger("app")
    app_logger.addHandler(collector)
    try:
        client = make_client(LOG_LEVEL="INFO")  # INFO so the pipeline's own line is emitted
        post_image(
            client,
            TREE_URL,
            client_id,
            data={"latitude": "27.717241", "longitude": "85.324021", "gps_accuracy_m": "123.456"},
        )
    finally:
        app_logger.removeHandler(collector)

    text = "\n".join(collector.lines)
    assert "analysis_completed" in text and "model_key=tree_classification" in text
    for secret in ("27.717241", "85.324021", "123.456", str(client_id)):
        assert secret not in text, secret
    assert str(client_id)[:8] in text  # the short form is logged


def test_coordinates_are_rounded_to_the_six_decimals_the_column_stores(
    api: TestClient, client_id: uuid.UUID
) -> None:
    created = post_image(
        api, TREE_URL, client_id, data={"latitude": "27.7172412345", "longitude": "85.3240198765"}
    ).json()

    fetched = api.get(f"{PREFIX}/analyses/{created['id']}", headers={"X-Client-Id": str(client_id)})

    # SQL Server rounds to DECIMAL(9,6) on its own; rounding first means the POST response
    # and a later GET can never disagree (SQLite would not have caught this).
    assert created["location"]["latitude"] == 27.717241
    assert created["location"]["longitude"] == 85.32402
    assert fetched.json()["location"] == created["location"]
