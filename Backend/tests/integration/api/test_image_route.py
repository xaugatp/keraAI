"""GET /analyses/{id}/image: signed links, public samples, caching, missing files."""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from app.core import signing
from app.core.config import Settings
from app.core.time import utcnow
from tests.integration.api.conftest import LogCapture
from tests.integration.api.helpers import (
    ANALYSES_URL,
    LEAF_URL,
    TREE_URL,
    assert_problem,
    headers_for,
    post_image,
    seed_analysis,
)

NOW = datetime(2026, 10, 1, 8, 0, 0)


def _image_url(analysis_id: object, **query: object) -> str:
    return f"{ANALYSES_URL}/{analysis_id}/image?{urlencode(query)}"


def _signed(
    settings: Settings, analysis_id: object, variant: str, *, exp: int | None = None
) -> str:
    exp = exp if exp is not None else int(utcnow().timestamp()) + 600
    sig = signing.sign(str(analysis_id), variant, exp, settings)
    return _image_url(analysis_id, variant=variant, exp=exp, sig=sig)


@pytest.fixture
def created(api: TestClient, client_id: uuid.UUID) -> dict[str, object]:
    body = post_image(api, TREE_URL, client_id).json()
    assert isinstance(body, dict)
    return body


# --- signature handling ----------------------------------------------------------------------


def test_signed_url_works_for_every_variant_and_needs_no_client_id(
    api: TestClient, client_id: uuid.UUID
) -> None:
    body = post_image(api, LEAF_URL, client_id).json()  # has all three variants
    expected = {"original": "image/jpeg", "thumbnail": "image/webp", "result": "image/png"}
    for variant, media_type in expected.items():
        response = api.get(body["image"][f"{variant}_url"])  # no X-Client-Id
        assert response.status_code == 200, variant
        assert response.headers["content-type"] == media_type
        assert response.headers["cache-control"] == "private, max-age=3600"
        assert response.content


def test_missing_signature_is_403(api: TestClient, created: dict[str, object]) -> None:
    for url in (
        _image_url(created["id"], variant="original"),
        _image_url(created["id"]),  # variant defaults to original, still unsigned
    ):
        assert_problem(api.get(url), 403, "INVALID_SIGNATURE")


def test_tampered_signature_is_403(api: TestClient, created: dict[str, object]) -> None:
    url = created["image"]["original_url"]  # type: ignore[index]
    sig = parse_qs(urlparse(url).query)["sig"][0]
    flipped = ("A" if sig[0] != "A" else "B") + sig[1:]
    assert_problem(api.get(url.replace(sig, flipped)), 403, "INVALID_SIGNATURE")
    assert_problem(api.get(url.replace(sig, "")), 403, "INVALID_SIGNATURE")
    assert_problem(api.get(url + "extra"), 403, "INVALID_SIGNATURE")


def test_tampered_expiry_is_403(api: TestClient, created: dict[str, object]) -> None:
    url = created["image"]["original_url"]  # type: ignore[index]
    exp = parse_qs(urlparse(url).query)["exp"][0]
    # Pushing the expiry forward invalidates the signature that covered the old value.
    assert_problem(
        api.get(url.replace(f"exp={exp}", f"exp={int(exp) + 3600}")), 403, "INVALID_SIGNATURE"
    )


@pytest.mark.parametrize("bad_exp", ["abc", "1e9", "", "12.5"])
def test_non_numeric_expiry_is_the_same_403_not_a_422(
    api: TestClient, created: dict[str, object], bad_exp: str
) -> None:
    url = _image_url(created["id"], variant="original", exp=bad_exp, sig="whatever")
    assert_problem(api.get(url), 403, "INVALID_SIGNATURE")


def test_expired_signature_is_403(
    api: TestClient, created: dict[str, object], settings: Settings
) -> None:
    past = int(utcnow().timestamp()) - 1
    assert_problem(
        api.get(_signed(settings, created["id"], "original", exp=past)), 403, "INVALID_SIGNATURE"
    )


def test_signature_is_bound_to_the_variant(
    api: TestClient, created: dict[str, object], settings: Settings
) -> None:
    exp = int(utcnow().timestamp()) + 600
    sig = signing.sign(str(created["id"]), "thumbnail", exp, settings)
    url = _image_url(created["id"], variant="original", exp=exp, sig=sig)
    assert_problem(api.get(url), 403, "INVALID_SIGNATURE")


def test_signature_is_bound_to_the_analysis(
    api: TestClient, created: dict[str, object], client_id: uuid.UUID, settings: Settings
) -> None:
    second = post_image(api, TREE_URL, client_id).json()
    exp = int(utcnow().timestamp()) + 600
    sig = signing.sign(str(second["id"]), "original", exp, settings)
    assert_problem(
        api.get(_image_url(created["id"], variant="original", exp=exp, sig=sig)),
        403,
        "INVALID_SIGNATURE",
    )


def test_unknown_variant_is_422(api: TestClient, created: dict[str, object]) -> None:
    assert_problem(api.get(_image_url(created["id"], variant="huge")), 422, "VALIDATION_ERROR")


def test_non_uuid_id_is_422(api: TestClient) -> None:
    assert_problem(api.get(f"{ANALYSES_URL}/nope/image?variant=original"), 422, "VALIDATION_ERROR")


def test_a_valid_link_works_for_whoever_holds_it(
    api: TestClient, created: dict[str, object], other_client_id: uuid.UUID
) -> None:
    # The signed link IS the credential (an <img> tag cannot send X-Client-Id), exactly
    # like an S3 pre-signed URL. Another client's id changes nothing either way.
    url = created["image"]["original_url"]  # type: ignore[index]
    assert api.get(url, headers=headers_for(other_client_id)).status_code == 200


# --- not found --------------------------------------------------------------------------------


def test_properly_signed_link_for_an_unknown_id_is_404(api: TestClient, settings: Settings) -> None:
    missing = uuid.uuid4()
    assert_problem(api.get(_signed(settings, missing, "original")), 404, "ANALYSIS_NOT_FOUND")


def test_unsigned_request_for_an_unknown_id_is_403_so_existence_is_not_revealed(
    api: TestClient, created: dict[str, object]
) -> None:
    unknown = api.get(_image_url(uuid.uuid4(), variant="original"))
    private = api.get(_image_url(created["id"], variant="original"))
    assert (unknown.status_code, unknown.json()["code"]) == (403, "INVALID_SIGNATURE")
    # Same answer for an id that does exist: a prober cannot tell the two apart.
    assert (private.status_code, private.json()["code"]) == (403, "INVALID_SIGNATURE")


def test_result_variant_of_an_analysis_without_overlay_is_404(
    api: TestClient, created: dict[str, object], settings: Settings
) -> None:
    assert_problem(api.get(_signed(settings, created["id"], "result")), 404, "ANALYSIS_NOT_FOUND")


def test_file_missing_on_disk_is_404_and_logged_as_an_error(
    api: TestClient,
    created: dict[str, object],
    data_root: Path,
    service_logs: LogCapture,
) -> None:
    for path in data_root.rglob("original.jpg"):
        path.unlink()

    response = api.get(created["image"]["original_url"])  # type: ignore[index]

    body = assert_problem(response, 404, "ANALYSIS_NOT_FOUND")
    assert "original.jpg" not in response.text and str(data_root) not in response.text
    assert body["detail"] == "Image not found."
    errors = [r for r in service_logs.records if r.getMessage() == "image_file_missing"]
    assert len(errors) == 1 and errors[0].levelname == "ERROR"
    # The thumbnail of the same analysis is unaffected.
    assert api.get(created["image"]["thumbnail_url"]).status_code == 200  # type: ignore[index]


# --- samples are public -------------------------------------------------------------------------


@pytest.fixture
def sample_id(session_factory: sessionmaker[Session], data_root: Path) -> uuid.UUID:
    return seed_analysis(
        session_factory,
        data_root,
        client_id=None,
        is_sample=True,
        created_at=NOW,
        with_result=True,
        title="Demo",
    )


@pytest.mark.parametrize("variant", ["original", "thumbnail", "result"])
def test_sample_images_need_no_signature_and_no_client_id(
    api: TestClient, sample_id: uuid.UUID, variant: str
) -> None:
    response = api.get(_image_url(sample_id, variant=variant))

    assert response.status_code == 200
    assert response.headers["cache-control"] == "public, max-age=86400"
    assert response.content


def test_sample_image_ignores_a_garbage_signature(api: TestClient, sample_id: uuid.UUID) -> None:
    assert api.get(_image_url(sample_id, variant="original", exp="x", sig="y")).status_code == 200


def test_deleted_sample_is_404_even_though_samples_are_public(
    api: TestClient,
    sample_id: uuid.UUID,
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        from app.db.models.analysis import Analysis

        row = session.get(Analysis, sample_id)
        assert row is not None
        row.deleted_at = NOW
        session.commit()
    # No longer public: it now needs a (valid) signature like any unknown row.
    assert api.get(_image_url(sample_id, variant="original")).status_code == 403


def test_failed_sample_is_not_public(
    api: TestClient, session_factory: sessionmaker[Session], data_root: Path
) -> None:
    failed = seed_analysis(
        session_factory,
        data_root,
        client_id=None,
        is_sample=True,
        status="failed",
        created_at=NOW,
    )
    assert api.get(_image_url(failed, variant="original")).status_code == 403
