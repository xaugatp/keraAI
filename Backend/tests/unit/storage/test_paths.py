from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.storage.base import ALLOWED_MODEL_KEYS, ImagePaths, InvalidStoragePathError, build_paths
from app.storage.local import LocalImageStorage

ANALYSIS_ID = uuid.UUID("6f1c2a9e-1b2c-4d3e-8f90-a1b2c3d4e5f6")
CREATED_AT = datetime(2026, 10, 1, 1, 12, 16, tzinfo=UTC)


def test_layout_matches_the_spec() -> None:
    paths = build_paths("tree_classification", ANALYSIS_ID, CREATED_AT)
    folder = "images/tree_classification/2026/10/6f1c2a9e-1b2c-4d3e-8f90-a1b2c3d4e5f6"
    assert paths == ImagePaths(
        original=f"{folder}/original.jpg",
        thumbnail=f"{folder}/thumb.webp",
        result=f"{folder}/result.png",
    )


def test_paths_use_forward_slashes_only() -> None:
    paths = build_paths("leaf_disease", ANALYSIS_ID, CREATED_AT)
    for path in (paths.original, paths.thumbnail, paths.result):
        assert "\\" not in path
        assert not path.startswith("/")


@pytest.mark.parametrize("model_key", ALLOWED_MODEL_KEYS)
def test_every_known_model_key_is_accepted(model_key: str) -> None:
    assert build_paths(model_key, ANALYSIS_ID, CREATED_AT).original.startswith(
        f"images/{model_key}/"
    )


def test_the_allowed_model_keys_are_the_three_from_the_spec() -> None:
    assert set(ALLOWED_MODEL_KEYS) == {"tree_classification", "leaf_segmentation", "leaf_disease"}


@pytest.mark.parametrize(
    "model_key",
    [
        "",
        "tree",
        "TREE_CLASSIFICATION",
        "../etc",
        "tree_classification/../x",
        "tree_classification ",
    ],
)
def test_unknown_model_keys_are_rejected(model_key: str) -> None:
    with pytest.raises(InvalidStoragePathError):
        build_paths(model_key, ANALYSIS_ID, CREATED_AT)


def test_month_is_zero_padded_and_year_is_four_digits() -> None:
    paths = build_paths("tree_classification", ANALYSIS_ID, datetime(2026, 3, 9, tzinfo=UTC))
    assert "/2026/03/" in paths.original


def test_naive_datetimes_are_treated_as_utc() -> None:
    paths = build_paths("tree_classification", ANALYSIS_ID, datetime(2026, 12, 31, 23, 59))
    assert "/2026/12/" in paths.original


def test_aware_datetimes_are_converted_to_utc_before_choosing_the_month() -> None:
    # 23:30 on Dec 31 at UTC-5 is already January in UTC.
    created = datetime(2026, 12, 31, 23, 30, tzinfo=timezone(timedelta(hours=-5)))
    assert "/2027/01/" in build_paths("tree_classification", ANALYSIS_ID, created).original


@pytest.mark.parametrize(
    "text",
    [
        "6F1C2A9E-1B2C-4D3E-8F90-A1B2C3D4E5F6",
        "{6f1c2a9e-1b2c-4d3e-8f90-a1b2c3d4e5f6}",
        "6f1c2a9e1b2c4d3e8f90a1b2c3d4e5f6",
        "urn:uuid:6f1c2a9e-1b2c-4d3e-8f90-a1b2c3d4e5f6",
    ],
)
def test_string_uuids_are_accepted_and_rendered_canonically(text: str) -> None:
    paths = build_paths("tree_classification", text, CREATED_AT)
    assert paths == build_paths("tree_classification", ANALYSIS_ID, CREATED_AT)


@pytest.mark.parametrize(
    "bad_id",
    [
        "",
        "not-a-uuid",
        "../../etc/passwd",
        "6f1c2a9e/../original",
        "123",
        "6f1c2a9e-1b2c",
        None,
        42,
    ],
)
def test_analysis_id_must_be_a_uuid(bad_id: object) -> None:
    with pytest.raises(InvalidStoragePathError):
        build_paths("tree_classification", bad_id, CREATED_AT)  # type: ignore[arg-type]


def test_built_paths_are_accepted_by_the_storage_guard(tmp_path: Path) -> None:
    storage = LocalImageStorage(str(tmp_path))
    paths = build_paths("leaf_segmentation", ANALYSIS_ID, CREATED_AT)
    for path in (paths.original, paths.thumbnail, paths.result):
        storage.save(path, b"data")
        assert storage.exists(path)
