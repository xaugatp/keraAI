from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

import pytest
from sqlalchemy import text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.models.analysis import Analysis
from tests.unit.db.factories import make_analysis


def _persist(session: Session, **overrides: Any) -> Analysis:
    row = make_analysis(**overrides)
    session.add(row)
    session.commit()
    return row


# --- defaults ----------------------------------------------------------------


def test_python_side_defaults_are_applied(db_session: Session) -> None:
    row = _persist(db_session)

    assert isinstance(row.id, uuid.UUID)
    assert row.is_sample is False
    assert row.is_uncertain is False
    assert row.deleted_at is None
    # UTC-naive, whole milliseconds (DATETIME2(3) precision) — see app/db/types.py.
    for stamp in (row.created_at, row.updated_at):
        assert stamp.tzinfo is None
        assert stamp.microsecond % 1000 == 0
    assert row.updated_at >= row.created_at


def test_id_defaults_to_a_fresh_uuid4_each_time(db_session: Session) -> None:
    a = Analysis(**_required())
    b = Analysis(**_required())
    db_session.add_all([a, b])
    db_session.commit()
    assert a.id != b.id
    assert a.id.version == 4


def _required() -> dict[str, Any]:
    return {
        "model_key": "tree_classification",
        "model_version": "tree_cls_v1",
        "status": "completed",
        "source": "upload",
        "original_image_path": "images/x/original.jpg",
        "thumbnail_path": "images/x/thumb.webp",
        "image_sha256": "a" * 64,
    }


def test_server_defaults_cover_raw_inserts(db_session: Session) -> None:
    # An INSERT that bypasses the ORM (e.g. typed in SSMS) still gets 0 for the BIT flags.
    db_session.execute(
        text(
            "INSERT INTO analyses (id, model_key, model_version, status, source, "
            "original_image_path, thumbnail_path, image_sha256, created_at, updated_at) "
            "VALUES ('00000000000000000000000000000001', 'tree_classification', 'v', "
            "'completed', 'upload', 'o', 't', :h, '2026-01-01 00:00:00', '2026-01-01 00:00:00')"
        ),
        {"h": "b" * 64},
    )
    flags = db_session.execute(text("SELECT is_sample, is_uncertain FROM analyses")).one()
    assert tuple(flags) == (0, 0)


def test_updated_at_moves_on_update_but_created_at_does_not(db_session: Session) -> None:
    row = _persist(db_session)
    created = row.created_at
    old = datetime(2020, 1, 1)
    db_session.execute(update(Analysis).where(Analysis.id == row.id).values(updated_at=old))
    db_session.commit()
    db_session.refresh(row)
    assert row.updated_at == old

    row.title = "Edited"
    db_session.commit()
    db_session.refresh(row)

    assert row.updated_at > old
    assert row.created_at == created


# --- round trip --------------------------------------------------------------


def test_full_row_round_trips(db_session: Session) -> None:
    client_id = uuid.uuid4()
    captured = datetime(2026, 10, 1, 1, 12, 15, 123000)
    details = {
        "kind": "tree_classification",
        "verdict": "banana_tree",
        "probabilities": [{"class_key": "banana_tree", "probability": 0.947}],
    }
    row = _persist(
        db_session,
        source="camera",
        client_id=client_id,
        title="केरा",  # non-ASCII must survive (NVARCHAR)
        description="d" * 1000,
        is_uncertain=True,
        latitude=27.71724,
        longitude=85.32402,
        gps_accuracy_m=4.2,
        captured_at=captured,
        result_image_path="images/x/result.png",
        original_filename="IMG_1.jpg",
        image_width=1920,
        image_height=1440,
        preprocess_ms=14,
        inference_ms=41,
        postprocess_ms=1,
        total_ms=102,
        details=details,
    )
    db_session.expunge_all()

    loaded = db_session.get(Analysis, row.id)
    assert loaded is not None
    assert loaded.client_id == client_id
    assert loaded.title == "केरा"
    assert loaded.description == "d" * 1000
    assert loaded.is_uncertain is True
    assert loaded.latitude == pytest.approx(27.71724)
    assert loaded.longitude == pytest.approx(85.32402)
    assert loaded.gps_accuracy_m == pytest.approx(4.2)
    assert loaded.captured_at == captured
    assert loaded.details == details
    assert (loaded.preprocess_ms, loaded.inference_ms, loaded.postprocess_ms) == (14, 41, 1)
    assert loaded.total_ms == 102
    assert (loaded.image_width, loaded.image_height) == (1920, 1440)
    assert len(loaded.image_sha256) == 64


def test_none_details_is_sql_null_not_json_null(db_session: Session) -> None:
    row = _persist(db_session, details=None)
    is_null = db_session.execute(
        text("SELECT details IS NULL FROM analyses WHERE id = :i"),
        {"i": row.id.hex},
    ).scalar_one()
    assert is_null == 1


# --- CHECK constraints -------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("model_key", "banana"),
        ("model_key", ""),
        ("status", "pending"),
        ("status", "COMPLETED"),
        ("source", "email"),
        ("confidence", -0.01),
        ("confidence", 1.01),
        ("latitude", -90.000001),
        ("latitude", 90.000001),
        ("longitude", -180.000001),
        ("longitude", 180.000001),
        ("gps_accuracy_m", -0.1),
    ],
)
def test_check_constraints_reject_out_of_range_values(
    db_session: Session, field: str, bad_value: object
) -> None:
    db_session.add(make_analysis(**{field: bad_value}))
    with pytest.raises(IntegrityError, match="CHECK constraint failed"):
        db_session.flush()


@pytest.mark.parametrize(
    ("field", "ok_value"),
    [
        ("model_key", "leaf_segmentation"),
        ("model_key", "leaf_disease"),
        ("status", "failed"),
        ("source", "camera"),
        ("source", "sample"),
        ("confidence", 0.0),
        ("confidence", 1.0),
        ("latitude", -90.0),
        ("latitude", 90.0),
        ("longitude", -180.0),
        ("longitude", 180.0),
        ("gps_accuracy_m", 0.0),
    ],
)
def test_check_constraints_accept_boundary_and_valid_values(
    db_session: Session, field: str, ok_value: object
) -> None:
    row = _persist(db_session, **{field: ok_value})
    assert db_session.get(Analysis, row.id) is not None


def test_nullable_checked_columns_accept_null(db_session: Session) -> None:
    # A failed inference has no confidence/location; NULL must pass every CHECK.
    row = _persist(
        db_session,
        status="failed",
        predicted_label=None,
        display_label=None,
        confidence=None,
        latitude=None,
        longitude=None,
        gps_accuracy_m=None,
        error_code="INFERENCE_FAILED",
        error_message="boom",
    )
    assert row.confidence is None


@pytest.mark.parametrize(
    "missing",
    [
        "model_key",
        "model_version",
        "status",
        "source",
        "original_image_path",
        "thumbnail_path",
        "image_sha256",
    ],
)
def test_not_null_columns_are_enforced(db_session: Session, missing: str) -> None:
    db_session.add(make_analysis(**{missing: None}))
    with pytest.raises(IntegrityError, match="NOT NULL"):
        db_session.flush()


def test_duplicate_primary_key_is_rejected(db_session: Session) -> None:
    row = _persist(db_session)
    db_session.expunge_all()  # otherwise the ORM itself refuses before the DB does
    db_session.add(make_analysis(id=row.id))
    with pytest.raises(IntegrityError):
        db_session.flush()
