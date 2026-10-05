"""Schema-level rules: constants stay in sync with the DB, UTC boundary, details union."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta, timezone
from typing import get_args

import pytest
from pydantic import TypeAdapter, ValidationError

from app.core import signing
from app.db.models.analysis import ANALYSIS_SOURCES, ANALYSIS_STATUSES, MODEL_KEYS, Analysis
from app.ml.base import ModelInfo
from app.repositories.analysis_repository import AnalysisScope
from app.schemas.analysis import (
    AnalysisDetail,
    AnalysisDetails,
    Prediction,
    TimingsMs,
)
from app.schemas.common import (
    AnalysisSource,
    AnalysisStatus,
    ClientSource,
    GeoLocation,
    ImageVariant,
    ModelInfoOut,
    ModelKey,
    Page,
    Scope,
)
from app.services.analysis_mapper import to_detail, to_summary
from tests.conftest import make_leaf_seg_details
from tests.fakes import make_tree_details
from tests.fixtures.settings import make_settings
from tests.unit.db.factories import make_analysis

# --- the Literals are pinned to their single source of truth ----------------------------------


def test_literals_match_the_database_constants() -> None:
    assert get_args(ModelKey) == MODEL_KEYS
    assert get_args(AnalysisStatus) == ANALYSIS_STATUSES
    assert get_args(AnalysisSource) == ANALYSIS_SOURCES


def test_storage_uses_the_same_model_key_tuple_as_the_database() -> None:
    from app.storage.base import ALLOWED_MODEL_KEYS

    assert ALLOWED_MODEL_KEYS is MODEL_KEYS


def test_clients_may_claim_every_source_except_sample() -> None:
    assert set(get_args(ClientSource)) == set(ANALYSIS_SOURCES) - {"sample"}


def test_image_variants_match_the_signing_module() -> None:
    assert get_args(ImageVariant) == signing.VARIANTS


def test_scope_is_the_repositorys_own_type() -> None:
    assert get_args(Scope) == get_args(AnalysisScope) == ("mine", "samples", "all_visible")


# --- UTC at the boundary -------------------------------------------------------------------------


def test_naive_datetimes_become_aware_utc_and_serialise_with_z() -> None:
    location = GeoLocation(captured_at=datetime(2026, 10, 1, 1, 12, 15))  # naive, as in the DB
    assert location.captured_at == datetime(2026, 10, 1, 1, 12, 15, tzinfo=UTC)
    assert location.model_dump_json() == (
        '{"latitude":null,"longitude":null,"accuracy_m":null,"captured_at":"2026-10-01T01:12:15Z"}'
    )


def test_offset_datetimes_are_converted_to_utc() -> None:
    nepal = timezone(timedelta(hours=5, minutes=45))
    location = GeoLocation(captured_at=datetime(2026, 10, 1, 6, 57, 15, tzinfo=nepal))
    assert location.captured_at == datetime(2026, 10, 1, 1, 12, 15, tzinfo=UTC)
    assert location.captured_at is not None and location.captured_at.utcoffset() == timedelta(0)


@pytest.mark.parametrize(
    "kwargs", [{"latitude": 91}, {"latitude": -91}, {"longitude": 181}, {"accuracy_m": -1}]
)
def test_geolocation_ranges(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        GeoLocation(**kwargs)


# --- details: discriminated union -------------------------------------------------------------------


def test_details_are_routed_by_kind() -> None:
    adapter: TypeAdapter[object] = TypeAdapter(AnalysisDetails)
    tree = adapter.validate_python(make_tree_details().model_dump(mode="json"))
    leaf = adapter.validate_python(make_leaf_seg_details().model_dump(mode="json"))
    assert type(tree).__name__ == "TreeDetails"
    assert type(leaf).__name__ == "LeafSegDetails"


def test_unknown_or_missing_kind_is_rejected() -> None:
    adapter: TypeAdapter[object] = TypeAdapter(AnalysisDetails)
    with pytest.raises(ValidationError):
        adapter.validate_python({"kind": "leaf_disease"})
    with pytest.raises(ValidationError):
        adapter.validate_python({"verdict": "banana_tree"})


# --- mapping ---------------------------------------------------------------------------------------------


SETTINGS = make_settings()


def _row(**kw: object) -> Analysis:
    values: dict[str, object] = {
        "details": make_tree_details().model_dump(mode="json"),
        "created_at": datetime(2026, 10, 1, 1, 12, 16),
        "image_width": 200,
        "image_height": 150,
        "source": "camera",
        # Column defaults only apply on INSERT; this row is never flushed.
        "is_sample": False,
        "is_uncertain": False,
        "total_ms": 12,
        "inference_ms": 7,
        "preprocess_ms": 3,
        "postprocess_ms": 1,
    }
    values.update(kw)
    return make_analysis(**values)


def test_completed_row_maps_to_the_spec_shape() -> None:
    row = _row(latitude=27.71724, longitude=85.32402, gps_accuracy_m=4.2)

    detail = to_detail(row, SETTINGS)

    assert detail.status == "completed" and detail.source == "camera"
    assert detail.prediction == Prediction(
        label="banana_tree", display_label="Banana tree", confidence=0.947, is_uncertain=False
    )
    assert detail.location == GeoLocation(latitude=27.71724, longitude=85.32402, accuracy_m=4.2)
    assert detail.timings_ms == TimingsMs(preprocess=3, inference=7, postprocess=1, total=12)
    assert detail.created_at == datetime(2026, 10, 1, 1, 12, 16, tzinfo=UTC)
    assert detail.image.width == 200 and detail.image.result_url is None
    assert detail.details is not None and detail.details.kind == "tree_classification"


def test_location_block_is_null_only_when_nothing_was_sent() -> None:
    assert to_detail(_row(), SETTINGS).location is None
    assert to_detail(_row(captured_at=datetime(2026, 10, 1)), SETTINGS).location is not None
    assert to_detail(_row(gps_accuracy_m=3.0), SETTINGS).location is not None


def test_failed_row_never_exposes_a_prediction_details_or_the_error() -> None:
    row = _row(
        status="failed",
        error_code="INFERENCE_FAILED",
        error_message="RuntimeError: C:\\secret\\weights.pt",
    )

    detail = to_detail(row, SETTINGS)

    assert detail.status == "failed"
    assert detail.prediction is None and detail.details is None
    assert "secret" not in detail.model_dump_json()
    assert "error" not in detail.model_dump()


def test_private_urls_are_signed_and_sample_urls_are_not() -> None:
    private = to_detail(_row(), SETTINGS)
    assert "&exp=" in private.image.original_url and "&sig=" in private.image.original_url

    sample = to_detail(_row(is_sample=True, source="sample", client_id=None), SETTINGS)
    assert sample.image.original_url.endswith("/image?variant=original")
    assert "sig=" not in sample.image.thumbnail_url


def test_result_url_only_when_there_is_a_result_image() -> None:
    row = _row(result_image_path="images/x/result.png")
    assert to_detail(row, SETTINGS).image.result_url is not None


def test_summary_has_exactly_the_spec_fields() -> None:
    summary = to_summary(_row(latitude=1.5, longitude=2.5, title="t"), SETTINGS)
    assert set(summary.model_dump()) == {
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
    assert (summary.latitude, summary.longitude, summary.title) == (1.5, 2.5, "t")


def test_a_leaf_segmentation_row_maps_its_details_branch() -> None:
    row = _row(
        model_key="leaf_segmentation",
        details=make_leaf_seg_details().model_dump(mode="json"),
        confidence=None,
    )
    detail = to_detail(row, SETTINGS)
    assert detail.details is not None and detail.details.kind == "leaf_segmentation"
    assert detail.prediction is not None and detail.prediction.confidence is None


def test_confidence_outside_0_1_is_rejected_by_the_schema() -> None:
    with pytest.raises(ValidationError):
        Prediction(label="x", display_label="X", confidence=1.5, is_uncertain=False)


def test_page_is_generic_and_validates_bounds() -> None:
    page = Page[int](items=[1, 2], page=1, page_size=2, total=5, total_pages=3)
    assert page.model_dump() == {
        "items": [1, 2],
        "page": 1,
        "page_size": 2,
        "total": 5,
        "total_pages": 3,
    }
    with pytest.raises(ValidationError):
        Page[int](items=[], page=0, page_size=1, total=0, total_pages=0)


def test_model_info_out_is_built_from_the_ml_dataclass() -> None:
    info = ModelInfo(
        key="tree_classification",
        display_name="Banana tree classification",
        version="v1",
        task="classify",
        classes=["a", "b"],
        status="ready",
        reason=None,
        is_placeholder=True,
    )
    out = ModelInfoOut.model_validate(info)
    assert out.is_placeholder is True and out.classes == ["a", "b"]


def test_detail_survives_a_json_round_trip() -> None:
    detail = to_detail(_row(), SETTINGS)
    again = AnalysisDetail.model_validate_json(detail.model_dump_json())
    assert isinstance(again.id, uuid.UUID)
    assert again == detail
