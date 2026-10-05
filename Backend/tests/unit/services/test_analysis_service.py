"""AnalysisService behaviour that is easiest to pin down without HTTP."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.errors import (
    AnalysisNotFoundError,
    InferenceError,
    InvalidSignatureError,
    ModelUnavailableError,
    SampleImmutableError,
)
from app.core.time import utcnow
from app.db.models.analysis import Analysis
from app.ml.registry import ModelRegistry
from app.schemas.analysis import CaptureMeta
from app.services.analysis_service import AnalysisFilter, AnalysisService
from tests.fakes import FakePredictor
from tests.fixtures.image_factory import corrupt_bytes, jpeg_bytes
from tests.unit.db.factories import make_analysis
from tests.unit.services.conftest import ListHandler

NO_META = CaptureMeta()
TREE = "tree_classification"


def analyze(service: AnalysisService, client_id: uuid.UUID | None = None, **kwargs: object):  # type: ignore[no-untyped-def]
    arguments: dict[str, object] = {
        "model_key": TREE,
        "upload_bytes": jpeg_bytes(),
        "original_filename": "photo.jpg",
        "meta": NO_META,
        "client_id": client_id if client_id is not None else uuid.uuid4(),
        "source": "upload",
    }
    arguments.update(kwargs)
    return service.analyze(**arguments)  # type: ignore[arg-type]


# --- pipeline order ---------------------------------------------------------------------------


def test_model_availability_is_checked_before_the_upload_is_looked_at(
    make_service: Callable[..., AnalysisService], storage: object
) -> None:
    down = FakePredictor(TREE, ready=False)
    service = make_service(ModelRegistry.from_predictors([down]))

    with pytest.raises(ModelUnavailableError):
        analyze(service, upload_bytes=corrupt_bytes())  # would be INVALID_IMAGE if decoded first
    assert down.predict_images == []


def test_invalid_image_is_rejected_without_calling_the_model_or_touching_the_db(
    service: AnalysisService, predictor: FakePredictor, db_session: Session
) -> None:
    from app.core.errors import InvalidImageError

    with pytest.raises(InvalidImageError):
        analyze(service, upload_bytes=corrupt_bytes())
    assert predictor.predict_images == []
    assert db_session.query(Analysis).count() == 0


def test_the_model_sees_the_normalised_image_not_the_raw_upload(
    service: AnalysisService, predictor: FakePredictor
) -> None:
    analyze(service, upload_bytes=jpeg_bytes(400, 300))
    (seen,) = predictor.predict_images
    assert seen.mode == "RGB" and seen.size == (400, 300)


# --- argument guards ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"is_sample": True, "source": "upload", "client_id": None},  # sample must say so
        {"is_sample": False, "source": "sample"},  # a client cannot claim "sample"
        {"is_sample": True, "source": "sample"},  # samples are anonymous (client_id given)
    ],
)
def test_inconsistent_sample_arguments_are_a_programming_error(
    service: AnalysisService, db_session: Session, kwargs: dict[str, object]
) -> None:
    kwargs = {"client_id": uuid.uuid4(), **kwargs}
    with pytest.raises(ValueError):
        service.analyze(TREE, jpeg_bytes(), "x.jpg", NO_META, **kwargs)  # type: ignore[arg-type]
    assert db_session.query(Analysis).count() == 0


def test_non_sample_without_a_client_id_is_a_programming_error(service: AnalysisService) -> None:
    with pytest.raises(ValueError):
        service.analyze(TREE, jpeg_bytes(), "x.jpg", NO_META, None, "upload")


def test_sample_creation_the_way_the_seed_script_will_do_it(
    service: AnalysisService, db_session: Session
) -> None:
    detail = service.analyze(
        TREE,
        jpeg_bytes(),
        "healthy.jpg",
        CaptureMeta(latitude=27.7, longitude=85.3),
        None,
        "sample",
        is_sample=True,
        title="Healthy tree",
        description="A banana tree",
    )

    assert detail.is_sample and detail.source == "sample" and detail.title == "Healthy tree"
    # Public: plain URL, no signature.
    assert detail.image.original_url == f"/api/v1/analyses/{detail.id}/image?variant=original"
    row = db_session.get(Analysis, detail.id)
    assert row is not None and row.client_id is None and row.is_sample


# --- model_version / details / timings come from the predictor -------------------------------------


def test_row_records_the_producing_model_version_and_details_json(
    service: AnalysisService, db_session: Session
) -> None:
    detail = analyze(service)
    row = db_session.get(Analysis, detail.id)
    assert row is not None
    assert row.model_version == "tree_fake_v1"
    assert row.details is not None and row.details["kind"] == "tree_classification"
    assert isinstance(row.details["probabilities"][0]["probability"], float)  # plain JSON types
    assert detail.model_version == "tree_fake_v1"


def test_total_ms_covers_the_whole_analysis(service: AnalysisService) -> None:
    detail = analyze(service, upload_bytes=jpeg_bytes(1200, 900))
    timings = detail.timings_ms
    assert None not in (timings.preprocess, timings.inference, timings.postprocess, timings.total)
    assert timings.total >= timings.preprocess + timings.inference + timings.postprocess  # type: ignore[operator]


def test_capture_time_is_stored_as_naive_utc_truncated_to_milliseconds(
    service: AnalysisService, db_session: Session
) -> None:
    nepal = timezone(timedelta(hours=5, minutes=45))
    captured = datetime(2026, 10, 1, 6, 57, 15, 123456, tzinfo=nepal)

    detail = analyze(service, meta=CaptureMeta(captured_at=captured))

    row = db_session.get(Analysis, detail.id)
    assert row is not None
    assert row.captured_at == datetime(2026, 10, 1, 1, 12, 15, 123000)
    assert detail.location is not None
    assert detail.location.captured_at == datetime(2026, 10, 1, 1, 12, 15, 123000, tzinfo=UTC)


# --- logging -------------------------------------------------------------------------------------------


def test_one_structured_log_line_without_coordinates_and_with_a_short_client_id(
    service: AnalysisService, logs: ListHandler
) -> None:
    client_id = uuid.UUID("6f1c2a9e-1b2c-4d3e-8f90-a1b2c3d4e5f6")

    detail = analyze(
        service,
        client_id=client_id,
        meta=CaptureMeta(latitude=27.71724, longitude=85.32402, gps_accuracy_m=123.456789),
        original_filename="IMG_1.jpg",
    )

    completed = [r for r in logs.records if r.getMessage().startswith("analysis_completed")]
    assert len(completed) == 1
    record = completed[0]
    assert record.levelname == "INFO"
    assert record.analysis_id == str(detail.id)  # type: ignore[attr-defined]
    assert record.model_key == TREE  # type: ignore[attr-defined]
    assert record.model_version == "tree_fake_v1"  # type: ignore[attr-defined]
    assert record.label == "banana_tree"  # type: ignore[attr-defined]
    assert record.confidence == 0.947  # type: ignore[attr-defined]
    assert record.total_ms == detail.timings_ms.total  # type: ignore[attr-defined]
    assert record.client_id == "6f1c2a9e"  # type: ignore[attr-defined]  # first 8 chars only
    # Privacy: no coordinates and no full client id anywhere in ANY record of the request.
    everything = " ".join(
        f"{r.getMessage()} {sorted(r.__dict__.items(), key=str)}" for r in logs.records
    )
    for secret in (
        "27.71724",
        "85.32402",
        "123.456789",
        str(client_id),
        "6f1c2a9e-1b2c",
        "latitude",
    ):
        assert secret not in everything, secret


# --- list(): the service imposes the caller's identity ---------------------------------------------


def _seed(db_session: Session, **kw: object) -> Analysis:
    row = make_analysis(**kw)
    db_session.add(row)
    db_session.commit()
    return row


def test_list_ignores_a_client_id_smuggled_into_the_filter(
    service: AnalysisService, db_session: Session
) -> None:
    me, victim = uuid.uuid4(), uuid.uuid4()
    mine = _seed(db_session, client_id=me)
    _seed(db_session, client_id=victim)

    page = service.list(AnalysisFilter(scope="mine", client_id=victim), 1, 20, me)

    assert [item.id for item in page.items] == [mine.id]


def test_list_ignores_a_status_in_the_filter_failed_rows_never_appear(
    service: AnalysisService, db_session: Session
) -> None:
    me = uuid.uuid4()
    _seed(
        db_session,
        client_id=me,
        status="failed",
        predicted_label=None,
        display_label=None,
        confidence=None,
    )

    page = service.list(AnalysisFilter(status="failed"), 1, 20, me)

    assert page.items == [] and page.total == 0


def test_list_clamps_paging_and_reports_the_effective_values(
    service: AnalysisService,
) -> None:
    page = service.list(AnalysisFilter(), 0, 1000, uuid.uuid4())
    assert (page.page, page.page_size) == (1, 100)


def test_total_pages_rounds_up(service: AnalysisService, db_session: Session) -> None:
    me = uuid.uuid4()
    for _ in range(5):
        _seed(db_session, client_id=me)
    page = service.list(AnalysisFilter(scope="mine"), 1, 2, me)
    assert (page.total, page.total_pages, len(page.items)) == (5, 3, 2)


# --- get_detail / delete access rule ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("owner_is_caller", "is_sample", "status", "visible"),
    [
        (True, False, "completed", True),  # my own
        (False, False, "completed", False),  # someone else's
        (False, True, "completed", True),  # a public sample
        (True, False, "failed", True),  # my own failed run
        (False, True, "failed", False),  # a failed sample is nobody's
    ],
)
def test_visibility_rule(
    service: AnalysisService,
    db_session: Session,
    owner_is_caller: bool,
    is_sample: bool,
    status: str,
    visible: bool,
) -> None:
    me = uuid.uuid4()
    owner = None if is_sample else (me if owner_is_caller else uuid.uuid4())
    row = _seed(
        db_session,
        client_id=owner,
        is_sample=is_sample,
        source="sample" if is_sample else "upload",
        status=status,
    )

    if visible:
        assert service.get_detail(row.id, me).id == row.id
    else:
        with pytest.raises(AnalysisNotFoundError):
            service.get_detail(row.id, me)


def test_soft_deleted_rows_are_not_found(service: AnalysisService, db_session: Session) -> None:
    me = uuid.uuid4()
    row = _seed(db_session, client_id=me, deleted_at=datetime(2026, 10, 1))
    with pytest.raises(AnalysisNotFoundError):
        service.get_detail(row.id, me)


def test_delete_sets_deleted_at_and_leaves_the_row(
    service: AnalysisService, db_session: Session
) -> None:
    me = uuid.uuid4()
    row = _seed(db_session, client_id=me)

    service.delete(row.id, me)

    db_session.expire_all()
    assert db_session.get(Analysis, row.id).deleted_at is not None  # type: ignore[union-attr]


def test_delete_of_a_sample_is_refused_and_of_a_foreign_row_is_not_found(
    service: AnalysisService, db_session: Session
) -> None:
    me = uuid.uuid4()
    sample = _seed(db_session, client_id=None, is_sample=True, source="sample")
    foreign = _seed(db_session, client_id=uuid.uuid4())

    with pytest.raises(SampleImmutableError):
        service.delete(sample.id, me)
    with pytest.raises(AnalysisNotFoundError):
        service.delete(foreign.id, me)


# --- open_image ----------------------------------------------------------------------------------------


def test_open_image_requires_a_signature_for_private_rows(
    service: AnalysisService, db_session: Session, settings: Settings
) -> None:
    row = _seed(db_session, client_id=uuid.uuid4())
    with pytest.raises(InvalidSignatureError):
        service.open_image(row.id, "original", None, None)
    with pytest.raises(InvalidSignatureError):
        service.open_image(row.id, "original", int(utcnow().timestamp()) + 60, "bogus")


def test_open_image_for_unknown_id_checks_the_signature_first(
    service: AnalysisService, settings: Settings
) -> None:
    from app.core import signing

    missing = uuid.uuid4()
    with pytest.raises(InvalidSignatureError):
        service.open_image(missing, "original", None, None)
    exp = int(utcnow().timestamp()) + 60
    with pytest.raises(AnalysisNotFoundError):
        service.open_image(
            missing, "original", exp, signing.sign(missing, "original", exp, settings)
        )


def test_open_image_of_a_stored_file_returns_path_media_type_and_cache_class(
    service: AnalysisService, storage: object, settings: Settings, tmp_path: Path
) -> None:
    from app.core import signing

    detail = analyze(service)
    exp = int(utcnow().timestamp()) + 60

    stored = service.open_image(
        detail.id, "thumbnail", exp, signing.sign(detail.id, "thumbnail", exp, settings)
    )

    assert stored.path.name == "thumb.webp" and stored.path.is_file()
    assert stored.media_type == "image/webp" and stored.is_public is False


def test_inference_failure_raises_a_clean_error_and_records_the_internal_one(
    make_service: Callable[..., AnalysisService], db_session: Session
) -> None:
    broken = FakePredictor(TREE, fail=True)
    service = make_service(ModelRegistry.from_predictors([broken]))

    with pytest.raises(InferenceError) as raised:
        analyze(service)

    assert raised.value.detail == "Inference failed"  # nothing internal in what the client gets
    (row,) = db_session.query(Analysis).all()
    assert row.status == "failed" and "fake inference failure" in (row.error_message or "")


def test_a_failing_rollback_does_not_mask_the_commit_error(
    service: AnalysisService,
    db_session: Session,
    logs: ListHandler,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    me = uuid.uuid4()
    row = _seed(db_session, client_id=me)

    def broken_commit() -> None:
        raise RuntimeError("commit failed")

    def broken_rollback() -> None:
        raise RuntimeError("rollback failed too")

    # A context so the patch is undone BEFORE the db_session fixture's own rollback runs.
    with monkeypatch.context() as patch:
        patch.setattr(db_session, "commit", broken_commit)
        patch.setattr(db_session, "rollback", broken_rollback)
        with pytest.raises(RuntimeError, match="commit failed"):  # the ORIGINAL error wins
            service.delete(row.id, me)
    assert any(r.getMessage() == "rollback_failed" for r in logs.records)
