"""Failure handling in the pipeline: INFERENCE_FAILED, storage and commit failures.

The invariant under test (spec 9, step 7): after ANY failure there are no orphan
files and no rows pointing at missing files.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session, sessionmaker

from app.api.deps import get_db, get_storage
from app.core.errors import AppError
from app.db.models.analysis import Analysis
from app.storage.base import ImageStorage
from app.storage.local import LocalImageStorage
from tests.fakes import FakePredictor
from tests.integration.api.conftest import LogCapture
from tests.integration.api.helpers import (
    ANALYSES_URL,
    LEAF_URL,
    TREE_URL,
    assert_problem,
    count_rows,
    files_under,
    headers_for,
    post_image,
)

# --- INFERENCE_FAILED -----------------------------------------------------------------


def test_inference_failure_persists_a_failed_row_with_the_image_and_hides_internals(
    api: TestClient,
    client_id: uuid.UUID,
    fake_tree: FakePredictor,
    session_factory: sessionmaker[Session],
    data_root: Path,
    service_logs: LogCapture,
) -> None:
    fake_tree.fail = True  # infer() raises RuntimeError("fake inference failure")

    response = post_image(api, TREE_URL, client_id)

    body = assert_problem(response, 500, "INFERENCE_FAILED")
    # Spec 11: nothing internal reaches the client.
    assert "fake inference failure" not in response.text
    assert "RuntimeError" not in response.text
    assert body["detail"] == "Inference failed"

    with session_factory() as session:
        (row,) = session.query(Analysis).all()
    assert row.status == "failed"
    assert row.error_code == "INFERENCE_FAILED"
    assert row.error_message is not None and "fake inference failure" in row.error_message
    assert row.client_id == client_id
    assert row.predicted_label is None and row.display_label is None and row.confidence is None
    assert row.details is None
    assert row.total_ms is not None

    # The image is kept so the failure can be debugged (spec 9 step 4, P-05).
    assert sorted(p.name for p in files_under(data_root)) == ["original.jpg", "thumb.webp"]
    assert any("analysis_failed" in message for message in service_logs.messages())


def test_failed_analysis_is_only_visible_to_its_owner_and_never_leaks_the_error(
    api: TestClient,
    client_id: uuid.UUID,
    other_client_id: uuid.UUID,
    fake_tree: FakePredictor,
    session_factory: sessionmaker[Session],
) -> None:
    fake_tree.fail = True
    post_image(api, TREE_URL, client_id)
    with session_factory() as session:
        (row,) = session.query(Analysis).all()
    url = f"{ANALYSES_URL}/{row.id}"

    own = api.get(url, headers=headers_for(client_id))
    assert own.status_code == 200
    detail = own.json()
    assert detail["status"] == "failed"
    assert detail["prediction"] is None and detail["details"] is None
    assert "error_message" not in detail and "error" not in detail
    assert "fake inference failure" not in own.text
    assert detail["image"]["original_url"]  # the saved image can still be shown

    assert_problem(api.get(url, headers=headers_for(other_client_id)), 404, "ANALYSIS_NOT_FOUND")

    # ... and it never shows up in a history list.
    listing = api.get(ANALYSES_URL, headers=headers_for(client_id)).json()
    assert listing["items"] == [] and listing["total"] == 0


def test_a_failed_analysis_can_be_deleted_by_its_owner(
    api: TestClient,
    client_id: uuid.UUID,
    fake_tree: FakePredictor,
    session_factory: sessionmaker[Session],
) -> None:
    fake_tree.fail = True
    post_image(api, TREE_URL, client_id)
    with session_factory() as session:
        (row,) = session.query(Analysis).all()
    assert api.delete(f"{ANALYSES_URL}/{row.id}", headers=headers_for(client_id)).status_code == 204


def test_failing_to_record_the_failure_still_returns_inference_failed(
    api: TestClient,
    client_id: uuid.UUID,
    fake_tree: FakePredictor,
    data_root: Path,
    service_logs: LogCapture,
) -> None:
    fake_tree.fail = True
    api.app.dependency_overrides[get_storage] = lambda: _FlakyStorage(  # type: ignore[attr-defined]
        LocalImageStorage(data_root), fail_on="thumb.webp"
    )

    response = post_image(api, TREE_URL, client_id)

    # The client still learns the true story; the operator gets the persistence error.
    assert_problem(response, 500, "INFERENCE_FAILED")
    assert files_under(data_root) == []  # and the half-saved failure left nothing behind
    assert any("failed_analysis_not_persisted" in m for m in service_logs.messages())


def test_predictor_raising_an_app_error_keeps_its_own_status(
    api: TestClient, client_id: uuid.UUID, fake_tree: FakePredictor, data_root: Path
) -> None:
    class Busy(AppError):
        status_code = 503
        code = "MODEL_UNAVAILABLE"
        title = "Model unavailable"

    fake_tree.fail = True
    fake_tree.error = Busy("warming up")
    assert_problem(post_image(api, TREE_URL, client_id), 503, "MODEL_UNAVAILABLE")
    assert files_under(data_root) == []  # not an InferenceError: nothing is persisted


# --- compensation ----------------------------------------------------------------------


class _FlakyStorage:
    """Real storage that raises on one chosen file name, to test the cleanup."""

    def __init__(self, inner: ImageStorage, fail_on: str) -> None:
        self._inner = inner
        self._fail_on = fail_on
        self.saved: list[str] = []

    def save(self, rel_path: str, data: bytes) -> None:
        if rel_path.endswith(self._fail_on):
            # A realistic driver message that must never reach a client.
            raise OSError(r"[Errno 28] No space left on device: 'C:\secret\kera-data\x'")
        self._inner.save(rel_path, data)
        self.saved.append(rel_path)

    def open(self, rel_path: str) -> Path:
        return self._inner.open(rel_path)

    def delete(self, rel_path: str) -> None:
        self._inner.delete(rel_path)

    def exists(self, rel_path: str) -> bool:
        return self._inner.exists(rel_path)


@pytest.mark.parametrize(
    ("url", "fail_on"),
    [
        (TREE_URL, "original.jpg"),  # first write fails
        (TREE_URL, "thumb.webp"),  # second write fails: the original must be removed
        (LEAF_URL, "result.png"),  # last write fails: original + thumbnail must be removed
    ],
)
def test_storage_failure_rolls_back_and_leaves_no_files(
    api: TestClient,
    client_id: uuid.UUID,
    session_factory: sessionmaker[Session],
    data_root: Path,
    url: str,
    fail_on: str,
) -> None:
    flaky = _FlakyStorage(LocalImageStorage(data_root), fail_on=fail_on)
    api.app.dependency_overrides[get_storage] = lambda: flaky  # type: ignore[attr-defined]

    response = post_image(api, url, client_id)

    body = assert_problem(response, 500, "INTERNAL_ERROR")
    assert "secret" not in response.text and "Errno" not in response.text
    assert body["detail"] == "An unexpected error occurred."
    assert files_under(data_root) == []
    assert not any(data_root.rglob("*.tmp"))
    assert count_rows(session_factory) == 0


class _CommitFails(Session):
    """A session whose COMMIT fails the way a dropped SQL Server connection does."""

    error: Exception = OperationalError("COMMIT", {}, Exception("08S01 link failure: host=SQLBOX1"))

    def commit(self) -> None:
        raise self.error


def _override_session(
    client: TestClient, factory: sessionmaker[Session], session_class: type[Session]
) -> list[Session]:
    created: list[Session] = []

    def failing_get_db() -> Iterator[Session]:
        session = session_class(bind=factory.kw["bind"], expire_on_commit=False)
        created.append(session)
        try:
            yield session
        finally:
            session.rollback()
            session.close()

    client.app.dependency_overrides[get_db] = failing_get_db  # type: ignore[attr-defined]
    return created


@pytest.mark.parametrize("url", [TREE_URL, LEAF_URL], ids=["tree", "leaf-with-result-image"])
def test_commit_failure_leaves_no_orphan_files_and_maps_to_503(
    api: TestClient,
    client_id: uuid.UUID,
    session_factory: sessionmaker[Session],
    data_root: Path,
    url: str,
) -> None:
    _override_session(api, session_factory, _CommitFails)

    response = post_image(api, url, client_id)

    body = assert_problem(response, 503, "DATABASE_UNAVAILABLE")
    assert (
        "SQLBOX1" not in response.text and "08S01" not in response.text
    )  # driver text stays in logs
    assert body["detail"].startswith("The database is temporarily unavailable")
    assert files_under(data_root) == []  # every file written before the commit was deleted
    assert count_rows(session_factory) == 0


def test_non_connection_database_error_is_500_without_sql_details(
    api: TestClient,
    client_id: uuid.UUID,
    session_factory: sessionmaker[Session],
    data_root: Path,
) -> None:
    class ConstraintFails(_CommitFails):
        error = IntegrityError(
            "INSERT INTO analyses ...", {"x": 1}, Exception("CK_analyses_confidence")
        )

    _override_session(api, session_factory, ConstraintFails)

    response = post_image(api, TREE_URL, client_id)

    assert_problem(response, 500, "INTERNAL_ERROR")
    assert "INSERT" not in response.text and "analyses" not in response.text
    assert files_under(data_root) == []


def test_compensation_failure_does_not_mask_the_original_error(
    api: TestClient,
    client_id: uuid.UUID,
    data_root: Path,
    session_factory: sessionmaker[Session],
    service_logs: LogCapture,
) -> None:
    class StuckDelete(_FlakyStorage):
        def delete(self, rel_path: str) -> None:
            raise PermissionError("antivirus has the file open")

    stuck = StuckDelete(LocalImageStorage(data_root), fail_on="thumb.webp")
    api.app.dependency_overrides[get_storage] = lambda: stuck  # type: ignore[attr-defined]

    response = post_image(api, TREE_URL, client_id)

    assert_problem(response, 500, "INTERNAL_ERROR")  # the ORIGINAL failure, not PermissionError
    assert any("compensation_delete_failed" in m for m in service_logs.messages())
    assert count_rows(session_factory) == 0


def test_database_down_on_a_read_route_is_503(
    api: TestClient, client_id: uuid.UUID, session_factory: sessionmaker[Session]
) -> None:
    class Broken(Session):
        def execute(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
            raise OperationalError("SELECT 1", {}, Exception("server unreachable"))

        def scalars(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
            raise OperationalError("SELECT 1", {}, Exception("server unreachable"))

        def scalar(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
            raise OperationalError("SELECT 1", {}, Exception("server unreachable"))

    _override_session(api, session_factory, Broken)
    response = api.get(ANALYSES_URL, headers=headers_for(client_id))
    assert_problem(response, 503, "DATABASE_UNAVAILABLE")
    assert "unreachable" not in response.text
