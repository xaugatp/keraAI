"""DELETE /analyses/{id}: soft delete, ownership, samples."""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.api.helpers import (
    ANALYSES_URL,
    TREE_URL,
    assert_problem,
    fetch_row,
    files_under,
    headers_for,
    post_image,
    seed_analysis,
)

NOW = datetime(2026, 10, 1, 8, 0, 0)


def test_delete_own_analysis_then_it_is_gone(
    api: TestClient,
    client_id: uuid.UUID,
    session_factory: sessionmaker[Session],
    data_root: Path,
) -> None:
    created = post_image(api, TREE_URL, client_id).json()
    url = f"{ANALYSES_URL}/{created['id']}"
    files_before = files_under(data_root)

    response = api.delete(url, headers=headers_for(client_id))

    assert response.status_code == 204
    assert response.content == b""
    # Gone from every read ...
    assert_problem(api.get(url, headers=headers_for(client_id)), 404, "ANALYSIS_NOT_FOUND")
    assert api.get(ANALYSES_URL, headers=headers_for(client_id)).json()["items"] == []
    # ... and a second delete is a 404, not a 204.
    assert_problem(api.delete(url, headers=headers_for(client_id)), 404, "ANALYSIS_NOT_FOUND")
    # Soft delete (P-07): the row is still in the table, flagged; files are untouched.
    row = fetch_row(session_factory, created["id"])
    assert row is not None and row.deleted_at is not None
    assert files_under(data_root) == files_before
    # Even a previously issued signed link stops working.
    assert_problem(api.get(created["image"]["original_url"]), 404, "ANALYSIS_NOT_FOUND")


def test_another_clients_analysis_cannot_be_deleted_and_looks_missing(
    api: TestClient,
    client_id: uuid.UUID,
    other_client_id: uuid.UUID,
    session_factory: sessionmaker[Session],
) -> None:
    created = post_image(api, TREE_URL, client_id).json()

    response = api.delete(f"{ANALYSES_URL}/{created['id']}", headers=headers_for(other_client_id))

    assert_problem(response, 404, "ANALYSIS_NOT_FOUND")
    row = fetch_row(session_factory, created["id"])
    assert row is not None and row.deleted_at is None  # untouched
    assert (
        api.get(f"{ANALYSES_URL}/{created['id']}", headers=headers_for(client_id)).status_code
        == 200
    )


def test_samples_cannot_be_deleted_by_anyone(
    api: TestClient,
    client_id: uuid.UUID,
    session_factory: sessionmaker[Session],
    data_root: Path,
) -> None:
    sample = seed_analysis(
        session_factory, data_root, client_id=None, is_sample=True, created_at=NOW, title="Demo"
    )

    response = api.delete(f"{ANALYSES_URL}/{sample}", headers=headers_for(client_id))

    assert_problem(response, 403, "SAMPLE_IMMUTABLE")
    row = fetch_row(session_factory, sample)
    assert row is not None and row.deleted_at is None
    assert api.get(f"{ANALYSES_URL}/{sample}", headers=headers_for(client_id)).status_code == 200


def test_deleting_a_missing_id_is_404(api: TestClient, client_id: uuid.UUID) -> None:
    response = api.delete(f"{ANALYSES_URL}/{uuid.uuid4()}", headers=headers_for(client_id))
    assert_problem(response, 404, "ANALYSIS_NOT_FOUND")


def test_delete_needs_a_valid_id_and_a_client_id(api: TestClient, client_id: uuid.UUID) -> None:
    assert_problem(
        api.delete(f"{ANALYSES_URL}/nope", headers=headers_for(client_id)), 422, "VALIDATION_ERROR"
    )
    assert_problem(api.delete(f"{ANALYSES_URL}/{uuid.uuid4()}"), 400, "MISSING_CLIENT_ID")


def test_commit_failure_while_deleting_is_503_and_nothing_is_deleted(
    api: TestClient,
    client_id: uuid.UUID,
    session_factory: sessionmaker[Session],
    data_root: Path,
) -> None:
    from sqlalchemy.exc import OperationalError

    from app.api.deps import get_db

    seeded = seed_analysis(session_factory, data_root, client_id=client_id, created_at=NOW)

    class CommitFails(Session):
        def commit(self) -> None:
            raise OperationalError("COMMIT", {}, Exception("link failure"))

    def failing_get_db():  # type: ignore[no-untyped-def]
        session = CommitFails(bind=session_factory.kw["bind"], expire_on_commit=False)
        try:
            yield session
        finally:
            session.rollback()
            session.close()

    api.app.dependency_overrides[get_db] = failing_get_db  # type: ignore[attr-defined]

    response = api.delete(f"{ANALYSES_URL}/{seeded}", headers=headers_for(client_id))

    assert_problem(response, 503, "DATABASE_UNAVAILABLE")
    row = fetch_row(session_factory, seeded)
    assert row is not None and row.deleted_at is None
