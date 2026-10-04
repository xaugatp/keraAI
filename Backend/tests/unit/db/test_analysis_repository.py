from __future__ import annotations

import uuid
from datetime import datetime, timedelta

import pytest
from sqlalchemy.orm import Session

from app.db.models.analysis import Analysis
from app.repositories.analysis_repository import (
    MAX_PAGE_SIZE,
    AnalysisFilter,
    AnalysisRepository,
)
from tests.unit.db.factories import at, make_analysis

CLIENT_A = uuid.UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
CLIENT_B = uuid.UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")


@pytest.fixture
def repo(db_session: Session) -> AnalysisRepository:
    return AnalysisRepository(db_session)


def _add(repo: AnalysisRepository, session: Session, **overrides: object) -> Analysis:
    row = make_analysis(**overrides)
    repo.add(row)
    session.flush()
    return row


def _ids(rows: list[Analysis]) -> set[uuid.UUID]:
    return {r.id for r in rows}


# --- add / get / soft_delete -------------------------------------------------


def test_add_then_get(repo: AnalysisRepository, db_session: Session) -> None:
    row = make_analysis()
    repo.add(row)
    db_session.commit()
    db_session.expunge_all()

    loaded = repo.get(row.id)
    assert loaded is not None
    assert loaded.id == row.id
    assert loaded.model_key == "tree_classification"


def test_get_unknown_id_returns_none(repo: AnalysisRepository) -> None:
    assert repo.get(uuid.uuid4()) is None


def test_add_does_not_commit(repo: AnalysisRepository, db_session: Session) -> None:
    row = make_analysis()
    repo.add(row)
    db_session.rollback()
    assert repo.get(row.id) is None


def test_get_excludes_soft_deleted(repo: AnalysisRepository, db_session: Session) -> None:
    row = _add(repo, db_session)
    repo.soft_delete(row)
    db_session.commit()
    assert repo.get(row.id) is None


def test_soft_delete_sets_naive_utc_timestamp_and_keeps_the_row(
    repo: AnalysisRepository, db_session: Session
) -> None:
    row = _add(repo, db_session)
    before = datetime.now() - timedelta(days=1)
    repo.soft_delete(row)
    db_session.commit()

    assert row.deleted_at is not None
    assert row.deleted_at.tzinfo is None
    assert row.deleted_at > before
    # The row is still physically there (audit trail, P-07).
    assert db_session.get(Analysis, row.id) is not None


def test_soft_delete_does_not_commit(repo: AnalysisRepository, db_session: Session) -> None:
    row = _add(repo, db_session)
    db_session.commit()

    repo.soft_delete(row)
    db_session.rollback()
    db_session.refresh(row)
    assert row.deleted_at is None


# --- list: scopes and client isolation ---------------------------------------


@pytest.fixture
def world(repo: AnalysisRepository, db_session: Session) -> dict[str, Analysis]:
    """A's two rows, B's row, two samples, and one soft-deleted row of A's."""
    rows = {
        "a1": _add(repo, db_session, client_id=CLIENT_A, created_at=at(2026, 10, 1)),
        "a2": _add(
            repo,
            db_session,
            client_id=CLIENT_A,
            created_at=at(2026, 10, 2),
            status="failed",
            model_key="leaf_segmentation",
        ),
        "b1": _add(repo, db_session, client_id=CLIENT_B, created_at=at(2026, 10, 3)),
        "s1": _add(
            repo,
            db_session,
            client_id=None,
            is_sample=True,
            source="sample",
            created_at=at(2026, 10, 4),
        ),
        "s2": _add(
            repo,
            db_session,
            client_id=None,
            is_sample=True,
            source="sample",
            model_key="leaf_segmentation",
            created_at=at(2026, 10, 5),
        ),
        "gone": _add(
            repo,
            db_session,
            client_id=CLIENT_A,
            created_at=at(2026, 10, 6),
            deleted_at=at(2026, 10, 7),
        ),
    }
    db_session.commit()
    return rows


def _list(repo: AnalysisRepository, **filter_kwargs: object) -> tuple[list[Analysis], int]:
    return repo.list(AnalysisFilter(**filter_kwargs), page=1, page_size=50)  # type: ignore[arg-type]


def test_scope_mine_returns_only_the_callers_rows(
    repo: AnalysisRepository, world: dict[str, Analysis]
) -> None:
    rows, total = _list(repo, scope="mine", client_id=CLIENT_A)
    assert _ids(rows) == {world["a1"].id, world["a2"].id}
    assert total == 2

    rows_b, total_b = _list(repo, scope="mine", client_id=CLIENT_B)
    assert _ids(rows_b) == {world["b1"].id}
    assert total_b == 1


def test_scope_mine_without_a_client_id_is_empty_not_the_samples(
    repo: AnalysisRepository, world: dict[str, Analysis]
) -> None:
    # `client_id IS NULL` would match every sample; "mine" for nobody must be empty.
    rows, total = _list(repo, scope="mine", client_id=None)
    assert rows == []
    assert total == 0


def test_scope_samples_returns_samples_whoever_asks(
    repo: AnalysisRepository, world: dict[str, Analysis]
) -> None:
    expected = {world["s1"].id, world["s2"].id}
    assert _ids(_list(repo, scope="samples", client_id=CLIENT_A)[0]) == expected
    assert _ids(_list(repo, scope="samples", client_id=CLIENT_B)[0]) == expected
    assert _ids(_list(repo, scope="samples", client_id=None)[0]) == expected


def test_scope_all_visible_is_samples_plus_own(
    repo: AnalysisRepository, world: dict[str, Analysis]
) -> None:
    rows_a, total_a = _list(repo, scope="all_visible", client_id=CLIENT_A)
    assert _ids(rows_a) == {world[k].id for k in ("a1", "a2", "s1", "s2")}
    assert total_a == 4

    rows_b, _ = _list(repo, scope="all_visible", client_id=CLIENT_B)
    assert _ids(rows_b) == {world[k].id for k in ("b1", "s1", "s2")}


def test_scope_all_visible_for_unknown_caller_is_samples_only(
    repo: AnalysisRepository, world: dict[str, Analysis]
) -> None:
    rows, _ = _list(repo, scope="all_visible", client_id=None)
    assert _ids(rows) == {world["s1"].id, world["s2"].id}


def test_default_scope_is_all_visible() -> None:
    assert AnalysisFilter().scope == "all_visible"


def test_one_client_never_sees_anothers_rows_in_any_scope(
    repo: AnalysisRepository, world: dict[str, Analysis]
) -> None:
    for scope in ("mine", "samples", "all_visible"):
        rows, _ = _list(repo, scope=scope, client_id=CLIENT_A)
        assert world["b1"].id not in _ids(rows), scope


def test_soft_deleted_rows_are_excluded_from_every_scope(
    repo: AnalysisRepository, world: dict[str, Analysis]
) -> None:
    for scope in ("mine", "samples", "all_visible"):
        rows, _ = _list(repo, scope=scope, client_id=CLIENT_A)
        assert world["gone"].id not in _ids(rows), scope


# --- list: filters -----------------------------------------------------------


def test_filter_by_model_key(repo: AnalysisRepository, world: dict[str, Analysis]) -> None:
    rows, total = _list(
        repo, scope="all_visible", client_id=CLIENT_A, model_key="leaf_segmentation"
    )
    assert _ids(rows) == {world["a2"].id, world["s2"].id}
    assert total == 2


def test_filter_by_status(repo: AnalysisRepository, world: dict[str, Analysis]) -> None:
    rows, total = _list(repo, scope="all_visible", client_id=CLIENT_A, status="failed")
    assert _ids(rows) == {world["a2"].id}
    assert total == 1

    completed, _ = _list(repo, scope="all_visible", client_id=CLIENT_A, status="completed")
    assert world["a2"].id not in _ids(completed)


def test_filters_combine(repo: AnalysisRepository, world: dict[str, Analysis]) -> None:
    rows, _ = _list(
        repo,
        scope="mine",
        client_id=CLIENT_A,
        model_key="leaf_segmentation",
        status="completed",
    )
    assert rows == []


# --- list: ordering and pagination -------------------------------------------


def test_default_order_is_newest_first(
    repo: AnalysisRepository, world: dict[str, Analysis]
) -> None:
    rows, _ = _list(repo, scope="all_visible", client_id=CLIENT_A)
    assert [r.id for r in rows] == [world[k].id for k in ("s2", "s1", "a2", "a1")]


def _seed_many(repo: AnalysisRepository, session: Session, count: int) -> list[Analysis]:
    base = datetime(2026, 1, 1)
    rows = [
        _add(repo, session, client_id=CLIENT_A, created_at=base + timedelta(minutes=i))
        for i in range(count)
    ]
    session.commit()
    return rows


def test_pagination_pages_and_total(repo: AnalysisRepository, db_session: Session) -> None:
    seeded = _seed_many(repo, db_session, 25)
    newest_first = [r.id for r in reversed(seeded)]
    filters = AnalysisFilter(scope="mine", client_id=CLIENT_A)

    page1, total = repo.list(filters, page=1, page_size=10)
    page2, _ = repo.list(filters, page=2, page_size=10)
    page3, _ = repo.list(filters, page=3, page_size=10)

    assert total == 25
    assert [len(p) for p in (page1, page2, page3)] == [10, 10, 5]
    assert [r.id for r in page1 + page2 + page3] == newest_first


def test_page_past_the_end_is_empty_but_total_is_kept(
    repo: AnalysisRepository, db_session: Session
) -> None:
    _seed_many(repo, db_session, 5)
    rows, total = repo.list(AnalysisFilter(scope="mine", client_id=CLIENT_A), page=99, page_size=10)
    assert rows == []
    assert total == 5


def test_empty_result_has_zero_total(repo: AnalysisRepository) -> None:
    rows, total = repo.list(AnalysisFilter(scope="mine", client_id=CLIENT_A))
    assert (rows, total) == ([], 0)


def test_page_size_is_capped_at_the_maximum(repo: AnalysisRepository, db_session: Session) -> None:
    assert MAX_PAGE_SIZE == 100
    _seed_many(repo, db_session, 120)
    filters = AnalysisFilter(scope="mine", client_id=CLIENT_A)

    rows, total = repo.list(filters, page=1, page_size=1000)
    assert len(rows) == 100
    assert total == 120

    exactly_max, _ = repo.list(filters, page=1, page_size=100)
    assert len(exactly_max) == 100
    second, _ = repo.list(filters, page=2, page_size=1000)
    assert len(second) == 20  # page 2 of the *capped* size


@pytest.mark.parametrize(("page", "page_size"), [(0, 10), (-3, 10), (1, 0), (1, -5)])
def test_nonsense_paging_values_are_clamped_not_errors(
    repo: AnalysisRepository, db_session: Session, page: int, page_size: int
) -> None:
    _seed_many(repo, db_session, 3)
    rows, total = repo.list(
        AnalysisFilter(scope="mine", client_id=CLIENT_A), page=page, page_size=page_size
    )
    assert total == 3
    assert 1 <= len(rows) <= 10


def test_rows_with_identical_timestamps_page_stably(
    repo: AnalysisRepository, db_session: Session
) -> None:
    # Same created_at on every row: without a tiebreaker OFFSET paging may repeat
    # or skip rows. `id` makes the order total.
    same_time = at(2026, 3, 3)
    seeded = [_add(repo, db_session, client_id=CLIENT_A, created_at=same_time) for _ in range(7)]
    db_session.commit()
    filters = AnalysisFilter(scope="mine", client_id=CLIENT_A)

    paged: list[uuid.UUID] = []
    for page in range(1, 5):
        rows, _ = repo.list(filters, page=page, page_size=2)
        paged.extend(r.id for r in rows)

    assert len(paged) == 7
    assert set(paged) == {r.id for r in seeded}  # nothing skipped or repeated
    # And the order is reproducible: asking again yields the identical sequence.
    again = [r.id for r in repo.list(filters, page=1, page_size=50)[0]]
    assert again == paged


# --- get_sample_by_hash ------------------------------------------------------


def _sample(repo: AnalysisRepository, session: Session, sha: str, **kw: object) -> Analysis:
    return _add(
        repo,
        session,
        is_sample=True,
        source="sample",
        client_id=None,
        image_sha256=sha,
        **kw,
    )


def test_get_sample_by_hash_finds_the_sample(repo: AnalysisRepository, db_session: Session) -> None:
    sha = "c" * 64
    row = _sample(repo, db_session, sha)
    assert repo.get_sample_by_hash("tree_classification", sha) is row


def test_get_sample_by_hash_is_scoped_by_model_key(
    repo: AnalysisRepository, db_session: Session
) -> None:
    sha = "d" * 64
    _sample(repo, db_session, sha)
    assert repo.get_sample_by_hash("leaf_segmentation", sha) is None


def test_get_sample_by_hash_ignores_non_samples(
    repo: AnalysisRepository, db_session: Session
) -> None:
    sha = "e" * 64
    _add(repo, db_session, image_sha256=sha, client_id=CLIENT_A)  # a user's upload
    assert repo.get_sample_by_hash("tree_classification", sha) is None


def test_get_sample_by_hash_unknown_hash(repo: AnalysisRepository) -> None:
    assert repo.get_sample_by_hash("tree_classification", "f" * 64) is None


def test_get_sample_by_hash_ignores_soft_deleted_samples(
    repo: AnalysisRepository, db_session: Session
) -> None:
    # `seed_samples --force` soft-deletes the old sample and re-creates it: the new
    # one must be found, the old one must not.
    sha = "1" * 64
    old = _sample(repo, db_session, sha)
    repo.soft_delete(old)
    assert repo.get_sample_by_hash("tree_classification", sha) is None

    new = _sample(repo, db_session, sha)
    assert repo.get_sample_by_hash("tree_classification", sha) is new
