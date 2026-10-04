"""Run the migration and the repository against a REAL SQL Server (KeraAI_Test).

Deselected by default; run with:  pytest -m mssql
Skips (with a reason) when the server/database/login is unavailable.
Destructive by design: it drops and recreates `analyses` in KeraAI_Test only.
"""

from __future__ import annotations

import uuid
from datetime import datetime

import pytest
from alembic import command
from sqlalchemy import Engine, inspect, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.repositories.analysis_repository import AnalysisFilter, AnalysisRepository
from tests.integration.db.helpers import make_alembic_config
from tests.unit.db.factories import make_analysis

pytestmark = pytest.mark.mssql

CLIENT_A = uuid.UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
CLIENT_B = uuid.UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")


def _has_table(engine: Engine, name: str) -> bool:
    return inspect(engine).has_table(name)


# --- migrations --------------------------------------------------------------


def test_migration_upgrade_and_downgrade_on_sql_server(
    mssql_url: URL, mssql_engine: Engine
) -> None:
    config = make_alembic_config(mssql_url)

    command.downgrade(config, "base")
    assert not _has_table(mssql_engine, "analyses")

    command.upgrade(config, "head")
    mssql_engine.dispose()
    assert _has_table(mssql_engine, "analyses")

    command.downgrade(config, "base")
    mssql_engine.dispose()
    assert not _has_table(mssql_engine, "analyses")

    # And it can come back: a downgrade that leaves debris would fail here.
    command.upgrade(config, "head")
    command.downgrade(config, "base")


def test_created_table_has_the_spec_column_types(mssql_schema: Engine) -> None:
    with mssql_schema.connect() as connection:
        rows = connection.execute(
            text(
                "SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, "
                "NUMERIC_PRECISION, NUMERIC_SCALE, DATETIME_PRECISION, IS_NULLABLE "
                "FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'analyses'"
            )
        ).all()
    columns = {r.COLUMN_NAME: r for r in rows}

    assert columns["id"].DATA_TYPE == "uniqueidentifier"
    assert columns["client_id"].DATA_TYPE == "uniqueidentifier"
    assert columns["client_id"].IS_NULLABLE == "YES"
    assert columns["model_key"].DATA_TYPE == "nvarchar"
    assert columns["model_key"].CHARACTER_MAXIMUM_LENGTH == 50
    assert columns["is_sample"].DATA_TYPE == "bit"
    assert columns["is_uncertain"].DATA_TYPE == "bit"
    assert columns["confidence"].DATA_TYPE == "float"
    assert columns["latitude"].DATA_TYPE == "decimal"
    assert (columns["latitude"].NUMERIC_PRECISION, columns["latitude"].NUMERIC_SCALE) == (9, 6)
    assert columns["image_sha256"].DATA_TYPE == "char"
    assert columns["image_sha256"].CHARACTER_MAXIMUM_LENGTH == 64
    assert columns["details"].DATA_TYPE == "nvarchar"
    assert columns["details"].CHARACTER_MAXIMUM_LENGTH == -1  # NVARCHAR(MAX)
    for name in ("created_at", "updated_at", "deleted_at", "captured_at"):
        assert columns[name].DATA_TYPE == "datetime2"
        assert columns[name].DATETIME_PRECISION == 3


def test_created_table_has_named_constraints_and_descending_indexes(mssql_schema: Engine) -> None:
    with mssql_schema.connect() as connection:
        checks = {
            r[0]
            for r in connection.execute(
                text(
                    "SELECT name FROM sys.check_constraints WHERE parent_object_id = OBJECT_ID('analyses')"
                )
            )
        }
        index_rows = connection.execute(
            text(
                "SELECT i.name AS index_name, c.name AS column_name, ic.is_descending_key "
                "FROM sys.indexes i "
                "JOIN sys.index_columns ic ON ic.object_id = i.object_id AND ic.index_id = i.index_id "
                "JOIN sys.columns c ON c.object_id = ic.object_id AND c.column_id = ic.column_id "
                "WHERE i.object_id = OBJECT_ID('analyses') AND i.is_primary_key = 0 "
                "ORDER BY i.name, ic.key_ordinal"
            )
        ).all()
        pk_name = connection.execute(
            text(
                "SELECT name FROM sys.key_constraints WHERE parent_object_id = OBJECT_ID('analyses') AND type = 'PK'"
            )
        ).scalar_one()

    assert pk_name == "pk_analyses"
    assert checks == {
        "ck_analyses_model_key",
        "ck_analyses_status",
        "ck_analyses_source",
        "ck_analyses_confidence_range",
        "ck_analyses_latitude_range",
        "ck_analyses_longitude_range",
        "ck_analyses_gps_accuracy_nonneg",
    }

    indexes: dict[str, list[tuple[str, bool]]] = {}
    for row in index_rows:
        indexes.setdefault(row.index_name, []).append(
            (row.column_name, bool(row.is_descending_key))
        )
    assert indexes == {
        "ix_analyses_client_created": [("client_id", False), ("created_at", True)],
        "ix_analyses_model_created": [("model_key", False), ("created_at", True)],
        "ix_analyses_sample": [("is_sample", False), ("model_key", False)],
        "ix_analyses_sha256": [("image_sha256", False)],
    }


# --- CRUD smoke --------------------------------------------------------------


def test_round_trip_on_sql_server(mssql_session_factory: sessionmaker[Session]) -> None:
    captured = datetime(2026, 10, 1, 1, 12, 15, 123000)
    details = {"kind": "tree_classification", "probabilities": [{"p": 0.947}]}
    original = make_analysis(
        client_id=CLIENT_A,
        title="केरा",
        latitude=27.71724,
        longitude=85.32402,
        gps_accuracy_m=4.2,
        captured_at=captured,
        details=details,
        is_uncertain=True,
    )
    with mssql_session_factory() as session:
        AnalysisRepository(session).add(original)
        session.commit()
        analysis_id, created_at = original.id, original.created_at

    with mssql_session_factory() as session:
        loaded = AnalysisRepository(session).get(analysis_id)
        assert loaded is not None
        assert loaded.id == analysis_id
        assert loaded.client_id == CLIENT_A
        assert loaded.title == "केरा"
        assert loaded.is_uncertain is True
        assert loaded.is_sample is False
        assert loaded.latitude == pytest.approx(27.71724)
        assert loaded.longitude == pytest.approx(85.32402)
        assert loaded.captured_at == captured
        # DATETIME2(3): the ms-truncated Python value is stored exactly.
        assert loaded.created_at == created_at
        assert loaded.created_at.tzinfo is None
        assert loaded.details == details


def test_repository_scopes_paging_and_soft_delete_on_sql_server(
    mssql_session_factory: sessionmaker[Session],
) -> None:
    with mssql_session_factory() as session:
        repo = AnalysisRepository(session)
        mine = [make_analysis(client_id=CLIENT_A) for _ in range(5)]
        theirs = make_analysis(client_id=CLIENT_B)
        sample = make_analysis(
            client_id=None, is_sample=True, source="sample", image_sha256="9" * 64
        )
        for row in (*mine, theirs, sample):
            repo.add(row)
        session.commit()

        rows, total = repo.list(AnalysisFilter(scope="mine", client_id=CLIENT_A), 1, 2)
        assert (len(rows), total) == (2, 5)

        visible, visible_total = repo.list(
            AnalysisFilter(scope="all_visible", client_id=CLIENT_A), 1, 50
        )
        assert visible_total == 6  # 5 own + 1 sample, never CLIENT_B's
        assert theirs.id not in {r.id for r in visible}

        assert repo.get_sample_by_hash("tree_classification", "9" * 64) is not None

        repo.soft_delete(mine[0])
        session.commit()
        assert repo.get(mine[0].id) is None
        assert repo.list(AnalysisFilter(scope="mine", client_id=CLIENT_A), 1, 50)[1] == 4


# --- constraints enforced by SQL Server itself -------------------------------


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("model_key", "banana"),
        ("status", "pending"),
        ("source", "email"),
        ("confidence", 1.5),
        ("latitude", 91.0),
        ("longitude", -181.0),
        ("gps_accuracy_m", -1.0),
    ],
)
def test_check_constraints_are_enforced_by_sql_server(
    mssql_session_factory: sessionmaker[Session], field: str, bad_value: object
) -> None:
    with mssql_session_factory() as session:
        session.add(make_analysis(**{field: bad_value}))
        with pytest.raises(IntegrityError, match="CHECK constraint"):
            session.flush()


def test_a_row_can_be_inserted_without_the_orm_defaults(
    mssql_session_factory: sessionmaker[Session],
) -> None:
    # The BIT server defaults apply to inserts that bypass the ORM (e.g. from SSMS).
    with mssql_session_factory() as session:
        session.execute(
            text(
                "INSERT INTO analyses (id, model_key, model_version, status, source, "
                "original_image_path, thumbnail_path, image_sha256, created_at, updated_at) "
                "VALUES (NEWID(), 'tree_classification', 'v', 'completed', 'upload', 'o', 't', "
                ":h, SYSUTCDATETIME(), SYSUTCDATETIME())"
            ),
            {"h": "b" * 64},
        )
        row = session.execute(text("SELECT is_sample, is_uncertain FROM analyses")).one()
        assert (row.is_sample, row.is_uncertain) == (False, False)
        session.rollback()
