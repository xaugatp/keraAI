"""Alembic migrations, run in-process against a temp SQLite file (fast suite).

The same migration is also applied to real SQL Server by the `mssql`-marked
tests in test_mssql.py; here we prove the mechanics (up, down, env.py URL
handling, model/migration agreement) and, via offline `--sql` output, the exact
T-SQL that SQL Server would receive.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, inspect

from app.db.base import Base
from tests.integration.db.helpers import make_alembic_config

EXPECTED_INDEXES = {
    "ix_analyses_client_created": ["client_id", "created_at"],
    "ix_analyses_model_created": ["model_key", "created_at"],
    "ix_analyses_sample": ["is_sample", "model_key"],
    "ix_analyses_sha256": ["image_sha256"],
}


@pytest.fixture
def db_url(tmp_path: Path) -> str:
    return f"sqlite:///{(tmp_path / 'migrations.db').as_posix()}"


def _tables(url: str) -> set[str]:
    engine = create_engine(url)
    try:
        return set(inspect(engine).get_table_names())
    finally:
        engine.dispose()


def test_there_is_a_single_linear_head_named_0001() -> None:
    script = ScriptDirectory.from_config(make_alembic_config())
    assert script.get_heads() == ["0001"]
    revision = script.get_revision("0001")
    assert revision is not None
    assert revision.down_revision is None


def test_upgrade_creates_the_analyses_table_with_constraints_and_indexes(db_url: str) -> None:
    command.upgrade(make_alembic_config(db_url), "head")

    engine = create_engine(db_url)
    try:
        inspector = inspect(engine)
        assert {"analyses", "alembic_version"} <= set(inspector.get_table_names())

        columns = {c["name"] for c in inspector.get_columns("analyses")}
        assert columns == {c.name for c in Base.metadata.tables["analyses"].columns}

        pk = inspector.get_pk_constraint("analyses")
        assert pk["constrained_columns"] == ["id"]
        assert pk["name"] == "pk_analyses"

        checks = {c["name"] for c in inspector.get_check_constraints("analyses")}
        assert checks == {
            "ck_analyses_model_key",
            "ck_analyses_status",
            "ck_analyses_source",
            "ck_analyses_confidence_range",
            "ck_analyses_latitude_range",
            "ck_analyses_longitude_range",
            "ck_analyses_gps_accuracy_nonneg",
        }

        indexes = {i["name"]: i["column_names"] for i in inspector.get_indexes("analyses")}
        assert indexes == EXPECTED_INDEXES
    finally:
        engine.dispose()


def test_downgrade_to_base_removes_the_table_and_its_indexes(db_url: str) -> None:
    config = make_alembic_config(db_url)
    command.upgrade(config, "head")
    command.downgrade(config, "base")

    assert "analyses" not in _tables(db_url)
    # alembic's own bookkeeping table stays, but must record "no revision".
    engine = create_engine(db_url)
    try:
        with engine.connect() as connection:
            assert MigrationContext.configure(connection).get_current_revision() is None
    finally:
        engine.dispose()


def test_up_down_up_round_trip_is_repeatable(db_url: str) -> None:
    config = make_alembic_config(db_url)
    for _ in range(2):
        command.upgrade(config, "head")
        assert "analyses" in _tables(db_url)
        command.downgrade(config, "base")
        assert "analyses" not in _tables(db_url)


def test_upgrade_is_idempotent_when_already_at_head(db_url: str) -> None:
    config = make_alembic_config(db_url)
    command.upgrade(config, "head")
    command.upgrade(config, "head")  # no-op, must not raise "table already exists"
    assert "analyses" in _tables(db_url)


def test_migration_and_model_agree(db_url: str) -> None:
    """If someone edits the model but forgets the migration, this fails.

    Same idea as `alembic check`: autogenerate against the migrated database and
    expect no pending operations.
    """
    command.upgrade(make_alembic_config(db_url), "head")
    engine = create_engine(db_url)
    try:
        with engine.connect() as connection:
            context = MigrationContext.configure(connection, opts={"compare_type": True})
            diff = compare_metadata(context, Base.metadata)
    finally:
        engine.dispose()
    assert diff == []


def test_env_py_falls_back_to_settings_when_no_override_is_given() -> None:
    """Offline mode with no URL override: env.py must build it from Settings.

    The default .env/test settings point at SQL Server, so the emitted SQL is
    T-SQL — which also proves the URL did NOT come from anywhere hard-coded.
    """
    buffer = io.StringIO()
    command.upgrade(make_alembic_config(output=buffer), "head", sql=True)
    assert "CREATE TABLE analyses" in buffer.getvalue()
    assert "UNIQUEIDENTIFIER" in buffer.getvalue()  # SQL Server dialect, not SQLite


@pytest.fixture(scope="module")
def mssql_upgrade_sql() -> str:
    buffer = io.StringIO()
    command.upgrade(make_alembic_config("mssql+pyodbc://", output=buffer), "head", sql=True)
    return buffer.getvalue()


@pytest.mark.parametrize(
    "fragment",
    [
        "id UNIQUEIDENTIFIER NOT NULL",
        "client_id UNIQUEIDENTIFIER NULL",
        "model_key NVARCHAR(50) NOT NULL",
        "is_sample BIT NOT NULL DEFAULT 0",
        "is_uncertain BIT NOT NULL DEFAULT 0",
        "confidence FLOAT NULL",
        "latitude DECIMAL(9, 6) NULL",
        "longitude DECIMAL(9, 6) NULL",
        "captured_at DATETIME2(3) NULL",
        "created_at DATETIME2(3) NOT NULL",
        "updated_at DATETIME2(3) NOT NULL",
        "deleted_at DATETIME2(3) NULL",
        "image_sha256 CHAR(64) NOT NULL",
        "details NVARCHAR(max) NULL",
        "CONSTRAINT pk_analyses PRIMARY KEY (id)",
        "CONSTRAINT ck_analyses_model_key CHECK (model_key IN "
        "('tree_classification', 'leaf_segmentation', 'leaf_disease'))",
        "CONSTRAINT ck_analyses_status CHECK (status IN ('completed', 'failed'))",
        "CONSTRAINT ck_analyses_source CHECK (source IN ('upload', 'camera', 'sample'))",
        "CONSTRAINT ck_analyses_confidence_range CHECK (confidence >= 0 AND confidence <= 1)",
        "CONSTRAINT ck_analyses_latitude_range CHECK (latitude >= -90 AND latitude <= 90)",
        "CONSTRAINT ck_analyses_longitude_range CHECK (longitude >= -180 AND longitude <= 180)",
        "CONSTRAINT ck_analyses_gps_accuracy_nonneg CHECK (gps_accuracy_m >= 0)",
        "CREATE INDEX ix_analyses_client_created ON analyses (client_id, created_at DESC)",
        "CREATE INDEX ix_analyses_model_created ON analyses (model_key, created_at DESC)",
        "CREATE INDEX ix_analyses_sample ON analyses (is_sample, model_key)",
        "CREATE INDEX ix_analyses_sha256 ON analyses (image_sha256)",
    ],
)
def test_offline_sql_server_output_matches_the_spec(mssql_upgrade_sql: str, fragment: str) -> None:
    assert fragment in mssql_upgrade_sql


def test_offline_sql_server_downgrade_drops_everything() -> None:
    buffer = io.StringIO()
    command.downgrade(make_alembic_config("mssql+pyodbc://", output=buffer), "0001:base", sql=True)
    sql = buffer.getvalue()
    for name in EXPECTED_INDEXES:
        assert f"DROP INDEX {name} ON analyses" in sql
    assert "DROP TABLE analyses" in sql


def test_x_argument_url_override(db_url: str, tmp_path: Path) -> None:
    # `alembic -x url=...` — the CLI escape hatch used for one-off runs.
    config = make_alembic_config()
    config.cmd_opts = _FakeCmdOpts([f"url={db_url}"])  # type: ignore[assignment]
    command.upgrade(config, "head")
    assert "analyses" in _tables(db_url)


class _FakeCmdOpts:
    """Just enough of argparse's Namespace for `context.get_x_argument`."""

    def __init__(self, x: list[str]) -> None:
        self.x = x
