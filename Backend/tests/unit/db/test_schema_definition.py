"""Static checks on the model's metadata and the DDL it compiles to.

No database is touched: SQLAlchemy can compile DDL for the SQL Server dialect
offline, which lets the fast suite prove the model maps to the spec §5.2 types
(DATETIME2(3), UNIQUEIDENTIFIER, ...) even on a machine with no SQL Server.
"""

from __future__ import annotations

import pytest
from sqlalchemy import CheckConstraint, Index
from sqlalchemy.dialects import mssql
from sqlalchemy.schema import CreateIndex, CreateTable

from app.db.base import NAMING_CONVENTION, Base
from app.db.models.analysis import (
    ANALYSIS_SOURCES,
    ANALYSIS_STATUSES,
    MODEL_KEYS,
    Analysis,
)

TABLE = Analysis.__table__


def test_naming_convention_is_applied_to_metadata() -> None:
    assert Base.metadata.naming_convention == NAMING_CONVENTION
    assert set(NAMING_CONVENTION) == {"ix", "uq", "ck", "fk", "pk"}


def test_primary_key_and_check_constraint_names_are_deterministic() -> None:
    assert TABLE.primary_key.name == "pk_analyses"
    check_names = {c.name for c in TABLE.constraints if isinstance(c, CheckConstraint)}
    assert check_names == {
        "ck_analyses_model_key",
        "ck_analyses_status",
        "ck_analyses_source",
        "ck_analyses_confidence_range",
        "ck_analyses_latitude_range",
        "ck_analyses_longitude_range",
        "ck_analyses_gps_accuracy_nonneg",
    }


def test_the_four_spec_indexes_exist() -> None:
    indexes = {i.name: i for i in TABLE.indexes}
    assert set(indexes) == {
        "ix_analyses_client_created",
        "ix_analyses_model_created",
        "ix_analyses_sample",
        "ix_analyses_sha256",
    }


def _create_index_sql(index: Index) -> str:
    return str(CreateIndex(index).compile(dialect=mssql.dialect()))


def test_history_indexes_are_descending_on_created_at() -> None:
    by_name = {i.name: i for i in TABLE.indexes}
    assert "created_at DESC" in _create_index_sql(by_name["ix_analyses_client_created"])
    assert "created_at DESC" in _create_index_sql(by_name["ix_analyses_model_created"])
    # The other two are plain ascending indexes.
    assert "DESC" not in _create_index_sql(by_name["ix_analyses_sample"])
    assert "DESC" not in _create_index_sql(by_name["ix_analyses_sha256"])


@pytest.fixture(scope="module")
def mssql_ddl() -> str:
    return str(CreateTable(TABLE).compile(dialect=mssql.dialect()))


@pytest.mark.parametrize(
    "fragment",
    [
        "id UNIQUEIDENTIFIER NOT NULL",
        "client_id UNIQUEIDENTIFIER NULL",
        "model_key NVARCHAR(50) NOT NULL",
        "description NVARCHAR(1000) NULL",
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
        "original_image_path NVARCHAR(400) NOT NULL",
        "details NVARCHAR(max) NULL",
        "CONSTRAINT pk_analyses PRIMARY KEY (id)",
    ],
)
def test_ddl_for_sql_server_uses_spec_types(mssql_ddl: str, fragment: str) -> None:
    assert fragment in mssql_ddl


def test_every_text_column_is_unicode_not_varchar(mssql_ddl: str) -> None:
    # String would render VARCHAR (lossy for Nepali/accents); Unicode => NVARCHAR.
    assert "VARCHAR(" not in mssql_ddl.replace("NVARCHAR(", "")


def test_value_sets_match_the_spec_and_feed_the_check_constraints() -> None:
    assert MODEL_KEYS == ("tree_classification", "leaf_segmentation", "leaf_disease")
    assert ANALYSIS_STATUSES == ("completed", "failed")
    assert ANALYSIS_SOURCES == ("upload", "camera", "sample")

    checks = {c.name: str(c.sqltext) for c in TABLE.constraints if isinstance(c, CheckConstraint)}
    for value in MODEL_KEYS:
        assert f"'{value}'" in checks["ck_analyses_model_key"]
    for value in ANALYSIS_STATUSES:
        assert f"'{value}'" in checks["ck_analyses_status"]
    for value in ANALYSIS_SOURCES:
        assert f"'{value}'" in checks["ck_analyses_source"]
