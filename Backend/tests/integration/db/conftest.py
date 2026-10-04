"""Fixtures for the `mssql`-marked tests (real SQL Server, database KeraAI_Test).

Everything here skips cleanly — with the reason — when the server, the
database or the login is not available, so a machine without SQL Server (or an
owner who has not yet run db/sql/001_create_database_and_login.sql) just sees
"skipped", never a failure.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from alembic import command
from dotenv import dotenv_values
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.db.session import build_database_url, create_db_engine, create_session_factory
from tests.integration.db.helpers import BACKEND_DIR, make_alembic_config

TEST_DATABASE = "KeraAI_Test"


def _mssql_settings() -> Settings:
    """Settings for the test database.

    tests/conftest.py forces a few DB_* env vars for hermeticity (e.g.
    DB_TRUSTED_CONNECTION=true), which would hide the owner's real choice. So the
    DB_* values are read straight from Backend/.env and passed as init kwargs
    (which outrank env vars); only the database name is forced to KeraAI_Test.
    """
    from_dotenv = {
        key.lower(): value
        for key, value in dotenv_values(BACKEND_DIR / ".env").items()
        if key.upper().startswith("DB_") and value is not None
    }
    from_dotenv["db_name"] = TEST_DATABASE
    return Settings(**from_dotenv)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def mssql_url() -> URL:
    url = build_database_url(_mssql_settings())
    # Hard guard: these tests drop and recreate tables. They must never run
    # against the real KeraAI database, whatever .env says.
    assert url.database == TEST_DATABASE

    probe = create_engine(url, connect_args={"timeout": 5})  # 5 s login timeout
    try:
        with probe.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception as exc:
        reason = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
        pytest.skip(
            f"SQL Server database {TEST_DATABASE!r} is not reachable ({reason}). "
            "Run Backend/db/sql/001_create_database_and_login.sql in SSMS and check the "
            "DB_* settings in Backend/.env (host, auth mode, TCP/IP, login)."
        )
    finally:
        probe.dispose()
    return url


@pytest.fixture
def mssql_engine(mssql_url: URL) -> Iterator[Engine]:
    engine = create_db_engine(_mssql_settings(), url=mssql_url)
    yield engine
    engine.dispose()


@pytest.fixture
def mssql_schema(mssql_url: URL, mssql_engine: Engine) -> Iterator[Engine]:
    """KeraAI_Test with migrations applied (from a clean slate), removed afterwards."""
    config = make_alembic_config(mssql_url)
    command.downgrade(config, "base")
    command.upgrade(config, "head")
    yield mssql_engine
    mssql_engine.dispose()  # drop pooled connections before DDL
    command.downgrade(config, "base")


@pytest.fixture
def mssql_session_factory(mssql_schema: Engine) -> sessionmaker[Session]:
    return create_session_factory(mssql_schema)
