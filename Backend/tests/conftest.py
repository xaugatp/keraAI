from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

# Must run before any `app.*` import: Settings is built eagerly at
# `app.main` import time (module-level `app = create_app()`), so these have
# to land in os.environ first. Real env vars outrank values from a local
# Backend/.env file, so this keeps tests hermetic regardless of what's in
# the developer's own .env.
os.environ.setdefault("SIGNING_SECRET", "x" * 40)
os.environ.setdefault("DATA_ROOT", str(Path(tempfile.gettempdir()) / "keraai-test-data"))
os.environ.setdefault("DB_TRUSTED_CONNECTION", "true")
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("ENABLE_DOCS", "true")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("CORS_ORIGINS", '["http://localhost:3000"]')

import pytest  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import Engine  # noqa: E402
from sqlalchemy.orm import Session, sessionmaker  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.db import models as _models  # noqa: E402,F401  (registers models on Base.metadata)
from app.db.base import Base  # noqa: E402
from app.db.session import create_db_engine, create_session_factory  # noqa: E402
from app.main import create_app  # noqa: E402


@pytest.fixture
def sqlite_url(tmp_path: Path) -> str:
    # A temp *file* (not :memory:) so every pooled connection / worker thread
    # sees the same database.
    return f"sqlite:///{(tmp_path / 'test.db').as_posix()}"


@pytest.fixture
def sqlite_engine(sqlite_url: str) -> Iterator[Engine]:
    """SQLite engine with the schema created straight from the models.

    `create_all` is for TESTS ONLY — the real schema comes from Alembic
    migrations (which have their own up/down tests).
    """
    engine = create_db_engine(get_settings(), url=sqlite_url)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def session_factory(sqlite_engine: Engine) -> sessionmaker[Session]:
    return create_session_factory(sqlite_engine)


@pytest.fixture
def db_session(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    session = session_factory()
    yield session
    session.rollback()
    session.close()


@pytest.fixture
def app(sqlite_engine: Engine) -> Iterator[FastAPI]:
    # The injected SQLite engine keeps the suite hermetic: the lifespan's DB
    # ping hits a temp file, never the developer's real SQL Server.
    get_settings.cache_clear()
    application = create_app(engine=sqlite_engine)
    yield application
    get_settings.cache_clear()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client
