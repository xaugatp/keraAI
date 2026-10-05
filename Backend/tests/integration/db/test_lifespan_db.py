"""Lifespan wiring of the database: engine, readiness check, degraded mode."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from app import main as app_main
from app.core.config import get_settings
from app.db.session import create_db_engine
from app.main import _database_readiness_check, create_app


class _ListHandler(logging.Handler):
    """Collects records. `caplog` can't be used: the lifespan runs dictConfig, which
    strips the root logger's handlers (including caplog's) at startup."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def log_records() -> Iterator[list[logging.LogRecord]]:
    handler = _ListHandler()
    names = ("app.main", "app.db.session")
    for name in names:
        logging.getLogger(name).addHandler(handler)
    yield handler.records
    for name in names:
        logging.getLogger(name).removeHandler(handler)


@pytest.fixture
def unreachable_engine(tmp_path: Path) -> Iterator[Engine]:
    # SQLite cannot create a file inside a directory that does not exist, so every
    # connection attempt fails with OperationalError — a stand-in for "SQL Server down".
    engine = create_db_engine(
        get_settings(), url=f"sqlite:///{(tmp_path / 'no-such-dir' / 'x.db').as_posix()}"
    )
    yield engine
    engine.dispose()


# --- healthy -----------------------------------------------------------------


def test_lifespan_stores_engine_and_session_factory_on_app_state(
    app: FastAPI, sqlite_engine: Engine
) -> None:
    with TestClient(app):
        assert app.state.engine is sqlite_engine
        with app.state.session_factory() as session:
            assert isinstance(session, Session)
            assert session.get_bind() is sqlite_engine


def test_readiness_reports_database_ok(client: TestClient) -> None:
    response = client.get("/health/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    # Since Phase 5 the lifespan also registers a `models` check (fake registry here).
    assert body["components"]["database"] == {"status": "ok", "detail": None}


# --- degraded mode -----------------------------------------------------------


def test_app_starts_when_the_database_is_unreachable(
    unreachable_engine: Engine, log_records: list[logging.LogRecord]
) -> None:
    app = create_app(engine=unreachable_engine)

    with TestClient(app) as client:  # must not raise
        # Liveness is independent of the DB.
        assert client.get("/health").status_code == 200

        response = client.get("/health/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["components"]["database"] == {
        "status": "unavailable",
        "detail": "Database unreachable",
    }
    # Internals (driver text, file paths) must not leak to the client.
    assert "no-such-dir" not in response.text
    assert "sqlite" not in response.text.lower()

    # The operator, on the other hand, gets the reason in the log.
    warnings = [r for r in log_records if r.levelno == logging.WARNING]
    messages = [r.getMessage() for r in warnings]
    assert any("Database ping failed" in m and "OperationalError" in m for m in messages)
    assert any("degraded mode" in m for m in messages)


def test_readiness_recovers_when_the_database_comes_back(
    tmp_path: Path, sqlite_engine: Engine
) -> None:
    # Start with a broken engine, then swap in a healthy one behind the same
    # check: models "SQL Server finished starting after the API did".
    broken = create_db_engine(
        get_settings(), url=f"sqlite:///{(tmp_path / 'nope' / 'x.db').as_posix()}"
    )
    app = create_app(engine=broken)
    with TestClient(app) as client:
        assert client.get("/health/ready").status_code == 503
        app.state.readiness_checks["database"] = _database_readiness_check(sqlite_engine)
        assert client.get("/health/ready").status_code == 200
    broken.dispose()


# --- engine ownership --------------------------------------------------------


class _SpyEngine:
    """Wraps a real engine and records dispose() calls."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self.disposed = 0

    def dispose(self) -> None:
        self.disposed += 1
        self._engine.dispose()

    def __getattr__(self, name: str) -> object:
        return getattr(self._engine, name)


def test_engine_created_by_the_lifespan_is_disposed_on_shutdown(
    monkeypatch: pytest.MonkeyPatch, sqlite_engine: Engine
) -> None:
    spy = _SpyEngine(sqlite_engine)
    monkeypatch.setattr(app_main, "create_db_engine", lambda settings: spy)

    app = create_app()  # no injected engine -> the lifespan builds its own
    with TestClient(app):
        assert app.state.engine is spy
        assert spy.disposed == 0
    assert spy.disposed == 1


def test_injected_engine_is_not_disposed_by_the_lifespan(sqlite_engine: Engine) -> None:
    spy = _SpyEngine(sqlite_engine)
    app = create_app(engine=spy)  # type: ignore[arg-type]
    with TestClient(app):
        pass
    assert spy.disposed == 0  # the creator owns it


def test_engine_is_disposed_even_if_the_app_errors_during_run(
    monkeypatch: pytest.MonkeyPatch, sqlite_engine: Engine
) -> None:
    spy = _SpyEngine(sqlite_engine)
    monkeypatch.setattr(app_main, "create_db_engine", lambda settings: spy)
    app = create_app()

    with pytest.raises(RuntimeError, match="kaboom"), TestClient(app):
        raise RuntimeError("kaboom")
    assert spy.disposed == 1


# --- threadpool --------------------------------------------------------------


def test_database_ping_runs_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch, sqlite_engine: Engine
) -> None:
    """pyodbc blocks, so the ping must run in a worker thread, not on the loop."""
    seen: list[tuple[str, bool]] = []

    def spy_ping(engine: Engine) -> tuple[bool, str | None]:
        try:
            asyncio.get_running_loop()
            on_loop = True
        except RuntimeError:
            on_loop = False
        seen.append((threading.current_thread().name, on_loop))
        return True, None

    monkeypatch.setattr(app_main, "ping_database", spy_ping)
    app = create_app(engine=sqlite_engine)
    with TestClient(app) as client:
        seen.clear()  # ignore the startup ping; test the readiness-check path
        assert client.get("/health/ready").status_code == 200

    assert seen, "readiness check never called the ping"
    assert all(on_loop is False for _, on_loop in seen)
