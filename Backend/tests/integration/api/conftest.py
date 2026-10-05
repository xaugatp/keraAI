from __future__ import annotations

import contextlib
import logging
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine

from app.core.config import Settings, get_settings
from app.main import create_app
from app.ml.registry import ModelRegistry

MakeClient = Callable[..., TestClient]


@pytest.fixture
def make_client(
    sqlite_engine: Engine,
    fake_registry: ModelRegistry,
    data_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[MakeClient]:
    """Factory for a fully wired app (SQLite temp file, fake models, tmp DATA_ROOT).

    ``make_client(MAX_UPLOAD_MB="1")`` overrides settings through the environment
    BEFORE the app is built (Settings is cached at creation); ``registry=`` swaps the
    fake models. ``raise_server_exceptions=False`` lets an unexpected 500 arrive as
    the problem+json a real client would see instead of re-raising inside the test.
    """
    with contextlib.ExitStack() as stack:

        def _make(*, registry: ModelRegistry | None = None, **env: str) -> TestClient:
            for name, value in env.items():
                monkeypatch.setenv(name.upper(), value)
            get_settings.cache_clear()
            application = create_app(engine=sqlite_engine, registry=registry or fake_registry)
            return stack.enter_context(TestClient(application, raise_server_exceptions=False))

        yield _make
    get_settings.cache_clear()


@pytest.fixture
def api(make_client: MakeClient) -> TestClient:
    return make_client()


@pytest.fixture
def settings(api: TestClient) -> Settings:
    """The Settings the running ``api`` app uses (depend on ``api`` first)."""
    return get_settings()


@pytest.fixture
def client_id() -> uuid.UUID:
    return uuid.uuid4()


@pytest.fixture
def other_client_id() -> uuid.UUID:
    return uuid.uuid4()


class LogCapture(logging.Handler):
    """Collects records. ``caplog`` cannot be used: the lifespan runs dictConfig,
    which strips the root logger's handlers (including caplog's) at startup."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def messages(self, level: int | None = None) -> list[str]:
        return [r.getMessage() for r in self.records if level is None or r.levelno == level]


@pytest.fixture
def service_logs() -> Iterator[LogCapture]:
    handler = LogCapture()
    logger = logging.getLogger("app.services.analysis_service")
    logger.addHandler(handler)
    yield handler
    logger.removeHandler(handler)
