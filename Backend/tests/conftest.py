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

from app.core.config import get_settings  # noqa: E402
from app.main import create_app  # noqa: E402


@pytest.fixture
def app() -> Iterator[FastAPI]:
    get_settings.cache_clear()
    application = create_app()
    yield application
    get_settings.cache_clear()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client
