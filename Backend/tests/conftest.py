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
from PIL import Image  # noqa: E402
from sqlalchemy import Engine  # noqa: E402
from sqlalchemy.orm import Session, sessionmaker  # noqa: E402

from app import main as app_main  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.core.rate_limit import limiter  # noqa: E402
from app.db import models as _models  # noqa: E402,F401  (registers models on Base.metadata)
from app.db.base import Base  # noqa: E402
from app.db.session import create_db_engine, create_session_factory  # noqa: E402
from app.main import create_app  # noqa: E402
from app.ml.registry import ModelRegistry  # noqa: E402
from app.schemas.leaf_disease import (  # noqa: E402
    DiseaseChannelMetrics,
    LeafDiseaseDetails,
    LeafDiseaseThresholds,
)
from app.schemas.leaf_seg import LeafSegDetails, LeafSegThresholds  # noqa: E402
from tests.fakes import FakePredictor  # noqa: E402


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


# --- model fakes ------------------------------------------------------------
# The default app/client fixtures must never build real torch models (fast tests
# import neither torch nor ultralytics), so they always get a registry of fakes.


def make_leaf_seg_details() -> LeafSegDetails:
    return LeafSegDetails(
        kind="leaf_segmentation",
        leaf_area_pct_of_image=61.5,
        affected_area_pct_of_leaf=12.25,
        lesion_count=3,
        largest_lesion_pct_of_leaf=6.5,
        mean_leaf_probability=0.91,
        mean_affected_probability=0.88,
        thresholds=LeafSegThresholds(
            leaf=0.5,
            affected=0.85,
            min_leaf_pct=3.0,
            min_lesion_pct=0.05,
            min_affected_pct=0.5,
            tta_hflip=True,
        ),
    )


def make_leaf_disease_details() -> LeafDiseaseDetails:
    return LeafDiseaseDetails(
        kind="leaf_disease",
        diagnosis="black_sigatoka",
        leaf_area_pct_of_image=58.0,
        mean_leaf_probability=0.9,
        diseases=[
            DiseaseChannelMetrics(
                channel="black_sigatoka",
                area_pct_of_leaf=15.0,
                lesion_count=2,
                largest_lesion_pct_of_leaf=9.0,
                mean_probability=0.82,
            ),
            DiseaseChannelMetrics(
                channel="yellow_sigatoka",
                area_pct_of_leaf=0.0,
                lesion_count=0,
                largest_lesion_pct_of_leaf=0.0,
                mean_probability=None,
            ),
        ],
        thresholds=LeafDiseaseThresholds(
            leaf=0.5,
            black_sigatoka=0.5,
            yellow_sigatoka=0.5,
            min_leaf_pct=3.0,
            min_lesion_pct=0.05,
            min_disease_pct=0.5,
            tta_hflip=True,
        ),
    )


def build_fake_tree() -> FakePredictor:
    return FakePredictor("tree_classification", version="tree_fake_v1")


def build_fake_leaf_seg() -> FakePredictor:
    # A segmentation model also produces an overlay image (stored as result.png).
    return FakePredictor(
        "leaf_segmentation",
        version="leaf_seg_fake_v1",
        task="segment",
        details=make_leaf_seg_details(),
        label="affected",
        display_label="Affected leaf",
        confidence=None,
        classes=["leaf", "affected"],
        result_image=Image.new("RGB", (40, 30), (10, 200, 10)),
    )


def build_fake_leaf_disease() -> FakePredictor:
    # A segmentation model also produces an overlay image (stored as result.png).
    return FakePredictor(
        "leaf_disease",
        version="leaf_disease_fake_v1",
        task="segment",
        details=make_leaf_disease_details(),
        label="black_sigatoka",
        display_label="Black Sigatoka detected",
        confidence=None,
        classes=["leaf", "black_sigatoka", "yellow_sigatoka"],
        result_image=Image.new("RGB", (40, 30), (200, 150, 10)),
    )


@pytest.fixture
def fake_tree() -> FakePredictor:
    return build_fake_tree()


@pytest.fixture
def fake_leaf_seg() -> FakePredictor:
    return build_fake_leaf_seg()


@pytest.fixture
def fake_leaf_disease() -> FakePredictor:
    return build_fake_leaf_disease()


@pytest.fixture
def fake_registry(
    fake_tree: FakePredictor, fake_leaf_seg: FakePredictor, fake_leaf_disease: FakePredictor
) -> ModelRegistry:
    return ModelRegistry.from_predictors([fake_tree, fake_leaf_seg, fake_leaf_disease])


@pytest.fixture(autouse=True)
def _lifespan_never_loads_real_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Safety net for tests that call `create_app()` without a registry: the lifespan
    would otherwise load the real weights (and torch) via `build_registry`. A test
    that wants the real builder patches `app.main.build_registry` itself."""
    monkeypatch.setattr(
        app_main,
        "build_registry",
        lambda settings: ModelRegistry.from_predictors([build_fake_tree(), build_fake_leaf_seg()]),
    )


@pytest.fixture(autouse=True)
def _fresh_rate_limiter() -> Iterator[None]:
    # The limiter keeps its counters in module-level memory, i.e. across tests.
    limiter.reset()
    yield
    limiter.reset()


@pytest.fixture
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A private DATA_ROOT per test, applied through the environment Settings reads."""
    root = tmp_path / "data"
    monkeypatch.setenv("DATA_ROOT", str(root))
    return root


@pytest.fixture
def app(sqlite_engine: Engine, fake_registry: ModelRegistry, data_root: Path) -> Iterator[FastAPI]:
    # The injected SQLite engine keeps the suite hermetic: the lifespan's DB
    # ping hits a temp file, never the developer's real SQL Server. Fake models
    # and a tmp DATA_ROOT do the same for inference and image storage.
    get_settings.cache_clear()
    application = create_app(engine=sqlite_engine, registry=fake_registry)
    yield application
    get_settings.cache_clear()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client
