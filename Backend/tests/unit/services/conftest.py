from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.ml.registry import ModelRegistry
from app.services.analysis_service import AnalysisService
from app.storage.local import LocalImageStorage
from tests.fakes import FakePredictor
from tests.fixtures.settings import make_settings


@pytest.fixture
def settings() -> Settings:
    return make_settings()


@pytest.fixture
def storage(tmp_path: Path) -> LocalImageStorage:
    return LocalImageStorage(tmp_path / "data")


@pytest.fixture
def predictor() -> FakePredictor:
    return FakePredictor("tree_classification", version="tree_fake_v1")


@pytest.fixture
def make_service(
    db_session: Session, storage: LocalImageStorage, settings: Settings, predictor: FakePredictor
) -> Callable[..., AnalysisService]:
    """Builds a service around the shared session/storage; pass ``registry=`` to swap models."""

    def _make(registry: ModelRegistry | None = None) -> AnalysisService:
        return AnalysisService(
            db_session, registry or ModelRegistry.from_predictors([predictor]), storage, settings
        )

    return _make


@pytest.fixture
def service(make_service: Callable[..., AnalysisService]) -> AnalysisService:
    return make_service()


class ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def logs() -> Iterator[ListHandler]:
    handler = ListHandler()
    logger = logging.getLogger("app.services.analysis_service")
    previous = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    yield handler
    logger.removeHandler(handler)
    logger.setLevel(previous)
