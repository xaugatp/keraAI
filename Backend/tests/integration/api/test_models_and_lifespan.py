"""GET /models, the registry's readiness check, and degraded-mode startup."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine

from app import main as app_main
from app.core.config import get_settings
from app.main import create_app
from app.ml.registry import ModelRegistry, build_registry
from tests.fakes import FakePredictor
from tests.integration.api.conftest import MakeClient
from tests.integration.api.helpers import (
    LEAF_URL,
    PREFIX,
    TREE_URL,
    assert_problem,
    post_image,
)

MODELS_URL = f"{PREFIX}/models"


def test_models_lists_every_model_with_is_placeholder(api: TestClient) -> None:
    response = api.get(MODELS_URL)  # no X-Client-Id needed

    assert response.status_code == 200
    models = {m["key"]: m for m in response.json()}
    assert set(models) == {"tree_classification", "leaf_segmentation", "leaf_disease"}

    tree = models["tree_classification"]
    assert set(tree) == {
        "key",
        "display_name",
        "version",
        "task",
        "classes",
        "status",
        "reason",
        "is_placeholder",
    }
    assert tree["status"] == "ready" and tree["reason"] is None
    assert tree["version"] == "tree_fake_v1"
    assert tree["classes"] == ["banana_tree", "non_banana"]
    assert tree["is_placeholder"] is False

    leaf_disease = models["leaf_disease"]  # built (ADR 0019), served by the default fake registry
    assert leaf_disease["status"] == "ready" and leaf_disease["reason"] is None
    assert leaf_disease["version"] == "leaf_disease_fake_v1"
    assert leaf_disease["classes"] == ["leaf", "black_sigatoka", "yellow_sigatoka"]


def test_models_reports_placeholder_weights(make_client: MakeClient) -> None:
    registry = ModelRegistry.from_predictors(
        [FakePredictor("tree_classification", is_placeholder=True)]
    )
    client = make_client(registry=registry)

    models = {m["key"]: m for m in client.get(MODELS_URL).json()}
    assert models["tree_classification"]["is_placeholder"] is True


def test_models_shows_an_unavailable_model_with_its_reason(make_client: MakeClient) -> None:
    down = FakePredictor("tree_classification", ready=False)
    down.mark_unavailable("weights file not found")
    client = make_client(registry=ModelRegistry.from_predictors([down]))

    tree = next(m for m in client.get(MODELS_URL).json() if m["key"] == "tree_classification")
    assert tree["status"] == "unavailable"
    assert tree["reason"] == "weights file not found"


def test_readiness_includes_the_models_check(api: TestClient) -> None:
    response = api.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["components"]["models"] == {"status": "ok", "detail": None}


def test_readiness_goes_503_when_an_enabled_model_is_down(make_client: MakeClient) -> None:
    down = FakePredictor("tree_classification", ready=False)
    down.mark_unavailable("weights file not found")
    client = make_client(registry=ModelRegistry.from_predictors([down]))

    response = client.get("/health/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["components"]["models"]["status"] == "unavailable"
    assert "weights file not found" in body["components"]["models"]["detail"]
    assert body["components"]["database"]["status"] == "ok"
    assert client.get("/health").status_code == 200  # liveness is unaffected


# --- degraded mode with the REAL registry builder (no torch needed: the files are missing) ----


def test_app_starts_without_weights_and_reports_why(
    sqlite_engine: Engine,
    client_id: uuid.UUID,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    data_root: Path,
) -> None:
    missing_tree = tmp_path / "nowhere" / "tree_cls_v1.pt"
    missing_leaf = tmp_path / "nowhere" / "leaf_seg_v1.pt"
    # Undo the suite-wide safety net: this test wants the real builder, which finds no
    # weights file and so never imports torch/ultralytics.
    monkeypatch.setattr(app_main, "build_registry", build_registry)
    monkeypatch.setenv("TREE_WEIGHTS_PATH", str(missing_tree))
    monkeypatch.setenv("LEAF_SEG_WEIGHTS_PATH", str(missing_leaf))
    get_settings.cache_clear()

    application = create_app(engine=sqlite_engine)
    with TestClient(application, raise_server_exceptions=False) as client:  # must not raise
        # Liveness is independent of the models ...
        assert client.get("/health").status_code == 200

        # ... readiness explains what is wrong, without leaking file paths.
        ready = client.get("/health/ready")
        assert ready.status_code == 503
        detail = ready.json()["components"]["models"]["detail"]
        assert "tree_classification" in detail and "leaf_segmentation" in detail
        assert "nowhere" not in ready.text and str(tmp_path) not in ready.text
        assert ready.json()["components"]["database"]["status"] == "ok"

        # GET /models says the same thing per model.
        models = {m["key"]: m for m in client.get(MODELS_URL).json()}
        for key in ("tree_classification", "leaf_segmentation"):
            assert models[key]["status"] == "unavailable"
            assert models[key]["reason"]
            assert "nowhere" not in models[key]["reason"]

        # And predict is a clean 503, not a crash.
        for url in (TREE_URL, LEAF_URL):
            assert_problem(post_image(client, url, client_id), 503, "MODEL_UNAVAILABLE")


def test_registry_is_built_in_the_threadpool_not_on_the_event_loop(
    sqlite_engine: Engine, monkeypatch: pytest.MonkeyPatch, data_root: Path
) -> None:
    seen: list[bool] = []

    def spy_build(settings: object) -> ModelRegistry:
        try:
            asyncio.get_running_loop()
            seen.append(True)  # on the event loop: bad
        except RuntimeError:
            seen.append(False)  # in a worker thread: good
        return ModelRegistry.from_predictors([FakePredictor("tree_classification")])

    monkeypatch.setattr(app_main, "build_registry", spy_build)
    get_settings.cache_clear()
    with TestClient(create_app(engine=sqlite_engine)):
        pass
    assert seen == [False]


def test_injected_registry_is_used_as_is_and_never_rebuilt(
    sqlite_engine: Engine, monkeypatch: pytest.MonkeyPatch, data_root: Path
) -> None:
    def explode(settings: object) -> ModelRegistry:
        raise AssertionError("build_registry must not run when a registry is injected")

    monkeypatch.setattr(app_main, "build_registry", explode)
    injected = ModelRegistry.from_predictors([FakePredictor("tree_classification")])
    get_settings.cache_clear()
    application = create_app(engine=sqlite_engine, registry=injected)
    with TestClient(application):
        assert application.state.registry is injected
