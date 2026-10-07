"""ModelRegistry: degraded mode, status reporting, readiness (spec §1, §8)."""

from __future__ import annotations

import asyncio
import logging

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.ml.registry as registry_module
from app.core.errors import InferenceError, ModelUnavailableError
from app.ml.base import ModelLoadError, Predictor
from app.ml.registry import (
    MODEL_SPECS,
    PLANNED_MODEL_INFOS,
    ModelRegistry,
    build_registry,
    import_factory,
    redact_paths,
)
from tests.fakes import FakePredictor
from tests.fixtures.settings import make_settings

TREE = "tree_classification"
LEAF = "leaf_segmentation"
LEAF_DISEASE = "leaf_disease"
WINDOWS_PATH = r"C:\Users\someone\KeraAI\Backend\weights\tree_cls_v1.pt"


def image() -> Image.Image:
    return Image.new("RGB", (64, 64), (1, 2, 3))


class Factories(dict):
    """`factories[key] = callable(settings) -> Predictor | Predictor | Exception`.

    Leave a key out to get the missing-module (ImportError) path. `.built` records
    which keys were constructed, in order.
    """

    def __init__(self) -> None:
        super().__init__()
        self.built: list[str] = []


@pytest.fixture
def factories(monkeypatch) -> Factories:
    """Replace the lazily imported predictor classes with test-controlled factories."""
    table = Factories()
    built = table.built
    by_import_string = {spec.factory: spec.key for spec in MODEL_SPECS}

    def fake_import_factory(path: str):
        key = by_import_string[path]
        if key not in table:
            raise ModuleNotFoundError(f"No module named 'fake_{key}'")

        def construct(settings):
            built.append(key)
            result = table[key]
            if isinstance(result, Exception):
                raise result
            return result(settings) if callable(result) else result

        return construct

    monkeypatch.setattr(registry_module, "import_factory", fake_import_factory)
    return table


def good(key: str, **kwargs):
    return lambda settings: FakePredictor(key, ready=False, **kwargs)


class TestBuildRegistry:
    def test_ready_models_are_served(self, factories):
        factories[TREE] = good(TREE)
        factories[LEAF] = good(LEAF, task="segment")
        factories[LEAF_DISEASE] = good(LEAF_DISEASE, task="segment")
        registry = build_registry(make_settings())

        assert registry.get(TREE).is_ready
        assert registry.get(LEAF).is_ready
        assert registry.get(LEAF_DISEASE).is_ready
        assert registry.readiness() == (True, None)
        statuses = {info.key: info.status for info in registry.all_info()}
        assert statuses == {TREE: "ready", LEAF: "ready", LEAF_DISEASE: "ready"}

    def test_each_predictor_is_loaded_then_warmed_up(self, factories):
        tree = FakePredictor(TREE, ready=False)
        factories[TREE] = tree
        factories[LEAF] = good(LEAF)
        factories[LEAF_DISEASE] = good(LEAF_DISEASE)
        build_registry(make_settings())
        assert tree.load_calls == 1
        assert tree.warmup_calls == 1
        assert tree.is_ready

    def test_disabled_model_is_not_constructed_or_loaded(self, factories):
        factories[TREE] = good(TREE)
        factories[LEAF] = good(LEAF)
        factories[LEAF_DISEASE] = good(LEAF_DISEASE)
        registry = build_registry(make_settings(tree_enabled=False))

        assert factories.built == [LEAF, LEAF_DISEASE]
        with pytest.raises(ModelUnavailableError):
            registry.get(TREE)
        info = next(i for i in registry.all_info() if i.key == TREE)
        assert info.status == "unavailable"
        assert info.reason == "Disabled by configuration"
        # A model that is switched off does not make the app "not ready".
        assert registry.readiness() == (True, None)

    def test_disabled_info_carries_configured_version_and_placeholder_flag(self, factories):
        registry = build_registry(
            make_settings(
                tree_enabled=False,
                leaf_seg_enabled=False,
                leaf_disease_enabled=False,
                tree_model_version="tree_cls_dummy_v0",
                tree_is_placeholder=True,
            )
        )
        info = next(i for i in registry.all_info() if i.key == TREE)
        assert info.version == "tree_cls_dummy_v0"
        assert info.is_placeholder is True
        assert info.task == "classify"

    def test_everything_disabled_still_builds_and_is_ready(self, factories):
        registry = build_registry(
            make_settings(tree_enabled=False, leaf_seg_enabled=False, leaf_disease_enabled=False)
        )
        assert factories.built == []
        assert registry.readiness() == (True, None)

    def test_load_error_marks_only_that_model_unavailable(self, factories, caplog):
        load_error = ModelLoadError(f"Weights file not found: '{WINDOWS_PATH}'.")
        factories[TREE] = FakePredictor(TREE, ready=False, load_error=load_error)
        factories[LEAF] = good(LEAF)
        with caplog.at_level(logging.ERROR, logger="app.ml.registry"):
            registry = build_registry(make_settings())  # must not raise

        with pytest.raises(ModelUnavailableError):
            registry.get(TREE)
        assert registry.get(LEAF).is_ready  # the other model is unaffected

        info = next(i for i in registry.all_info() if i.key == TREE)
        assert info.status == "unavailable"
        assert "Weights file not found" in (info.reason or "")
        # Client-visible reason has no filesystem path...
        assert "Users" not in (info.reason or "") and "C:" not in (info.reason or "")
        assert "tree_cls_v1.pt" in (info.reason or "")
        # ...while the log keeps the exact one for the operator.
        assert WINDOWS_PATH in caplog.text

        ok, detail = registry.readiness()
        assert ok is False
        assert detail is not None and TREE in detail and "Weights file not found" in detail

    def test_generic_exception_in_load_marks_unavailable(self, factories):
        factories[TREE] = lambda s: FakePredictor(
            TREE, ready=False, load_error=RuntimeError("kaboom")
        )
        factories[LEAF] = good(LEAF)
        registry = build_registry(make_settings())
        info = next(i for i in registry.all_info() if i.key == TREE)
        assert info.status == "unavailable"
        assert "Unexpected error while loading: RuntimeError: kaboom" in (info.reason or "")

    def test_failing_warmup_marks_unavailable_with_the_reason(self, factories):
        broken = FakePredictor(TREE, ready=False, warmup_error=InferenceError("shape mismatch"))
        factories[TREE] = broken
        factories[LEAF] = good(LEAF)
        registry = build_registry(make_settings())

        assert broken.load_calls == 1  # it did load; the warm-up is what failed
        assert broken.is_ready is False  # reverted from the transient ready state
        info = next(i for i in registry.all_info() if i.key == TREE)
        assert "Warm-up inference failed: InferenceError: shape mismatch" in (info.reason or "")
        with pytest.raises(ModelUnavailableError):
            registry.get(TREE)

    def test_unexpected_warmup_exception_is_contained(self, factories):
        factories[TREE] = FakePredictor(TREE, ready=False, warmup_error=MemoryError("oom"))
        factories[LEAF] = good(LEAF)
        registry = build_registry(make_settings())
        assert registry.readiness()[0] is False

    def test_missing_model_module_marks_unavailable(self, factories):
        factories[TREE] = good(TREE)  # LEAF has no factory -> ModuleNotFoundError
        registry = build_registry(make_settings())
        assert registry.get(TREE).is_ready
        info = next(i for i in registry.all_info() if i.key == LEAF)
        assert info.status == "unavailable"
        assert "Model code failed to load: ModuleNotFoundError" in (info.reason or "")
        assert info.task == "segment"  # static fallback metadata
        assert registry.readiness()[0] is False

    def test_constructor_failure_marks_unavailable(self, factories):
        factories[TREE] = TypeError("__init__() got an unexpected keyword argument")
        factories[LEAF] = good(LEAF)
        registry = build_registry(make_settings())
        info = next(i for i in registry.all_info() if i.key == TREE)
        assert "TypeError" in (info.reason or "")
        assert registry.get(LEAF).is_ready

    def test_every_model_failing_still_builds(self, factories):
        registry = build_registry(make_settings())  # no factories at all
        ok, detail = registry.readiness()
        assert ok is False
        assert detail is not None and TREE in detail and LEAF in detail
        for key in (TREE, LEAF, LEAF_DISEASE):
            with pytest.raises(ModelUnavailableError):
                registry.get(key)

    def test_only_restricts_what_is_loaded(self, factories):
        factories[TREE] = good(TREE)
        factories[LEAF] = good(LEAF)
        registry = build_registry(make_settings(), only=[TREE])
        assert factories.built == [TREE]
        info = next(i for i in registry.all_info() if i.key == LEAF)
        assert info.reason == "Not loaded (not requested)"
        assert registry.readiness() == (True, None)

    def test_real_specs_point_at_the_expected_classes(self):
        assert {s.key: s.factory for s in MODEL_SPECS} == {
            TREE: "app.ml.tree_classifier:TreeClassifier",
            LEAF: "app.ml.leaf_segmenter:LeafSegmenter",
            LEAF_DISEASE: "app.ml.leaf_disease_identifier:LeafDiseaseIdentifier",
        }

    def test_import_factory_resolves_module_class_strings(self):
        from app.ml.tree_classifier import TreeClassifier

        assert import_factory("app.ml.tree_classifier:TreeClassifier") is TreeClassifier

    def test_import_factory_propagates_import_errors(self):
        with pytest.raises(ImportError):
            import_factory("app.ml.does_not_exist:Nope")
        with pytest.raises(AttributeError):
            import_factory("app.ml.tree_classifier:Nope")

    def test_real_tree_classifier_without_weights_degrades_instead_of_crashing(self, tmp_path):
        settings = make_settings(
            tree_weights_path=str(tmp_path / "missing.pt"), leaf_seg_enabled=False
        )
        registry = build_registry(settings)  # real TreeClassifier, no torch needed to fail
        info = next(i for i in registry.all_info() if i.key == TREE)
        assert info.status == "unavailable"
        assert "Weights file not found" in (info.reason or "")
        assert str(tmp_path) not in (info.reason or "")
        assert "missing.pt" in (info.reason or "")


class TestGet:
    def test_unknown_key_is_model_unavailable(self):
        registry = ModelRegistry.from_predictors([FakePredictor(TREE)])
        with pytest.raises(ModelUnavailableError) as raised:
            registry.get("nonexistent")
        assert raised.value.status_code == 503
        assert raised.value.code == "MODEL_UNAVAILABLE"

    def test_not_ready_predictor_is_model_unavailable_without_leaking_the_reason(self):
        fake = FakePredictor(TREE, ready=False)
        fake.mark_unavailable(f"weights at {WINDOWS_PATH} are corrupt")
        registry = ModelRegistry.from_predictors([fake])
        with pytest.raises(ModelUnavailableError) as raised:
            registry.get(TREE)
        assert "corrupt" not in raised.value.detail
        assert "C:" not in raised.value.detail

    def test_ready_predictor_is_returned_and_usable(self):
        fake = FakePredictor(TREE)
        registry = ModelRegistry.from_predictors([fake])
        predictor = registry.get(TREE)
        assert predictor is fake
        assert predictor.predict(image()).label == "banana_tree"

    def test_a_model_that_goes_down_later_is_reported_unavailable(self):
        fake = FakePredictor(TREE)
        registry = ModelRegistry.from_predictors([fake])
        fake.mark_unavailable("GPU lost")
        with pytest.raises(ModelUnavailableError):
            registry.get(TREE)
        assert registry.readiness()[0] is False


class TestAllInfo:
    def test_planned_model_infos_is_empty_now_every_claude_md_model_is_built(self):
        # All three CLAUDE.md models now have a predictor + settings (ADR 0019);
        # this tuple stays as the place a genuinely new, unbuilt model would go.
        assert PLANNED_MODEL_INFOS == ()

    def test_registry_order_matches_registration_order(self):
        registry = ModelRegistry.from_predictors([FakePredictor(TREE), FakePredictor(LEAF)])
        assert [info.key for info in registry.all_info()] == [TREE, LEAF]
        assert registry.readiness() == (True, None)

    def test_ready_info_exposes_classes_and_version(self):
        registry = ModelRegistry.from_predictors([FakePredictor(TREE, version="tree_cls_v1")])
        info = registry.all_info()[0]
        assert (info.status, info.version, info.classes) == (
            "ready",
            "tree_cls_v1",
            ["banana_tree", "non_banana"],
        )

    def test_empty_registry_lists_nothing(self):
        assert ModelRegistry.from_predictors([]).all_info() == []
        assert ModelRegistry.from_predictors([]).readiness() == (True, None)

    def test_registering_the_same_key_twice_replaces_the_predictor(self):
        first, second = FakePredictor(TREE, version="a"), FakePredictor(TREE, version="b")
        registry = ModelRegistry.from_predictors([first, second])
        assert registry.get(TREE) is second
        assert [i.key for i in registry.all_info()].count(TREE) == 1


class TestReadiness:
    def test_all_ready(self):
        registry = ModelRegistry.from_predictors([FakePredictor(TREE), FakePredictor(LEAF)])
        assert registry.readiness() == (True, None)

    def test_one_down_makes_it_not_ready_and_names_it(self):
        down = FakePredictor(LEAF, ready=False)
        down.mark_unavailable("Weights file not found: 'leaf_seg_v1.pt'.")
        registry = ModelRegistry.from_predictors([FakePredictor(TREE), down])
        ok, detail = registry.readiness()
        assert ok is False
        assert (
            detail
            == "unavailable models — leaf_segmentation: Weights file not found: 'leaf_seg_v1.pt'."
        )

    def test_placeholder_hint_when_a_ready_model_is_a_placeholder(self):
        registry = ModelRegistry.from_predictors([FakePredictor(TREE, is_placeholder=True)])
        ok, detail = registry.readiness()
        assert ok is True  # placeholder weights are a warning, not an outage
        assert detail is not None
        assert "placeholder weights in use (results are not real)" in detail
        assert TREE in detail

    def test_problems_and_placeholders_are_both_reported(self):
        down = FakePredictor(LEAF, ready=False)
        registry = ModelRegistry.from_predictors([FakePredictor(TREE, is_placeholder=True), down])
        ok, detail = registry.readiness()
        assert ok is False
        assert detail is not None
        assert "unavailable models" in detail and "placeholder" in detail

    def test_unavailable_placeholder_is_only_reported_as_unavailable(self):
        down = FakePredictor(TREE, ready=False, is_placeholder=True)
        ok, detail = ModelRegistry.from_predictors([down]).readiness()
        assert ok is False
        assert detail is not None and "placeholder" not in detail

    def test_async_check_matches_the_health_contract(self):
        registry = ModelRegistry.from_predictors([FakePredictor(TREE)])
        assert asyncio.run(registry.readiness_check()) == (True, None)

    def test_registered_on_the_real_health_endpoint(self, client: TestClient):
        down = FakePredictor(LEAF, ready=False)
        down.mark_unavailable("no weights")
        registry = ModelRegistry.from_predictors([FakePredictor(TREE, is_placeholder=True), down])
        client.app.state.readiness_checks["models"] = registry.readiness_check  # type: ignore[attr-defined]

        response = client.get("/health/ready")

        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "degraded"
        component = body["components"]["models"]
        assert component["status"] == "unavailable"
        assert "leaf_segmentation: no weights" in component["detail"]

    def test_healthy_models_component_on_the_real_health_endpoint(self, client: TestClient):
        registry = ModelRegistry.from_predictors([FakePredictor(TREE)])
        client.app.state.readiness_checks["models"] = registry.readiness_check  # type: ignore[attr-defined]
        response = client.get("/health/ready")
        assert response.json()["components"]["models"] == {"status": "ok", "detail": None}


class TestRedactPaths:
    def test_quoted_windows_path_keeps_only_the_file_name(self):
        assert redact_paths(f"Weights file not found: '{WINDOWS_PATH}'. Place it there.") == (
            "Weights file not found: 'tree_cls_v1.pt'. Place it there."
        )

    def test_quoted_windows_path_with_spaces(self):
        text = r"Weights file not found: 'C:\Users\Jane Doe\My Models\tree.pt'."
        assert redact_paths(text) == "Weights file not found: 'tree.pt'."

    def test_forward_slash_and_posix_paths(self):
        assert redact_paths("bad '/srv/kera/weights/tree.pt' here") == "bad 'tree.pt' here"
        assert redact_paths("bad 'C:/kera/weights/tree.pt' here") == "bad 'tree.pt' here"

    def test_repr_style_doubled_backslashes(self):
        assert redact_paths(repr(WINDOWS_PATH)) == "'tree_cls_v1.pt'"

    def test_bare_windows_path_and_trailing_punctuation(self):
        assert redact_paths(f"cannot read {WINDOWS_PATH}.") == "cannot read tree_cls_v1.pt."
        assert redact_paths(f"cannot read ({WINDOWS_PATH})") == "cannot read (tree_cls_v1.pt)"

    def test_several_paths_in_one_message(self):
        text = rf"'{WINDOWS_PATH}' and 'D:\other\x.pt' failed"
        assert redact_paths(text) == "'tree_cls_v1.pt' and 'x.pt' failed"

    def test_text_without_paths_is_unchanged(self):
        text = "expected classify, got detect — wrong weights? (task='detect', nc=80)"
        assert redact_paths(text) == text

    def test_non_path_quotes_are_untouched(self):
        assert redact_paths("Model 'tree_classification' is not available.") == (
            "Model 'tree_classification' is not available."
        )


def test_all_predictors_in_the_registry_are_predictors():
    # from_predictors takes any Predictor subclass; FakePredictor is one.
    assert isinstance(FakePredictor(), Predictor)
