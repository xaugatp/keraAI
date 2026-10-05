"""Model 1 against REAL weights (`pytest -m model`; deselected by default).

Works with either the dummy checkpoint from `python -m scripts.make_dummy_weights`
or the real trained one: it checks the CONTRACT (loads, task/classes, valid
output, no network), never specific predictions.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from PIL import Image

from app.core.errors import ModelUnavailableError
from app.ml.base import ModelLoadError
from app.ml.registry import build_registry
from app.ml.tree_classifier import BACKEND_DIR, TreeClassifier, resolve_weights_path
from app.schemas.tree import TreeDetails
from tests.fixtures.image_factory import gradient_image
from tests.fixtures.settings import make_settings

pytestmark = pytest.mark.model

WEIGHTS = resolve_weights_path("weights/tree_cls_v1.pt")
MISSING_REASON = "tree_cls_v1.pt missing - run `python -m scripts.make_dummy_weights`"

needs_weights = pytest.mark.skipif(not WEIGHTS.is_file(), reason=MISSING_REASON)


@pytest.fixture(scope="module")
def settings():
    return make_settings(tree_weights_path=str(WEIGHTS), leaf_seg_enabled=False)


@pytest.fixture(scope="module")
def predictor(settings) -> TreeClassifier:
    """One loaded + warmed-up classifier shared by the module (loading costs seconds)."""
    pytest.importorskip("ultralytics")
    classifier = TreeClassifier(settings)
    classifier.load()
    classifier.mark_ready()
    classifier.warmup()
    return classifier


def _save_checkpoint(path: Path, *, yaml: str, names: dict[int, str] | None) -> None:
    """Write a random-init Ultralytics checkpoint (no network, no pretrained weights)."""
    from ultralytics import YOLO
    from ultralytics.nn.tasks import ClassificationModel

    wrapper = YOLO(yaml)
    if names is not None:  # classifier with custom class names
        network = ClassificationModel(wrapper.model.yaml, nc=len(names), verbose=False)
        network.names = dict(names)
        network.task = "classify"
        wrapper.model = network
    wrapper.save(path)


@needs_weights
class TestRealWeights:
    def test_loads_as_a_classifier_with_the_configured_classes(self, predictor, settings):
        assert predictor.task == "classify"
        assert len(predictor.classes) >= 2
        assert settings.tree.positive_class in predictor.classes
        assert set(settings.tree.display_names) <= set(predictor.classes)

    def test_not_ready_until_the_registry_says_so(self, settings):
        classifier = TreeClassifier(settings)
        classifier.load()
        assert classifier.is_ready is False  # load() alone never marks ready
        with pytest.raises(ModelUnavailableError):
            classifier.predict(gradient_image())

    def test_predict_returns_a_valid_tree_details(self, predictor):
        output = predictor.predict(gradient_image(320, 240))

        assert isinstance(output.details, TreeDetails)
        probs = [p.probability for p in output.details.probabilities]
        assert len(probs) == len(predictor.classes)
        assert probs == sorted(probs, reverse=True)
        assert math.isclose(sum(probs), 1.0, abs_tol=len(probs) * 5e-5)
        assert output.label == output.details.probabilities[0].class_key
        assert output.confidence is not None and 0.0 <= output.confidence <= 1.0
        assert output.is_uncertain == (output.confidence < predictor._cfg.uncertain_threshold)
        assert output.details.verdict in {"banana_tree", "not_banana_tree", "uncertain"}
        if output.is_uncertain:
            assert output.display_label == "Uncertain"
        assert output.timings.inference_ms >= 0

    def test_prediction_is_deterministic(self, predictor):
        picture = gradient_image(300, 300)
        first = predictor.predict(picture).details.model_dump()
        second = predictor.predict(picture).details.model_dump()
        assert first == second

    def test_accepts_non_square_and_tiny_inputs(self, predictor):
        for size in [(64, 64), (1200, 300), (300, 1200)]:
            assert predictor.predict(Image.new("RGB", size, (30, 120, 40))).details is not None

    def test_works_through_the_registry(self, settings):
        registry = build_registry(settings)
        assert registry.readiness()[0] is True
        info = registry.all_info()[0]
        assert (info.key, info.status, info.task) == ("tree_classification", "ready", "classify")
        assert registry.get("tree_classification").predict(gradient_image()).label is not None


class TestRejectedWeights:
    def test_a_detection_checkpoint_is_rejected(self, tmp_path, settings):
        pytest.importorskip("ultralytics")
        detect = tmp_path / "detect.pt"
        _save_checkpoint(detect, yaml="yolov8n.yaml", names=None)  # task=detect, 80 classes

        classifier = TreeClassifier(make_settings(tree_weights_path=str(detect)))
        with pytest.raises(ModelLoadError, match="expected classify, got detect"):
            classifier.load()
        assert classifier.is_ready is False

    def test_class_names_missing_from_the_model_are_rejected(self, tmp_path):
        pytest.importorskip("ultralytics")
        wrong = tmp_path / "cats.pt"
        _save_checkpoint(wrong, yaml="yolov8n-cls.yaml", names={0: "cat", 1: "dog"})

        classifier = TreeClassifier(make_settings(tree_weights_path=str(wrong)))
        with pytest.raises(ModelLoadError) as raised:
            classifier.load()
        message = str(raised.value)
        assert "['cat', 'dog']" in message  # what the model has
        assert "banana_tree" in message and "non_banana" in message  # what the config wants

    def test_a_single_class_model_is_rejected(self, tmp_path):
        pytest.importorskip("ultralytics")
        single = tmp_path / "one.pt"
        _save_checkpoint(single, yaml="yolov8n-cls.yaml", names={0: "banana_tree"})
        classifier = TreeClassifier(
            make_settings(tree_weights_path=str(single), tree_display_names={"banana_tree": "x"})
        )
        with pytest.raises(ModelLoadError, match="at least 2 classes"):
            classifier.load()

    def test_a_corrupt_zip_is_reported_as_a_load_error_not_a_crash(self, tmp_path):
        pytest.importorskip("ultralytics")
        import zipfile

        broken = tmp_path / "broken.pt"
        with zipfile.ZipFile(broken, "w") as archive:
            archive.writestr("broken/data.pkl", b"not a pickle")
        with pytest.raises(ModelLoadError, match="Ultralytics could not load"):
            TreeClassifier(make_settings(tree_weights_path=str(broken))).load()


_NETWORK_GUARD = textwrap.dedent(
    """
    import socket, sys

    attempts = []

    def deny(*args, **kwargs):
        attempts.append(repr(args)[:80])
        raise OSError("network disabled for this test")

    # Everything that can leave the machine goes through one of these.
    socket.getaddrinfo = deny
    socket.create_connection = deny
    socket.socket.connect = deny
    socket.socket.connect_ex = deny

    mode, weights = sys.argv[1], sys.argv[2]
    if mode == "raw":
        # Control run: plain `import ultralytics` (no offline switch) DOES try to
        # resolve hostnames, which proves the guard above would catch an app that
        # forgot to go offline.
        import ultralytics  # noqa: F401
    else:
        from PIL import Image
        from app.ml.registry import build_registry
        from tests.fixtures.settings import make_settings

        registry = build_registry(
            make_settings(tree_weights_path=weights, leaf_seg_enabled=False)
        )
        output = registry.get("tree_classification").predict(Image.new("RGB", (256, 256), (9, 99, 9)))
        assert output.label
        import ultralytics.utils as u
        assert u.ONLINE is False, "ultralytics believes it is online"

    print("ATTEMPTS=" + str(len(attempts)))
    """
)


def _run_guard(mode: str) -> subprocess.CompletedProcess[str]:
    # Fresh interpreter: ultralytics decides "online or not" once, at import time,
    # so this process (which may have imported it already) cannot test that.
    # YOLO_* variables are stripped: an earlier in-process load() sets them, and
    # inheriting them would hide whether the app itself turns the switch on.
    env = {k: v for k, v in os.environ.items() if not k.startswith("YOLO_")}
    return subprocess.run(
        [sys.executable, "-c", _NETWORK_GUARD, mode, str(WEIGHTS)],
        cwd=BACKEND_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


@needs_weights
class TestNoNetwork:
    def test_load_warmup_and_predict_make_no_network_calls(self):
        result = _run_guard("app")
        assert result.returncode == 0, result.stdout + result.stderr
        assert "ATTEMPTS=0" in result.stdout

    def test_the_guard_would_catch_an_online_ultralytics(self):
        result = _run_guard("raw")
        assert result.returncode == 0, result.stdout + result.stderr
        attempts = int(result.stdout.rsplit("ATTEMPTS=", 1)[1].split()[0])
        assert attempts > 0, "ultralytics no longer probes the network on import; revisit this test"
