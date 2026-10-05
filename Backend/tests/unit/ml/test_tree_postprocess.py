"""Tree postprocess + weights-file validation. Torch-free (fast run)."""

from __future__ import annotations

import math
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from PIL import Image

import app.ml.tree_classifier as tree_module
from app.core.errors import InferenceError, ModelUnavailableError
from app.ml.base import ModelLoadError
from app.ml.tree_classifier import (
    BACKEND_DIR,
    TreeClassifier,
    TreeRawOutput,
    build_tree_output,
    configure_ultralytics_environment,
    extracted_checkpoint_hint,
    resolve_weights_path,
    stray_coco_folder_hint,
    validate_weights_file,
)
from app.schemas.tree import TreeDetails, TreeProbability
from tests.fixtures.settings import make_settings

NAMES = ("banana_tree", "non_banana")
DISPLAY = {"banana_tree": "Banana tree", "non_banana": "Not a banana tree"}


def run(probs, names=NAMES, *, threshold=0.6, positive="banana_tree", display=DISPLAY):
    return build_tree_output(
        probs,
        names,
        positive_class=positive,
        uncertain_threshold=threshold,
        display_names=display,
    )


class TestThreshold:
    def test_below_threshold_is_uncertain(self):
        out = run([0.59, 0.41])
        assert out.is_uncertain is True
        assert out.details.verdict == "uncertain"
        assert out.display_label == "Uncertain"
        # The raw class is still reported so the UI can show "leaning banana".
        assert out.label == "banana_tree"

    def test_exactly_at_threshold_is_not_uncertain(self):
        out = run([0.6, 0.4])
        assert out.is_uncertain is False
        assert out.details.verdict == "banana_tree"
        assert out.display_label == "Banana tree"

    def test_above_threshold_is_confident(self):
        out = run([0.61, 0.39])
        assert out.is_uncertain is False
        assert out.confidence == pytest.approx(0.61)

    @pytest.mark.parametrize("top", [0.5, 0.59, 0.6, 0.61, 0.99])
    def test_is_uncertain_is_exactly_confidence_below_threshold(self, top):
        out = run([top, 1 - top])
        assert out.is_uncertain == (out.confidence < 0.6)

    def test_threshold_comes_from_the_argument_and_is_echoed(self):
        out = run([0.7, 0.3], threshold=0.8)
        assert out.is_uncertain is True
        assert out.details.threshold == 0.8


class TestVerdict:
    def test_positive_class_top_is_banana_tree(self):
        assert run([0.9, 0.1]).details.verdict == "banana_tree"

    def test_other_class_top_is_not_banana_tree(self):
        out = run([0.1, 0.9])
        assert out.details.verdict == "not_banana_tree"
        assert out.label == "non_banana"
        assert out.display_label == "Not a banana tree"
        assert out.confidence == pytest.approx(0.9)

    def test_positive_class_is_looked_up_by_name_not_position(self):
        out = run([0.2, 0.8], names=("non_banana", "banana_tree"))
        assert out.label == "banana_tree"
        assert out.details.verdict == "banana_tree"

    def test_unknown_positive_class_is_never_the_verdict(self):
        assert run([0.9, 0.1], positive="oak").details.verdict == "not_banana_tree"

    def test_n_class_input(self):
        names = ("weed", "banana_tree", "palm", "grass")
        out = run([0.05, 0.7, 0.15, 0.1], names=names, display={"banana_tree": "Banana tree"})
        assert out.details.verdict == "banana_tree"
        assert [p.class_key for p in out.details.probabilities] == [
            "banana_tree",
            "palm",
            "grass",
            "weed",
        ]

    def test_n_class_non_positive_top_with_missing_display_name_falls_back_to_key(self):
        out = run([0.1, 0.2, 0.7], names=("a", "banana_tree", "palm"), display={})
        assert out.details.verdict == "not_banana_tree"
        assert out.display_label == "palm"
        assert out.details.probabilities[0].display_name == "palm"

    def test_ties_resolve_to_the_lowest_class_index(self):
        assert run([0.5, 0.5], threshold=0.5).label == "banana_tree"
        out = run([0.5, 0.5], names=("non_banana", "banana_tree"), threshold=0.5)
        assert out.label == "non_banana"
        assert [p.class_key for p in out.details.probabilities] == ["non_banana", "banana_tree"]

    def test_three_way_tie_keeps_index_order(self):
        third = 1 / 3
        out = run([third, third, third], names=("x", "y", "z"), threshold=0.0)
        assert [p.class_key for p in out.details.probabilities] == ["x", "y", "z"]


class TestDistribution:
    def test_probabilities_sorted_descending(self):
        out = run([0.2, 0.8])
        values = [p.probability for p in out.details.probabilities]
        assert values == sorted(values, reverse=True)

    def test_rounded_to_four_decimals_and_still_sums_to_one(self):
        out = run([0.123456789, 0.876543211])
        probs = [p.probability for p in out.details.probabilities]
        assert all(round(p, 4) == p for p in probs)
        assert math.isclose(sum(probs), 1.0, abs_tol=len(probs) * 5e-5)

    def test_many_classes_rounding_error_stays_small(self):
        raw = [1 / 7] * 7
        out = run(raw, names=tuple(f"c{i}" for i in range(7)))
        assert math.isclose(
            sum(p.probability for p in out.details.probabilities), 1.0, abs_tol=7 * 5e-5
        )

    def test_tiny_float_drift_is_renormalised(self):
        out = run([0.6000004, 0.4000004])
        assert math.isclose(
            sum(p.probability for p in out.details.probabilities), 1.0, abs_tol=1e-4
        )

    def test_confidence_is_the_raw_top1_value(self):
        out = run([0.947312, 0.052688])
        assert out.confidence == 0.947312
        assert out.details.probabilities[0].probability == 0.9473

    def test_float32_style_values_work(self):
        # What `tensor.tolist()` gives for a float32 softmax.
        out = run([0.60000002384185791015625, 0.39999997615814208984375])
        assert out.is_uncertain is False

    def test_details_shape_matches_the_api_contract(self):
        out = run([0.947, 0.053])
        assert out.details.model_dump(mode="json") == {
            "kind": "tree_classification",
            "verdict": "banana_tree",
            "threshold": 0.6,
            "probabilities": [
                {"class_key": "banana_tree", "display_name": "Banana tree", "probability": 0.947},
                {
                    "class_key": "non_banana",
                    "display_name": "Not a banana tree",
                    "probability": 0.053,
                },
            ],
        }

    def test_timings_are_left_for_the_template_method(self):
        assert run([0.9, 0.1]).timings.model_total_ms == 0

    def test_no_result_image_for_a_classifier(self):
        assert run([0.9, 0.1]).result_image is None


class TestRejectsGarbage:
    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_non_finite_probabilities(self, bad):
        with pytest.raises(InferenceError, match="non-finite"):
            run([bad, 0.5])

    def test_negative_probability(self):
        with pytest.raises(InferenceError, match="outside"):
            run([1.2, -0.2])

    def test_probability_above_one(self):
        with pytest.raises(InferenceError, match="outside"):
            run([1.5, 0.0])

    def test_logits_instead_of_probabilities(self):
        with pytest.raises(InferenceError, match="not a probability distribution"):
            run([0.9, 0.9])

    def test_sum_far_from_one(self):
        with pytest.raises(InferenceError, match="sum="):
            run([0.2, 0.2])

    def test_empty(self):
        with pytest.raises(InferenceError):
            run([], names=())

    def test_length_mismatch(self):
        with pytest.raises(InferenceError, match="3 probabilities for 2 classes"):
            run([0.5, 0.3, 0.2])


class TestSchema:
    def test_probability_is_bounded(self):
        with pytest.raises(ValueError):
            TreeProbability(class_key="a", display_name="A", probability=1.2)

    def test_kind_and_verdict_are_literals(self):
        with pytest.raises(ValueError):
            TreeDetails(kind="other", verdict="banana_tree", threshold=0.6, probabilities=[])
        with pytest.raises(ValueError):
            TreeDetails(
                kind="tree_classification", verdict="maybe", threshold=0.6, probabilities=[]
            )

    def test_kind_is_required(self):
        with pytest.raises(ValueError):
            TreeDetails(verdict="banana_tree", threshold=0.6, probabilities=[])  # type: ignore[call-arg]


class TestTreeClassifierWithoutTorch:
    """The predictor's own non-torch behaviour: wiring from Settings and the stages."""

    def test_wires_settings_into_the_base_class(self):
        settings = make_settings(
            tree_model_version="tree_cls_dummy_v0", tree_is_placeholder=True, tree_imgsz=160
        )
        predictor = TreeClassifier(settings)
        assert predictor.key == "tree_classification"
        assert predictor.version == "tree_cls_dummy_v0"
        assert predictor.task == "classify"
        assert predictor.is_placeholder is True
        assert predictor.info.status == "unavailable"
        assert predictor.info.reason == "not loaded yet"

    def test_postprocess_uses_the_configured_policy(self):
        settings = make_settings(
            tree_uncertain_threshold=0.9,
            tree_positive_class="non_banana",
            tree_display_names={"banana_tree": "Tree", "non_banana": "Other"},
        )
        predictor = TreeClassifier(settings)
        out = predictor.postprocess(
            TreeRawOutput(probabilities=[0.2, 0.8], class_names=["banana_tree", "non_banana"])
        )
        assert out.is_uncertain is True  # 0.8 < 0.9
        confident = predictor.postprocess(
            TreeRawOutput(probabilities=[0.05, 0.95], class_names=["banana_tree", "non_banana"])
        )
        assert confident.details.verdict == "banana_tree"  # positive class is now non_banana
        assert confident.display_label == "Other"

    def test_preprocess_passes_rgb_through_untouched(self):
        predictor = TreeClassifier(make_settings())
        image = Image.new("RGB", (80, 60))
        assert predictor.preprocess(image) is image

    def test_preprocess_converts_other_modes_to_rgb(self):
        predictor = TreeClassifier(make_settings())
        assert predictor.preprocess(Image.new("RGBA", (80, 60))).mode == "RGB"
        assert predictor.preprocess(Image.new("L", (80, 60))).mode == "RGB"

    def test_infer_before_load_is_a_clear_error(self):
        predictor = TreeClassifier(make_settings())
        with pytest.raises(ModelUnavailableError, match="not loaded"):
            predictor.infer(Image.new("RGB", (80, 60)))

    def test_class_validation_accepts_matching_names(self):
        TreeClassifier(make_settings())._validate_classes(["banana_tree", "non_banana"])

    def test_class_validation_lists_both_sets_when_names_do_not_match(self):
        predictor = TreeClassifier(make_settings())
        with pytest.raises(ModelLoadError) as raised:
            predictor._validate_classes(["cat", "dog"])
        message = str(raised.value)
        assert "['cat', 'dog']" in message  # what the model has
        assert "TREE_POSITIVE_CLASS='banana_tree'" in message  # what is configured
        assert "['banana_tree', 'non_banana']" in message
        assert "not found in the model: ['banana_tree', 'non_banana']" in message

    def test_class_validation_checks_every_display_name_key(self):
        settings = make_settings(
            tree_display_names={"banana_tree": "Banana tree", "weed": "Weed"},
        )
        with pytest.raises(ModelLoadError, match=r"not found in the model: \['weed'\]"):
            TreeClassifier(settings)._validate_classes(["banana_tree", "non_banana"])

    def test_class_validation_checks_the_positive_class(self):
        settings = make_settings(tree_positive_class="banana")
        with pytest.raises(ModelLoadError, match=r"not found in the model: \['banana'\]"):
            TreeClassifier(settings)._validate_classes(["banana_tree", "non_banana"])

    def test_class_validation_needs_at_least_two_classes(self):
        settings = make_settings(tree_display_names={"banana_tree": "Banana tree"})
        with pytest.raises(ModelLoadError, match="at least 2 classes"):
            TreeClassifier(settings)._validate_classes(["banana_tree"])

    def test_unlabelled_model_class_only_warns(self, caplog):
        with caplog.at_level("WARNING", logger="app.ml.tree_classifier"):
            TreeClassifier(make_settings())._validate_classes(["banana_tree", "non_banana", "palm"])
        assert "palm" in caplog.text

    def test_load_with_missing_weights_fails_before_touching_ultralytics(self, tmp_path):
        missing = tmp_path / "nope.pt"
        predictor = TreeClassifier(make_settings(tree_weights_path=str(missing)))
        with pytest.raises(ModelLoadError, match="not found"):
            predictor.load()
        assert predictor.is_ready is False


class TestWeightsPath:
    def test_relative_path_is_anchored_on_backend_not_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        resolved = resolve_weights_path("weights/tree_cls_v1.pt")
        assert resolved == (BACKEND_DIR / "weights" / "tree_cls_v1.pt").resolve()
        assert BACKEND_DIR.name == "Backend"

    def test_absolute_path_passes_through(self, tmp_path):
        target = tmp_path / "custom.pt"
        assert resolve_weights_path(str(target)) == target


def _write_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", b"x")


class TestValidateWeightsFile:
    def test_valid_zip_passes(self, tmp_path):
        weights = tmp_path / "ok.pt"
        _write_zip(weights)
        validate_weights_file(weights)  # no exception

    def test_uppercase_extension_is_accepted(self, tmp_path):
        weights = tmp_path / "ok.PT"
        _write_zip(weights)
        validate_weights_file(weights)

    def test_missing_file_names_the_path_and_the_fix(self, tmp_path):
        weights = tmp_path / "missing.pt"
        with pytest.raises(ModelLoadError) as info:
            validate_weights_file(weights)
        message = str(info.value)
        assert str(weights) in message
        assert "make_dummy_weights" in message
        assert "TREE_WEIGHTS_PATH" in message

    def test_directory_instead_of_file(self, tmp_path):
        folder = tmp_path / "tree_cls_v1.pt"
        folder.mkdir()
        with pytest.raises(ModelLoadError, match="directory"):
            validate_weights_file(folder)

    def test_directory_that_is_an_extracted_checkpoint(self, tmp_path):
        folder = tmp_path / "tree_cls_v1.pt"
        folder.mkdir()
        (folder / "data.pkl").write_bytes(b"x")
        with pytest.raises(ModelLoadError, match="UNZIPPED"):
            validate_weights_file(folder)

    def test_sibling_folder_named_like_the_weights(self, tmp_path):
        weights = tmp_path / "tree_cls_v1.pt"  # missing...
        extracted = tmp_path / "tree_cls_v1"  # ...but its unzipped twin exists
        (extracted / "tree_cls_v1").mkdir(parents=True)
        (extracted / "tree_cls_v1" / "data.pkl").write_bytes(b"x")
        with pytest.raises(ModelLoadError, match="UNZIPPED"):
            validate_weights_file(weights)

    def test_wrong_extension(self, tmp_path):
        weights = tmp_path / "model.onnx"
        _write_zip(weights)
        with pytest.raises(ModelLoadError, match=r"\.pt extension"):
            validate_weights_file(weights)

    def test_not_a_zip(self, tmp_path):
        weights = tmp_path / "garbage.pt"
        weights.write_bytes(b"this is not a zip archive")
        with pytest.raises(ModelLoadError, match="not a valid .pt archive"):
            validate_weights_file(weights)

    def test_stray_coco_folder_is_mentioned_when_weights_are_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tree_module, "BACKEND_DIR", tmp_path)
        coco = tmp_path / "yolov8n" / "yolov8n"
        coco.mkdir(parents=True)
        (coco / "data.pkl").write_bytes(b"x")
        with pytest.raises(ModelLoadError) as info:
            validate_weights_file(tmp_path / "weights" / "tree_cls_v1.pt")
        assert "COCO" in str(info.value)
        assert "task=detect" in str(info.value)

    def test_stray_coco_folder_hint_detects_nested_and_flat_layouts(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tree_module, "BACKEND_DIR", tmp_path)
        assert stray_coco_folder_hint() is None
        (tmp_path / "yolov8n").mkdir()
        assert stray_coco_folder_hint() is None  # a folder without a checkpoint inside
        (tmp_path / "yolov8n" / "data.pkl").write_bytes(b"x")  # flat layout
        assert "COCO" in (stray_coco_folder_hint() or "")

    def test_no_hint_for_an_ordinary_missing_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tree_module, "BACKEND_DIR", tmp_path)
        assert extracted_checkpoint_hint(tmp_path / "weights" / "tree_cls_v1.pt") is None


class TestUltralyticsEnvironment:
    def test_forces_offline_quiet_no_autoinstall(self, monkeypatch):
        # setenv first so monkeypatch restores the original state afterwards.
        for name in ("YOLO_OFFLINE", "YOLO_AUTOINSTALL", "YOLO_VERBOSE"):
            monkeypatch.setenv(name, "to-be-overridden")
        configure_ultralytics_environment()
        assert os.environ["YOLO_OFFLINE"] == "1"
        assert os.environ["YOLO_AUTOINSTALL"] == "False"
        assert os.environ["YOLO_VERBOSE"] == "False"

    def test_stray_online_setting_is_overridden(self, monkeypatch):
        monkeypatch.setenv("YOLO_OFFLINE", "0")
        configure_ultralytics_environment()
        assert os.environ["YOLO_OFFLINE"] == "1"


def test_importing_the_ml_modules_does_not_import_torch_or_ultralytics():
    """The degraded-mode and fast-test guarantee, checked in a clean interpreter
    (this process may legitimately have torch loaded by a `model` test)."""
    code = (
        "import sys\n"
        "import app.ml.base, app.ml.registry, app.ml.tree_classifier, app.schemas.tree\n"
        "bad = [m for m in ('torch', 'ultralytics') if m in sys.modules]\n"
        "assert not bad, bad\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=BACKEND_DIR, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
