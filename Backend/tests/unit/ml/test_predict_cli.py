"""scripts/predict_cli.py: the real decode/normalize path + a fake predictor (no torch)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from PIL import Image
from pydantic import BaseModel
from scripts import predict_cli

from app.ml.base import PredictionOutput
from app.ml.registry import ModelRegistry
from tests.fakes import FakePredictor
from tests.fixtures.image_factory import encode, gradient_image
from tests.fixtures.settings import make_settings

TREE = "tree_classification"
LEAF = "leaf_segmentation"


class SegDetails(BaseModel):
    """Stands in for a segmentation model's details (anything that is not TreeDetails)."""

    kind: str = "leaf_segmentation"
    leaf_area_pct_of_image: float = 41.5


@pytest.fixture
def jpeg(tmp_path: Path) -> Path:
    path = tmp_path / "leaf.jpg"
    path.write_bytes(encode(gradient_image(200, 150), "JPEG"))
    return path


@pytest.fixture
def use_registry(monkeypatch):
    """Point the CLI at a registry made of fakes (and hermetic settings)."""

    def install(*predictors: FakePredictor) -> ModelRegistry:
        registry = ModelRegistry.from_predictors(predictors)
        monkeypatch.setattr(predict_cli, "get_settings", lambda: make_settings())
        monkeypatch.setattr(predict_cli, "build_registry", lambda settings, only=None: registry)
        return registry

    return install


def run_cli(*argv: str) -> tuple[int, str, str]:
    args = predict_cli.build_parser().parse_args(list(argv))
    out, err = io.StringIO(), io.StringIO()
    code = predict_cli.run(args, out, err)
    return code, out.getvalue(), err.getvalue()


class TestTextOutput:
    def test_prints_label_confidence_timings_and_distribution(self, use_registry, jpeg):
        use_registry(FakePredictor(TREE))
        code, out, err = run_cli("--model", TREE, str(jpeg))

        assert code == 0 and err == ""
        assert "== leaf.jpg" in out
        assert "label:         banana_tree" in out
        assert "display label: Banana tree" in out
        assert "0.9470" in out
        assert "timings (ms):" in out
        assert "distribution:" in out
        assert "banana_tree" in out and "non_banana" in out
        assert "uncertain: no" in out

    def test_default_model_is_the_tree_classifier(self, use_registry, jpeg):
        use_registry(FakePredictor(TREE))
        code, out, _ = run_cli(str(jpeg))
        assert code == 0 and f"[{TREE} fake_v1]" in out

    def test_several_images_and_several_models(self, use_registry, jpeg, tmp_path):
        second = tmp_path / "second.jpg"
        second.write_bytes(encode(gradient_image(180, 180), "JPEG"))
        use_registry(FakePredictor(TREE), FakePredictor(LEAF, task="segment"))
        code, out, _ = run_cli("--model", TREE, "--model", LEAF, str(jpeg), str(second))
        assert code == 0
        assert out.count("== ") == 4  # 2 images x 2 models

    def test_non_tree_details_are_printed_as_json_block(self, use_registry, jpeg):
        use_registry(FakePredictor(LEAF, details=SegDetails(), task="segment"))
        code, out, _ = run_cli("--model", LEAF, str(jpeg))
        assert code == 0
        assert '"leaf_area_pct_of_image": 41.5' in out
        assert "distribution:" not in out

    def test_unconfident_prediction_is_flagged(self, use_registry, jpeg):
        use_registry(
            FakePredictor(TREE, display_label="Uncertain", confidence=0.51, is_uncertain=True)
        )
        _, out, _ = run_cli(str(jpeg))
        assert "uncertain: yes" in out

    def test_missing_confidence_prints_na(self, use_registry, jpeg):
        use_registry(FakePredictor(TREE, confidence=None))
        _, out, _ = run_cli(str(jpeg))
        assert "confidence:    n/a" in out


class TestJsonOutput:
    def test_stdout_is_pure_json(self, use_registry, jpeg):
        use_registry(FakePredictor(TREE, is_placeholder=True))
        code, out, err = run_cli("--json", str(jpeg))

        rows = json.loads(out)  # raises if the banner leaked into stdout
        assert code == 0
        assert len(rows) == 1
        row = rows[0]
        assert row["file"] == "leaf.jpg"
        assert row["model_key"] == TREE
        assert row["model_version"] == "fake_v1"
        assert row["is_placeholder"] is True
        assert row["label"] == "banana_tree"
        assert row["confidence"] == 0.947
        assert row["is_uncertain"] is False
        assert set(row["timings_ms"]) == {"preprocess", "inference", "postprocess"}
        assert row["details"]["kind"] == "tree_classification"
        assert row["error"] is None
        assert "PLACEHOLDER" in err  # the banner went to stderr instead

    def test_errors_are_reported_in_json(self, use_registry, tmp_path):
        use_registry(FakePredictor(TREE))
        bad = tmp_path / "bad.jpg"
        bad.write_bytes(b"not an image")
        code, out, _ = run_cli("--json", str(bad))
        assert code == 1
        row = json.loads(out)[0]
        assert row["error"]["code"] == "INVALID_IMAGE"
        assert row["label"] is None and row["details"] is None and row["timings_ms"] is None


class TestPlaceholderBanner:
    def test_loud_banner_when_the_model_is_a_placeholder(self, use_registry, jpeg):
        use_registry(FakePredictor(TREE, is_placeholder=True))
        _, out, _ = run_cli(str(jpeg))
        assert "PLACEHOLDER WEIGHTS" in out
        assert "NOT real predictions" in out
        assert out.index("PLACEHOLDER WEIGHTS") < out.index("== leaf.jpg")

    def test_no_banner_for_real_weights(self, use_registry, jpeg):
        use_registry(FakePredictor(TREE))
        _, out, _ = run_cli(str(jpeg))
        assert "PLACEHOLDER" not in out


class TestOverlay:
    def test_result_image_is_saved_when_requested(self, use_registry, jpeg, tmp_path):
        overlay = Image.new("RGB", (40, 30), (255, 0, 0))
        use_registry(FakePredictor(LEAF, result_image=overlay, task="segment"))
        target = tmp_path / "overlays"
        code, out, _ = run_cli("--model", LEAF, "--save-overlay", str(target), str(jpeg))

        saved = target / f"leaf.{LEAF}.png"
        assert code == 0 and saved.is_file()
        assert Image.open(saved).size == (40, 30)
        assert f"overlay saved: {saved}" in out

    def test_nothing_saved_without_the_flag_or_without_an_overlay(
        self, use_registry, jpeg, tmp_path
    ):
        use_registry(FakePredictor(TREE))  # classifiers have no result_image
        target = tmp_path / "overlays"
        run_cli("--save-overlay", str(target), str(jpeg))
        assert target.is_dir() and list(target.iterdir()) == []


class TestFailures:
    def test_unreadable_and_invalid_files_do_not_stop_the_batch(self, use_registry, jpeg, tmp_path):
        use_registry(FakePredictor(TREE))
        corrupt = tmp_path / "corrupt.jpg"
        corrupt.write_bytes(b"\x00\x01nonsense")
        tiny = tmp_path / "tiny.png"
        tiny.write_bytes(encode(Image.new("RGB", (16, 16)), "PNG"))
        missing = tmp_path / "missing.jpg"

        code, out, _ = run_cli(str(corrupt), str(tiny), str(missing), str(jpeg))

        assert code == 1
        assert "INVALID_IMAGE" in out
        assert "IMAGE_TOO_SMALL" in out
        assert "FILE_ERROR" in out
        assert "label:         banana_tree" in out  # the good image was still analysed

    def test_unavailable_model_is_reported_and_fails_the_run(self, use_registry, jpeg):
        down = FakePredictor(TREE, ready=False)
        down.mark_unavailable("Weights file not found: 'tree_cls_v1.pt'.")
        use_registry(down)
        code, out, err = run_cli(str(jpeg))
        assert code == 1
        assert "UNAVAILABLE" in err and "Weights file not found" in err
        assert "check_env" in err
        assert out == ""

    def test_one_model_down_does_not_block_the_other(self, use_registry, jpeg):
        down = FakePredictor(LEAF, ready=False)
        use_registry(FakePredictor(TREE), down)
        code, out, err = run_cli("--model", TREE, "--model", LEAF, str(jpeg))
        assert code == 1
        assert f"[{TREE} " in out
        assert LEAF in err

    def test_inference_failure_is_reported_per_image(self, use_registry, jpeg):
        use_registry(FakePredictor(TREE, fail=True))
        code, out, _ = run_cli(str(jpeg))
        assert code == 1
        assert "INFERENCE_FAILED" in out

    def test_unknown_model_name_is_rejected_by_argparse(self):
        with pytest.raises(SystemExit):
            predict_cli.build_parser().parse_args(["--model", "nope", "x.jpg"])

    def test_requires_at_least_one_path(self):
        with pytest.raises(SystemExit):
            predict_cli.build_parser().parse_args(["--model", TREE])


def test_row_json_for_output_without_result():
    row = predict_cli.Row("a.jpg", TREE, "v1")
    assert row.as_json()["label"] is None


def test_prediction_output_type_is_what_the_cli_prints(use_registry, jpeg):
    # Guards the contract between Predictor.predict and the CLI formatter.
    fake = FakePredictor(TREE)
    use_registry(fake)
    assert isinstance(fake.predict(Image.new("RGB", (64, 64))), PredictionOutput)
