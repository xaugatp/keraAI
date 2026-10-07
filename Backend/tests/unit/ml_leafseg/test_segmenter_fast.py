"""LeafSegmenter without torch: plumbing, config wiring and error paths.

The torch-dependent parts (load, infer) are covered by the `model`-marked tests in
tests/model/test_leaf_segmenter.py. Here `infer` is replaced by a stub that
returns synthetic probabilities, so the real preprocess -> postprocess -> overlay
chain still runs end to end through `Predictor.predict`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.ml.base import ModelLoadError, PredictionOutput
from app.ml.leaf_segmenter import (
    LeafSegInput,
    LeafSegmenter,
    LeafSegRaw,
    _mismatch_summary,
    _short,
    resolve_weights_path,
)
from app.schemas.leaf_seg import LeafSegDetails
from tests.fixtures.settings import make_settings
from tests.unit.ml_leafseg.helpers import make_probs

IMGSZ = 64
BACKEND_DIR = Path(__file__).resolve().parents[3]


def make_segmenter(**overrides: object) -> LeafSegmenter:
    values: dict[str, object] = {"leaf_seg_imgsz": IMGSZ, "leaf_seg_tta_hflip": False}
    values.update(overrides)
    return LeafSegmenter(make_settings(**values))


def stub_infer(
    segmenter: LeafSegmenter, monkeypatch: pytest.MonkeyPatch, probs: np.ndarray
) -> list[LeafSegInput]:
    seen: list[LeafSegInput] = []

    def infer(x: LeafSegInput) -> LeafSegRaw:
        seen.append(x)
        return LeafSegRaw(original=x.original, probs=probs)

    monkeypatch.setattr(segmenter, "infer", infer)
    return seen


# --- identity & config wiring -------------------------------------------------


def test_identity_and_info_before_load() -> None:
    segmenter = make_segmenter(leaf_seg_model_version="leaf_seg_v9", leaf_seg_is_placeholder=True)
    info = segmenter.info
    assert LeafSegmenter.key == "leaf_segmentation"
    assert (info.key, info.display_name, info.task) == (
        "leaf_segmentation",
        "Leaf segmentation",
        "segment",
    )
    assert info.version == "leaf_seg_v9" and info.is_placeholder is True
    assert info.status == "unavailable" and info.reason == "not loaded yet"
    assert not segmenter.is_ready


def test_warmup_image_matches_the_configured_size() -> None:
    assert make_segmenter(leaf_seg_imgsz=96).warmup_image().size == (96, 96)
    assert make_segmenter().warmup_image().mode == "RGB"


def test_preprocess_carries_the_original_image_and_network_input() -> None:
    segmenter = make_segmenter()
    image = Image.new("RGB", (300, 200), (10, 20, 30))
    prepared = segmenter.preprocess(image)
    assert isinstance(prepared, LeafSegInput)
    assert prepared.original is image
    assert prepared.array.shape == (1, 3, IMGSZ, IMGSZ) and prepared.array.dtype == np.float32


def test_infer_before_load_is_a_clear_error() -> None:
    segmenter = make_segmenter()
    with pytest.raises(RuntimeError, match="before load"):
        segmenter.infer(segmenter.preprocess(Image.new("RGB", (64, 64))))


# --- weights path resolution --------------------------------------------------


def test_relative_weights_path_is_anchored_on_the_backend_dir_not_the_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert (
        resolve_weights_path("weights/leaf_seg_v1.pt") == BACKEND_DIR / "weights" / "leaf_seg_v1.pt"
    )


def test_absolute_weights_path_is_used_as_is(tmp_path: Path) -> None:
    target = tmp_path / "w.pt"
    assert resolve_weights_path(str(target)) == target


def test_load_missing_file_reports_the_exact_path_and_a_hint(tmp_path: Path) -> None:
    missing = tmp_path / "nope.pt"
    segmenter = make_segmenter(leaf_seg_weights_path=str(missing))
    with pytest.raises(ModelLoadError) as excinfo:
        segmenter.load()
    message = str(excinfo.value)
    assert str(missing) in message and "LEAF_SEG_WEIGHTS_PATH" in message
    assert not segmenter.is_ready


def test_load_directory_instead_of_file_is_a_load_error(tmp_path: Path) -> None:
    with pytest.raises(ModelLoadError, match="not found"):
        make_segmenter(leaf_seg_weights_path=str(tmp_path)).load()


# --- full predict() through the template method (stubbed network) --------------


def test_predict_runs_the_whole_chain_and_fills_the_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segmenter = make_segmenter()
    seen = stub_infer(
        segmenter,
        monkeypatch,
        make_probs(IMGSZ, IMGSZ, leaf=[(0, IMGSZ, 0, 40)], lesions=[(20, 30, 10, 20)]),
    )
    segmenter.mark_ready()

    photo = Image.new("RGB", (640, 480), (90, 140, 70))
    out = segmenter.predict(photo)

    assert isinstance(out, PredictionOutput)
    assert len(seen) == 1 and seen[0].original is photo
    assert out.label == "affected"
    assert out.display_label == "Damaged tissue found"
    assert out.confidence is None and out.is_uncertain is False
    assert isinstance(out.details, LeafSegDetails)
    assert out.details.kind == "leaf_segmentation"
    assert out.details.lesion_count == 1
    assert out.details.leaf_area_pct_of_image == pytest.approx(62.5, abs=1.0)  # 40 of 64 columns
    assert out.details.thresholds.leaf == 0.5 and out.details.thresholds.affected == 0.90
    assert out.details.thresholds.tta_hflip is False
    assert out.result_image is not None
    assert out.result_image.mode == "RGB" and out.result_image.size == (640, 480)
    assert out.timings.model_total_ms >= 0


def test_predict_overlay_has_the_original_photo_size_not_the_network_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segmenter = make_segmenter()
    stub_infer(segmenter, monkeypatch, make_probs(IMGSZ, IMGSZ))
    segmenter.mark_ready()
    out = segmenter.predict(Image.new("RGB", (2000, 1000), (1, 2, 3)))
    assert out.result_image is not None and out.result_image.size == (1280, 640)
    assert out.label == "no_leaf"
    assert out.details.model_dump()["mean_leaf_probability"] is None


def test_thresholds_from_settings_reach_postprocessing(monkeypatch: pytest.MonkeyPatch) -> None:
    probs = make_probs(IMGSZ, IMGSZ, leaf=[(0, IMGSZ, 0, 40)], lesions=[(20, 30, 10, 20)])
    probs[1] = np.where(probs[1] > 0, 0.9, 0.0)
    loose = make_segmenter(leaf_seg_affected_threshold=0.85)
    strict = make_segmenter(leaf_seg_affected_threshold=0.95, leaf_seg_tta_hflip=True)
    for segmenter in (loose, strict):
        stub_infer(segmenter, monkeypatch, probs)
        segmenter.mark_ready()
    assert loose.predict(Image.new("RGB", (64, 64))).label == "affected"
    strict_out = strict.predict(Image.new("RGB", (64, 64)))
    assert strict_out.label == "healthy"
    assert strict_out.details.model_dump()["thresholds"]["affected"] == 0.95
    assert strict_out.details.model_dump()["thresholds"]["tta_hflip"] is True


def test_non_rgb_input_surfaces_as_an_inference_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.core.errors import InferenceError

    segmenter = make_segmenter()
    stub_infer(segmenter, monkeypatch, make_probs(IMGSZ, IMGSZ))
    segmenter.mark_ready()
    with pytest.raises(InferenceError):
        segmenter.predict(Image.new("RGBA", (64, 64)))


# --- load() helpers that need no torch checkpoint ------------------------------


def test_short_collapses_whitespace_and_caps_length() -> None:
    assert _short("a\n  b\tc") == "a b c"
    capped = _short("x" * 1000, limit=50)
    assert len(capped) == 50 and capped.endswith("...")


class _FakeTensor:
    def __init__(self, *shape: int) -> None:
        self.shape = shape


def test_mismatch_summary_lists_missing_unexpected_and_shape_problems() -> None:
    expected = {"a": _FakeTensor(3), "b": _FakeTensor(2, 2), "c": _FakeTensor(1)}
    state = {"a": _FakeTensor(4), "b": _FakeTensor(2, 2), "z": _FakeTensor(1)}
    summary = _mismatch_summary(expected, state)
    assert "1 missing keys (e.g. c)" in summary
    assert "1 unexpected keys (e.g. z)" in summary
    assert "a: checkpoint (4,) vs model (3,)" in summary


def test_mismatch_summary_truncates_long_lists() -> None:
    expected = {f"k{i}": _FakeTensor(1) for i in range(10)}
    assert "+7 more" in _mismatch_summary(expected, {})
    assert _mismatch_summary({}, {}) == "no key/shape difference found"


# --- the fast run must stay torch-free -----------------------------------------


def test_importing_the_pipeline_does_not_import_torch() -> None:
    # A subprocess gives a clean interpreter: other tests may already have
    # imported torch into THIS process, which would make an in-process check lie.
    code = (
        "import sys\n"
        "import app.ml.leaf_seg.preprocess, app.ml.leaf_seg.postprocess\n"
        "import app.ml.leaf_seg.overlay, app.schemas.leaf_seg, app.ml.leaf_segmenter\n"
        "from app.core.config import Settings\n"
        "from app.ml.leaf_segmenter import LeafSegmenter\n"
        "LeafSegmenter(Settings(_env_file=None, signing_secret='x'*40, data_root='d'))\n"
        "bad = [m for m in ('torch', 'ultralytics', 'segmentation_models_pytorch') if m in sys.modules]\n"
        "sys.exit(f'imported: {bad}' if bad else 0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=BACKEND_DIR, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr or result.stdout
