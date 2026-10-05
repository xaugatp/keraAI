"""Model 2 with a real (randomly initialised) U-Net — needs torch + segmentation-models-pytorch.

Run with `pytest -m model tests/model/test_leaf_segmenter.py`.

The checkpoints are built here, in a temp dir, from `smp.Unet(..., encoder_weights=None)`:
random weights prove the plumbing (formats, strict loading, shapes, determinism, no
network) but say nothing about segmentation QUALITY — that needs the trained
`banana_leaf_segmentation_best.pt`. Most tests use the lighter resnet18 encoder to keep
the checkpoint files small; the production architecture (resnet34 @ 512 px) has its own test.

torch/smp are imported lazily (inside fixtures) so merely *collecting* this module in
the default fast run never imports them.
"""

from __future__ import annotations

import math
import os
import socket
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageOps

from app.ml.base import ModelLoadError, Timings
from app.ml.leaf_segmenter import LeafSegmenter, _extract_state_dict
from app.schemas.leaf_seg import LeafSegDetails
from tests.fixtures.settings import make_settings

pytestmark = pytest.mark.model

IMGSZ = 64
LIGHT_ENCODER = "resnet18"
BACKEND_DIR = Path(__file__).resolve().parents[2]
LABELS = {"affected", "healthy", "no_leaf"}


class _Evil:
    """Unpickling this would run os.mkdir(marker): a stand-in for hostile code."""

    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self) -> tuple[Any, tuple[str]]:
        return (os.mkdir, (str(self.marker),))


# --- fixtures ------------------------------------------------------------------


@pytest.fixture(scope="module")
def torch_module() -> Any:
    return pytest.importorskip("torch")


@pytest.fixture(scope="module")
def smp_module() -> Any:
    return pytest.importorskip("segmentation_models_pytorch")


@pytest.fixture(scope="module")
def build_state(torch_module: Any, smp_module: Any) -> Any:
    def build(encoder: str = LIGHT_ENCODER, classes: int = 2, seed: int = 0) -> dict[str, Any]:
        torch_module.manual_seed(seed)
        model = smp_module.Unet(
            encoder_name=encoder,
            encoder_weights=None,  # never touch the network
            in_channels=3,
            classes=classes,
            activation=None,
        )
        return {k: v.clone() for k, v in model.state_dict().items()}

    return build


@pytest.fixture(scope="module")
def ckpts(tmp_path_factory: pytest.TempPathFactory, torch_module: Any, build_state: Any) -> Any:
    """Checkpoint files in every format the loader promises to accept."""
    root = tmp_path_factory.mktemp("leafseg_ckpts")
    state = build_state()
    paths = {
        "bare": root / "bare.pt",
        "state_dict": root / "wrapped_state_dict.pt",
        "model_state_dict": root / "wrapped_model_state_dict.pt",
        "module_prefix": root / "module_prefix.pt",
    }
    torch_module.save(state, paths["bare"])
    torch_module.save({"state_dict": state, "epoch": 7}, paths["state_dict"])
    torch_module.save({"model_state_dict": state, "optimizer": {}}, paths["model_state_dict"])
    torch_module.save({f"module.{k}": v for k, v in state.items()}, paths["module_prefix"])
    return paths


def seg_for(path: Path | str, **overrides: Any) -> LeafSegmenter:
    values: dict[str, Any] = {
        "leaf_seg_weights_path": str(path),
        "leaf_seg_imgsz": IMGSZ,
        "leaf_seg_encoder": LIGHT_ENCODER,
        "leaf_seg_tta_hflip": False,
        "inference_device": "cpu",
    }
    values.update(overrides)
    return LeafSegmenter(make_settings(**values))


def loaded(path: Path | str, **overrides: Any) -> LeafSegmenter:
    segmenter = seg_for(path, **overrides)
    segmenter.load()
    return segmenter


def photo(width: int = 320, height: int = 240) -> Image.Image:
    """Synthetic 'leaf photo': a green blob on a brown background with a darker patch."""
    rng = np.random.default_rng(3)
    pixels = np.full((height, width, 3), (120, 90, 60), dtype=np.uint8)
    pixels[height // 6 : -height // 6, width // 8 : -width // 3] = (60, 150, 60)
    pixels[height // 3 : height // 2, width // 5 : width // 3] = (110, 80, 40)
    noise = rng.integers(-8, 9, pixels.shape, dtype=np.int16)
    return Image.fromarray(np.clip(pixels + noise, 0, 255).astype(np.uint8))


def assert_valid_output(out: Any, *, size: tuple[int, int]) -> None:
    assert isinstance(out.details, LeafSegDetails)
    d = out.details
    numbers = [
        d.leaf_area_pct_of_image,
        d.affected_area_pct_of_leaf,
        d.largest_lesion_pct_of_leaf,
        d.lesion_count,
        *(v for v in (d.mean_leaf_probability, d.mean_affected_probability) if v is not None),
    ]
    assert all(math.isfinite(n) for n in numbers)
    assert out.label in LABELS
    assert out.confidence is None and out.is_uncertain is False
    assert isinstance(out.result_image, Image.Image) and out.result_image.mode == "RGB"
    assert max(out.result_image.size) <= 1280
    assert out.result_image.size[0] / out.result_image.size[1] == pytest.approx(
        size[0] / size[1], rel=0.01
    )
    assert isinstance(out.timings, Timings)
    assert out.timings.inference_ms > 0 and out.timings.model_total_ms >= out.timings.inference_ms


# --- happy paths ----------------------------------------------------------------


def test_bare_state_dict_loads_strictly_and_predicts(ckpts: Any) -> None:
    segmenter = loaded(ckpts["bare"])
    assert segmenter.is_ready
    assert segmenter.info.status == "ready" and segmenter.info.reason is None
    assert segmenter.info.classes == ["leaf", "affected"]

    out = segmenter.predict(photo(1600, 1200))
    assert_valid_output(out, size=(1600, 1200))
    assert out.result_image is not None and out.result_image.size == (1280, 960)
    assert out.details.kind == "leaf_segmentation"
    assert out.details.thresholds.tta_hflip is False


def test_infer_returns_float32_probabilities_in_unit_range(ckpts: Any) -> None:
    segmenter = loaded(ckpts["bare"])
    raw = segmenter.infer(segmenter.preprocess(photo()))
    assert raw.probs.shape == (2, IMGSZ, IMGSZ) and raw.probs.dtype == np.float32
    assert raw.probs.flags["C_CONTIGUOUS"]
    assert np.isfinite(raw.probs).all() and raw.probs.min() >= 0.0 and raw.probs.max() <= 1.0
    # Sigmoid (multi-label), NOT softmax: the two channels are independent, so they
    # must not be forced to sum to 1 at every pixel.
    assert not np.allclose(raw.probs.sum(axis=0), 1.0, atol=1e-3)


@pytest.mark.parametrize("tta", [False, True])
def test_prediction_is_deterministic_with_and_without_tta(ckpts: Any, tta: bool) -> None:
    segmenter = loaded(ckpts["bare"], leaf_seg_tta_hflip=tta)
    image = photo()
    first, second = segmenter.predict(image), segmenter.predict(image)
    assert first.details == second.details
    assert first.label == second.label
    assert first.result_image is not None and second.result_image is not None
    assert np.array_equal(np.asarray(first.result_image), np.asarray(second.result_image))
    assert first.details.thresholds.tta_hflip is tta  # echoed so a stored row says which was used
    p1 = segmenter.infer(segmenter.preprocess(image)).probs
    p2 = segmenter.infer(segmenter.preprocess(image)).probs
    assert np.array_equal(p1, p2)


def test_tta_averages_with_the_mirrored_pass(ckpts: Any) -> None:
    plain = loaded(ckpts["bare"], leaf_seg_tta_hflip=False)
    tta = loaded(ckpts["bare"], leaf_seg_tta_hflip=True)
    image = photo()
    p_plain = plain.infer(plain.preprocess(image)).probs
    p_tta = tta.infer(tta.preprocess(image)).probs
    mirrored = plain.infer(plain.preprocess(ImageOps.mirror(image))).probs
    # TTA output == average of the normal pass and the flipped-back mirrored pass...
    assert np.allclose(p_tta, (p_plain + mirrored[..., ::-1]) / 2, atol=1e-4)
    # ...and therefore genuinely differs from the single pass (a random net is not equivariant).
    assert not np.allclose(p_tta, p_plain, atol=1e-4)
    # TTA makes the model mirror-equivariant: predicting the mirrored photo gives the mirrored maps.
    p_tta_mirrored = tta.infer(tta.preprocess(ImageOps.mirror(image))).probs
    assert np.allclose(p_tta_mirrored, p_tta[..., ::-1], atol=1e-4)


@pytest.mark.parametrize("fmt", ["state_dict", "model_state_dict", "module_prefix"])
def test_wrapped_and_prefixed_checkpoints_load_the_same_weights(ckpts: Any, fmt: str) -> None:
    image = photo()
    reference = loaded(ckpts["bare"])
    other = loaded(ckpts[fmt])
    expected = reference.infer(reference.preprocess(image)).probs
    actual = other.infer(other.preprocess(image)).probs
    assert np.allclose(actual, expected, atol=1e-6)
    assert_valid_output(other.predict(image), size=(320, 240))


def test_warmup_succeeds(ckpts: Any) -> None:
    segmenter = loaded(ckpts["bare"])
    segmenter.warmup()
    assert segmenter.is_ready


def test_production_architecture_resnet34_at_512(
    tmp_path: Path, torch_module: Any, build_state: Any
) -> None:
    path = tmp_path / "r34.pt"
    torch_module.save(build_state("resnet34"), path)
    settings = make_settings(leaf_seg_weights_path=str(path), inference_device="cpu")
    assert settings.leaf_seg.imgsz == 512 and settings.leaf_seg.encoder == "resnet34"
    segmenter = LeafSegmenter(settings)
    segmenter.load()
    raw = segmenter.infer(segmenter.preprocess(photo(800, 600)))
    assert raw.probs.shape == (2, 512, 512)
    assert_valid_output(segmenter.predict(photo(800, 600)), size=(800, 600))


def test_repo_weights_file_loads_when_present() -> None:
    path = BACKEND_DIR / "weights" / "leaf_seg_v1.pt"
    if not path.is_file():
        pytest.skip(f"{path} not present (create it with scripts/make_dummy_weights.py)")
    segmenter = LeafSegmenter(make_settings(inference_device="cpu"))  # default relative path
    segmenter.load()
    assert segmenter.is_ready
    assert_valid_output(segmenter.predict(photo(640, 480)), size=(640, 480))


# --- failure modes ----------------------------------------------------------------


def test_missing_file_is_a_load_error_with_the_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pt"
    segmenter = seg_for(missing)
    with pytest.raises(ModelLoadError, match="not found") as excinfo:
        segmenter.load()
    assert str(missing) in str(excinfo.value)
    assert not segmenter.is_ready


def test_encoder_mismatch_explains_how_to_fix_it(
    tmp_path: Path, torch_module: Any, build_state: Any
) -> None:
    path = tmp_path / "r18.pt"
    torch_module.save(build_state("resnet18"), path)
    segmenter = seg_for(path, leaf_seg_encoder="resnet34")
    with pytest.raises(ModelLoadError) as excinfo:
        segmenter.load()
    message = str(excinfo.value)
    assert "2-class U-Net with encoder 'resnet34'" in message
    assert "LEAF_SEG_ENCODER" in message
    assert "keys" in message or "shape mismatches" in message  # short key/shape summary
    assert not segmenter.is_ready


def test_wrong_class_count_reports_shape_mismatches(
    tmp_path: Path, torch_module: Any, build_state: Any
) -> None:
    path = tmp_path / "three_class.pt"
    torch_module.save(build_state(LIGHT_ENCODER, classes=3), path)
    with pytest.raises(ModelLoadError, match="shape mismatches"):
        seg_for(path).load()


def test_unknown_encoder_name_is_a_load_error(ckpts: Any) -> None:
    with pytest.raises(ModelLoadError, match="LEAF_SEG_ENCODER"):
        seg_for(ckpts["bare"], leaf_seg_encoder="not-a-real-encoder").load()


@pytest.mark.parametrize("content", [b"definitely not a torch file", b"", b"PK\x03\x04garbage"])
def test_garbage_file_is_a_load_error_not_a_raw_exception(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "garbage.pt"
    path.write_bytes(content)
    segmenter = seg_for(path)
    with pytest.raises(ModelLoadError, match="not a loadable PyTorch checkpoint"):
        segmenter.load()
    assert not segmenter.is_ready


@pytest.mark.parametrize(
    "payload",
    [
        {"epoch": 3, "best_dice": 0.7},  # a training-state dict with no weights
        {"state_dict": {"w": 1}},  # wrapper holding non-tensors
        {"state_dict": {}},  # empty
        [1, 2, 3],
    ],
)
def test_checkpoint_that_is_not_a_state_dict_is_rejected(
    tmp_path: Path, torch_module: Any, payload: Any
) -> None:
    path = tmp_path / "weird.pt"
    torch_module.save(payload, path)
    with pytest.raises(ModelLoadError, match="state_dict"):
        seg_for(path).load()
    with pytest.raises(ModelLoadError, match="state_dict"):
        _extract_state_dict(payload, path)


def test_a_bare_tensor_checkpoint_is_rejected(tmp_path: Path, torch_module: Any) -> None:
    path = tmp_path / "tensor.pt"
    torch_module.save(torch_module.zeros(3), path)
    with pytest.raises(ModelLoadError, match="state_dict"):
        seg_for(path).load()


def test_tampered_pickle_is_never_executed(tmp_path: Path, torch_module: Any) -> None:
    marker = tmp_path / "pwned"
    path = tmp_path / "evil.pt"
    torch_module.save(_Evil(marker), path)

    # Control: an UNSAFE load (weights_only=False) really would run the payload, so
    # the assertion below is meaningful and not vacuous.
    torch_module.load(path, map_location="cpu", weights_only=False)
    assert marker.exists()
    marker.rmdir()

    with pytest.raises(ModelLoadError, match="not a loadable PyTorch checkpoint"):
        seg_for(path).load()
    assert not marker.exists(), "weights_only=True must refuse to unpickle arbitrary callables"


# --- no network -------------------------------------------------------------------


def test_load_and_predict_make_no_network_connections(
    ckpts: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    # Control: the guard is live (a connection attempt is refused).
    with pytest.raises(AssertionError, match="network access attempted"):
        socket.create_connection(("127.0.0.1", 9))

    for tta in (False, True):
        segmenter = loaded(ckpts["bare"], leaf_seg_tta_hflip=tta)
        assert_valid_output(segmenter.predict(photo()), size=(320, 240))
