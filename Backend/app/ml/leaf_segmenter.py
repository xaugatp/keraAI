"""Model 2: banana-leaf segmentation (healthy leaf tissue vs affected/damaged area).

A U-Net (segmentation-models-pytorch, ResNet34 encoder) with TWO raw-logit output
channels, in the order ["leaf", "affected"]. It is multi-label (a damaged pixel is
BOTH leaf and affected), so we apply a per-channel sigmoid, never a softmax.

The pre/post-processing lives in `app.ml.leaf_seg` (torch-free, unit-tested with
synthetic data); this class only owns the torch-specific parts: loading the
checkpoint and running the network. torch and smp are imported lazily inside
`load()` / `infer()` so the app still starts (degraded mode) and the fast test
run still works on a machine without them.

Known weaknesses (from the training notebook, repeated here so nobody trusts the
output more than the data allows): 50 training images, only 2 healthy leaves (the
"healthy" verdict is unvalidated), 8 test images, small scattered lesions get
merged into blobs, and the 0.85 affected threshold was tuned on the test split.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from app.core.config import Settings
from app.ml.base import ModelLoadError, PredictionOutput, Predictor
from app.ml.device import resolve_device
from app.ml.leaf_seg.overlay import render_overlay
from app.ml.leaf_seg.postprocess import (
    DISPLAY_LABELS,
    average_flip_probs,
    build_details,
    postprocess_probabilities,
)
from app.ml.leaf_seg.preprocess import prepare_input
from app.schemas.leaf_seg import LeafSegThresholds

logger = logging.getLogger(__name__)

# Backend/ — anchored on this file, NOT the working directory, so a relative
# LEAF_SEG_WEIGHTS_PATH resolves the same under uvicorn, pytest or a script.
_BACKEND_DIR = Path(__file__).resolve().parents[2]

_CHANNELS = ["leaf", "affected"]
_WRAPPER_KEYS = ("state_dict", "model_state_dict")
_DATA_PARALLEL_PREFIX = "module."


@dataclass(frozen=True, eq=False)
class LeafSegInput:
    """Output of `preprocess`. The ORIGINAL image travels with the network input
    because `postprocess(raw)` takes a single argument yet needs the photo's
    pixels (overlay) and size (mapping masks back to full resolution)."""

    original: Image.Image
    array: NDArray[np.float32]  # (1, 3, S, S), normalised


@dataclass(frozen=True, eq=False)
class LeafSegRaw:
    """Output of `infer`: sigmoid probabilities, plus the photo they belong to."""

    original: Image.Image
    probs: NDArray[np.float32]  # (2, S, S): [leaf, affected]


def resolve_weights_path(weights_path: str) -> Path:
    path = Path(weights_path)
    return path if path.is_absolute() else _BACKEND_DIR / path


def _short(text: object, limit: int = 300) -> str:
    """Single-line, length-capped exception text (torch's pickle errors can be huge)."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


def _extract_state_dict(checkpoint: Any, path: Path) -> dict[str, Any]:
    """Accept the notebook's bare state_dict or one wrapped under a known key."""
    import torch

    if isinstance(checkpoint, dict):
        for wrapper_key in _WRAPPER_KEYS:
            inner = checkpoint.get(wrapper_key)
            if isinstance(inner, dict):
                checkpoint = inner
                break
    if (
        not isinstance(checkpoint, dict)
        or not checkpoint
        or not all(isinstance(k, str) and torch.is_tensor(v) for k, v in checkpoint.items())
    ):
        raise ModelLoadError(
            f"{path} does not contain a state_dict of tensors (expected the output of "
            "torch.save(model.state_dict(), ...), optionally wrapped under "
            f"{' or '.join(repr(k) for k in _WRAPPER_KEYS)})."
        )
    state: dict[str, Any] = dict(checkpoint)
    # A model saved while wrapped in nn.DataParallel prefixes every key with "module.".
    if all(k.startswith(_DATA_PARALLEL_PREFIX) for k in state):
        state = {k[len(_DATA_PARALLEL_PREFIX) :]: v for k, v in state.items()}
    return state


def _mismatch_summary(expected: dict[str, Any], state: dict[str, Any], limit: int = 3) -> str:
    """Short, human-readable diff between the model's keys/shapes and the checkpoint's."""
    missing = sorted(expected.keys() - state.keys())
    unexpected = sorted(state.keys() - expected.keys())
    wrong_shape = [
        f"{k}: checkpoint {tuple(state[k].shape)} vs model {tuple(expected[k].shape)}"
        for k in sorted(expected.keys() & state.keys())
        if tuple(state[k].shape) != tuple(expected[k].shape)
    ]
    parts: list[str] = []
    for title, items in (
        ("missing keys", missing),
        ("unexpected keys", unexpected),
        ("shape mismatches", wrong_shape),
    ):
        if items:
            more = f", +{len(items) - limit} more" if len(items) > limit else ""
            parts.append(f"{len(items)} {title} (e.g. {'; '.join(items[:limit])}{more})")
    return "; ".join(parts) or "no key/shape difference found"


class LeafSegmenter(Predictor):
    key: ClassVar[str] = "leaf_segmentation"

    def __init__(self, settings: Settings) -> None:
        cfg = settings.leaf_seg  # built once: the property creates a new object per access
        super().__init__(
            display_name="Leaf segmentation",
            version=cfg.model_version,
            task="segment",
            is_placeholder=cfg.is_placeholder,
        )
        self._settings = settings
        self._cfg = cfg
        self._thresholds = LeafSegThresholds(
            leaf=cfg.leaf_threshold,
            affected=cfg.affected_threshold,
            min_leaf_pct=cfg.min_leaf_pct,
            min_lesion_pct=cfg.min_lesion_pct,
            min_affected_pct=cfg.min_affected_pct,
            tta_hflip=cfg.tta_hflip,
        )
        self._model: Any = None  # torch.nn.Module once loaded (typed Any: torch is lazy)
        self._device = "cpu"

    # --- lifecycle -------------------------------------------------------
    def load(self) -> None:
        path = resolve_weights_path(self._cfg.weights_path)
        # Checked BEFORE importing torch: a missing file is the common first-run
        # problem and should be reported instantly and without the heavy import.
        if not path.is_file():
            raise ModelLoadError(
                f"Leaf-segmentation weights not found at {path}. Place the trained "
                "checkpoint (the notebook's banana_leaf_segmentation_best.pt) there, or set "
                "LEAF_SEG_WEIGHTS_PATH; for a demo run use scripts/make_dummy_weights.py."
            )
        try:
            import torch

            # import_module (not `import ... as smp`): smp ships no type stubs, and
            # this keeps mypy quiet without a `type: ignore` that would go stale.
            smp = importlib.import_module("segmentation_models_pytorch")
        except ImportError as exc:
            raise ModelLoadError(
                f"torch / segmentation-models-pytorch are not installed: {exc}"
            ) from exc

        try:
            # SECURITY: weights_only=True restricts unpickling to tensors and plain
            # containers, so a tampered .pt file cannot execute code on load.
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ModelLoadError(
                f"{path} is not a loadable PyTorch checkpoint "
                f"({type(exc).__name__}: {_short(exc)}). Expected the file produced by "
                "torch.save(model.state_dict(), ...)."
            ) from exc
        state = _extract_state_dict(checkpoint, path)

        try:
            # encoder_weights=None is essential: "imagenet" would DOWNLOAD the
            # pretrained encoder at startup. We only need the architecture; the
            # real weights come from the checkpoint.
            model = smp.Unet(
                encoder_name=self._cfg.encoder,
                encoder_weights=None,
                in_channels=3,
                classes=len(_CHANNELS),
                activation=None,
            )
        except Exception as exc:
            raise ModelLoadError(
                f"Cannot build a U-Net with encoder {self._cfg.encoder!r} "
                f"(check LEAF_SEG_ENCODER): {type(exc).__name__}: {_short(exc)}"
            ) from exc

        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise ModelLoadError(
                f"{path} does not match a {len(_CHANNELS)}-class U-Net with encoder "
                f"{self._cfg.encoder!r} (check LEAF_SEG_ENCODER): "
                f"{_mismatch_summary(model.state_dict(), state)}."
            ) from exc

        device = resolve_device(self._settings.inference_device)
        try:
            model.eval()  # BatchNorm/dropout must use inference behaviour
            model.to(device)
        except Exception as exc:
            raise ModelLoadError(
                f"Cannot move the model to device {device!r} (check INFERENCE_DEVICE): "
                f"{type(exc).__name__}: {_short(exc)}"
            ) from exc

        self._model = model
        self._device = device
        self.classes = list(_CHANNELS)
        # predict() refuses to run until ready, and warmup() goes through predict(),
        # so a successfully loaded model must be marked ready here.
        self.mark_ready()
        logger.info(
            "leaf_seg_model_loaded",
            extra={
                "model_key": self.key,
                "weights": str(path),
                "encoder": self._cfg.encoder,
                "device": device,
            },
        )

    def warmup_image(self) -> Image.Image:
        side = self._cfg.imgsz
        return Image.new("RGB", (side, side), (128, 128, 128))

    # --- pipeline stages -------------------------------------------------
    def preprocess(self, image: Image.Image) -> LeafSegInput:
        return LeafSegInput(original=image, array=prepare_input(image, self._cfg.imgsz))

    def infer(self, x: LeafSegInput) -> LeafSegRaw:
        if self._model is None:
            raise RuntimeError("infer() called before load()")
        import torch

        tta = self._cfg.tta_hflip
        with torch.inference_mode():
            batch = torch.from_numpy(x.array).to(self._device)
            if tta:
                # One batch of [image, mirrored image]: BatchNorm is in eval mode so
                # samples are independent, and a batch of 2 is cheaper than 2 calls.
                batch = torch.cat([batch, torch.flip(batch, dims=[3])], dim=0)
            # Raw logits -> per-channel sigmoid (multi-label, NOT softmax).
            probs = torch.sigmoid(self._model(batch)).float().cpu().numpy()
        if tta:
            merged = average_flip_probs(probs[0], probs[1])
        else:
            merged = np.ascontiguousarray(probs[0], dtype=np.float32)
        return LeafSegRaw(original=x.original, probs=merged)

    def postprocess(self, raw: LeafSegRaw) -> PredictionOutput:
        width, height = raw.original.size
        result = postprocess_probabilities(raw.probs, (height, width), self._thresholds)
        overlay = render_overlay(raw.original, result.leaf_mask, result.affected_mask)
        return PredictionOutput(
            label=result.label,
            display_label=DISPLAY_LABELS[result.label],
            # D-06: a segmentation has no single honest confidence number, so none is invented.
            confidence=None,
            is_uncertain=False,
            details=build_details(result, self._thresholds),
            result_image=overlay,
        )
