"""Model 3: banana-leaf disease identification (ADR 0019).

A U-Net (segmentation-models-pytorch, ResNet34 encoder) with THREE raw-logit output
channels, in the order ["leaf", "black_sigatoka", "yellow_sigatoka"]. Multi-label (a
diseased pixel is BOTH leaf and that disease), so we apply a per-channel sigmoid,
never a softmax. Architecturally identical to Model 2 (`app.ml.leaf_segmenter`),
just with one extra output channel and a per-image diagnosis derived by pooling.

The pre/post-processing lives in `app.ml.leaf_disease` (torch-free, unit-tested with
synthetic data); this class only owns the torch-specific parts: loading the
checkpoint and running the network. torch and smp are imported lazily inside
`load()` / `infer()` so the app still starts (degraded mode) and the fast test
run still works on a machine without them.

Known weaknesses (from the training notebook, repeated here so nobody trusts the
output more than the data allows): 159 training images total, `yellow_sigatoka`
backed by only 7 of them, `cordana` and one unlabeled image excluded outright.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from app.core.config import Settings
from app.ml.base import ModelLoadError, PredictionOutput, Predictor
from app.ml.device import resolve_device
from app.ml.leaf_disease.overlay import render_overlay
from app.ml.leaf_disease.postprocess import (
    DISEASE_CHANNELS,
    DISPLAY_LABELS,
    average_flip_probs,
    build_details,
    postprocess_probabilities,
)
from app.ml.leaf_seg.preprocess import prepare_input

# Reuse Model 2's checkpoint-loading helpers verbatim: the same wrapper-key /
# DataParallel-prefix / mismatch-summary logic applies unchanged to a 3-class U-Net.
from app.ml.leaf_segmenter import (
    _extract_state_dict,
    _mismatch_summary,
    _short,
)
from app.ml.leaf_segmenter import resolve_weights_path as _resolve_weights_path
from app.schemas.leaf_disease import LeafDiseaseThresholds

logger = logging.getLogger(__name__)

_CHANNELS = ("leaf", *DISEASE_CHANNELS)

resolve_weights_path = _resolve_weights_path


@dataclass(frozen=True, eq=False)
class LeafDiseaseInput:
    original: Image.Image
    array: NDArray[np.float32]  # (1, 3, S, S), normalised


@dataclass(frozen=True, eq=False)
class LeafDiseaseRaw:
    original: Image.Image
    probs: NDArray[np.float32]  # (3, S, S): [leaf, black_sigatoka, yellow_sigatoka]


class LeafDiseaseIdentifier(Predictor):
    key: ClassVar[str] = "leaf_disease"

    def __init__(self, settings: Settings) -> None:
        cfg = settings.leaf_disease  # built once: the property creates a new object per access
        super().__init__(
            display_name="Leaf disease identification",
            version=cfg.model_version,
            task="segment",
            is_placeholder=cfg.is_placeholder,
        )
        self._settings = settings
        self._cfg = cfg
        self._thresholds = LeafDiseaseThresholds(
            leaf=cfg.leaf_threshold,
            black_sigatoka=cfg.black_sigatoka_threshold,
            yellow_sigatoka=cfg.yellow_sigatoka_threshold,
            min_leaf_pct=cfg.min_leaf_pct,
            min_lesion_pct=cfg.min_lesion_pct,
            min_disease_pct=cfg.min_disease_pct,
            tta_hflip=cfg.tta_hflip,
        )
        self._model: Any = None
        self._device = "cpu"

    # --- lifecycle -------------------------------------------------------
    def load(self) -> None:
        path = resolve_weights_path(self._cfg.weights_path)
        if not path.is_file():
            raise ModelLoadError(
                f"Leaf-disease weights not found at {path}. Place the trained "
                "checkpoint there, or set LEAF_DISEASE_WEIGHTS_PATH; for a demo run use "
                "scripts/make_dummy_weights.py."
            )
        try:
            import torch

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
                f"(check LEAF_DISEASE_ENCODER): {type(exc).__name__}: {_short(exc)}"
            ) from exc

        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise ModelLoadError(
                f"{path} does not match a {len(_CHANNELS)}-class U-Net with encoder "
                f"{self._cfg.encoder!r} (check LEAF_DISEASE_ENCODER): "
                f"{_mismatch_summary(model.state_dict(), state)}."
            ) from exc

        device = resolve_device(self._settings.inference_device)
        try:
            model.eval()
            model.to(device)
        except Exception as exc:
            raise ModelLoadError(
                f"Cannot move the model to device {device!r} (check INFERENCE_DEVICE): "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        self._model = model
        self._device = device
        self.classes = list(_CHANNELS)
        self.mark_ready()
        logger.info(
            "leaf_disease_model_loaded",
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
    def preprocess(self, image: Image.Image) -> LeafDiseaseInput:
        return LeafDiseaseInput(original=image, array=prepare_input(image, self._cfg.imgsz))

    def infer(self, x: LeafDiseaseInput) -> LeafDiseaseRaw:
        if self._model is None:
            raise RuntimeError("infer() called before load()")
        import torch

        tta = self._cfg.tta_hflip
        with torch.inference_mode():
            batch = torch.from_numpy(x.array).to(self._device)
            if tta:
                batch = torch.cat([batch, torch.flip(batch, dims=[3])], dim=0)
            probs = torch.sigmoid(self._model(batch)).float().cpu().numpy()
        if tta:
            merged = average_flip_probs(probs[0], probs[1])
        else:
            merged = np.ascontiguousarray(probs[0], dtype=np.float32)
        return LeafDiseaseRaw(original=x.original, probs=merged)

    def postprocess(self, raw: LeafDiseaseRaw) -> PredictionOutput:
        width, height = raw.original.size
        result = postprocess_probabilities(raw.probs, (height, width), self._thresholds)
        overlay = render_overlay(
            raw.original, result.leaf_mask, {name: d.mask for name, d in result.diseases.items()}
        )
        return PredictionOutput(
            label=result.diagnosis,
            display_label=DISPLAY_LABELS[result.diagnosis],
            # D-06: a segmentation has no single honest confidence number, so none is invented.
            confidence=None,
            is_uncertain=False,
            details=build_details(result, self._thresholds),
            result_image=overlay,
        )
