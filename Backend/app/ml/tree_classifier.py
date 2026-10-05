"""Model 1 — banana tree classification (YOLOv8-cls wrapper, spec §8).

Import-time rule: this module must stay cheap to import and must NOT import
torch/ultralytics at module level. They are imported lazily inside `load()`, so
the fast test run and the degraded "no weights" startup never pay for (or need)
them. Everything that can be tested without torch — weights-file validation and
`build_tree_output` (the postprocess logic) — is a plain module-level function.
"""

from __future__ import annotations

import logging
import math
import os
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from PIL import Image

from app.core.config import Settings
from app.core.errors import InferenceError, ModelUnavailableError
from app.ml.base import ModelLoadError, PredictionOutput, Predictor
from app.ml.device import resolve_device
from app.schemas.tree import TreeDetails, TreeProbability

logger = logging.getLogger(__name__)

# Backend/ — anchored on this file (app/ml/tree_classifier.py), NOT on the
# process's working directory, so `uvicorn` / pytest / scripts all resolve a
# relative TREE_WEIGHTS_PATH to the same place no matter where they are started.
BACKEND_DIR = Path(__file__).resolve().parents[2]

# Probabilities from a softmax in float32 sum to 1 within ~1e-6. Anything off by
# more than this is not float drift but a bug (logits instead of probabilities,
# a wrong head, ...), and we refuse to publish it rather than "fix" it.
_SUM_TOLERANCE = 1e-3
_PROBABILITY_DECIMALS = 4

# Environment switches read by Ultralytics when it is first imported (see
# `configure_ultralytics_environment`).
_ULTRALYTICS_ENV = {
    # Ultralytics resolves hostnames (one.one.one.one, dns.google) at *import
    # time* to decide whether it is "online"; being online also enables its
    # usage analytics (Google Analytics events) and crash reporting, and lets it
    # download assets (fonts, weights by name). The server must make no outbound
    # calls, so we declare it offline before the first import.
    "YOLO_OFFLINE": "1",
    # Never let a library `pip install` packages from inside a request/startup.
    "YOLO_AUTOINSTALL": "False",
    # Keep Ultralytics' console chatter out of our structured logs.
    "YOLO_VERBOSE": "False",
}


def configure_ultralytics_environment() -> None:
    """Force Ultralytics offline/quiet. Must run BEFORE `import ultralytics`.

    Forced (not `setdefault`) on purpose: a stray YOLO_OFFLINE=0 left in the
    owner's shell from training must not silently turn the API server "online".
    The values are read once at import, so if Ultralytics was already imported
    elsewhere this is too late; `TreeClassifier.load` warns in that case.
    """
    for name, value in _ULTRALYTICS_ENV.items():
        os.environ[name] = value


# --- weights file handling -------------------------------------------------


def resolve_weights_path(configured: str) -> Path:
    """Absolute paths pass through; relative ones are anchored on Backend/."""
    path = Path(configured).expanduser()
    return path if path.is_absolute() else (BACKEND_DIR / path).resolve()


def _contains_checkpoint_archive(directory: Path) -> bool:
    """True if `directory` looks like an unzipped torch checkpoint.

    Unzipping a `.pt` yields `data.pkl` (+ `data/`, `version`, ...), either right
    inside the folder or one level down (an archive that has a root folder).
    """
    if not directory.is_dir():
        return False
    if (directory / "data.pkl").is_file():
        return True
    return any((child / "data.pkl").is_file() for child in directory.iterdir() if child.is_dir())


def _unzipped_message(folder: Path) -> str:
    return (
        f"Found an UNZIPPED checkpoint folder at '{folder}'. A .pt file is itself a zip "
        "archive and must be used as the original single file — never extracted or re-zipped."
    )


def stray_coco_folder_hint() -> str | None:
    """The known spec §1 situation: `Backend/yolov8n/` is an UNZIPPED copy of
    Ultralytics' stock COCO `yolov8n.pt` (task=detect, 80 classes) — not the banana
    tree model. Returns an explanation if that folder is present, else None."""
    folder = BACKEND_DIR / "yolov8n"
    if not _contains_checkpoint_archive(folder):
        return None
    return (
        f"{_unzipped_message(folder)} Backend/yolov8n/ is the stock COCO detector "
        "(task=detect, 80 classes), not the banana tree model; do not use or re-zip it."
    )


def extracted_checkpoint_hint(path: Path) -> str | None:
    """Explain the 'someone unzipped the .pt' situation (spec §1), or return None.

    Detects (1) the configured path itself being a folder, (2) a folder named
    like the weights file (`tree_cls_v1/` next to `tree_cls_v1.pt`), and (3) the
    known `Backend/yolov8n/` folder (see `stray_coco_folder_hint`).
    """
    for candidate in (path, path.with_suffix("")):
        if _contains_checkpoint_archive(candidate):
            return _unzipped_message(candidate)
    coco_hint = stray_coco_folder_hint()
    if coco_hint is not None:
        return coco_hint
    if path.is_dir():
        return (
            f"'{path}' is a folder, but a single .pt file is expected. If it was extracted "
            "from a .pt, copy the original .pt file back instead."
        )
    return None


def validate_weights_file(path: Path) -> None:
    """Cheap, torch-free checks that give actionable errors before Ultralytics runs.

    Raises ModelLoadError if the file is missing, is a folder, is not a `.pt`,
    or is not a zip archive (modern torch/Ultralytics checkpoints always are).
    """
    if path.is_dir():
        raise ModelLoadError(
            f"Weights path '{path}' is a directory, not a file. {extracted_checkpoint_hint(path)}"
        )
    if not path.exists():
        message = (
            f"Weights file not found: '{path}'. Place the trained Ultralytics classifier there "
            "(set TREE_WEIGHTS_PATH to use another location), or run "
            "`python -m scripts.make_dummy_weights` for a random stand-in."
        )
        hint = extracted_checkpoint_hint(path)
        raise ModelLoadError(f"{message} {hint}" if hint else message)
    if path.suffix.lower() != ".pt":
        # Also protects against Ultralytics' "bare model name -> download it" feature.
        raise ModelLoadError(f"Weights file '{path}' must have a .pt extension.")
    if not zipfile.is_zipfile(path):
        message = (
            f"Weights file '{path}' is not a valid .pt archive (not a zip — corrupt, truncated "
            "or not a PyTorch checkpoint). Re-copy the original file."
        )
        hint = extracted_checkpoint_hint(path)
        raise ModelLoadError(f"{message} {hint}" if hint else message)


# --- postprocess (pure) ----------------------------------------------------


@dataclass(frozen=True)
class TreeRawOutput:
    """What `infer` hands to `postprocess`: plain Python data, no torch tensors,
    so postprocessing is a pure function that is testable without torch."""

    probabilities: list[float]  # full distribution, in class-index order
    class_names: list[str]  # same order as `probabilities`


def build_tree_output(
    probabilities: Sequence[float],
    class_names: Sequence[str],
    *,
    positive_class: str,
    uncertain_threshold: float,
    display_names: Mapping[str, str],
) -> PredictionOutput:
    """Turn the model's class distribution into the API-facing prediction.

    Rules (spec §8):
    - the full distribution is sorted by probability, descending; exact ties keep
      class-index order (stable sort), so the result is deterministic;
    - `is_uncertain = top1_confidence < uncertain_threshold`, so a confidence of
      exactly 0.60 with threshold 0.60 is NOT uncertain;
    - verdict is "uncertain" if uncertain, else "banana_tree" when the top class
      is the configured positive class, else "not_banana_tree".

    The uncertain decision and `confidence` use the model's raw top-1 value (so
    `is_uncertain == (confidence < threshold)` always holds for consumers); only
    the listed probabilities are renormalised and rounded for display.

    Raises InferenceError for output that is not a probability distribution
    (NaN/Inf, negative, wrong length, sum far from 1): we never emit garbage.
    """
    count = len(probabilities)
    if count == 0 or count != len(class_names):
        raise InferenceError(
            f"Model returned {count} probabilities for {len(class_names)} classes."
        )
    if not all(math.isfinite(p) for p in probabilities):
        raise InferenceError("Model returned non-finite probabilities (NaN/Inf).")
    if any(p < 0.0 or p > 1.0 + _SUM_TOLERANCE for p in probabilities):
        raise InferenceError("Model returned probabilities outside the range 0..1.")
    total = math.fsum(probabilities)
    if abs(total - 1.0) > _SUM_TOLERANCE:
        raise InferenceError(f"Model output is not a probability distribution (sum={total:.6f}).")

    # Stable sort + reverse=True keeps the original (class-index) order among ties.
    order = sorted(range(count), key=lambda i: probabilities[i], reverse=True)
    top = order[0]
    top1 = class_names[top]
    top1conf = float(probabilities[top])

    is_uncertain = top1conf < uncertain_threshold
    if is_uncertain:
        verdict = "uncertain"
    elif top1 == positive_class:
        verdict = "banana_tree"
    else:
        verdict = "not_banana_tree"

    details = TreeDetails(
        kind="tree_classification",
        verdict=verdict,
        threshold=uncertain_threshold,
        probabilities=[
            TreeProbability(
                class_key=class_names[i],
                # A class the owner did not give a label to still gets shown, by key.
                display_name=display_names.get(class_names[i], class_names[i]),
                probability=round(probabilities[i] / total, _PROBABILITY_DECIMALS),
            )
            for i in order
        ],
    )
    return PredictionOutput(
        label=top1,
        display_label="Uncertain" if is_uncertain else display_names.get(top1, top1),
        confidence=top1conf,
        is_uncertain=is_uncertain,
        details=details,
    )


# --- the predictor ---------------------------------------------------------


class TreeClassifier(Predictor):
    key: ClassVar[str] = "tree_classification"

    def __init__(self, settings: Settings) -> None:
        cfg = settings.tree
        super().__init__(
            display_name="Banana tree classification",
            version=cfg.model_version,
            task="classify",
            is_placeholder=cfg.is_placeholder,
        )
        self._cfg = cfg
        self._device_preference = settings.inference_device
        self._device = "cpu"
        # Typed Any on purpose: ultralytics is imported lazily, and this keeps
        # the module importable (and type-checkable) without it.
        self._model: Any = None

    def load(self) -> None:
        path = resolve_weights_path(self._cfg.weights_path)
        validate_weights_file(path)

        configure_ultralytics_environment()
        try:
            from ultralytics import YOLO
            from ultralytics.utils import ONLINE

            if ONLINE:
                logger.warning(
                    "Ultralytics was imported before the offline switch was applied; "
                    "it may contact the network. Import it only through TreeClassifier.load()."
                )
            model = YOLO(str(path))
        except Exception as exc:  # corrupt pickle, missing torch, incompatible version, ...
            raise ModelLoadError(
                f"Ultralytics could not load '{path}': {type(exc).__name__}: {exc}"
            ) from exc

        task = getattr(model, "task", None)
        if task != "classify":
            raise ModelLoadError(
                f"Weights '{path}' are not a classifier: expected classify, got {task} "
                "— wrong weights?"
            )

        names: dict[int, str] = dict(model.names)
        class_names = [names[index] for index in sorted(names)]
        self._validate_classes(class_names)

        try:
            self._device = resolve_device(self._device_preference)
            model.to(self._device)
        except Exception as exc:
            raise ModelLoadError(
                f"Could not move the model to device '{self._device_preference}': "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        self._model = model
        self.classes = class_names
        # Deliberately NOT mark_ready() here: the registry marks the model ready
        # only after a warm-up inference has also succeeded.
        logger.info(
            "Loaded tree classifier '%s' (classes=%s, device=%s, placeholder=%s)",
            self.version,
            class_names,
            self._device,
            self.is_placeholder,
        )

    def _validate_classes(self, class_names: list[str]) -> None:
        """The configured class names must exist in the checkpoint (spec §4)."""
        wanted = {self._cfg.positive_class, *self._cfg.display_names}
        missing = sorted(wanted - set(class_names))
        if missing or len(class_names) < 2:
            raise ModelLoadError(
                f"Configured classes do not match the model. Model classes: {class_names}. "
                f"TREE_POSITIVE_CLASS={self._cfg.positive_class!r}, TREE_DISPLAY_NAMES keys="
                f"{sorted(self._cfg.display_names)}; not found in the model: {missing or 'none'}"
                + ("; a classifier needs at least 2 classes." if len(class_names) < 2 else ".")
            )
        unlabelled = [name for name in class_names if name not in self._cfg.display_names]
        if unlabelled:
            logger.warning(
                "Model classes without a TREE_DISPLAY_NAMES entry (shown by key): %s", unlabelled
            )

    # --- stages ------------------------------------------------------------
    def preprocess(self, image: Image.Image) -> Image.Image:
        # Ultralytics applies its own classify transforms (resize shortest edge
        # to imgsz, then centre-crop), exactly as in training/validation. Doing
        # our own would silently shift the input distribution, so we only make
        # sure the mode is RGB (the shared pipeline already guarantees it).
        return image if image.mode == "RGB" else image.convert("RGB")

    def infer(self, x: Image.Image) -> TreeRawOutput:
        if self._model is None:
            raise ModelUnavailableError(f"Model '{self.key}' is not loaded.")
        result = self._model.predict(x, imgsz=self._cfg.imgsz, device=self._device, verbose=False)[
            0
        ]
        if result.probs is None:
            raise InferenceError("Classifier returned no probabilities.")
        # Detach from torch here so postprocess is a pure function of plain floats.
        values = [float(v) for v in result.probs.data.detach().cpu().tolist()]
        return TreeRawOutput(probabilities=values, class_names=list(self.classes))

    def postprocess(self, raw: TreeRawOutput) -> PredictionOutput:
        return build_tree_output(
            raw.probabilities,
            raw.class_names,
            positive_class=self._cfg.positive_class,
            uncertain_threshold=self._cfg.uncertain_threshold,
            display_names=self._cfg.display_names,
        )
