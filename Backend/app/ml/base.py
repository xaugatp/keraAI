"""Model-agnostic prediction contract (spec §8).

Adding Model 2/3 means a new `Predictor` subclass — the service, storage,
repository and history endpoints never learn which model they are talking to.
This module must stay importable WITHOUT torch/ultralytics: the fast test run
and the degraded "no weights" startup both rely on that. Subclasses import the
heavy libraries lazily, inside `load()`.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from PIL import Image
from pydantic import BaseModel

from app.core.errors import AppError, InferenceError, ModelUnavailableError

logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    """Weights are missing/invalid. Startup-only: the registry catches it and
    marks the model `unavailable` instead of crashing the app (degraded mode)."""


@dataclass(frozen=True)
class Timings:
    preprocess_ms: int = 0
    inference_ms: int = 0
    postprocess_ms: int = 0

    @property
    def model_total_ms(self) -> int:
        return self.preprocess_ms + self.inference_ms + self.postprocess_ms


@dataclass(frozen=True)
class ModelInfo:
    key: str
    display_name: str
    version: str
    task: str
    classes: list[str]
    status: Literal["ready", "unavailable"]
    reason: str | None
    # Owner-approved addition to the spec's ModelInfo: lets the UI badge results
    # as "demo weights" while the checkpoint is a randomly-initialised stand-in.
    is_placeholder: bool = False


@dataclass
class PredictionOutput:
    label: str | None  # raw class key, e.g. "banana_tree"
    display_label: str  # human label, or "Uncertain"
    confidence: float | None  # 0..1; None when a single number would be invented
    is_uncertain: bool
    details: BaseModel  # model-specific payload (TreeDetails, LeafSegDetails, ...)
    result_image: Image.Image | None = None  # overlay for segmentation models
    timings: Timings = field(default_factory=Timings)


class Predictor(ABC):
    """Template-method base: preprocess -> infer -> postprocess, timed and locked."""

    key: ClassVar[str]

    def __init__(
        self,
        *,
        display_name: str,
        version: str,
        task: str,
        is_placeholder: bool = False,
    ) -> None:
        self.display_name = display_name
        self.version = version
        self.task = task
        self.is_placeholder = is_placeholder
        self.classes: list[str] = []
        self._ready = False
        self._reason: str | None = "not loaded yet"
        # P-03: one model copy in memory and Ultralytics/torch modules are not
        # guaranteed thread-safe, while the service calls us from FastAPI's
        # threadpool. Serialising predict() per model is the simple, safe answer.
        self._lock = threading.Lock()

    # --- lifecycle -------------------------------------------------------
    @abstractmethod
    def load(self) -> None:
        """Load + validate weights. Raise ModelLoadError with an actionable reason."""

    def warmup_image(self) -> Image.Image:
        return Image.new("RGB", (256, 256), (128, 128, 128))

    def warmup(self) -> None:
        """One dummy inference so the first real request doesn't pay lazy-init cost."""
        self.predict(self.warmup_image())

    def mark_ready(self) -> None:
        self._ready = True
        self._reason = None

    def mark_unavailable(self, reason: str) -> None:
        self._ready = False
        self._reason = reason

    @property
    def is_ready(self) -> bool:
        return self._ready

    # --- pipeline stages -------------------------------------------------
    @abstractmethod
    def preprocess(self, image: Image.Image) -> Any:
        """Model-specific input prep (resize/normalise/tensor). RGB PIL in.

        Segmentation models need the ORIGINAL size/pixels again in postprocess
        (to map masks back and draw the overlay). The spec's stage signatures
        stay as-is, so the carrier is the value flowing through the stages:
        return a small dataclass here that includes the original image, have
        `infer` pass it along with the raw outputs, and unpack it in postprocess.
        """

    @abstractmethod
    def infer(self, x: Any) -> Any:
        """Run the network. Returns raw outputs only — no business rules."""

    @abstractmethod
    def postprocess(self, raw: Any) -> PredictionOutput:
        """Raw outputs -> PredictionOutput (thresholds, cleanup, labels, overlay)."""

    def predict(self, image: Image.Image) -> PredictionOutput:
        if not self._ready:
            raise ModelUnavailableError(f"Model '{self.key}' is not available.")
        with self._lock:
            try:
                t0 = time.perf_counter()
                x = self.preprocess(image)
                t1 = time.perf_counter()
                raw = self.infer(x)
                t2 = time.perf_counter()
                output = self.postprocess(raw)
                t3 = time.perf_counter()
            except AppError:
                raise
            except Exception as exc:
                # Detail stays internal (logged); clients only ever see INFERENCE_FAILED.
                logger.exception("inference_failed", extra={"model_key": self.key})
                raise InferenceError(f"{type(exc).__name__}: {exc}") from exc
        output.timings = Timings(
            preprocess_ms=round((t1 - t0) * 1000),
            inference_ms=round((t2 - t1) * 1000),
            postprocess_ms=round((t3 - t2) * 1000),
        )
        return output

    # --- introspection ---------------------------------------------------
    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            key=self.key,
            display_name=self.display_name,
            version=self.version,
            task=self.task,
            classes=list(self.classes),
            status="ready" if self._ready else "unavailable",
            reason=self._reason,
            is_placeholder=self.is_placeholder,
        )
