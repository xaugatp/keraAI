"""Test doubles for the ML layer.

`FakePredictor` stands in for a real model in the service/API tests (Phase 5) and
in registry tests: no torch, no weights, instant, deterministic. Because it is a
real `Predictor` subclass, the template method in `Predictor.predict` (lock,
timings, error wrapping, readiness gate) runs for real around it.

Typical use::

    registry = ModelRegistry.from_predictors([FakePredictor("tree_classification")])
    broken = FakePredictor("tree_classification", fail=True)       # -> InferenceError
    down = FakePredictor("tree_classification", ready=False)       # -> 503 MODEL_UNAVAILABLE
"""

from __future__ import annotations

from typing import Any, ClassVar

from PIL import Image
from pydantic import BaseModel

from app.ml.base import PredictionOutput, Predictor
from app.schemas.tree import TreeDetails, TreeProbability


def make_tree_details(
    *,
    top: str = "banana_tree",
    confidence: float = 0.947,
    threshold: float = 0.6,
) -> TreeDetails:
    """A valid two-class `TreeDetails` (top class + the rest), for fakes and assertions."""
    other = "non_banana" if top == "banana_tree" else "banana_tree"
    names = {"banana_tree": "Banana tree", "non_banana": "Not a banana tree"}
    if confidence < threshold:
        verdict = "uncertain"
    else:
        verdict = "banana_tree" if top == "banana_tree" else "not_banana_tree"
    return TreeDetails(
        kind="tree_classification",
        verdict=verdict,
        threshold=threshold,
        probabilities=[
            TreeProbability(class_key=top, display_name=names[top], probability=confidence),
            TreeProbability(
                class_key=other, display_name=names[other], probability=round(1 - confidence, 4)
            ),
        ],
    )


class FakePredictor(Predictor):
    """Deterministic predictor with knobs for every behaviour tests need.

    Args:
        key: model key this fake answers to (e.g. "tree_classification").
        details: the model-specific payload to return. Defaults to a valid
            `TreeDetails` for `label`. A fresh deep copy is returned on every call
            (the base class mutates the output's timings, so sharing would leak
            state between calls).
        label / display_label / confidence / is_uncertain: the prediction fields.
        result_image: optional overlay image (copied per call), as Model 2/3 produce.
        fail: when True, `infer` raises -> `Predictor.predict` wraps it in
            `InferenceError`. Mutable at runtime (`fake.fail = True`).
        error: the exception raised when `fail` is set. Default `RuntimeError`; pass
            an `AppError` to check that `predict` lets those through unwrapped.
        ready: initial readiness. `False` -> `predict()` raises ModelUnavailableError.
        load_error / warmup_error: raised from `load()` / `warmup()` to simulate
            startup failures (registry degraded-mode tests).
    """

    key: ClassVar[str] = "fake_model"

    def __init__(
        self,
        key: str = "fake_model",
        *,
        details: BaseModel | None = None,
        label: str | None = "banana_tree",
        display_label: str = "Banana tree",
        confidence: float | None = 0.947,
        is_uncertain: bool = False,
        result_image: Image.Image | None = None,
        fail: bool = False,
        error: Exception | None = None,
        ready: bool = True,
        load_error: Exception | None = None,
        warmup_error: Exception | None = None,
        version: str = "fake_v1",
        display_name: str = "Fake model",
        task: str = "classify",
        classes: list[str] | None = None,
        is_placeholder: bool = False,
    ) -> None:
        super().__init__(
            display_name=display_name,
            version=version,
            task=task,
            is_placeholder=is_placeholder,
        )
        # `key` is a ClassVar on the real predictors; a fake needs it per instance.
        self.key = key  # type: ignore[misc]
        self.classes = list(classes) if classes is not None else ["banana_tree", "non_banana"]
        self.details: BaseModel = details if details is not None else make_tree_details()
        self.label = label
        self.display_label = display_label
        self.confidence = confidence
        self.is_uncertain = is_uncertain
        self.result_image = result_image
        self.fail = fail
        self.error = error
        self.load_error = load_error
        self.warmup_error = warmup_error
        # Observability for assertions.
        self.load_calls = 0
        self.warmup_calls = 0
        self.predict_images: list[Image.Image] = []
        if ready:
            self.mark_ready()

    # --- lifecycle ---------------------------------------------------------
    def load(self) -> None:
        self.load_calls += 1
        if self.load_error is not None:
            raise self.load_error

    def warmup(self) -> None:
        self.warmup_calls += 1
        if self.warmup_error is not None:
            raise self.warmup_error
        super().warmup()

    # --- stages ------------------------------------------------------------
    def preprocess(self, image: Image.Image) -> Image.Image:
        self.predict_images.append(image)
        return image

    def infer(self, x: Any) -> Any:
        if self.fail:
            raise self.error if self.error is not None else RuntimeError("fake inference failure")
        return x

    def postprocess(self, raw: Any) -> PredictionOutput:
        return PredictionOutput(
            label=self.label,
            display_label=self.display_label,
            confidence=self.confidence,
            is_uncertain=self.is_uncertain,
            details=self.details.model_copy(deep=True),
            result_image=self.result_image.copy() if self.result_image is not None else None,
        )
