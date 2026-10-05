"""Model registry: key -> Predictor, with load/warm-up/status (spec §8).

Degraded mode is the whole point of this module: a missing weights file, a
broken model module or a failing warm-up must never stop the API from starting.
Such a model is simply marked `unavailable` (with the reason), `/health/ready`
reports it, and `get()` answers 503 MODEL_UNAVAILABLE for it.

Predictor classes are referenced by "module:Class" import strings and imported
lazily, so even a SyntaxError in one model's module only takes that one model
down. This module itself never imports torch/ultralytics.
"""

from __future__ import annotations

import importlib
import logging
import re
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass
from typing import cast

from app.core.config import LeafSegModelSettings, Settings, TreeModelSettings
from app.core.errors import ModelUnavailableError
from app.ml.base import ModelInfo, ModelLoadError, Predictor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    """Static facts about one model the registry knows how to build."""

    key: str
    factory: str  # "package.module:ClassName", imported lazily
    # Used only when the predictor object cannot be built (disabled, import
    # error): a built predictor reports its own display name / task / version.
    display_name: str
    task: str
    settings_group: Callable[[Settings], TreeModelSettings | LeafSegModelSettings]

    def enabled(self, settings: Settings) -> bool:
        return self.settings_group(settings).enabled


# Adding Model 3 = one new line here (plus its Predictor class and settings).
MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="tree_classification",
        factory="app.ml.tree_classifier:TreeClassifier",
        display_name="Banana tree classification",
        task="classify",
        settings_group=lambda s: s.tree,
    ),
    ModelSpec(
        key="leaf_segmentation",
        factory="app.ml.leaf_segmenter:LeafSegmenter",
        display_name="Leaf segmentation",
        task="segment",
        settings_group=lambda s: s.leaf_seg,
    ),
)

# Shown by GET /models as "coming soon". It has no predictor and no settings:
# it exists only so the UI can list what is planned (CLAUDE.md model table).
PLANNED_MODEL_INFOS: tuple[ModelInfo, ...] = (
    ModelInfo(
        key="leaf_disease",
        display_name="Leaf disease detection",
        version="n/a",
        task="n/a",
        classes=[],
        status="unavailable",
        reason="Planned — not built yet",
        is_placeholder=False,
    ),
)


# --- reasons shown to clients -----------------------------------------------

# /health/ready and GET /models are client-visible (the API sits behind a public
# tunnel) and spec §11 says never leak file paths. Load errors deliberately name
# the exact path for the *logs* and for `scripts.check_env`, so the registry
# keeps the full text in the log and exposes only a path-free version.
_QUOTED_PATH = re.compile(r"""(['"])(?:[A-Za-z]:)?[\\/][^'"\r\n]*\1""")
_BARE_WINDOWS_PATH = re.compile(r"""[A-Za-z]:[\\/][^\s'"]*[^\s'".,;:)]""")


def _basename(path_text: str) -> str:
    parts = [part for part in re.split(r"[\\/]", path_text) if part]
    return parts[-1] if parts else path_text


def redact_paths(text: str) -> str:
    """Replace absolute file paths in `text` by their file name."""

    def quoted(match: re.Match[str]) -> str:
        quote = match.group(1)
        return f"{quote}{_basename(match.group(0)[1:-1])}{quote}"

    text = _QUOTED_PATH.sub(quoted, text)
    return _BARE_WINDOWS_PATH.sub(lambda match: _basename(match.group(0)), text)


# --- construction ------------------------------------------------------------


def import_factory(path: str) -> Callable[[Settings], Predictor]:
    """Resolve a "module:Class" string to the class (imports the module)."""
    module_name, _, attribute = path.partition(":")
    module = importlib.import_module(module_name)
    return cast(Callable[[Settings], Predictor], getattr(module, attribute))


def _bring_up(predictor: Predictor) -> str | None:
    """load -> warm up -> ready. Returns None on success, else the failure reason.

    `Predictor.warmup()` runs through `predict()`, which refuses while the model
    is not ready, so the model is flagged ready just before warming up and
    flagged unavailable again if the warm-up fails. Nothing can observe the
    brief ready state: this runs during startup, before any request is served.
    """
    try:
        predictor.load()
    except ModelLoadError as exc:
        return str(exc)  # already an actionable, human-written message
    except Exception as exc:
        logger.exception("Loading model '%s' crashed unexpectedly", predictor.key)
        return f"Unexpected error while loading: {type(exc).__name__}: {exc}"

    predictor.mark_ready()
    try:
        predictor.warmup()
    except Exception as exc:
        logger.exception("Warm-up of model '%s' failed", predictor.key)
        return f"Warm-up inference failed: {type(exc).__name__}: {exc}"
    return None


class ModelRegistry:
    """Holds every predictor the app can serve and reports their status.

    Build it with `build_registry(settings)` (production) or
    `ModelRegistry.from_predictors([...])` (tests).
    """

    def __init__(self) -> None:
        self._predictors: dict[str, Predictor] = {}
        # Models without a Predictor object (disabled, or construction failed).
        self._static_info: dict[str, ModelInfo] = {}
        self._order: list[str] = []
        # Keys that SHOULD be serving; /health/ready is only "ok" if all are ready.
        self._enabled: set[str] = set()

    @classmethod
    def from_predictors(cls, predictors: Iterable[Predictor]) -> ModelRegistry:
        """Registry around already-built predictors (tests, fakes). All count as enabled."""
        registry = cls()
        for predictor in predictors:
            registry._add_predictor(predictor)
        return registry

    # --- internal registration ---------------------------------------------
    def _add_predictor(self, predictor: Predictor) -> None:
        if predictor.key not in self._order:
            self._order.append(predictor.key)
        self._predictors[predictor.key] = predictor
        self._enabled.add(predictor.key)

    def _add_static(self, info: ModelInfo, *, enabled: bool) -> None:
        if info.key not in self._order:
            self._order.append(info.key)
        self._static_info[info.key] = info
        if enabled:
            self._enabled.add(info.key)

    # --- lookups -------------------------------------------------------------
    def get(self, key: str) -> Predictor:
        """The ready predictor for `key`, or ModelUnavailableError (503).

        The client-facing detail is generic on purpose; the reason is in the
        logs, `/health/ready` and `GET /models`.
        """
        predictor = self._predictors.get(key)
        if predictor is None or not predictor.is_ready:
            raise ModelUnavailableError(f"Model '{key}' is not available.")
        return predictor

    def all_info(self) -> list[ModelInfo]:
        """Status of every known model, including planned ones (for GET /models)."""
        infos = [
            self._predictors[key].info if key in self._predictors else self._static_info[key]
            for key in self._order
        ]
        known = {info.key for info in infos}
        infos.extend(planned for planned in PLANNED_MODEL_INFOS if planned.key not in known)
        return infos

    # --- readiness -------------------------------------------------------------
    def readiness(self) -> tuple[bool, str | None]:
        """(ok, detail): ok only if EVERY enabled model is ready."""
        problems: list[str] = []
        placeholders: list[str] = []
        for info in self.all_info():
            if info.key in self._enabled and info.status != "ready":
                problems.append(f"{info.key}: {info.reason or 'unavailable'}")
            elif info.status == "ready" and info.is_placeholder:
                placeholders.append(info.key)

        details: list[str] = []
        if problems:
            details.append("unavailable models — " + "; ".join(problems))
        if placeholders:
            details.append(
                "placeholder weights in use (results are not real): " + ", ".join(placeholders)
            )
        return not problems, " | ".join(details) or None

    async def readiness_check(self) -> tuple[bool, str | None]:
        """Matches `app.api.v1.endpoints.health.ReadinessCheck`; register with
        `app.state.readiness_checks["models"] = registry.readiness_check`.

        Pure in-memory read, so it is safe (and cheap) on the event loop.
        """
        return self.readiness()


def _static_unavailable_info(spec: ModelSpec, settings: Settings, reason: str) -> ModelInfo:
    """ModelInfo for a model that has no usable predictor object."""
    group = spec.settings_group(settings)
    return ModelInfo(
        key=spec.key,
        display_name=spec.display_name,
        version=group.model_version,
        task=spec.task,
        classes=[],
        status="unavailable",
        reason=reason,
        is_placeholder=group.is_placeholder,
    )


def build_registry(settings: Settings, only: Collection[str] | None = None) -> ModelRegistry:
    """Build, load and warm up every ENABLED model; never raises for a model failure.

    `only` (used by the CLI tools) restricts loading to the named models so that
    e.g. `predict_cli --model tree_classification` does not also pay to load the
    segmentation network; the others are reported as "not requested".
    """
    registry = ModelRegistry()
    for spec in MODEL_SPECS:
        if only is not None and spec.key not in only:
            info = _static_unavailable_info(spec, settings, "Not loaded (not requested)")
            registry._add_static(info, enabled=False)
            continue
        if not spec.enabled(settings):
            logger.info("Model '%s' is disabled by configuration; not loading it", spec.key)
            info = _static_unavailable_info(spec, settings, "Disabled by configuration")
            registry._add_static(info, enabled=False)
            continue

        try:
            predictor = import_factory(spec.factory)(settings)
        except Exception as exc:  # ImportError, SyntaxError in the module, bad constructor, ...
            logger.exception("Model '%s' could not be constructed (%s)", spec.key, spec.factory)
            reason = redact_paths(f"Model code failed to load: {type(exc).__name__}: {exc}")
            registry._add_static(_static_unavailable_info(spec, settings, reason), enabled=True)
            continue

        failure = _bring_up(predictor)
        registry._add_predictor(predictor)
        if failure is None:
            logger.info(
                "Model '%s' is READY (version=%s, placeholder=%s)",
                spec.key,
                predictor.version,
                predictor.is_placeholder,
            )
        else:
            logger.error("Model '%s' is UNAVAILABLE: %s", spec.key, failure)
            predictor.mark_unavailable(redact_paths(failure))
    return registry
