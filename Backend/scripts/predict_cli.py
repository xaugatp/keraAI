"""Run the REAL prediction path on image files, without a database or storage.

    python -m scripts.predict_cli --model tree_classification path\\to\\a.jpg [b.jpg ...]
    python -m scripts.predict_cli --model tree_classification --model leaf_segmentation a.jpg \\
        --save-overlay out_dir --json

It goes through exactly what production does: read bytes -> `imaging.processing.decode`
-> `normalize` -> the registry's predictor (load, warm-up, template-method `predict`).
Use it to sanity-check a checkpoint on known images before trusting the API.

With `--json`, stdout contains ONLY the JSON document (banners and errors go to
stderr), so it can be piped into other tools.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from app.core.config import get_settings
from app.core.errors import AppError
from app.imaging import processing
from app.ml.base import ModelInfo, PredictionOutput, Predictor
from app.ml.registry import MODEL_SPECS, ModelRegistry, build_registry
from app.schemas.tree import TreeDetails

BAR_WIDTH = 30


@dataclass
class Row:
    """One (image, model) result, shaped for both text and JSON output."""

    file: str
    model_key: str
    model_version: str | None = None
    is_placeholder: bool = False
    output: PredictionOutput | None = None
    overlay_path: str | None = None
    error: dict[str, str] | None = None

    def as_json(self) -> dict[str, Any]:
        out = self.output
        return {
            "file": self.file,
            "model_key": self.model_key,
            "model_version": self.model_version,
            "is_placeholder": self.is_placeholder,
            "label": out.label if out else None,
            "display_label": out.display_label if out else None,
            "confidence": out.confidence if out else None,
            "is_uncertain": out.is_uncertain if out else None,
            "timings_ms": (
                {
                    "preprocess": out.timings.preprocess_ms,
                    "inference": out.timings.inference_ms,
                    "postprocess": out.timings.postprocess_ms,
                }
                if out
                else None
            ),
            "details": out.details.model_dump(mode="json") if out else None,
            "overlay_path": self.overlay_path,
            "error": self.error,
        }


def _placeholder_banner(predictor: Predictor) -> str:
    bar = "!" * 78
    return (
        f"{bar}\n"
        f"!!  PLACEHOLDER WEIGHTS: {predictor.key} ({predictor.version})\n"
        "!!  These are random/dummy weights - the results below are NOT real predictions.\n"
        f"{bar}"
    )


def _bar(probability: float) -> str:
    return "#" * round(probability * BAR_WIDTH)


def _print_row(row: Row, out: TextIO) -> None:
    print(f"\n== {row.file}   [{row.model_key} {row.model_version}]", file=out)
    if row.error is not None:
        print(f"   ERROR {row.error['code']}: {row.error['detail']}", file=out)
        return
    result = row.output
    assert result is not None
    confidence = "n/a" if result.confidence is None else f"{result.confidence:.4f}"
    print(f"   label:         {result.label}", file=out)
    print(f"   display label: {result.display_label}", file=out)
    print(
        f"   confidence:    {confidence}   uncertain: {'yes' if result.is_uncertain else 'no'}",
        file=out,
    )
    timings = result.timings
    print(
        f"   timings (ms):  preprocess={timings.preprocess_ms} inference={timings.inference_ms} "
        f"postprocess={timings.postprocess_ms}",
        file=out,
    )
    details = result.details
    if isinstance(details, TreeDetails):
        print(f"   verdict:       {details.verdict}   (threshold {details.threshold})", file=out)
        print("   distribution:", file=out)
        width = max(len(p.class_key) for p in details.probabilities)
        for p in details.probabilities:
            print(
                f"     {p.class_key.ljust(width)}  {p.probability:.4f}  {_bar(p.probability)}"
                f"  ({p.display_name})",
                file=out,
            )
    else:
        print("   details:", file=out)
        block = json.dumps(details.model_dump(mode="json"), indent=2)
        print("\n".join(f"     {line}" for line in block.splitlines()), file=out)
    if row.overlay_path:
        print(f"   overlay saved: {row.overlay_path}", file=out)


def _unavailable_reason(registry: ModelRegistry, key: str) -> str:
    info: ModelInfo | None = next((i for i in registry.all_info() if i.key == key), None)
    if info is None:
        return "unknown model"
    return info.reason or "unavailable"


def run(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    settings = get_settings()
    models: list[str] = list(dict.fromkeys(args.model or ["tree_classification"]))
    registry = build_registry(settings, only=models)

    predictors: dict[str, Predictor] = {}
    rows: list[Row] = []
    failures = 0
    for key in models:
        try:
            predictor = registry.get(key)
        except AppError:
            print(
                f"Model '{key}' is UNAVAILABLE: {_unavailable_reason(registry, key)}\n"
                "Run `python -m scripts.check_env` for details.",
                file=err,
            )
            failures += 1
            continue
        predictors[key] = predictor
        if predictor.is_placeholder:
            # In --json mode stdout must stay pure JSON, so the banner goes to stderr.
            print(_placeholder_banner(predictor), file=err if args.json else out)

    overlay_dir = Path(args.save_overlay) if args.save_overlay else None
    if overlay_dir is not None:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    for raw_path in args.paths:
        path = Path(raw_path)
        try:
            data = path.read_bytes()
            normalized = processing.normalize(processing.decode(data, settings), settings)
        except (OSError, AppError) as exc:
            code = exc.code if isinstance(exc, AppError) else "FILE_ERROR"
            detail = exc.detail if isinstance(exc, AppError) else f"{type(exc).__name__}: {exc}"
            for key in predictors:
                rows.append(
                    Row(
                        path.name,
                        key,
                        predictors[key].version,
                        error={"code": code, "detail": detail},
                    )
                )
            failures += 1
            continue

        for key, predictor in predictors.items():
            row = Row(path.name, key, predictor.version, is_placeholder=predictor.is_placeholder)
            try:
                row.output = predictor.predict(normalized.image)
            except AppError as exc:
                row.error = {"code": exc.code, "detail": exc.detail}
                failures += 1
            else:
                image = row.output.result_image
                if overlay_dir is not None and image is not None:
                    target = overlay_dir / f"{path.stem}.{key}.png"
                    image.save(target)
                    row.overlay_path = str(target)
            rows.append(row)

    if args.json:
        print(json.dumps([row.as_json() for row in rows], indent=2), file=out)
    else:
        for row in rows:
            _print_row(row, out)
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a KeraAI model on image files (no database, no storage)."
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=[spec.key for spec in MODEL_SPECS],
        help="model to run; repeat for several (default: tree_classification)",
    )
    parser.add_argument("paths", nargs="+", help="image file(s) to analyse")
    parser.add_argument("--save-overlay", metavar="DIR", help="save result overlays (PNG) here")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON only")
    return parser


def main(argv: list[str] | None = None) -> int:
    # Windows consoles may default to a legacy code page; never crash on output.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    return run(args, sys.stdout, sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
