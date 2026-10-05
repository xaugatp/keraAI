"""Create DUMMY (randomly initialised) weights so the whole stack can run end-to-end
before the real trained checkpoints exist.

    python -m scripts.make_dummy_weights [--model all|tree_classification|leaf_segmentation] [--force]

The predictions of these files are MEANINGLESS. They exist to prove the plumbing
(load -> warm up -> predict -> save -> history -> UI), nothing more. Nothing is
downloaded: both networks are built from their architecture definitions with
`torch.manual_seed(0)`, on CPU, with no pretrained weights.

File formats are identical to the real thing, so a real file is a drop-in:
- Model 1: a normal Ultralytics `.pt` zip (what `YOLO.save` / training writes),
  2 classes {0: banana_tree, 1: non_banana}, task=classify.
- Model 2: a BARE `state_dict` saved with `torch.save` — exactly like the
  training notebook's `banana_leaf_segmentation_best.pt`.

Existing files are never overwritten unless `--force` is given, so a real
checkpoint cannot be clobbered by accident. This script does NOT edit `.env`.
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.core.config import Settings, get_settings
from app.ml.tree_classifier import configure_ultralytics_environment, resolve_weights_path

SEED = 0
# Class order matters: index 0 must be the positive class, like the real model.
TREE_CLASS_NAMES = {0: "banana_tree", 1: "non_banana"}

ENV_LINES = (
    "TREE_MODEL_VERSION=tree_cls_dummy_v0",
    "TREE_IS_PLACEHOLDER=true",
    "LEAF_SEG_MODEL_VERSION=leaf_seg_dummy_v0",
    "LEAF_SEG_IS_PLACEHOLDER=true",
)


def _atomic_target(path: Path) -> Path:
    """Write next to the destination, then `os.replace`, so a crash never leaves a half file."""
    return path.with_name(path.name + ".tmp")


def make_tree_dummy(path: Path) -> None:
    """2-class YOLOv8n-cls, random init, saved as a normal Ultralytics .pt."""
    # Must happen before the first `import ultralytics` (offline, no auto-install).
    configure_ultralytics_environment()
    import torch
    from ultralytics import YOLO
    from ultralytics.nn.tasks import ClassificationModel

    torch.manual_seed(SEED)
    # "yolov8n-cls.yaml" is the architecture file shipped inside the package: no
    # download, no pretrained weights. It defaults to ImageNet's 1000 classes...
    wrapper: Any = YOLO("yolov8n-cls.yaml")  # Any: Ultralytics' typing is looser than its runtime
    # ...so rebuild the network with 2 outputs (same thing the trainer does with
    # `nc` from the dataset), then put it back into the YOLO wrapper.
    network: Any = ClassificationModel(wrapper.model.yaml, nc=len(TREE_CLASS_NAMES), verbose=False)
    network.names = dict(TREE_CLASS_NAMES)
    network.task = "classify"
    wrapper.model = network

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _atomic_target(path)
    wrapper.save(tmp)
    os.replace(tmp, path)

    # Verify the way the app will use it: load by path, check task/names, predict.
    from PIL import Image

    reloaded: Any = YOLO(str(path))
    if reloaded.task != "classify" or dict(reloaded.names) != TREE_CLASS_NAMES:
        raise RuntimeError(f"Verification failed: task={reloaded.task}, names={reloaded.names}")
    probs = reloaded.predict(
        Image.new("RGB", (256, 256), (128, 128, 128)), imgsz=224, device="cpu", verbose=False
    )[0].probs
    if probs is None or len(probs.data) != len(TREE_CLASS_NAMES):
        raise RuntimeError("Verification failed: the reloaded model returned no 2-class output.")


def make_leaf_seg_dummy(path: Path, encoder: str) -> None:
    """U-Net (random init) saved as a bare state_dict, like the training notebook."""
    import segmentation_models_pytorch as smp  # type: ignore[import-untyped]
    import torch

    def build() -> torch.nn.Module:
        # encoder_weights=None -> nothing is downloaded. classes=2 (leaf, affected)
        # and activation=None (raw logits; the app applies the sigmoid) mirror the
        # notebook's model definition.
        return smp.Unet(
            encoder_name=encoder,
            encoder_weights=None,
            in_channels=3,
            classes=2,
            activation=None,
        )

    torch.manual_seed(SEED)
    model = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _atomic_target(path)
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, path)

    # Verify: a fresh model must load the file strictly and run a forward pass.
    state = torch.load(path, map_location="cpu", weights_only=True)
    check = build()
    check.load_state_dict(state, strict=True)
    check.eval()
    with torch.no_grad():
        out = check(torch.zeros(1, 3, 64, 64))
    if tuple(out.shape) != (1, 2, 64, 64) or not zipfile.is_zipfile(path):
        raise RuntimeError(f"Verification failed: unexpected output shape {tuple(out.shape)}.")


def _banner(written: list[str]) -> str:
    bar = "!" * 78
    lines = [
        bar,
        "!!  DUMMY WEIGHTS - RANDOM NUMBERS, NOT A TRAINED MODEL".ljust(76) + "!!",
        "!!  Predictions from these files are MEANINGLESS. Plumbing tests only.".ljust(76) + "!!",
        bar,
        "",
        "Written: " + (", ".join(written) if written else "(nothing)"),
        "",
        "Put these lines in Backend/.env so rows made during the dummy era are",
        "identifiable in the database and the UI can badge them as demo results:",
        "",
        *(f"    {line}" for line in ENV_LINES),
        "",
        "When the real weights arrive: replace the files, set the *_MODEL_VERSION to the",
        "real version (e.g. tree_cls_v1) and set both *_IS_PLACEHOLDER lines to false.",
        "This script did not modify .env.",
    ]
    return "\n".join(lines)


def _targets(settings: Settings, selection: str) -> list[tuple[str, Path, Callable[[Path], None]]]:
    tree_path = resolve_weights_path(settings.tree.weights_path)
    leaf_path = resolve_weights_path(settings.leaf_seg.weights_path)
    available: list[tuple[str, Path, Callable[[Path], None]]] = [
        ("tree_classification", tree_path, make_tree_dummy),
        (
            "leaf_segmentation",
            leaf_path,
            lambda path: make_leaf_seg_dummy(path, settings.leaf_seg.encoder),
        ),
    ]
    return [item for item in available if selection in ("all", item[0])]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write random-weight stand-in checkpoints (meaningless predictions)."
    )
    parser.add_argument(
        "--model",
        choices=["all", "tree_classification", "leaf_segmentation"],
        default="all",
        help="which dummy to create (default: all)",
    )
    parser.add_argument(
        "--force", action="store_true", help="overwrite existing weights files (DANGEROUS)"
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    written: list[str] = []
    refused: list[str] = []
    for key, path, make in _targets(settings, args.model):
        if path.exists() and not args.force:
            refused.append(f"{key}: '{path}' already exists")
            continue
        print(f"Creating dummy weights for {key} -> {path}")
        make(path)
        written.append(f"{path} ({path.stat().st_size / 1_000_000:.1f} MB)")

    if written:
        print()
        print(_banner(written))
    for line in refused:
        print(
            f"REFUSING to overwrite - {line}. Use --force if you really mean it.", file=sys.stderr
        )
    return 1 if refused else 0


if __name__ == "__main__":
    raise SystemExit(main())
