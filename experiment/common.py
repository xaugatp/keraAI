"""
Shared utilities for the KeraAI experiment notebooks.

Kept as a plain .py module (not copy-pasted into every notebook) so that the three
notebooks share one tested implementation of: zip extraction, reproducible seeding,
device selection, YOLO-polygon mask decoding, group-aware splitting, and the small
set of segmentation metrics all three problems need in some form.

Import from a notebook with:
    import sys; sys.path.insert(0, ".")
    from common import *
"""
from __future__ import annotations

import os
import random
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


def save_fig(path, fig=None) -> None:
    """Save the current (or given) matplotlib figure to disk before `plt.show()` clears it.
    Used by every notebook chart so graphs survive outside the .ipynb's embedded output too."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    (fig or plt.gcf()).savefig(path, dpi=150, bbox_inches="tight")


# --------------------------------------------------------------------------------------
# Reproducibility & device
# --------------------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Seed python/numpy/torch so notebook re-runs are comparable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def detect_hardware() -> dict:
    """Report what this machine actually has, so notebooks size batch/epoch budgets from
    measured capacity instead of a guessed constant (mirrors the `pos_weight` philosophy
    elsewhere in this module: measure, don't assume).
    """
    import psutil

    vm = psutil.virtual_memory()
    info = {
        "cpu_logical_cores": os.cpu_count(),
        "ram_total_gb": round(vm.total / 1e9, 1),
        "ram_available_gb": round(vm.available / 1e9, 1),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": None,
        "vram_total_gb": None,
    }
    if info["cuda_available"]:
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["vram_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    return info


def recommend_batch_size(hw: dict, base_batch: int, base_ram_gb: float = 8.0) -> int:
    """Scale a baseline batch size by available RAM (CUDA: by VRAM instead), clamped to a
    sane range. `base_batch` is what the baseline assumes at `base_ram_gb` available memory.
    """
    if hw["cuda_available"] and hw["vram_total_gb"]:
        budget_gb = hw["vram_total_gb"]
    else:
        budget_gb = hw["ram_available_gb"]
    scale = max(0.5, min(3.0, budget_gb / base_ram_gb))
    return max(2, int(round(base_batch * scale)))


def pick_device() -> str:
    """CUDA if available, else CPU — matches the backend's own auto-detect policy
    (see CLAUDE.md: 'Inference device: auto-detected'). This laptop has an Intel Arc
    GPU, not NVIDIA, so torch reports no CUDA and every notebook here trains on CPU;
    the code is unchanged if later run on a CUDA machine."""
    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------------------
# Dataset extraction
# --------------------------------------------------------------------------------------

def extract_zip_once(zip_path: Path, extract_to: Path) -> Path:
    """Extract `zip_path` into `extract_to` unless it looks already extracted.

    The raw Roboflow exports live as zips under `../Data` (git-ignored, not duplicated
    into the repo). We extract once into `data_cache/` (also git-ignored) so re-running
    a notebook doesn't re-unzip ~50MB every time.
    """
    extract_to.mkdir(parents=True, exist_ok=True)
    marker = extract_to / ".extracted"
    if marker.exists():
        return extract_to
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_to)
    marker.write_text("ok")
    return extract_to


# --------------------------------------------------------------------------------------
# YOLO-format polygon/box label parsing
# --------------------------------------------------------------------------------------

ROBOFLOW_HASH_SUFFIX = re.compile(r"_(jpg|jpeg|png)\.rf\.[a-f0-9]+$", re.IGNORECASE)


def strip_roboflow_suffix(stem: str) -> str:
    """Strip Roboflow's '_jpg.rf.<hash>' export suffix, leaving the original filename.

    NOTE: verified during the data audit (see notebook 01, Section 2.5) that this is
    NOT a reliable "same source photo" key for the tree-detection set — multiple
    distinct crops from the same capture burst share this stem. It is still useful as
    a *capture-session* grouping key for a leakage-safe split (same reasoning as the
    reference repo's `extra.name` grouping), just not as a duplicate-image detector.
    """
    return ROBOFLOW_HASH_SUFFIX.sub("", stem)


def read_yolo_label_file(path: Path) -> list[list[float]]:
    """Parse a YOLO .txt label file into a list of [class_id, *coords] float rows.
    Works for both bbox rows (5 numbers) and polygon rows (class_id + variable-length
    x,y pairs), since we only ever split on whitespace and cast to float."""
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        rows.append([float(p) for p in parts])
    return rows


def polygon_row_to_mask(row: list[float], h: int, w: int) -> np.ndarray:
    """Rasterize one YOLO-seg polygon row ([class_id, x1,y1,x2,y2,...], normalized
    [0,1]) into a (h, w) uint8 binary mask via cv2.fillPoly.

    We use our own OpenCV rasterizer rather than pycocotools (used by the reference
    repo): the Roboflow exports here already ship YOLO-polygon .txt labels, so this
    avoids an extra dependency with known Windows build friction, for an identical
    result on polygon-type annotations.
    """
    coords = np.array(row[1:], dtype=np.float32).reshape(-1, 2)
    coords[:, 0] *= w
    coords[:, 1] *= h
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [coords.round().astype(np.int32)], color=1)
    return mask


def bbox_row_to_mask(row: list[float], h: int, w: int) -> np.ndarray:
    """Rasterize one YOLO-bbox row ([class_id, cx, cy, w, h], normalized) into a
    (h, w) uint8 binary mask (filled rectangle)."""
    _, cx, cy, bw, bh = row
    x1 = int(round((cx - bw / 2) * w))
    y1 = int(round((cy - bh / 2) * h))
    x2 = int(round((cx + bw / 2) * w))
    y2 = int(round((cy + bh / 2) * h))
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[max(y1, 0):min(y2, h), max(x1, 0):min(x2, w)] = 1
    return mask


# --------------------------------------------------------------------------------------
# Group-aware splitting
# --------------------------------------------------------------------------------------

def group_shuffle_split_3way(
    items: list,
    groups: list[str],
    train_size: float = 0.70,
    val_size: float = 0.15,
    seed: int = 42,
    n_seed_trials: int = 60,
):
    """3-way split that keeps every member of a group in the same split.

    Mirrors the reference repo's approach (`GroupShuffleSplit` run twice, best-of-N
    seeds to land close to the target ratios since group sizes are uneven), generalized
    into a reusable function. Returns (train_idx, val_idx, test_idx) — positional
    indices into `items`/`groups`.
    """
    from sklearn.model_selection import GroupShuffleSplit

    idx = np.arange(len(items))
    test_size = 1.0 - train_size - val_size
    assert test_size > 0

    best = None
    for s in range(n_seed_trials):
        gss1 = GroupShuffleSplit(n_splits=1, train_size=train_size + val_size, random_state=s)
        trainval_idx, test_idx = next(gss1.split(idx, groups=groups))
        trainval_groups = [groups[i] for i in trainval_idx]
        gss2 = GroupShuffleSplit(n_splits=1, train_size=train_size / (train_size + val_size), random_state=s)
        tr_rel, va_rel = next(gss2.split(trainval_idx, groups=trainval_groups))
        train_idx = trainval_idx[tr_rel]
        val_idx = trainval_idx[va_rel]

        total = len(train_idx) + len(val_idx) + len(test_idx)
        score = (
            abs(len(train_idx) / total - train_size)
            + abs(len(val_idx) / total - val_size)
            + abs(len(test_idx) / total - test_size)
        )
        if best is None or score < best[0]:
            best = (score, s, train_idx, val_idx, test_idx)

    _, best_seed, train_idx, val_idx, test_idx = best

    g = np.array(groups)
    assert not (set(g[train_idx]) & set(g[val_idx])), "LEAKAGE: train/val share a group"
    assert not (set(g[train_idx]) & set(g[test_idx])), "LEAKAGE: train/test share a group"
    assert not (set(g[val_idx]) & set(g[test_idx])), "LEAKAGE: val/test share a group"

    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist(), best_seed


# --------------------------------------------------------------------------------------
# Segmentation metrics (shared by notebooks 02 and 03)
# --------------------------------------------------------------------------------------

def dice_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-7) -> torch.Tensor:
    """Per-channel Dice over a batch of sigmoid logits vs. binary targets.
    logits, targets: (B, C, H, W). Returns a (C,) tensor averaged over the batch."""
    probs = (torch.sigmoid(logits) > threshold).float()
    probs = probs.flatten(2)
    targets = targets.flatten(2)
    intersection = (probs * targets).sum(-1)
    union = probs.sum(-1) + targets.sum(-1)
    return ((2 * intersection + eps) / (union + eps)).mean(0)


def pixel_confusion(pred: torch.Tensor, target: torch.Tensor) -> tuple[int, int, int, int]:
    tp = ((pred == 1) & (target == 1)).sum().item()
    fp = ((pred == 1) & (target == 0)).sum().item()
    fn = ((pred == 0) & (target == 1)).sum().item()
    tn = ((pred == 0) & (target == 0)).sum().item()
    return tp, fp, fn, tn


def compute_pos_weight(mean_ratio: float) -> float:
    """BCE pos_weight from a measured positive-pixel ratio (not a guessed constant) —
    same formula as the reference repo: weight = (1 - p) / p."""
    mean_ratio = float(np.clip(mean_ratio, 1e-4, 1 - 1e-4))
    return (1 - mean_ratio) / mean_ratio


class DiceLoss(torch.nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.flatten(2)
        targets = targets.flatten(2)
        intersection = (probs * targets).sum(-1)
        union = probs.sum(-1) + targets.sum(-1)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class ComboLoss(torch.nn.Module):
    """Weighted BCE + Dice, same combination the reference repo and ADR 0012 use:
    Dice optimizes region overlap (robust to class imbalance), BCE gives calibrated
    per-pixel probabilities (which Dice alone doesn't)."""

    def __init__(self, pos_weight: torch.Tensor, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.view(1, -1, 1, 1))
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, targets) + self.dice_weight * self.dice(logits, targets)
