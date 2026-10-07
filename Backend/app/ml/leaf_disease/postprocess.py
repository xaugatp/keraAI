"""Model 3 post-processing: probability maps -> clean masks, metrics, diagnosis.

Pure functions on numpy arrays (no torch, no I/O, deterministic) — same shape as
`app.ml.leaf_seg.postprocess` (ADR 0012), extended from 2 channels (leaf, affected) to 3
(leaf, black_sigatoka, yellow_sigatoka) per ADR 0019.

Pipeline (order matters, mirrors Model 2):

1. Upsample the PROBABILITIES to the original photo size (bilinear).
2. Leaf: threshold -> morphological close + open -> fill interior holes -> drop speckle components.
3. Each disease channel: threshold, restricted to a slightly dilated leaf, drop tiny blobs, clip
   to the leaf. Each disease gets its OWN threshold (rarer classes often need a lower bar).
4. Metrics from the FINAL masks only (D-06: nothing is invented).
5. Diagnosis: whichever disease channel covers the largest leaf-relative area above its own
   threshold wins; no leaf -> `no_leaf`; no disease above threshold -> `healthy`.

HONESTY NOTE: `yellow_sigatoka` has only 7 images in the whole training dataset (see the training
notebook's "Known limitations") — treat any result on that channel as directional, not reliable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

# Reuse Model 2's mask clean-up exactly — same morphology, same component filtering.
from app.ml.leaf_seg.postprocess import clean_affected_mask, clean_leaf_mask
from app.schemas.leaf_disease import (
    DiseaseChannelMetrics,
    LeafDiseaseDetails,
    LeafDiseaseThresholds,
)

LeafDiseaseLabel = Literal["healthy", "black_sigatoka", "yellow_sigatoka", "no_leaf"]

DISEASE_CHANNELS: tuple[Literal["black_sigatoka", "yellow_sigatoka"], ...] = (
    "black_sigatoka",
    "yellow_sigatoka",
)
CHANNEL_NAMES = ("leaf", *DISEASE_CHANNELS)

DISPLAY_LABELS: dict[LeafDiseaseLabel, str] = {
    "healthy": "No disease detected",
    "black_sigatoka": "Black Sigatoka detected",
    "yellow_sigatoka": "Yellow Sigatoka detected",
    "no_leaf": "No banana leaf found",
}

_AREA_EPSILON = 1e-6


@dataclass(frozen=True, eq=False)
class DiseaseResult:
    mask: NDArray[np.bool_]
    area_pct_of_leaf: float
    lesion_count: int
    largest_lesion_pct_of_leaf: float
    mean_probability: float | None


@dataclass(frozen=True, eq=False)
class DiagnosisResult:
    """Final masks at ORIGINAL resolution plus every metric derived from them."""

    leaf_mask: NDArray[np.bool_]
    leaf_area_pct_of_image: float
    mean_leaf_probability: float | None
    diseases: dict[str, DiseaseResult]
    diagnosis: LeafDiseaseLabel


def _pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return min(100.0, round(100.0 * numerator / denominator, 2))


def _mean_prob(prob: NDArray[np.float32], mask: NDArray[np.bool_]) -> float | None:
    if not mask.any():
        return None
    return round(float(prob[mask].mean(dtype=np.float64)), 4)


def _label_components(mask: NDArray[np.bool_]) -> NDArray[np.int64]:
    import cv2

    _, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return stats[1:, cv2.CC_STAT_AREA].astype(np.int64)


def postprocess_probabilities(
    probs: NDArray[np.float32],
    original_size: tuple[int, int],
    thresholds: LeafDiseaseThresholds,
) -> DiagnosisResult:
    """Probability maps (3, S, S) [leaf, black_sigatoka, yellow_sigatoka] -> result."""
    full = upsample_probabilities_3ch(probs, original_size)
    p_leaf = full[0]
    height, width = p_leaf.shape

    raw_leaf = p_leaf >= np.float32(thresholds.leaf)
    leaf = clean_leaf_mask(raw_leaf)
    leaf_pixels = int(leaf.sum())
    leaf_area_pct = _pct(leaf_pixels, height * width)

    disease_thresholds = {
        "black_sigatoka": thresholds.black_sigatoka,
        "yellow_sigatoka": thresholds.yellow_sigatoka,
    }
    diseases: dict[str, DiseaseResult] = {}
    for idx, channel in enumerate(DISEASE_CHANNELS, start=1):
        p_channel = full[idx]
        raw = p_channel >= np.float32(disease_thresholds[channel])
        mask = clean_affected_mask(raw, leaf, thresholds.min_lesion_pct)
        areas = _label_components(mask)
        largest = int(areas.max()) if areas.size else 0
        diseases[channel] = DiseaseResult(
            mask=mask,
            area_pct_of_leaf=_pct(int(mask.sum()), leaf_pixels),
            lesion_count=int(areas.size),
            largest_lesion_pct_of_leaf=_pct(largest, leaf_pixels),
            mean_probability=_mean_prob(p_channel, mask),
        )

    diagnosis = decide_diagnosis(leaf_area_pct, diseases, thresholds)
    return DiagnosisResult(
        leaf_mask=leaf,
        leaf_area_pct_of_image=leaf_area_pct,
        mean_leaf_probability=_mean_prob(p_leaf, leaf),
        diseases=diseases,
        diagnosis=diagnosis,
    )


def decide_diagnosis(
    leaf_area_pct_of_image: float,
    diseases: dict[str, DiseaseResult],
    thresholds: LeafDiseaseThresholds,
) -> LeafDiseaseLabel:
    if leaf_area_pct_of_image <= 0.0 or leaf_area_pct_of_image < thresholds.min_leaf_pct:
        return "no_leaf"
    eligible = {
        name: result.area_pct_of_leaf
        for name, result in diseases.items()
        if result.area_pct_of_leaf + _AREA_EPSILON >= thresholds.min_disease_pct
    }
    if not eligible:
        return "healthy"
    winner = max(eligible, key=lambda name: eligible[name])
    return winner  # type: ignore[return-value]  # winner is always a DISEASE_CHANNELS member


def upsample_probabilities_3ch(
    probs: NDArray[np.float32], original_size: tuple[int, int]
) -> NDArray[np.float32]:
    """(3, S, S) -> (3, H, W). Thin wrapper: Model 2's helper only asserts 2 channels."""
    height, width = original_size
    if probs.ndim != 3 or probs.shape[0] != len(CHANNEL_NAMES):
        raise ValueError(
            f"expected probability maps of shape ({len(CHANNEL_NAMES)}, S, S), got {probs.shape}"
        )
    import cv2

    probs = np.asarray(probs, dtype=np.float32)
    if probs.shape[1:] == (height, width):
        return probs
    return np.stack(
        [
            cv2.resize(
                np.ascontiguousarray(channel), (width, height), interpolation=cv2.INTER_LINEAR
            )
            for channel in probs
        ]
    )


def build_details(result: DiagnosisResult, thresholds: LeafDiseaseThresholds) -> LeafDiseaseDetails:
    return LeafDiseaseDetails(
        kind="leaf_disease",
        diagnosis=result.diagnosis,
        leaf_area_pct_of_image=result.leaf_area_pct_of_image,
        mean_leaf_probability=result.mean_leaf_probability,
        diseases=[
            DiseaseChannelMetrics(
                channel=channel,
                area_pct_of_leaf=disease.area_pct_of_leaf,
                lesion_count=disease.lesion_count,
                largest_lesion_pct_of_leaf=disease.largest_lesion_pct_of_leaf,
                mean_probability=disease.mean_probability,
            )
            for channel, disease in result.diseases.items()
        ],
        thresholds=thresholds,
    )


def average_flip_probs(
    probs: NDArray[np.float32], probs_flipped: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Same averaging rule as Model 2 (ADR 0012): mirror the flipped pass back, then average
    probabilities (not logits or binary masks) so the result stays calibrated for the thresholds.
    """
    if probs.shape != probs_flipped.shape:
        raise ValueError(f"shape mismatch: {probs.shape} vs {probs_flipped.shape}")
    return np.ascontiguousarray(
        (probs + probs_flipped[..., ::-1]) * np.float32(0.5), dtype=np.float32
    )


__all__ = [
    "CHANNEL_NAMES",
    "DISEASE_CHANNELS",
    "DISPLAY_LABELS",
    "DiagnosisResult",
    "DiseaseResult",
    "LeafDiseaseLabel",
    "average_flip_probs",
    "build_details",
    "decide_diagnosis",
    "postprocess_probabilities",
    "upsample_probabilities_3ch",
]
