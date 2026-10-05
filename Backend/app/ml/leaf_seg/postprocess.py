"""Model 2 post-processing: probability maps -> clean masks, metrics, label.

Pure functions on numpy arrays (no torch, no I/O, deterministic), so every rule
below is unit-tested with synthetic maps whose answers are known analytically.

Pipeline (order matters):

1. Upsample the PROBABILITIES to the original photo size (bilinear). Thresholding
   a smooth map at full resolution gives smooth boundaries; upsampling a binary
   512x512 mask would give staircase edges.
2. Leaf: threshold -> morphological close + open -> fill interior holes ->
   drop speckle components.
3. Affected: threshold, restricted to a slightly dilated leaf (damage outside the
   leaf is a false positive by construction: in the training data `affected` is
   a sub-region of `leaf`), drop tiny blobs, then clip to the leaf.
4. Metrics from the FINAL masks only (D-06: nothing is invented).
5. Label from the metrics and the configured thresholds.

HONESTY NOTES (the training set is tiny â€” see the training notebook):
* 50 images, 8 test images, only 2 fully healthy leaves. The "healthy" verdict
  is therefore UNVALIDATED: it means "no damage above our size thresholds was
  found", not "the leaf is healthy".
* The affected threshold (0.85) was tuned on the 8-image test split, so it is
  optimistic; small scattered lesions tend to be merged into blobs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from app.schemas.leaf_seg import LeafSegDetails, LeafSegThresholds

LeafSegLabel = Literal["affected", "healthy", "no_leaf"]

DISPLAY_LABELS: dict[LeafSegLabel, str] = {
    "affected": "Damaged tissue found",
    "healthy": "No damage detected",
    "no_leaf": "No banana leaf found",
}

# --- post-processing parameters that are NOT in the notebook -----------------
# All are fractions of the image's SHORTER side (so they scale with photo size)
# or of areas; they are deliberately module constants, not settings: they are
# implementation detail of the clean-up, while the user-facing policy knobs
# (thresholds and min-area percentages) live in `LeafSegThresholds`/settings.
MORPH_KERNEL_FRACTION = 0.005  # close/open kernel diameter, ~0.5% of the short side
DILATION_KERNEL_FRACTION = 0.01  # leaf-edge tolerance for lesions, ~1% (kernel diameter)
MIN_KERNEL_PX = 3  # smaller than a 3x3 structuring element does nothing
MIN_LEAF_COMPONENT_IMAGE_FRACTION = 0.01  # leaf blobs under 1% of the image are speckle
MIN_LEAF_COMPONENT_VS_LARGEST = 0.20  # ...or under 20% of the largest blob (keeps several leaves)
MIN_LESION_PIXELS = 4  # absolute floor, whatever min_lesion_pct says
# Area comparisons are ">= threshold" on floats derived from percentages; this
# slack stops 1 ulp of float error (12.000000000000002 vs 12) from flipping a
# blob that is exactly at the threshold.
_AREA_EPSILON = 1e-6


@dataclass(frozen=True, eq=False)  # eq=False: numpy fields make generated __eq__ meaningless
class SegmentationResult:
    """Final masks at ORIGINAL resolution plus every metric derived from them.

    Percentages are rounded to 2 dp and probabilities to 4 dp here, at the
    source, and the label is decided from these rounded numbers. That way a
    stored row (which only keeps the rounded values) always reproduces its own
    label from the stored thresholds.
    """

    leaf_mask: NDArray[np.bool_]
    affected_mask: NDArray[np.bool_]
    leaf_area_pct_of_image: float
    affected_area_pct_of_leaf: float
    lesion_count: int
    largest_lesion_pct_of_leaf: float
    mean_leaf_probability: float | None
    mean_affected_probability: float | None
    label: LeafSegLabel


# --- helpers ------------------------------------------------------------------


def _odd_ellipse(short_side: int, fraction: float) -> NDArray[np.uint8]:
    """Elliptical structuring element, odd-sized so it has a centre pixel (no
    half-pixel shift of the mask), min 3 px, ~`fraction` of the short side."""
    size = max(MIN_KERNEL_PX, round(short_side * fraction))
    if size % 2 == 0:
        size += 1
    return np.asarray(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)), dtype=np.uint8)


def _label_components(mask: NDArray[np.bool_]) -> tuple[NDArray[np.int32], NDArray[np.int64]]:
    """8-connected components: (label image, areas of components 1..n)."""
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return np.asarray(labels, dtype=np.int32), stats[1:, cv2.CC_STAT_AREA].astype(np.int64)


def _keep_components(labels: NDArray[np.int32], keep: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Mask of the components flagged in `keep` (index i <-> component i+1)."""
    lookup = np.zeros(keep.shape[0] + 1, dtype=np.bool_)
    lookup[1:] = keep
    return lookup[labels]


def _pct(numerator: int, denominator: int) -> float:
    """100 * n / d rounded to 2 dp; 0.0 for an empty denominator (nothing to be a share of)."""
    if denominator <= 0:
        return 0.0
    return min(100.0, round(100.0 * numerator / denominator, 2))


def _mean_prob(prob: NDArray[np.float32], mask: NDArray[np.bool_]) -> float | None:
    """Mean probability over `mask`, rounded to 4 dp; None (not 0.0) for an empty region."""
    if not mask.any():
        return None
    return round(float(prob[mask].mean(dtype=np.float64)), 4)


# --- step 1: resolution ------------------------------------------------------


def upsample_probabilities(
    probs: NDArray[np.float32], original_size: tuple[int, int]
) -> NDArray[np.float32]:
    """(2, S, S) -> (2, H, W), bilinear. `original_size` is (height, width)."""
    height, width = original_size
    if height <= 0 or width <= 0:
        raise ValueError(f"original_size must be positive, got {original_size}")
    if probs.ndim != 3 or probs.shape[0] != 2:
        raise ValueError(f"expected probability maps of shape (2, S, S), got {probs.shape}")
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


# --- step 2: leaf ------------------------------------------------------------


def fill_holes(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Fill background regions that do not touch the image border.

    Background is 4-connected and foreground 8-connected (the matching pair), so
    a leaf outline closed only diagonally still counts as closed. A notch that
    opens onto the image border is NOT a hole and stays untouched. A 1 px
    background frame joins all border-touching background into one component, so
    a single labelling pass finds "the outside".
    """
    padded = np.pad(~mask, 1, mode="constant", constant_values=True)
    _, labels = cv2.connectedComponents(padded.astype(np.uint8), connectivity=4)
    outside = labels == labels[0, 0]
    return mask | ~outside[1:-1, 1:-1]


def clean_leaf_mask(raw: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Close, open, fill holes, drop speckle components."""
    height, width = raw.shape
    if not raw.any():
        return raw.copy()
    kernel = _odd_ellipse(min(height, width), MORPH_KERNEL_FRACTION)
    mask = cv2.morphologyEx(raw.astype(np.uint8), cv2.MORPH_CLOSE, kernel)  # bridge thin gaps
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # shave off hairlines/speckle
    filled = fill_holes(mask.astype(np.bool_))

    labels, areas = _label_components(filled)
    if areas.size == 0:
        return filled
    # A component survives only if it is big in absolute terms (>= 1% of the
    # photo) AND comparable to the biggest one (>= 20%): several similar leaves
    # are all kept, a stray patch of background that fooled the model is not.
    min_area = max(
        MIN_LEAF_COMPONENT_IMAGE_FRACTION * height * width,
        MIN_LEAF_COMPONENT_VS_LARGEST * float(areas.max()),
    )
    return _keep_components(labels, areas + _AREA_EPSILON >= min_area)


# --- step 3: affected --------------------------------------------------------


def clean_affected_mask(
    raw: NDArray[np.bool_], leaf: NDArray[np.bool_], min_lesion_pct: float
) -> NDArray[np.bool_]:
    """Restrict lesions to the leaf (with edge tolerance) and drop tiny blobs.

    The stored/drawn lesion mask is: (raw & dilated leaf) -> blob-size filter ->
    & leaf. The dilation (~1% of the short side) only matters for the blob-size
    decision: a real lesion running onto a leaf edge that the leaf mask undershot
    by a few pixels is judged by its full size, then clipped to the leaf so it
    never contributes area outside it.
    """
    height, width = leaf.shape
    leaf_pixels = int(leaf.sum())
    if leaf_pixels == 0 or not raw.any():
        return np.zeros_like(leaf)

    kernel = _odd_ellipse(min(height, width), DILATION_KERNEL_FRACTION)
    tolerant_leaf = cv2.dilate(leaf.astype(np.uint8), kernel).astype(np.bool_)
    candidate = raw & tolerant_leaf

    labels, areas = _label_components(candidate)
    min_area = max(float(MIN_LESION_PIXELS), min_lesion_pct / 100.0 * leaf_pixels)
    kept = _keep_components(labels, areas + _AREA_EPSILON >= min_area) & leaf

    # Clipping can leave a sliver; apply the absolute floor again so every lesion
    # that is counted/drawn is at least MIN_LESION_PIXELS pixels.
    labels, areas = _label_components(kept)
    return _keep_components(labels, areas >= MIN_LESION_PIXELS)


# --- step 5: label -----------------------------------------------------------


def decide_label(
    leaf_area_pct_of_image: float,
    affected_area_pct_of_leaf: float,
    thresholds: LeafSegThresholds,
) -> LeafSegLabel:
    """Label from the (rounded) metrics.

    WARNING: "healthy" is UNVALIDATED. Only 2 of the 50 training images are
    healthy leaves, so the model has barely seen what a clean leaf looks like.
    Read it as "no damage above the size thresholds was found".
    """
    if leaf_area_pct_of_image <= 0.0 or leaf_area_pct_of_image < thresholds.min_leaf_pct:
        return "no_leaf"
    if affected_area_pct_of_leaf >= thresholds.min_affected_pct:
        return "affected"
    return "healthy"


# --- orchestration -----------------------------------------------------------


def postprocess_probabilities(
    probs: NDArray[np.float32],
    original_size: tuple[int, int],
    thresholds: LeafSegThresholds,
) -> SegmentationResult:
    """Probability maps (2, S, S) [leaf, affected] (sigmoid applied) -> result.

    `original_size` is (height, width) of the photo the masks must align with.
    """
    full = upsample_probabilities(probs, original_size)
    p_leaf, p_affected = full[0], full[1]
    height, width = p_leaf.shape

    # float32 thresholds so a probability exactly equal to the threshold passes
    # (>=) without float64 promotion surprises.
    raw_leaf = p_leaf >= np.float32(thresholds.leaf)
    leaf = clean_leaf_mask(raw_leaf)
    raw_affected = p_affected >= np.float32(thresholds.affected)
    affected = clean_affected_mask(raw_affected, leaf, thresholds.min_lesion_pct)

    leaf_pixels = int(leaf.sum())
    affected_pixels = int(affected.sum())
    _, lesion_areas = _label_components(affected)
    largest_lesion = int(lesion_areas.max()) if lesion_areas.size else 0

    leaf_area_pct = _pct(leaf_pixels, height * width)
    affected_pct = _pct(affected_pixels, leaf_pixels)
    return SegmentationResult(
        leaf_mask=leaf,
        affected_mask=affected,
        leaf_area_pct_of_image=leaf_area_pct,
        affected_area_pct_of_leaf=affected_pct,
        lesion_count=int(lesion_areas.size),
        largest_lesion_pct_of_leaf=_pct(largest_lesion, leaf_pixels),
        mean_leaf_probability=_mean_prob(p_leaf, leaf),
        mean_affected_probability=_mean_prob(p_affected, affected),
        label=decide_label(leaf_area_pct, affected_pct, thresholds),
    )


def build_details(result: SegmentationResult, thresholds: LeafSegThresholds) -> LeafSegDetails:
    """Result -> the API/DB `details` payload (thresholds echoed for reproducibility)."""
    return LeafSegDetails(
        leaf_area_pct_of_image=result.leaf_area_pct_of_image,
        affected_area_pct_of_leaf=result.affected_area_pct_of_leaf,
        lesion_count=result.lesion_count,
        largest_lesion_pct_of_leaf=result.largest_lesion_pct_of_leaf,
        mean_leaf_probability=result.mean_leaf_probability,
        mean_affected_probability=result.mean_affected_probability,
        thresholds=thresholds,
    )


# --- test-time augmentation --------------------------------------------------


def average_flip_probs(
    probs: NDArray[np.float32], probs_flipped: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Average the normal pass with the horizontally-flipped pass.

    `probs_flipped` is the network output for the mirrored INPUT, so it is
    mirrored too: flip it back along the width axis before averaging. Averaging
    PROBABILITIES (not logits or binary masks) keeps the maps calibrated for the
    thresholds, which were tuned on single-pass probabilities.
    """
    if probs.shape != probs_flipped.shape:
        raise ValueError(f"shape mismatch: {probs.shape} vs {probs_flipped.shape}")
    return np.ascontiguousarray(
        (probs + probs_flipped[..., ::-1]) * np.float32(0.5), dtype=np.float32
    )


__all__ = [
    "DISPLAY_LABELS",
    "LeafSegLabel",
    "SegmentationResult",
    "average_flip_probs",
    "build_details",
    "clean_affected_mask",
    "clean_leaf_mask",
    "decide_label",
    "fill_holes",
    "postprocess_probabilities",
    "upsample_probabilities",
]
