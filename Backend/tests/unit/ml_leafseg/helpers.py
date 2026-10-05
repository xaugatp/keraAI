"""Synthetic probability maps with analytically known answers (torch-free)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from app.schemas.leaf_seg import LeafSegThresholds

Box = tuple[int, int, int, int]  # (row0, row1, col0, col1), end-exclusive


def make_thresholds(**overrides: Any) -> LeafSegThresholds:
    values: dict[str, Any] = {
        "leaf": 0.5,
        "affected": 0.85,
        "min_leaf_pct": 3.0,
        "min_lesion_pct": 0.05,
        "min_affected_pct": 0.5,
        "tta_hflip": False,
    }
    values.update(overrides)
    return LeafSegThresholds(**values)


def make_probs(
    height: int = 200,
    width: int = 200,
    *,
    leaf: Iterable[Box] = (),
    lesions: Iterable[Box] = (),
    holes: Iterable[Box] = (),
    leaf_p: float = 0.95,
    affected_p: float = 0.95,
) -> NDArray[np.float32]:
    """(2, H, W) float32 maps: `leaf_p` inside leaf boxes (minus holes), `affected_p`
    inside lesion boxes, 0 elsewhere. Lesion boxes are NOT clipped to the leaf, so
    tests can put damage outside it on purpose."""
    probs = np.zeros((2, height, width), dtype=np.float32)
    for r0, r1, c0, c1 in leaf:
        probs[0, r0:r1, c0:c1] = leaf_p
    for r0, r1, c0, c1 in holes:
        probs[0, r0:r1, c0:c1] = 0.1
    for r0, r1, c0, c1 in lesions:
        probs[1, r0:r1, c0:c1] = affected_p
    return probs


# A full-height strip (columns 0..119) of a 200x200 image: it touches the top and
# bottom borders, which morphological close/open leave EXACTLY intact (a free-standing
# rectangle would lose its 4 corner pixels to the opening). 24000 px = 60% of the image.
STRIP: Box = (0, 200, 0, 120)
STRIP_PIXELS = 200 * 120
