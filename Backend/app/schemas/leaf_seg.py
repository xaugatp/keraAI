"""Model 2 (leaf segmentation) result payload — the `details` JSON of an analysis.

Owner-approved contract (D-06: real data only). Every number here is *measured*
from the final masks / probability maps; nothing is a model "confidence" in
disguise. Conventions:

* percentages are 0-100 floats rounded to 2 decimals,
* probabilities are 0-1 floats rounded to 4 decimals,
* a probability mean is `None` when its region is empty (a mean over zero
  pixels is not 0.0 — it is "not measured").

The rounding is done once, in `app.ml.leaf_seg.postprocess`, so the label rule
and the stored numbers can never disagree; these models only validate ranges.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LeafSegThresholds(BaseModel):
    """The exact policy that produced a result, echoed so a stored row is reproducible."""

    leaf: float = Field(gt=0, lt=1, description="p(leaf) >= this -> leaf pixel")
    affected: float = Field(gt=0, lt=1, description="p(affected) >= this -> affected pixel")
    min_leaf_pct: float = Field(
        ge=0, le=100, description="leaf share of the image (%) below which the result is 'no_leaf'"
    )
    min_lesion_pct: float = Field(
        ge=0, le=100, description="lesion blobs smaller than this % of the leaf area are dropped"
    )
    min_affected_pct: float = Field(
        ge=0, le=100, description="affected share of the leaf (%) below which the leaf is 'healthy'"
    )
    tta_hflip: bool = Field(description="probabilities averaged with the horizontally flipped pass")


class LeafSegDetails(BaseModel):
    kind: Literal["leaf_segmentation"] = "leaf_segmentation"
    leaf_area_pct_of_image: float = Field(ge=0, le=100)
    affected_area_pct_of_leaf: float = Field(ge=0, le=100)
    lesion_count: int = Field(ge=0)
    largest_lesion_pct_of_leaf: float = Field(ge=0, le=100)
    mean_leaf_probability: float | None = Field(default=None, ge=0, le=1)
    mean_affected_probability: float | None = Field(default=None, ge=0, le=1)
    thresholds: LeafSegThresholds
