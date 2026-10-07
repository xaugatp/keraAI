"""Model 3 (leaf disease identification) result payload — the `details` JSON of an analysis.

Same conventions as `app.schemas.leaf_seg` (ADR 0012), extended to one channel per disease
(ADR 0019): percentages are 0-100 floats rounded to 2 decimals, probabilities are 0-1 floats
rounded to 4 decimals, and a probability mean is `None` when its region is empty.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Keep in sync with app.ml.leaf_disease.postprocess.DISEASE_CHANNELS.
DiseaseChannel = Literal["black_sigatoka", "yellow_sigatoka"]


class LeafDiseaseThresholds(BaseModel):
    """The exact policy that produced a result, echoed so a stored row is reproducible."""

    leaf: float = Field(gt=0, lt=1, description="p(leaf) >= this -> leaf pixel")
    black_sigatoka: float = Field(
        gt=0, lt=1, description="p(black_sigatoka) >= this -> affected pixel"
    )
    yellow_sigatoka: float = Field(
        gt=0, lt=1, description="p(yellow_sigatoka) >= this -> affected pixel"
    )
    min_leaf_pct: float = Field(
        ge=0, le=100, description="leaf share of the image (%) below which the result is 'no_leaf'"
    )
    min_lesion_pct: float = Field(
        ge=0, le=100, description="lesion blobs smaller than this % of the leaf area are dropped"
    )
    min_disease_pct: float = Field(
        ge=0, le=100, description="a disease's share of the leaf (%) below which it is ignored"
    )
    tta_hflip: bool = Field(description="probabilities averaged with the horizontally flipped pass")


class DiseaseChannelMetrics(BaseModel):
    """Per-disease-channel measurements, one of these per entry in `LeafDiseaseDetails.diseases`."""

    channel: DiseaseChannel
    area_pct_of_leaf: float = Field(ge=0, le=100)
    lesion_count: int = Field(ge=0)
    largest_lesion_pct_of_leaf: float = Field(ge=0, le=100)
    mean_probability: float | None = Field(default=None, ge=0, le=1)


class LeafDiseaseDetails(BaseModel):
    # No default on purpose: see LeafSegDetails.kind for why.
    kind: Literal["leaf_disease"]
    diagnosis: Literal["healthy", "black_sigatoka", "yellow_sigatoka", "no_leaf"]
    leaf_area_pct_of_image: float = Field(ge=0, le=100)
    mean_leaf_probability: float | None = Field(default=None, ge=0, le=1)
    diseases: list[DiseaseChannelMetrics]
    thresholds: LeafDiseaseThresholds
