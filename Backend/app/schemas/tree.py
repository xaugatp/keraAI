"""Model 1 (banana tree classification) response payloads — spec §10.

`TreeDetails` is the model-specific blob stored in `analyses.details` and
returned as `details` in the API. Field names are snake_case (the frontend
generates its TypeScript types from the OpenAPI document).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TreeProbability(BaseModel):
    class_key: str  # raw class name from the model, e.g. "banana_tree"
    display_name: str  # human label from TREE_DISPLAY_NAMES, e.g. "Banana tree"
    probability: float = Field(ge=0.0, le=1.0)


class TreeDetails(BaseModel):
    # No default on purpose: once a second model exists this becomes the
    # discriminator of a union, and OpenAPI/TypeScript should see it as required.
    kind: Literal["tree_classification"]
    # "uncertain" wins over everything when top-1 confidence < threshold;
    # otherwise the positive class vs. anything else.
    verdict: Literal["banana_tree", "not_banana_tree", "uncertain"]
    threshold: float  # the uncertain threshold in force when this was computed
    # Full distribution, sorted by probability descending.
    probabilities: list[TreeProbability]
