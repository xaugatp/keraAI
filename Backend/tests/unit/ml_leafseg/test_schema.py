"""The owner-approved Model 2 `details` contract."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from app.schemas.leaf_seg import LeafSegDetails, LeafSegThresholds
from tests.unit.ml_leafseg.helpers import make_thresholds


def details(**overrides: Any) -> LeafSegDetails:
    values: dict[str, Any] = {
        "kind": "leaf_segmentation",
        "leaf_area_pct_of_image": 60.0,
        "affected_area_pct_of_leaf": 0.83,
        "lesion_count": 2,
        "largest_lesion_pct_of_leaf": 0.42,
        "mean_leaf_probability": 0.95,
        "mean_affected_probability": 0.9,
        "thresholds": make_thresholds(),
    }
    values.update(overrides)
    return LeafSegDetails(**values)


def test_json_shape_is_the_approved_contract() -> None:
    assert details().model_dump(mode="json") == {
        "kind": "leaf_segmentation",
        "leaf_area_pct_of_image": 60.0,
        "affected_area_pct_of_leaf": 0.83,
        "lesion_count": 2,
        "largest_lesion_pct_of_leaf": 0.42,
        "mean_leaf_probability": 0.95,
        "mean_affected_probability": 0.9,
        "thresholds": {
            "leaf": 0.5,
            "affected": 0.85,
            "min_leaf_pct": 3.0,
            "min_lesion_pct": 0.05,
            "min_affected_pct": 0.5,
            "tta_hflip": False,
        },
    }


def test_probability_means_default_to_none() -> None:
    d = LeafSegDetails(
        kind="leaf_segmentation",
        leaf_area_pct_of_image=0.0,
        affected_area_pct_of_leaf=0.0,
        lesion_count=0,
        largest_lesion_pct_of_leaf=0.0,
        thresholds=make_thresholds(),
    )
    assert d.mean_leaf_probability is None and d.mean_affected_probability is None


def test_round_trips_through_json() -> None:
    original = details(mean_affected_probability=None)
    assert LeafSegDetails.model_validate_json(original.model_dump_json()) == original


@pytest.mark.parametrize(
    "override",
    [
        {"leaf_area_pct_of_image": 100.01},
        {"leaf_area_pct_of_image": -0.01},
        {"affected_area_pct_of_leaf": 101},
        {"largest_lesion_pct_of_leaf": -1},
        {"lesion_count": -1},
        {"mean_leaf_probability": 1.01},
        {"mean_affected_probability": -0.1},
        {"kind": "tree_classification"},
    ],
)
def test_out_of_range_values_are_rejected(override: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        details(**override)


@pytest.mark.parametrize(
    "override",
    [
        {"leaf": 0.0},
        {"leaf": 1.0},
        {"affected": 1.5},
        {"min_leaf_pct": 101},
        {"min_lesion_pct": -1},
    ],
)
def test_threshold_ranges_are_validated(override: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        make_thresholds(**override)


def test_thresholds_are_required_fields() -> None:
    with pytest.raises(ValidationError):
        LeafSegThresholds()  # type: ignore[call-arg]
