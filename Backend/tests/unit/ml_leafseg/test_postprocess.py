"""Post-processing on synthetic maps with analytically known answers."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.ml.leaf_seg.postprocess import (
    DISPLAY_LABELS,
    SegmentationResult,
    average_flip_probs,
    build_details,
    clean_affected_mask,
    clean_leaf_mask,
    decide_label,
    fill_holes,
    postprocess_probabilities,
    upsample_probabilities,
)
from tests.unit.ml_leafseg.helpers import STRIP, STRIP_PIXELS, make_probs, make_thresholds


def run(
    probs: np.ndarray, size: tuple[int, int] | None = None, **thr: object
) -> SegmentationResult:
    height, width = probs.shape[1:]
    return postprocess_probabilities(probs, size or (height, width), make_thresholds(**thr))


def assert_invariants(result: SegmentationResult) -> None:
    assert result.leaf_mask.dtype == np.bool_ and result.affected_mask.dtype == np.bool_
    assert result.leaf_mask.shape == result.affected_mask.shape
    assert not (result.affected_mask & ~result.leaf_mask).any(), "lesion outside the leaf"
    assert 0 <= result.leaf_area_pct_of_image <= 100
    assert 0 <= result.affected_area_pct_of_leaf <= 100
    assert 0 <= result.largest_lesion_pct_of_leaf <= result.affected_area_pct_of_leaf + 0.01


# --- the headline case --------------------------------------------------------


def test_rectangular_leaf_with_two_lesions_has_exact_metrics() -> None:
    lesions = [(20, 30, 20, 30), (100, 110, 50, 60)]  # two 10x10 squares = 100 px each
    result = run(make_probs(leaf=[STRIP], lesions=lesions))

    assert int(result.leaf_mask.sum()) == STRIP_PIXELS
    assert int(result.affected_mask.sum()) == 200
    assert result.leaf_area_pct_of_image == 60.0  # 24000 / 40000
    assert result.affected_area_pct_of_leaf == 0.83  # 200 / 24000 = 0.8333...
    assert result.lesion_count == 2
    assert result.largest_lesion_pct_of_leaf == 0.42  # 100 / 24000 = 0.41666...
    assert result.mean_leaf_probability == 0.95
    assert result.mean_affected_probability == 0.95
    assert result.label == "affected"
    assert_invariants(result)


def test_postprocess_is_deterministic() -> None:
    probs = make_probs(leaf=[STRIP], lesions=[(20, 30, 20, 30)])
    first, second = run(probs), run(probs)
    assert np.array_equal(first.leaf_mask, second.leaf_mask)
    assert np.array_equal(first.affected_mask, second.affected_mask)
    assert first.label == second.label


# --- lesion cleaning ----------------------------------------------------------


@pytest.mark.parametrize(
    ("rows", "cols", "survives"),
    [
        (3, 3, False),  # 9 px  < 12 px (0.05% of 24000)
        (3, 4, True),  # 12 px == threshold: ">=" keeps it
        (4, 4, True),  # 16 px
    ],
)
def test_min_lesion_area_boundary(rows: int, cols: int, survives: bool) -> None:
    result = run(make_probs(leaf=[STRIP], lesions=[(50, 50 + rows, 50, 50 + cols)]))
    assert result.lesion_count == (1 if survives else 0)
    assert int(result.affected_mask.sum()) == (rows * cols if survives else 0)


def test_absolute_four_pixel_floor_applies_even_with_zero_min_lesion_pct() -> None:
    kept = run(make_probs(leaf=[STRIP], lesions=[(50, 52, 50, 52)]), min_lesion_pct=0.0)
    dropped = run(make_probs(leaf=[STRIP], lesions=[(50, 51, 50, 53)]), min_lesion_pct=0.0)
    assert kept.lesion_count == 1  # 2x2 = 4 px
    assert dropped.lesion_count == 0  # 1x3 = 3 px


def test_lesion_outside_leaf_is_removed() -> None:
    result = run(make_probs(leaf=[STRIP], lesions=[(50, 70, 150, 170)]))
    assert result.lesion_count == 0
    assert not result.affected_mask.any()
    assert result.label == "healthy"
    assert result.mean_affected_probability is None


def test_lesion_straddling_leaf_edge_is_kept_only_inside() -> None:
    # 20x20 lesion, columns 110..129; the strip ends at column 119.
    result = run(make_probs(leaf=[STRIP], lesions=[(50, 70, 110, 130)]))
    assert int(result.affected_mask.sum()) == 20 * 10
    assert not result.affected_mask[:, 120:].any()
    assert result.lesion_count == 1
    assert_invariants(result)


def test_blob_size_is_judged_on_leaf_plus_edge_band_not_on_the_part_outside() -> None:
    # A huge blob (10 x 82 = 820 px) mostly OUTSIDE the leaf, overlapping it by 2
    # columns. Its in-leaf + tolerance-band part is only 30 px, below
    # min_lesion_pct=0.5% of the leaf (120 px): damage that is really "outside the
    # leaf" must not be rescued by its own size.
    probs = make_probs(leaf=[STRIP], lesions=[(50, 60, 118, 200)])
    assert run(probs, min_lesion_pct=0.5).lesion_count == 0
    # With the default (0.05% = 12 px) the 20 px that are inside the leaf are kept.
    assert int(run(probs).affected_mask.sum()) == 20


def test_sliver_left_after_clipping_is_dropped() -> None:
    # With min_lesion_pct=0 only the 4 px floor applies. The blob's tolerance-band
    # candidate is 4 px (passes), but only 2 px lie inside the leaf (column 119):
    # below the floor once clipped, so it must not be counted as a lesion.
    probs = make_probs(leaf=[STRIP], lesions=[(50, 52, 119, 130)])
    result = run(probs, min_lesion_pct=0.0)
    assert result.lesion_count == 0 and not result.affected_mask.any()


def test_lesion_filling_a_hole_in_the_leaf_is_counted() -> None:
    # Necrotic centre: the net says "not leaf" but "affected" there. Hole-filling
    # makes it part of the leaf so the damage is measured instead of discarded.
    box = (80, 100, 40, 60)
    result = run(make_probs(leaf=[STRIP], holes=[box], lesions=[box]))
    assert int(result.affected_mask.sum()) == 20 * 20
    assert int(result.leaf_mask.sum()) == STRIP_PIXELS
    assert_invariants(result)


# --- leaf cleaning ------------------------------------------------------------


def test_interior_hole_in_leaf_is_filled() -> None:
    result = run(make_probs(leaf=[STRIP], holes=[(80, 100, 40, 60)]))
    assert result.leaf_mask[90, 50]
    assert int(result.leaf_mask.sum()) == STRIP_PIXELS
    assert result.leaf_area_pct_of_image == 60.0


def test_two_comparable_leaves_kept_and_tiny_blob_dropped() -> None:
    probs = make_probs(
        leaf=[
            (20, 80, 20, 80),  # leaf A: 60x60 = 3600 px (9% of the image)
            (120, 170, 120, 170),  # leaf B: 50x50 = 2500 px, >= 20% of A -> kept
            (100, 110, 5, 15),  # 100 px: below 1% of the image (400 px) -> speckle
        ]
    )
    result = run(probs)
    assert result.leaf_mask[50, 50] and result.leaf_mask[145, 145]
    assert not result.leaf_mask[105, 10]
    # Rectangles lose their 4 corner pixels to the morphological opening.
    assert abs(int(result.leaf_mask.sum()) - (3600 + 2500)) <= 8


def test_blob_comparable_to_image_but_small_next_to_the_main_leaf_is_dropped() -> None:
    probs = make_probs(
        leaf=[
            (10, 110, 10, 110),  # main leaf: 100x100 = 10000 px (25%)
            (150, 176, 150, 176),  # 26x26 = 676 px: >= 1% of image (400) but < 20% of 10000
        ]
    )
    result = run(probs)
    assert result.leaf_mask[60, 60]
    assert not result.leaf_mask[160, 160]


def test_single_blob_under_one_percent_of_the_image_is_dropped_even_if_it_is_the_largest() -> None:
    # 10 x 20 = 200 px = 0.5% of the image; it is "comparable to the largest" (it
    # IS the largest), so only the absolute 1%-of-image rule can reject it.
    result = run(make_probs(leaf=[(50, 60, 50, 70)]), min_leaf_pct=0.0)
    assert not result.leaf_mask.any()
    assert result.label == "no_leaf"


def test_edge_tolerance_band_scales_with_image_size() -> None:
    # 2000 px photo -> ~10 px tolerance each side of the leaf edge. The blob has
    # 100 px inside the leaf and 200 px in the 10 px band: judged at 300 px it
    # passes a 200 px minimum, so the in-leaf 100 px are kept. With a fixed 1 px
    # band it would measure only 120 px and be rejected.
    probs = make_probs(2000, 2000, leaf=[(0, 2000, 0, 1200)], lesions=[(500, 520, 1195, 1210)])
    result = run(probs, min_lesion_pct=200 / (2000 * 1200) * 100)
    assert result.lesion_count == 1
    assert int(result.affected_mask.sum()) == 5 * 20


def test_morphological_close_bridges_a_thin_gap() -> None:
    gap_probs = make_probs(leaf=[(0, 200, 0, 99), (0, 200, 100, 200)])  # 1 px gap at column 99
    gap_probs[0, :, 99] = 0.0
    result = run(gap_probs)
    assert result.leaf_mask[:, 99].all()


def test_hairline_leaf_prediction_is_erased_by_the_opening() -> None:
    # A 1 px wide "leaf" cannot contain the structuring element: pure noise.
    result = run(make_probs(leaf=[(50, 51, 10, 190)]))
    assert not result.leaf_mask.any()
    assert result.label == "no_leaf"


@pytest.mark.parametrize("which", ["zeros", "ones"])
@pytest.mark.parametrize("shape", [(200, 200), (7, 13), (2, 3), (1, 1)])
def test_degenerate_maps_do_not_crash(which: str, shape: tuple[int, int]) -> None:
    fill = np.zeros if which == "zeros" else np.ones
    result = postprocess_probabilities(
        fill((2, *shape), dtype=np.float32), shape, make_thresholds()
    )
    assert_invariants(result)
    if which == "zeros":
        assert result.label == "no_leaf"
        assert result.leaf_area_pct_of_image == 0.0
    else:
        assert result.leaf_area_pct_of_image == 100.0


def test_all_ones_is_one_leaf_fully_damaged() -> None:
    result = run(np.ones((2, 120, 90), dtype=np.float32))
    assert result.leaf_area_pct_of_image == 100.0
    assert result.affected_area_pct_of_leaf == 100.0
    assert result.lesion_count == 1
    assert result.largest_lesion_pct_of_leaf == 100.0
    assert result.mean_leaf_probability == 1.0
    assert result.mean_affected_probability == 1.0
    assert result.label == "affected"


# --- labels -------------------------------------------------------------------


def test_no_leaf_when_nothing_is_predicted() -> None:
    result = run(make_probs())
    assert result.label == "no_leaf"
    assert result.leaf_area_pct_of_image == 0.0
    assert result.affected_area_pct_of_leaf == 0.0
    assert result.lesion_count == 0
    assert result.largest_lesion_pct_of_leaf == 0.0
    assert result.mean_leaf_probability is None
    assert result.mean_affected_probability is None


def test_leaf_below_min_leaf_pct_is_no_leaf_even_with_damage_on_it() -> None:
    probs = make_probs(leaf=[(50, 70, 50, 90)], lesions=[(55, 65, 55, 85)])  # 800 px = 2% of image
    small = run(probs)
    assert small.leaf_area_pct_of_image < 3.0
    assert small.label == "no_leaf"
    # The numbers are still reported honestly; only the verdict changes.
    assert small.lesion_count == 1
    assert run(probs, min_leaf_pct=1.0).label == "affected"


def test_min_leaf_pct_boundary_is_inclusive() -> None:
    probs = make_probs(leaf=[STRIP])  # exactly 60.0 %
    assert run(probs, min_leaf_pct=60.0).label == "healthy"
    assert run(probs, min_leaf_pct=60.01).label == "no_leaf"


def test_leaf_without_damage_is_healthy() -> None:
    result = run(make_probs(leaf=[STRIP]))
    assert result.label == "healthy"
    assert result.affected_area_pct_of_leaf == 0.0
    assert result.lesion_count == 0
    assert result.largest_lesion_pct_of_leaf == 0.0
    assert result.mean_affected_probability is None  # not measured, NOT 0.0
    assert result.mean_leaf_probability == 0.95


@pytest.mark.parametrize(
    ("lesion_rows", "pct", "label"),
    [
        (11, 0.46, "healthy"),  # 110 px  = 0.458% < 0.5%
        (12, 0.5, "affected"),  # 120 px  = 0.5%   (>= is inclusive)
        (13, 0.54, "affected"),  # 130 px  = 0.542%
    ],
)
def test_min_affected_pct_flips_the_label(lesion_rows: int, pct: float, label: str) -> None:
    result = run(make_probs(leaf=[STRIP], lesions=[(50, 50 + lesion_rows, 50, 60)]))
    assert result.affected_area_pct_of_leaf == pct
    assert result.label == label
    assert result.lesion_count == 1  # a "healthy" leaf still reports the small lesion it has


@pytest.mark.parametrize(
    ("leaf_pct", "affected_pct", "expected"),
    [
        (0.0, 0.0, "no_leaf"),
        (0.0, 50.0, "no_leaf"),
        (2.99, 80.0, "no_leaf"),
        (3.0, 0.49, "healthy"),
        (3.0, 0.5, "affected"),
        (100.0, 100.0, "affected"),
    ],
)
def test_decide_label_rule(leaf_pct: float, affected_pct: float, expected: str) -> None:
    assert decide_label(leaf_pct, affected_pct, make_thresholds()) == expected


def test_min_leaf_pct_zero_with_no_leaf_pixels_is_still_no_leaf() -> None:
    assert decide_label(0.0, 0.0, make_thresholds(min_leaf_pct=0.0)) == "no_leaf"


def test_display_labels_cover_every_label() -> None:
    assert DISPLAY_LABELS == {
        "affected": "Damaged tissue found",
        "healthy": "No damage detected",
        "no_leaf": "No banana leaf found",
    }


# --- thresholds ---------------------------------------------------------------


def test_leaf_threshold_is_inclusive() -> None:
    at = make_probs(leaf=[STRIP], leaf_p=0.5)
    below = make_probs(leaf=[STRIP], leaf_p=0.49)
    assert run(at).leaf_area_pct_of_image == 60.0
    assert run(below).leaf_area_pct_of_image == 0.0


def test_affected_threshold_is_inclusive() -> None:
    lesion = [(50, 70, 50, 70)]
    at = run(make_probs(leaf=[STRIP], lesions=lesion, affected_p=0.85))
    just_below = make_probs(leaf=[STRIP], lesions=lesion)
    just_below[1, 50:70, 50:70] = np.nextafter(np.float32(0.85), np.float32(0))
    assert at.lesion_count == 1
    assert run(just_below).lesion_count == 0


def test_stricter_affected_threshold_removes_weak_lesions() -> None:
    probs = make_probs(leaf=[STRIP], lesions=[(50, 70, 50, 70)], affected_p=0.9)
    assert run(probs, affected=0.85).lesion_count == 1
    assert run(probs, affected=0.95).lesion_count == 0


# --- resolution ---------------------------------------------------------------


def test_masks_come_back_at_the_original_non_square_resolution() -> None:
    small = np.zeros((2, 50, 50), dtype=np.float32)
    small[0, :, :30] = 0.95
    small[1, 10:20, 5:15] = 0.95
    result = postprocess_probabilities(small, (300, 500), make_thresholds())
    assert result.leaf_mask.shape == (300, 500)
    assert result.affected_mask.shape == (300, 500)
    assert 59.0 <= result.leaf_area_pct_of_image <= 61.0  # 30 of 50 columns, +-1 px edges
    assert_invariants(result)


def test_integer_upscale_keeps_region_boundaries_exact() -> None:
    # Bilinear upsampling of a 0 -> 0.95 step, thresholded at 0.5, lands the edge
    # exactly on the scaled boundary for any integer factor.
    small = make_probs(100, 100, leaf=[(0, 100, 0, 60)])
    result = postprocess_probabilities(small, (200, 200), make_thresholds())
    assert int(result.leaf_mask.sum()) == STRIP_PIXELS
    assert result.leaf_mask[:, 119].all() and not result.leaf_mask[:, 120].any()


def test_downscaled_original_is_supported_too() -> None:
    big = make_probs(200, 200, leaf=[STRIP])
    result = postprocess_probabilities(big, (100, 100), make_thresholds())
    assert result.leaf_mask.shape == (100, 100)
    assert result.leaf_area_pct_of_image == pytest.approx(60.0, abs=1.0)


def test_upsample_probabilities_identity_and_validation() -> None:
    probs = make_probs(20, 30, leaf=[(0, 10, 0, 10)])
    assert upsample_probabilities(probs, (20, 30)).shape == (2, 20, 30)
    assert upsample_probabilities(probs, (40, 90)).shape == (2, 40, 90)
    assert upsample_probabilities(probs, (40, 90)).dtype == np.float32
    with pytest.raises(ValueError, match="original_size"):
        upsample_probabilities(probs, (0, 30))
    with pytest.raises(ValueError, match="shape"):
        upsample_probabilities(probs[0], (20, 30))
    with pytest.raises(ValueError, match="shape"):
        upsample_probabilities(np.zeros((3, 20, 30), dtype=np.float32), (20, 30))


# --- building blocks ----------------------------------------------------------


def test_fill_holes_fills_enclosed_background_only() -> None:
    ring = np.zeros((20, 20), dtype=bool)
    ring[4:16, 4:16] = True
    ring[8:12, 8:12] = False
    assert fill_holes(ring)[8:12, 8:12].all()

    # A notch open to the image border is part of the outside, not a hole.
    notch = np.ones((20, 20), dtype=bool)
    notch[:, 9:11] = False
    notch[0, :] = True  # closes the top...
    notch[19, :] = False  # ...but the gap still reaches the bottom border
    assert not fill_holes(notch)[10, 9:11].any()


def test_fill_holes_treats_a_diagonally_closed_ring_as_closed() -> None:
    # Cells with |dy|+|dx| == 3 around (4, 4): a ring whose neighbours touch only
    # diagonally. Foreground is 8-connected, so background must be 4-connected,
    # and the ring therefore seals its interior.
    diamond = np.zeros((9, 9), dtype=bool)
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            if abs(dy) + abs(dx) == 3:
                diamond[4 + dy, 4 + dx] = True
    filled = fill_holes(diamond)
    assert filled[4, 4] and filled[3, 4] and filled[4, 3]
    assert not filled[0, 0]


def test_fill_holes_on_empty_and_full_masks() -> None:
    assert not fill_holes(np.zeros((5, 5), dtype=bool)).any()
    assert fill_holes(np.ones((5, 5), dtype=bool)).all()


def test_clean_leaf_mask_returns_a_copy_for_an_empty_mask() -> None:
    empty = np.zeros((10, 10), dtype=bool)
    cleaned = clean_leaf_mask(empty)
    assert not cleaned.any() and cleaned is not empty


def test_clean_affected_mask_with_no_leaf_is_empty() -> None:
    raw = np.ones((20, 20), dtype=bool)
    assert not clean_affected_mask(raw, np.zeros((20, 20), dtype=bool), 0.05).any()


def test_morphology_kernel_scales_with_the_image() -> None:
    # The same 6 px gap is bridged on a 2000 px photo (kernel ~11 px) but not on a
    # 200 px one (minimum 3 px kernel): the clean-up is relative to image size.
    big = make_probs(2000, 2000, leaf=[(0, 2000, 0, 997), (0, 2000, 1003, 2000)])
    assert run(big).leaf_mask[:, 1000].all()
    small = make_probs(200, 200, leaf=[(0, 200, 0, 97), (0, 200, 103, 200)])
    assert not run(small).leaf_mask[:, 100].any()


def test_large_non_square_photo_is_processed_exactly() -> None:
    probs = make_probs(400, 2000, leaf=[(0, 400, 0, 1200)], lesions=[(100, 200, 100, 300)])
    result = run(probs)
    assert int(result.leaf_mask.sum()) == 400 * 1200
    assert int(result.affected_mask.sum()) == 100 * 200


# --- test-time augmentation ---------------------------------------------------


def test_average_flip_probs_of_an_equivariant_net_is_the_original() -> None:
    rng = np.random.default_rng(0)
    probs = rng.random((2, 8, 12), dtype=np.float32)
    # An equivariant network's output on the mirrored input is the mirrored output.
    averaged = average_flip_probs(probs, probs[..., ::-1])
    assert np.allclose(averaged, probs, atol=1e-7)


def test_average_flip_probs_flips_back_along_width_then_averages() -> None:
    p = np.zeros((2, 2, 4), dtype=np.float32)
    p[:, :, 0] = 1.0  # evidence only at the LEFT edge
    flipped = np.zeros((2, 2, 4), dtype=np.float32)
    flipped[:, :, 0] = 1.0  # the flipped pass also says "left edge" in ITS frame (= right edge)
    out = average_flip_probs(p, flipped)
    assert out[0, 0].tolist() == [0.5, 0.0, 0.0, 0.5]


def test_average_flip_probs_is_mirror_symmetric_for_symmetric_inputs() -> None:
    rng = np.random.default_rng(1)
    p = rng.random((2, 5, 7), dtype=np.float32)
    out = average_flip_probs(p, p)
    assert np.allclose(out, out[..., ::-1])
    assert out.dtype == np.float32 and out.flags["C_CONTIGUOUS"]


def test_average_flip_probs_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        average_flip_probs(np.zeros((2, 4, 4), np.float32), np.zeros((2, 4, 5), np.float32))


# --- details / contract --------------------------------------------------------


def test_build_details_echoes_metrics_and_thresholds() -> None:
    thresholds = make_thresholds(tta_hflip=True)
    result = run(make_probs(leaf=[STRIP], lesions=[(20, 30, 20, 30)]))
    details = build_details(result, thresholds)
    dumped = details.model_dump()
    assert dumped["kind"] == "leaf_segmentation"
    assert dumped["lesion_count"] == 1
    assert dumped["leaf_area_pct_of_image"] == 60.0
    assert dumped["thresholds"] == thresholds.model_dump()
    assert dumped["thresholds"]["tta_hflip"] is True


def test_build_details_keeps_none_means_for_empty_regions() -> None:
    details = build_details(run(make_probs()), make_thresholds())
    assert details.mean_leaf_probability is None
    assert details.mean_affected_probability is None
    assert details.model_dump(mode="json")["mean_affected_probability"] is None


# --- fuzz: invariants hold on arbitrary smooth random maps --------------------


@pytest.mark.parametrize("seed", range(8))
def test_invariants_on_random_maps(seed: int) -> None:
    rng = np.random.default_rng(seed)
    noise = rng.random((2, 24, 40)).astype(np.float32)
    probs = np.stack([cv2.GaussianBlur(ch, (0, 0), 2.0) for ch in noise])
    probs = np.clip((probs - probs.min()) / (np.ptp(probs) + 1e-9), 0, 1).astype(np.float32)
    result = postprocess_probabilities(probs, (97, 161), make_thresholds(leaf=0.4, affected=0.5))
    assert_invariants(result)
    assert result.leaf_mask.shape == (97, 161)
    again = postprocess_probabilities(probs, (97, 161), make_thresholds(leaf=0.4, affected=0.5))
    assert np.array_equal(result.leaf_mask, again.leaf_mask)
    assert result.label == again.label
