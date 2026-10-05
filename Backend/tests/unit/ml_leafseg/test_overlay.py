"""Result-image rendering (torch-free)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from app.ml.leaf_seg.overlay import LEAF_COLOUR, LESION_ALPHA, LESION_FILL_COLOUR, render_overlay

GREY = (100, 100, 100)


def scene(width: int = 200, height: int = 100) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    image = Image.new("RGB", (width, height), GREY)
    leaf = np.zeros((height, width), dtype=bool)
    leaf[10:90, 10:190] = True
    lesion = np.zeros((height, width), dtype=bool)
    lesion[40:60, 50:100] = True
    return image, leaf, lesion


def px(image: Image.Image, row: int, col: int) -> tuple[int, int, int]:
    r, g, b = image.getpixel((col, row))  # type: ignore[misc]
    return int(r), int(g), int(b)


def test_lesion_is_filled_with_semi_transparent_red() -> None:
    image, leaf, lesion = scene()
    out = render_overlay(image, leaf, lesion)
    expected = tuple(
        round(g * (1 - LESION_ALPHA) + c * LESION_ALPHA)
        for g, c in zip(GREY, LESION_FILL_COLOUR, strict=True)
    )
    inside = px(out, 50, 75)
    assert all(abs(a - b) <= 1 for a, b in zip(inside, expected, strict=True))
    # Semi-transparent: the photo still shows through (not pure fill colour).
    assert inside != LESION_FILL_COLOUR and inside[0] > inside[1]


def test_pixels_away_from_masks_are_untouched() -> None:
    image, leaf, lesion = scene()
    out = render_overlay(image, leaf, lesion)
    assert px(out, 2, 2) == GREY  # outside the leaf
    assert px(out, 30, 150) == GREY  # inside the leaf but outside the lesion
    assert px(out, 50, 20) == GREY  # leaf interior, left of the lesion


def test_leaf_contour_is_green_over_a_white_halo() -> None:
    image, leaf, lesion = scene()
    out = render_overlay(image, leaf, lesion)
    column = [px(out, row, 150) for row in range(6, 16)]  # crossing the leaf's top edge
    assert any(p[1] > 150 and p[0] < 80 and p[2] < 80 for p in column), "no green contour"
    assert any(min(p) > 200 for p in column), "no white halo"
    assert LEAF_COLOUR[1] > 150  # the colour constant the assertion above relies on


def test_lesion_has_a_darker_red_contour() -> None:
    image, leaf, lesion = scene()
    out = render_overlay(image, leaf, lesion)
    row = [px(out, 50, col) for col in range(46, 56)]  # crossing the lesion's left edge
    assert any(p[0] < 140 and p[1] < 60 and p[2] < 60 for p in row), "no dark-red contour"


def test_empty_masks_return_the_photo_unchanged() -> None:
    image, _, _ = scene()
    empty = np.zeros((100, 200), dtype=bool)
    out = render_overlay(image, empty, empty)
    assert np.array_equal(np.asarray(out), np.asarray(image))
    assert out.mode == "RGB"


def test_output_is_never_upscaled() -> None:
    image = Image.new("RGB", (40, 30), GREY)
    mask = np.zeros((30, 40), dtype=bool)
    assert render_overlay(image, mask, mask).size == (40, 30)
    assert render_overlay(image, mask, mask, max_side=1280).size == (40, 30)


@pytest.mark.parametrize(
    ("size", "max_side", "expected"),
    [
        ((2000, 1000), 1280, (1280, 640)),
        ((1000, 2000), 1280, (640, 1280)),
        ((200, 100), 100, (100, 50)),
        ((1280, 720), 1280, (1280, 720)),  # exactly at the limit: left alone
    ],
)
def test_output_is_downscaled_to_fit_max_side(
    size: tuple[int, int], max_side: int, expected: tuple[int, int]
) -> None:
    width, height = size
    mask = np.zeros((height, width), dtype=bool)
    out = render_overlay(Image.new("RGB", size, GREY), mask, mask, max_side=max_side)
    assert out.size == expected
    assert max(out.size) <= max_side


def test_masks_are_scaled_with_the_photo_when_downscaling() -> None:
    image, leaf, lesion = scene(400, 200)  # lesion at rows 40:60, cols 50:100 of 400x200
    out = render_overlay(image, leaf, lesion, max_side=200)
    assert out.size == (200, 100)
    assert px(out, 25, 38)[0] > px(out, 25, 38)[1]  # (50, 76) in source -> red tint
    assert px(out, 90, 190) == GREY  # an untouched corner


def test_inputs_are_not_modified() -> None:
    image, leaf, lesion = scene()
    image_before = np.asarray(image).copy()
    leaf_before, lesion_before = leaf.copy(), lesion.copy()
    render_overlay(image, leaf, lesion)
    assert np.array_equal(np.asarray(image), image_before)
    assert np.array_equal(leaf, leaf_before) and np.array_equal(lesion, lesion_before)


def test_legible_on_dark_and_bright_photos() -> None:
    _, leaf, lesion = scene()
    for background in ((5, 5, 5), (250, 250, 250)):
        out = render_overlay(Image.new("RGB", (200, 100), background), leaf, lesion)
        column = [px(out, row, 150) for row in range(6, 16)]
        # The outline differs strongly from the background on both extremes.
        assert any(
            max(abs(a - b) for a, b in zip(p, background, strict=True)) > 100 for p in column
        )


def test_non_rgb_input_is_converted() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    out = render_overlay(Image.new("L", (10, 10), 50), mask, mask)
    assert out.mode == "RGB" and px(out, 5, 5) == (50, 50, 50)


def test_mask_shape_mismatch_is_rejected() -> None:
    image, leaf, lesion = scene()
    with pytest.raises(ValueError, match="leaf_mask"):
        render_overlay(image, leaf[:50], lesion)
    with pytest.raises(ValueError, match="affected_mask"):
        render_overlay(image, leaf, lesion[:, :50])


def test_invalid_max_side_and_empty_image_are_rejected() -> None:
    image, leaf, lesion = scene()
    with pytest.raises(ValueError, match="max_side"):
        render_overlay(image, leaf, lesion, max_side=0)
    with pytest.raises(ValueError, match="no pixels"):
        render_overlay(Image.new("RGB", (0, 5)), np.zeros((5, 0), bool), np.zeros((5, 0), bool))
