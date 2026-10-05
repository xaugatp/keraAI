"""Model 2 input preparation (torch-free)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from PIL import Image

from app.ml.leaf_seg import preprocess
from app.ml.leaf_seg.preprocess import IMAGENET_MEAN, IMAGENET_STD, prepare_input

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def expected(rgb: tuple[int, int, int]) -> np.ndarray:
    return np.array([(v / 255 - m) / s for v, m, s in zip(rgb, MEAN, STD, strict=True)])


def test_constants_match_the_training_notebook() -> None:
    assert tuple(IMAGENET_MEAN.tolist()) == pytest.approx(MEAN)
    assert tuple(IMAGENET_STD.tolist()) == pytest.approx(STD)


def test_output_is_float32_nchw_and_contiguous() -> None:
    out = prepare_input(Image.new("RGB", (500, 300), (10, 20, 30)), 64)
    assert out.shape == (1, 3, 64, 64)
    assert out.dtype == np.float32
    assert out.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize("rgb", [(124, 116, 104), (0, 0, 0), (255, 255, 255), (255, 0, 128)])
@pytest.mark.parametrize("size", [(64, 64), (300, 500), (40, 20)])
def test_constant_image_gets_exact_imagenet_normalisation(
    rgb: tuple[int, int, int], size: tuple[int, int]
) -> None:
    out = prepare_input(Image.new("RGB", size, rgb), 64)
    for channel, value in enumerate(expected(rgb)):
        assert np.allclose(out[0, channel], value, rtol=1e-5, atol=1e-5)


def test_mid_grey_is_close_to_zero_mean_per_channel() -> None:
    # (124, 116, 104)/255 ~ the ImageNet mean, so the normalised value is ~0.
    out = prepare_input(Image.new("RGB", (64, 64), (124, 116, 104)), 64)
    assert np.abs(out).max() < 0.02


def test_non_square_input_is_stretched_not_padded() -> None:
    # Left half red, right half blue, 3:1 aspect. A direct stretch keeps both
    # halves filling the whole square; letterboxing would add constant bars.
    image = Image.new("RGB", (120, 40), (255, 0, 0))
    image.paste((0, 0, 255), (60, 0, 120, 40))
    out = prepare_input(image, 64)
    red, blue = expected((255, 0, 0)), expected((0, 0, 255))
    assert np.allclose(out[0, :, 0, 5], red, atol=1e-4)  # top-left corner area
    assert np.allclose(out[0, :, 63, 5], red, atol=1e-4)  # bottom-left: rows are all used
    assert np.allclose(out[0, :, 0, 58], blue, atol=1e-4)
    assert np.allclose(out[0, :, 63, 58], blue, atol=1e-4)


def test_channel_order_is_rgb() -> None:
    out = prepare_input(Image.new("RGB", (64, 64), (255, 0, 0)), 64)
    assert out[0, 0, 0, 0] > 2.0  # red channel high ((1-.485)/.229 = 2.25)
    assert out[0, 2, 0, 0] < -1.5  # blue channel low


def test_preprocessing_is_deterministic_and_does_not_mutate_the_input() -> None:
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 256, (90, 140, 3), dtype=np.uint8))
    before = np.asarray(image).copy()
    first, second = prepare_input(image, 64), prepare_input(image, 64)
    assert np.array_equal(first, second)
    assert np.array_equal(np.asarray(image), before)


class _ResizeSpy:
    """Records (target size, interpolation) of every cv2.resize call, then delegates."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: list[tuple[tuple[int, int], int]] = []
        real = cv2.resize

        def spy(src: np.ndarray, dsize: tuple[int, int], *, interpolation: int) -> np.ndarray:
            self.calls.append((dsize, interpolation))
            return real(src, dsize, interpolation=interpolation)

        monkeypatch.setattr(preprocess.cv2, "resize", spy)


@pytest.mark.parametrize(
    ("size", "expected_calls"),
    [
        # shrinking both axes -> INTER_AREA only
        ((2048, 2048), [((64, 64), cv2.INTER_AREA)]),
        ((200, 100), [((64, 64), cv2.INTER_AREA)]),
        # enlarging both axes -> INTER_LINEAR only
        ((32, 32), [((64, 64), cv2.INTER_LINEAR)]),
        # shrink one axis, enlarge the other: AREA on the shrink first, then LINEAR
        ((128, 32), [((64, 32), cv2.INTER_AREA), ((64, 64), cv2.INTER_LINEAR)]),
        ((32, 128), [((32, 64), cv2.INTER_AREA), ((64, 64), cv2.INTER_LINEAR)]),
        # already the right size -> no resampling at all
        ((64, 64), []),
    ],
)
def test_interpolation_follows_shrink_or_enlarge(
    size: tuple[int, int],
    expected_calls: list[tuple[tuple[int, int], int]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = _ResizeSpy(monkeypatch)
    prepare_input(Image.new("RGB", size, (1, 2, 3)), 64)
    assert spy.calls == expected_calls


def test_area_interpolation_averages_instead_of_aliasing() -> None:
    # Alternating 1-px stripes (0/255) shrunk 4x: INTER_AREA gives the mean
    # (~127.5 everywhere); a point-sampling filter would give 0 or 255.
    stripes = np.zeros((256, 256, 3), dtype=np.uint8)
    stripes[:, ::2] = 255
    out = prepare_input(Image.fromarray(stripes), 64)
    value = out[0, 0] * IMAGENET_STD[0] + IMAGENET_MEAN[0]  # undo normalisation of channel 0
    assert np.allclose(value, 0.5, atol=0.01)


@pytest.mark.parametrize("mode", ["L", "RGBA", "P", "CMYK", "1"])
def test_non_rgb_modes_are_rejected(mode: str) -> None:
    with pytest.raises(ValueError, match="RGB"):
        prepare_input(Image.new(mode, (10, 10)), 64)


@pytest.mark.parametrize("size", [(0, 10), (10, 0), (0, 0)])
def test_zero_size_images_are_rejected(size: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="no pixels"):
        prepare_input(Image.new("RGB", size), 64)


@pytest.mark.parametrize("imgsz", [0, -32])
def test_non_positive_imgsz_is_rejected(imgsz: int) -> None:
    with pytest.raises(ValueError, match="imgsz"):
        prepare_input(Image.new("RGB", (10, 10)), imgsz)
