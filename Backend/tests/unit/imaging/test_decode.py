from __future__ import annotations

import io
import warnings
from concurrent.futures import ThreadPoolExecutor

import pytest
from PIL import Image, ImageFile

from app.core.errors import (
    ImageTooLargeDimensionsError,
    ImageTooSmallError,
    InvalidImageError,
    UnsupportedMediaTypeError,
)
from app.imaging.processing import decode
from tests.fixtures import image_factory as factory
from tests.fixtures.settings import make_settings

# --- happy path ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "expected_format"),
    [
        (factory.jpeg_bytes(), "JPEG"),
        (factory.png_bytes(), "PNG"),
        (factory.webp_bytes(), "WEBP"),
    ],
    ids=["jpeg", "png", "webp"],
)
def test_valid_images_decode_fully(data: bytes, expected_format: str) -> None:
    image = decode(data, make_settings())
    assert image.format == expected_format
    assert image.size == (200, 150)
    # Fully decoded: pixels are readable even though the source buffer is gone.
    assert image.getpixel((0, 0)) is not None


def test_decode_returns_the_original_orientation_unrotated() -> None:
    data = factory.jpeg_bytes(100, 64, exif=factory.exif_bytes(6))
    assert decode(data, make_settings()).size == (100, 64)


def test_allowed_formats_are_matched_case_insensitively() -> None:
    settings = make_settings(allowed_image_formats=["jpeg"])
    assert decode(factory.jpeg_bytes(), settings).format == "JPEG"


# --- corrupt / truncated ----------------------------------------------------


@pytest.mark.parametrize(
    "data",
    [b"", factory.corrupt_bytes(), b"\xff\xd8\xff"],
    ids=["empty", "plain-text", "jpeg-magic-only"],
)
def test_garbage_is_rejected_as_invalid_image(data: bytes) -> None:
    with pytest.raises(InvalidImageError) as exc_info:
        decode(data, make_settings())
    assert exc_info.value.code == "INVALID_IMAGE"
    assert exc_info.value.status_code == 400


@pytest.mark.parametrize(
    "data",
    [
        factory.truncated(factory.jpeg_bytes(400, 300), 0.5),
        factory.truncated(factory.png_bytes(factory.gradient_image(400, 300)), 0.5),
        factory.truncated(factory.webp_bytes(factory.gradient_image(400, 300)), 0.5),
    ],
    ids=["jpeg", "png", "webp"],
)
def test_truncated_files_are_rejected(data: bytes) -> None:
    with pytest.raises(InvalidImageError):
        decode(data, make_settings())


def test_truncated_images_stay_rejected_even_if_something_enabled_pillows_tolerance() -> None:
    # Another library flipping this global must not make us accept half-grey photos.
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    data = factory.truncated(factory.jpeg_bytes(400, 300), 0.5)
    with pytest.raises(InvalidImageError):
        decode(data, make_settings())
    assert ImageFile.LOAD_TRUNCATED_IMAGES is False


def test_decode_error_chains_the_underlying_cause() -> None:
    with pytest.raises(InvalidImageError) as exc_info:
        decode(factory.corrupt_bytes(), make_settings())
    assert exc_info.value.__cause__ is not None


# --- format allowlist -----------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "name"),
    [(factory.bmp_bytes(), "BMP"), (factory.gif_bytes(), "GIF")],
    ids=["bmp", "gif"],
)
def test_disallowed_formats_are_rejected_with_415(data: bytes, name: str) -> None:
    with pytest.raises(UnsupportedMediaTypeError) as exc_info:
        decode(data, make_settings())
    assert exc_info.value.status_code == 415
    assert exc_info.value.code == "UNSUPPORTED_MEDIA_TYPE"
    assert name in exc_info.value.detail


def test_a_format_added_to_the_allowlist_is_accepted() -> None:
    settings = make_settings(allowed_image_formats=["JPEG", "PNG", "WEBP", "GIF"])
    assert decode(factory.gif_bytes(), settings).format == "GIF"


def test_the_allowlist_is_enforced_for_formats_that_are_normally_fine() -> None:
    settings = make_settings(allowed_image_formats=["JPEG"])
    with pytest.raises(UnsupportedMediaTypeError):
        decode(factory.png_bytes(), settings)


def test_format_is_judged_by_content_not_by_anything_the_client_claims() -> None:
    # decode() takes only bytes: a BMP is a BMP whatever Content-Type/extension
    # the client sent, and PNG bytes are accepted whatever they were named.
    with pytest.raises(UnsupportedMediaTypeError):
        decode(factory.bmp_bytes(), make_settings())
    assert decode(factory.png_bytes(), make_settings()).format == "PNG"


def test_multi_picture_jpeg_from_phones_counts_as_jpeg() -> None:
    data = factory.mpo_bytes()
    assert Image.open(io.BytesIO(data)).format == "MPO"  # the premise of this test
    assert decode(data, make_settings()).size == (100, 100)
    with pytest.raises(UnsupportedMediaTypeError):
        decode(data, make_settings(allowed_image_formats=["PNG"]))


# --- minimum size ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("width", "height", "ok"),
    [
        (64, 64, True),
        (63, 64, False),
        (64, 63, False),
        (63, 63, False),
        (64, 600, True),
        (600, 64, True),
        (600, 63, False),
        (1, 1, False),
    ],
)
def test_minimum_side_threshold_edges(width: int, height: int, ok: bool) -> None:
    data = factory.png_bytes(factory.solid_image(width, height))
    if ok:
        assert decode(data, make_settings()).size == (width, height)
    else:
        with pytest.raises(ImageTooSmallError) as exc_info:
            decode(data, make_settings())
        assert exc_info.value.code == "IMAGE_TOO_SMALL"
        assert exc_info.value.status_code == 400


def test_minimum_side_comes_from_settings() -> None:
    data = factory.png_bytes(factory.solid_image(100, 100))
    assert decode(data, make_settings(min_image_side_px=100)).size == (100, 100)
    with pytest.raises(ImageTooSmallError):
        decode(data, make_settings(min_image_side_px=101))


@pytest.mark.parametrize(
    ("width", "height", "orientation", "ok"),
    [
        (100, 64, 6, True),  # becomes 64x100
        (100, 63, 6, False),  # becomes 63x100
        (63, 100, 8, False),  # becomes 100x63
        (64, 100, 8, True),
        (100, 63, 1, False),  # upright already: same verdict
        (100, 64, 1, True),
        (100, 63, 3, False),  # 180 degrees: dimensions do not swap
    ],
)
def test_minimum_side_is_measured_on_the_exif_upright_image(
    width: int, height: int, orientation: int, ok: bool
) -> None:
    data = factory.jpeg_bytes(width, height, exif=factory.exif_bytes(orientation))
    if ok:
        decode(data, make_settings())
    else:
        with pytest.raises(ImageTooSmallError) as exc_info:
            decode(data, make_settings())
        # The message reports the upright size, as the user will see it.
        assert f"{min(width, height)}" in exc_info.value.detail


def test_unreadable_exif_does_not_break_the_size_check(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(self: Image.Image) -> Image.Exif:
        raise ValueError("corrupt EXIF")

    monkeypatch.setattr(Image.Image, "getexif", broken)
    assert decode(factory.png_bytes(), make_settings()).size == (200, 150)


# --- decompression bombs ---------------------------------------------------------


def test_image_far_over_the_pixel_limit_is_rejected() -> None:
    # 10,000 pixels against a 1,000 limit: Pillow raises DecompressionBombError (> 2x).
    data = factory.png_bytes(factory.solid_image(100, 100))
    with pytest.raises(ImageTooLargeDimensionsError) as exc_info:
        decode(data, make_settings(max_image_pixels=1000))
    assert exc_info.value.code == "IMAGE_TOO_LARGE_DIMENSIONS"
    assert exc_info.value.status_code == 400


def test_image_between_one_and_two_times_the_limit_is_rejected_too() -> None:
    # 1,600 pixels vs a 1,000 limit: Pillow only *warns* here; decode must escalate it.
    data = factory.png_bytes(factory.solid_image(40, 40))
    settings = make_settings(max_image_pixels=1000, min_image_side_px=10)
    with pytest.raises(ImageTooLargeDimensionsError):
        decode(data, settings)


def test_image_exactly_at_the_pixel_limit_is_accepted() -> None:
    data = factory.png_bytes(factory.solid_image(40, 25))  # 1,000 pixels
    settings = make_settings(max_image_pixels=1000, min_image_side_px=10)
    assert decode(data, settings).size == (40, 25)
    one_more = factory.png_bytes(factory.solid_image(1001, 1))
    with pytest.raises(ImageTooLargeDimensionsError):
        decode(one_more, make_settings(max_image_pixels=1000, min_image_side_px=1))


def test_pillows_global_pixel_limit_is_taken_from_settings() -> None:
    decode(factory.png_bytes(), make_settings(max_image_pixels=12_345_678))
    assert Image.MAX_IMAGE_PIXELS == 12_345_678


def test_bomb_handling_leaves_the_warning_filters_untouched() -> None:
    before = list(warnings.filters)
    with pytest.raises(ImageTooLargeDimensionsError):
        decode(factory.png_bytes(factory.solid_image(40, 40)), make_settings(max_image_pixels=500))
    assert list(warnings.filters) == before


def test_concurrent_decodes_get_the_right_verdicts() -> None:
    # decode() runs in a threadpool in the real app; the warning escalation must
    # not leak between threads.
    bomb = factory.png_bytes(factory.solid_image(40, 40))
    fine = factory.png_bytes(factory.solid_image(40, 25))
    settings = make_settings(max_image_pixels=1000, min_image_side_px=10)

    def attempt(index: int) -> bool:
        try:
            decode(bomb if index % 2 else fine, settings)
        except ImageTooLargeDimensionsError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(attempt, range(40)))
    assert results == [index % 2 == 0 for index in range(40)]


# --- multi-frame ---------------------------------------------------------------


def test_animated_files_decode_to_their_first_frame() -> None:
    settings = make_settings(allowed_image_formats=["JPEG", "PNG", "WEBP", "GIF"])
    for data in (factory.animated_gif(), factory.animated_webp(), factory.animated_png()):
        image = decode(data, settings)
        assert image.tell() == 0
