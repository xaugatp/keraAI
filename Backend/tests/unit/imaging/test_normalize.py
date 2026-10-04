from __future__ import annotations

import dataclasses
import hashlib
import io

import pytest
from PIL import Image

from app.core.config import Settings
from app.core.errors import InvalidImageError
from app.imaging import processing
from app.imaging.processing import NormalizedImage, decode, encode_png, normalize
from tests.fixtures import image_factory as factory
from tests.fixtures.settings import make_settings

Pixel = tuple[int, ...]


def run(data: bytes, settings: Settings | None = None) -> NormalizedImage:
    settings = settings or make_settings()
    return normalize(decode(data, settings), settings)


def stored(normalized: NormalizedImage) -> Image.Image:
    return Image.open(io.BytesIO(normalized.jpeg_bytes))


def thumbnail(normalized: NormalizedImage) -> Image.Image:
    return Image.open(io.BytesIO(normalized.thumb_bytes))


def pixel(image: Image.Image, xy: tuple[int, int]) -> Pixel:
    value = image.getpixel(xy)
    assert isinstance(value, tuple)
    return value


# --- result shape -----------------------------------------------------------------


def test_result_matches_the_spec_dataclass() -> None:
    result = run(factory.jpeg_bytes(300, 200))
    assert isinstance(result, NormalizedImage)
    assert result.image.mode == "RGB"
    assert (result.width, result.height) == (300, 200) == result.image.size
    assert stored(result).format == "JPEG"
    assert stored(result).size == (300, 200)
    assert thumbnail(result).format == "WEBP"


def test_normalized_image_is_frozen() -> None:
    result = run(factory.jpeg_bytes())
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.width = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    "data",
    [factory.jpeg_bytes(), factory.png_bytes(), factory.webp_bytes()],
    ids=["jpeg", "png", "webp"],
)
def test_every_allowed_format_normalizes_to_rgb(data: bytes) -> None:
    assert run(data).image.mode == "RGB"


# --- EXIF orientation ---------------------------------------------------------------

_RED, _GREEN, _BLUE, _YELLOW = (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)
_COLOURS = {"R": _RED, "G": _GREEN, "B": _BLUE, "Y": _YELLOW}


def _quadrants(width: int = 200, height: int = 100) -> Image.Image:
    """Top-left red, top-right green, bottom-left blue, bottom-right yellow."""
    image = Image.new("RGB", (width, height))
    image.paste(_RED, (0, 0, width // 2, height // 2))
    image.paste(_GREEN, (width // 2, 0, width, height // 2))
    image.paste(_BLUE, (0, height // 2, width // 2, height))
    image.paste(_YELLOW, (width // 2, height // 2, width, height))
    return image


def _nearest_colour(value: Pixel) -> str:
    return min(
        _COLOURS,
        key=lambda name: sum((a - b) ** 2 for a, b in zip(value[:3], _COLOURS[name], strict=True)),
    )


def _corners(image: Image.Image) -> str:
    """Colour names at top-left, top-right, bottom-left, bottom-right (8 px inset)."""
    width, height = image.size
    points = [(8, 8), (width - 9, 8), (8, height - 9), (width - 9, height - 9)]
    return "".join(_nearest_colour(pixel(image, point)) for point in points)


# For each EXIF orientation: the corner colours of the *displayed* picture.
_EXPECTED_CORNERS = {
    1: "RGBY",
    2: "GRYB",  # mirror horizontal
    3: "YBGR",  # rotate 180
    4: "BYRG",  # mirror vertical
    5: "RBGY",  # transpose
    6: "BRYG",  # rotate 90 clockwise
    7: "YGBR",  # transverse
    8: "GYRB",  # rotate 90 counter-clockwise
}


@pytest.mark.parametrize("orientation", range(1, 9))
def test_exif_orientation_is_applied(orientation: int) -> None:
    data = factory.jpeg_bytes(image=_quadrants(), exif=factory.exif_bytes(orientation))
    result = run(data)
    expected_size = (100, 200) if orientation >= 5 else (200, 100)
    assert result.image.size == expected_size
    assert _corners(result.image) == _EXPECTED_CORNERS[orientation]
    # The stored file is upright too: re-opening it needs no further rotation.
    assert stored(result).size == expected_size
    assert _corners(stored(result)) == _EXPECTED_CORNERS[orientation]


@pytest.mark.parametrize("fmt", ["PNG", "WEBP"])
def test_exif_orientation_is_applied_for_png_and_webp_too(fmt: str) -> None:
    data = factory.encode(_quadrants(), fmt, exif=factory.exif_bytes(6))
    result = run(data)
    assert result.image.size == (100, 200)
    assert _corners(result.image) == _EXPECTED_CORNERS[6]


def test_corrupt_exif_does_not_fail_the_upload(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(image: Image.Image, **_: object) -> Image.Image:
        raise ValueError("bad EXIF")

    monkeypatch.setattr(processing.ImageOps, "exif_transpose", boom)
    assert run(factory.jpeg_bytes(200, 100)).image.size == (200, 100)


# --- metadata stripping -------------------------------------------------------------


def test_stored_jpeg_and_thumbnail_carry_no_exif_gps_or_icc() -> None:
    data = factory.jpeg_bytes(
        200, 100, exif=factory.exif_bytes(6), icc_profile=factory.display_p3_profile_bytes()
    )
    assert b"KeraTestMake" in data  # the input really has the metadata
    result = run(data)
    for blob in (result.jpeg_bytes, result.thumb_bytes):
        assert b"KeraTestMake" not in blob
        assert b"Exif" not in blob
        assert b"ICC_PROFILE" not in blob
        assert b"Display P3" not in blob
    assert dict(stored(result).getexif()) == {}
    assert "icc_profile" not in stored(result).info
    assert dict(thumbnail(result).getexif()) == {}
    assert "icc_profile" not in thumbnail(result).info
    assert result.image.info == {}


# --- transparency --------------------------------------------------------------------


def test_rgba_is_composited_on_white_not_black() -> None:
    result = run(factory.png_bytes(factory.rgba_image()))
    assert pixel(result.image, (10, 50)) == (255, 0, 0)  # opaque red stays
    assert pixel(result.image, (90, 50)) == (255, 255, 255)  # transparent -> white
    assert pixel(stored(result), (90, 50))[0] > 250


def test_semi_transparent_pixels_blend_with_white() -> None:
    image = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
    result = run(factory.png_bytes(image))
    red, green, blue = pixel(result.image, (50, 50))
    assert red == 255
    assert green == blue
    assert 125 <= green <= 130  # half red over white


def test_la_is_composited_on_white() -> None:
    result = run(factory.png_bytes(factory.la_image()))
    assert pixel(result.image, (10, 50)) == (0, 0, 0)
    assert pixel(result.image, (90, 50)) == (255, 255, 255)


def test_palette_image_with_transparency_is_composited_on_white() -> None:
    result = run(factory.palette_transparent_png())
    assert pixel(result.image, (10, 50)) == (0, 0, 255)
    assert pixel(result.image, (90, 50)) == (255, 255, 255)


@pytest.mark.parametrize(
    ("mode", "bad_transparency"), [("P", (0, 0, 0)), ("RGB", b"\x00\x00\x00\x00")]
)
def test_malformed_transparency_info_does_not_fail_normalisation(
    mode: str, bad_transparency: object
) -> None:
    image = Image.new(mode, (100, 100), (10, 200, 30) if mode == "RGB" else 0)
    image.info["transparency"] = bad_transparency
    assert normalize(image, make_settings()).image.mode == "RGB"


def test_opaque_palette_image_is_converted_to_rgb() -> None:
    image = Image.new("P", (100, 100), 1)
    image.putpalette([0, 0, 0, 10, 200, 30] + [0] * (254 * 3))
    assert pixel(run(factory.png_bytes(image)).image, (50, 50)) == (10, 200, 30)


def test_transparent_color_key_png_is_composited_on_white() -> None:
    image = Image.new("RGB", (100, 100), (255, 0, 0))
    image.paste((0, 255, 0), (50, 0, 100, 100))
    result = run(factory.png_bytes(image, transparency=(0, 255, 0)))
    assert pixel(result.image, (10, 50)) == (255, 0, 0)
    assert pixel(result.image, (90, 50)) == (255, 255, 255)


# --- other colour modes --------------------------------------------------------------


def test_cmyk_jpeg_becomes_rgb_with_the_right_colour() -> None:
    result = run(factory.cmyk_jpeg())
    red, green, blue = pixel(result.image, (50, 50))
    assert red > 230
    assert green < 40
    assert blue < 40


def test_cmyk_with_an_unusable_profile_falls_back_to_plain_conversion() -> None:
    result = run(factory.cmyk_jpeg(icc_profile=b"not a profile" * 10))
    red, green, blue = pixel(result.image, (50, 50))
    assert red > 230
    assert green < 40
    assert blue < 40


def test_cmyk_with_an_rgb_profile_ignores_the_mismatched_profile() -> None:
    plain = run(factory.cmyk_jpeg())
    mismatched = run(factory.cmyk_jpeg(icc_profile=factory.display_p3_profile_bytes()))
    assert pixel(mismatched.image, (50, 50)) == pixel(plain.image, (50, 50))


def test_grayscale_jpeg_becomes_rgb() -> None:
    gray = Image.linear_gradient("L").resize((100, 100))
    result = run(factory.encode(gray, "JPEG"))
    red, green, blue = pixel(result.image, (50, 50))
    assert red == green == blue


def test_one_bit_image_becomes_rgb() -> None:
    bilevel = Image.new("1", (100, 100), 1)
    assert pixel(run(factory.png_bytes(bilevel)).image, (50, 50)) == (255, 255, 255)


def test_16_bit_grayscale_is_scaled_not_clipped() -> None:
    result = run(factory.gray16_png())
    # Column x holds 257*x, i.e. x on an 8-bit scale. Pillow's own convert("L")
    # would turn everything above 255 (all but the first column) pure white.
    for x in (0, 1, 64, 128, 200, 255):
        red, green, blue = pixel(result.image, (x, 10))
        assert red == green == blue
        assert abs(red - x) <= 1, f"column {x} -> {red}"


def _single_row(mode: str, values: list[float]) -> Image.Image:
    image = Image.new(mode, (len(values), 1))
    image.putdata(values)
    return image


def _normalized_row(image: Image.Image) -> list[int]:
    result = normalize(image, make_settings())
    return [pixel(result.image, (x, 0))[0] for x in range(result.width)]


def test_32_bit_integer_data_within_16_bit_range_uses_full_scale() -> None:
    assert _normalized_row(_single_row("I", [0, 16384, 32768, 65535])) == [0, 64, 128, 255]


def test_32_bit_integer_data_outside_16_bit_range_is_stretched() -> None:
    assert _normalized_row(_single_row("I", [-100, 0, 100])) == [0, 128, 255]


def test_float_data_in_unit_range_is_scaled_by_255() -> None:
    assert _normalized_row(_single_row("F", [0.0, 0.25, 1.0])) == [0, 64, 255]


def test_float_data_outside_unit_range_is_stretched() -> None:
    assert _normalized_row(_single_row("F", [-5.0, 0.0, 5.0])) == [0, 128, 255]


def test_constant_out_of_range_integer_image_does_not_crash() -> None:
    assert _normalized_row(_single_row("I", [70000, 70000])) == [0, 0]


def test_a_conversion_failure_is_reported_as_an_invalid_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unsupported(*_: object) -> tuple[Image.Image, bool]:
        raise ValueError("conversion not supported")

    monkeypatch.setattr(processing, "_to_rgb", unsupported)
    with pytest.raises(InvalidImageError) as exc_info:
        normalize(Image.new("RGB", (8, 8)), make_settings())
    assert isinstance(exc_info.value.__cause__, ValueError)


def test_uncommon_pillow_modes_still_convert() -> None:
    for mode in ("LAB", "HSV", "YCbCr"):
        assert normalize(Image.new(mode, (8, 8)), make_settings()).image.mode == "RGB"


# --- colour management ---------------------------------------------------------------

_SKIN = (200, 100, 100)


def _flat_png(colour: Pixel, **params: object) -> bytes:
    return factory.png_bytes(factory.solid_image(100, 100, colour), **params)


def test_display_p3_pixels_are_converted_towards_srgb() -> None:
    plain = pixel(run(_flat_png(_SKIN)).image, (50, 50))
    converted = pixel(
        run(_flat_png(_SKIN, icc_profile=factory.display_p3_profile_bytes())).image, (50, 50)
    )
    assert plain == _SKIN  # no profile: untouched
    # The same numbers mean a more saturated colour in the wider P3 gamut, so in
    # sRGB the dominant channel goes up and the others come down.
    assert converted[0] > plain[0]
    assert converted[1] < plain[1]
    assert converted[2] < plain[2]


def test_display_p3_green_is_converted_the_other_way_round() -> None:
    green = (100, 200, 100)
    plain = pixel(run(_flat_png(green)).image, (50, 50))
    converted = pixel(
        run(_flat_png(green, icc_profile=factory.display_p3_profile_bytes())).image, (50, 50)
    )
    assert converted[1] > plain[1]
    assert converted[0] < plain[0]


def test_display_p3_jpeg_from_a_phone_is_converted_too() -> None:
    image = factory.solid_image(100, 100, _SKIN)
    with_profile = run(
        factory.jpeg_bytes(image=image, icc_profile=factory.display_p3_profile_bytes())
    )
    without = run(factory.jpeg_bytes(image=image))
    assert pixel(with_profile.image, (50, 50))[0] > pixel(without.image, (50, 50))[0]
    assert "icc_profile" not in stored(with_profile).info


def test_an_srgb_profile_leaves_pixels_exactly_as_they_were() -> None:
    result = run(_flat_png(_SKIN, icc_profile=factory.srgb_profile_bytes()))
    assert pixel(result.image, (50, 50)) == _SKIN


def test_a_corrupt_profile_is_ignored_without_failing() -> None:
    result = run(_flat_png(_SKIN, icc_profile=b"definitely not an ICC profile" * 8))
    assert pixel(result.image, (50, 50)) == _SKIN


def test_a_profile_of_the_wrong_colour_space_is_ignored() -> None:
    # An RGB image carrying a non-RGB (here Lab) profile must not be mangled.
    result = run(_flat_png(_SKIN, icc_profile=factory.lab_profile_bytes()))
    assert pixel(result.image, (50, 50)) == _SKIN


# --- multi-frame --------------------------------------------------------------------


@pytest.mark.parametrize(
    "data",
    [factory.animated_gif(), factory.animated_webp(), factory.animated_png()],
    ids=["gif", "webp", "apng"],
)
def test_only_the_first_frame_of_an_animation_is_used(data: bytes) -> None:
    settings = make_settings(allowed_image_formats=["JPEG", "PNG", "WEBP", "GIF"])
    result = run(data, settings)
    red, _, blue = pixel(result.image, (50, 50))
    assert red > 240
    assert blue < 15  # the second frame is solid blue


def test_a_decoder_left_on_a_later_frame_is_rewound_to_the_first() -> None:
    settings = make_settings(allowed_image_formats=["JPEG", "PNG", "WEBP", "GIF"])
    image = decode(factory.animated_gif(), settings)
    image.seek(1)
    red, _, blue = pixel(normalize(image, settings).image, (50, 50))
    assert red > 240
    assert blue < 15


# --- resizing -----------------------------------------------------------------------


def test_large_images_are_downscaled_to_the_store_limit() -> None:
    result = run(factory.jpeg_bytes(3000, 2000))
    assert result.image.size == (2048, 1365)
    assert stored(result).size == (2048, 1365)


def test_portrait_images_are_limited_on_their_longest_side() -> None:
    result = run(factory.jpeg_bytes(1000, 3000))
    assert result.image.size == (683, 2048)


def test_store_limit_comes_from_settings() -> None:
    result = run(factory.jpeg_bytes(400, 200), make_settings(store_max_side_px=100))
    assert result.image.size == (100, 50)


@pytest.mark.parametrize(
    ("side", "expected_longest"), [(2047, 2047), (2048, 2048), (2049, 2048), (1000, 1000)]
)
def test_size_threshold_edges_around_the_store_limit(side: int, expected_longest: int) -> None:
    data = factory.png_bytes(factory.solid_image(side, 100))
    assert max(run(data).image.size) == expected_longest


def test_small_images_are_never_upscaled() -> None:
    result = run(factory.jpeg_bytes(300, 200))
    assert result.image.size == (300, 200)
    assert thumbnail(result).size == (300, 200)  # smaller than the 320 thumbnail side too


def test_thumbnail_longest_side_is_the_configured_size() -> None:
    result = run(factory.jpeg_bytes(1600, 1200))
    assert thumbnail(result).size == (320, 240)
    portrait = run(factory.jpeg_bytes(900, 1800))
    assert thumbnail(portrait).size == (160, 320)


def test_thumbnail_side_comes_from_settings() -> None:
    result = run(factory.jpeg_bytes(1600, 1200), make_settings(thumbnail_side_px=80))
    assert thumbnail(result).size == (80, 60)


def test_thumbnail_at_exactly_the_configured_size_is_not_resized() -> None:
    assert thumbnail(run(factory.jpeg_bytes(320, 200))).size == (320, 200)


def test_extreme_aspect_ratio_never_resizes_a_side_to_zero() -> None:
    settings = make_settings(min_image_side_px=1, store_max_side_px=100, thumbnail_side_px=50)
    result = run(factory.png_bytes(factory.solid_image(2000, 1)), settings)
    assert result.image.size == (100, 1)
    assert thumbnail(result).size == (50, 1)


# --- hashing and encoding -----------------------------------------------------------


def test_sha256_is_the_hash_of_the_stored_jpeg_bytes() -> None:
    result = run(factory.jpeg_bytes())
    assert result.sha256 == hashlib.sha256(result.jpeg_bytes).hexdigest()
    assert len(result.sha256) == 64


def test_the_same_upload_always_produces_the_same_bytes_and_hash() -> None:
    data = factory.jpeg_bytes(500, 400, exif=factory.exif_bytes(6))
    first, second = run(data), run(data)
    assert first.jpeg_bytes == second.jpeg_bytes
    assert first.thumb_bytes == second.thumb_bytes
    assert first.sha256 == second.sha256


def test_different_pictures_have_different_hashes() -> None:
    assert run(factory.jpeg_bytes(200, 150)).sha256 != run(factory.jpeg_bytes(201, 150)).sha256


def test_input_metadata_does_not_influence_the_hash() -> None:
    # The hash covers the *normalised* output (no EXIF, no container), which is
    # what makes it usable for sample idempotency.
    image = factory.gradient_image(200, 150)
    plain = run(factory.png_bytes(image))
    with_exif = run(factory.png_bytes(image, exif=factory.exif_bytes(1)))
    assert plain.sha256 == with_exif.sha256


def test_encode_png_round_trips_losslessly_including_alpha() -> None:
    overlay = factory.rgba_image(40, 40)
    decoded = Image.open(io.BytesIO(encode_png(overlay)))
    assert decoded.format == "PNG"
    assert decoded.mode == "RGBA"
    assert decoded.tobytes() == overlay.tobytes()
