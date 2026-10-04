"""Programmatic test images, so no binary fixtures live in the repository.

Each helper returns encoded ``bytes`` (what a client would upload) unless the
name says it returns an ``Image``. Images are small; the interesting
properties (orientation, alpha, colour profile, bit depth, ...) are what
matter, not the picture.
"""

from __future__ import annotations

import io
import struct
from typing import Any

from PIL import Image, ImageCms

# --------------------------------------------------------------------------
# Plain pictures
# --------------------------------------------------------------------------


def gradient_image(width: int = 200, height: int = 150) -> Image.Image:
    """RGB image with a horizontal red ramp, vertical green ramp and constant blue."""
    vertical = Image.linear_gradient("L").resize((width, height))
    horizontal = vertical.transpose(Image.Transpose.ROTATE_90).resize((width, height))
    blue = Image.new("L", (width, height), 128)
    return Image.merge("RGB", (horizontal, vertical, blue))


def solid_image(
    width: int, height: int, colour: tuple[int, ...] = (200, 40, 40), mode: str = "RGB"
) -> Image.Image:
    return Image.new(mode, (width, height), colour)


def encode(image: Image.Image, fmt: str, **params: Any) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **params)
    return buffer.getvalue()


def jpeg_bytes(
    width: int = 200,
    height: int = 150,
    *,
    image: Image.Image | None = None,
    quality: int = 90,
    exif: bytes | None = None,
    icc_profile: bytes | None = None,
) -> bytes:
    params: dict[str, Any] = {"quality": quality}
    if exif is not None:
        params["exif"] = exif
    if icc_profile is not None:
        params["icc_profile"] = icc_profile
    return encode(image or gradient_image(width, height), "JPEG", **params)


def png_bytes(image: Image.Image | None = None, **params: Any) -> bytes:
    return encode(image or gradient_image(), "PNG", **params)


def webp_bytes(image: Image.Image | None = None, **params: Any) -> bytes:
    return encode(image or gradient_image(), "WEBP", **params)


def bmp_bytes(width: int = 100, height: int = 100) -> bytes:
    return encode(gradient_image(width, height), "BMP")


def gif_bytes(width: int = 100, height: int = 100) -> bytes:
    return encode(gradient_image(width, height), "GIF")


# --------------------------------------------------------------------------
# Broken files
# --------------------------------------------------------------------------


def corrupt_bytes() -> bytes:
    return b"this is definitely not an image" * 20


def truncated(data: bytes, keep: float = 0.5) -> bytes:
    return data[: int(len(data) * keep)]


# --------------------------------------------------------------------------
# EXIF
# --------------------------------------------------------------------------


def exif_bytes(orientation: int = 1) -> bytes:
    """EXIF block with an orientation plus device and GPS tags that must be stripped."""
    exif = Image.Exif()
    exif[0x010F] = "KeraTestMake"
    exif[0x0110] = "KeraTestModel"
    exif[0x0112] = orientation
    gps = exif.get_ifd(0x8825)
    gps[1] = "N"
    gps[2] = (27.0, 43.0, 2.0)
    return exif.tobytes()


# --------------------------------------------------------------------------
# Colour profiles
# --------------------------------------------------------------------------


def srgb_profile_bytes() -> bytes:
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def lab_profile_bytes() -> bytes:
    """A valid ICC profile whose colour space is not RGB (Lab)."""
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("LAB")).tobytes()


def _s15f16(value: float) -> bytes:
    return struct.pack(">i", round(value * 65536))


def _xyz_tag(x: float, y: float, z: float) -> bytes:
    return b"XYZ \0\0\0\0" + _s15f16(x) + _s15f16(y) + _s15f16(z)


def _gamma_tag(gamma: float) -> bytes:
    # 'curv' with a single entry is a pure gamma encoded as u8Fixed8.
    return b"curv\0\0\0\0" + struct.pack(">IH", 1, round(gamma * 256))


def _text_tag(text: str) -> bytes:
    return b"text\0\0\0\0" + text.encode("ascii") + b"\0"


def _desc_tag(text: str) -> bytes:
    ascii_text = text.encode("ascii") + b"\0"
    return (
        b"desc\0\0\0\0"
        + struct.pack(">I", len(ascii_text))
        + ascii_text
        + struct.pack(">II", 0, 0)  # empty Unicode description
        + struct.pack(">HB", 0, 0)  # empty ScriptCode description
        + b"\0" * 67
    )


def display_p3_profile_bytes() -> bytes:
    """A minimal ICC v2 matrix/TRC RGB profile with Display P3 primaries.

    Built by hand so the tests need no binary profile. Colorants are the
    D50-adapted Display P3 values; the transfer curve is a plain gamma 2.2
    (the real profile uses the sRGB curve, which does not matter for checking
    that wide-gamut pixels are moved towards sRGB).
    """
    tags: list[tuple[bytes, bytes]] = [
        (b"desc", _desc_tag("Display P3")),
        (b"cprt", _text_tag("Test profile")),
        (b"wtpt", _xyz_tag(0.9642, 1.0, 0.8249)),
        (b"rXYZ", _xyz_tag(0.5151, 0.2412, -0.0011)),
        (b"gXYZ", _xyz_tag(0.2920, 0.6922, 0.0419)),
        (b"bXYZ", _xyz_tag(0.1571, 0.0666, 0.7841)),
        (b"rTRC", _gamma_tag(2.2)),
        (b"gTRC", _gamma_tag(2.2)),
        (b"bTRC", _gamma_tag(2.2)),
    ]
    table_size = 4 + 12 * len(tags)
    offset = 128 + table_size
    table = struct.pack(">I", len(tags))
    body = b""
    for signature, data in tags:
        padded = data + b"\0" * (-len(data) % 4)
        table += signature + struct.pack(">II", offset + len(body), len(data))
        body += padded

    header = bytearray(128)
    struct.pack_into(">I", header, 0, 128 + len(table) + len(body))
    header[8:12] = bytes([2, 0x10, 0, 0])  # version 2.1
    header[12:16] = b"mntr"
    header[16:20] = b"RGB "
    header[20:24] = b"XYZ "
    header[24:36] = struct.pack(">6H", 2024, 1, 1, 0, 0, 0)
    header[36:40] = b"acsp"
    header[68:80] = _s15f16(0.9642) + _s15f16(1.0) + _s15f16(0.8249)  # D50 illuminant
    return bytes(header) + table + body


# --------------------------------------------------------------------------
# Modes with special handling
# --------------------------------------------------------------------------


def rgba_image(width: int = 100, height: int = 100) -> Image.Image:
    """Left half opaque red, right half fully transparent *black*.

    Dropping the alpha channel (or compositing on black) turns the right half
    black; compositing on white turns it white.
    """
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    image.paste((255, 0, 0, 255), (0, 0, width // 2, height))
    return image


def la_image(width: int = 100, height: int = 100) -> Image.Image:
    """Left half opaque black, right half fully transparent."""
    image = Image.new("LA", (width, height), (0, 0))
    image.paste((0, 255), (0, 0, width // 2, height))
    return image


def palette_transparent_png(width: int = 100, height: int = 100) -> bytes:
    """P-mode PNG whose index 0 (right half) is transparent and index 1 (left) is blue."""
    image = Image.new("P", (width, height), 0)
    image.putpalette([0, 0, 0, 0, 0, 255] + [0] * (254 * 3))
    image.paste(1, (0, 0, width // 2, height))
    return encode(image, "PNG", transparency=0)


def cmyk_jpeg(width: int = 100, height: int = 100, **params: Any) -> bytes:
    """CMYK JPEG that is 100% magenta + 100% yellow, i.e. red."""
    return encode(Image.new("CMYK", (width, height), (0, 255, 255, 0)), "JPEG", **params)


def gray16_png(width: int = 256, height: int = 64) -> bytes:
    """16-bit grayscale PNG with a horizontal ramp from 0 to 65535."""
    ramp = [round(x * 65535 / (width - 1)) for x in range(width)]
    row = b"".join(value.to_bytes(2, "little") for value in ramp)
    image = Image.frombytes("I;16", (width, height), row * height)
    return encode(image, "PNG")


# --------------------------------------------------------------------------
# Animations / multi-frame
# --------------------------------------------------------------------------


def _two_frames(width: int, height: int) -> tuple[Image.Image, Image.Image]:
    return (
        Image.new("RGB", (width, height), (255, 0, 0)),
        Image.new("RGB", (width, height), (0, 0, 255)),
    )


def animated_gif(width: int = 100, height: int = 100) -> bytes:
    first, second = _two_frames(width, height)
    return encode(first, "GIF", save_all=True, append_images=[second], duration=100, loop=0)


def animated_webp(width: int = 100, height: int = 100) -> bytes:
    first, second = _two_frames(width, height)
    return encode(first, "WEBP", save_all=True, append_images=[second], duration=100, lossless=True)


def animated_png(width: int = 100, height: int = 100) -> bytes:
    first, second = _two_frames(width, height)
    return encode(first, "PNG", save_all=True, append_images=[second], duration=100)


def mpo_bytes(width: int = 100, height: int = 100) -> bytes:
    """Multi-picture JPEG (what some phones produce); Pillow reports format 'MPO'."""
    first, second = _two_frames(width, height)
    return encode(first, "MPO", save_all=True, append_images=[second])
