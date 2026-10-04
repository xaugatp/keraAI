"""Shared image handling for every model: read, validate, normalise, encode.

Pipeline (spec section 6): ``read_upload_limited`` -> ``decode`` -> ``normalize``.
Everything here is a pure function with settings passed in. Model-specific
preprocessing (resize / letterbox / tensor conversion) lives in each predictor.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import logging
import threading
import warnings
from dataclasses import dataclass
from typing import Protocol, cast

from PIL import Image, ImageCms, ImageFile, ImageOps

from app.core.config import Settings
from app.core.errors import (
    AppError,
    ImageTooLargeDimensionsError,
    ImageTooSmallError,
    InvalidImageError,
    PayloadTooLargeError,
    UnsupportedMediaTypeError,
)

logger = logging.getLogger(__name__)

_CHUNK_BYTES = 1024 * 1024
_BYTES_PER_MB = 1024 * 1024

_JPEG_QUALITY = 90
_THUMBNAIL_QUALITY = 80

_EXIF_ORIENTATION_TAG = 0x0112
# EXIF orientations 5-8 are the 90-degree rotations/transposes, i.e. width and
# height swap when the orientation is applied.
_SWAPPED_ORIENTATIONS = frozenset({5, 6, 7, 8})

# Pillow reports multi-picture JPEGs (some Samsung/Pixel/depth-map phone
# photos) as "MPO". Its first frame is an ordinary JPEG, so treating it as
# JPEG avoids rejecting real phone photos with a 415.
_FORMAT_ALIASES = {"MPO": "JPEG"}

_ALPHA_MODES = frozenset({"RGBA", "LA", "PA", "RGBa", "La"})
_FULL_SCALE_16_BIT = 65535

# warnings.catch_warnings() mutates interpreter-global state and is not
# thread-safe, while decode() runs in FastAPI's threadpool. Serialising only
# the (cheap, header-only) Image.open calls keeps the filter stack consistent.
_PIL_GUARD = threading.Lock()


class AsyncReadable(Protocol):
    """The slice of ``starlette.UploadFile`` that ``read_upload_limited`` uses."""

    async def read(self, size: int = -1, /) -> bytes: ...


@dataclass(frozen=True)
class NormalizedImage:
    """A validated, upright, sRGB, size-capped image plus its stored encodings."""

    image: Image.Image  # RGB, ready for a predictor (lossless, pre-JPEG pixels)
    width: int
    height: int
    jpeg_bytes: bytes  # what gets stored as original.jpg
    thumb_bytes: bytes  # what gets stored as thumb.webp
    sha256: str  # hex digest of jpeg_bytes


def max_upload_bytes(settings: Settings) -> int:
    return settings.max_upload_mb * _BYTES_PER_MB


def _too_large(max_bytes: int) -> PayloadTooLargeError:
    return PayloadTooLargeError(f"Image exceeds the {max_bytes / _BYTES_PER_MB:g} MB limit.")


async def read_upload_limited(
    upload: AsyncReadable, max_bytes: int, content_length: int | None = None
) -> bytes:
    """Read an upload in 1 MB chunks, aborting as soon as it exceeds ``max_bytes``.

    ``content_length`` (the request header, if the client sent one) lets us
    refuse an oversized body before reading a single byte. It cannot be
    trusted on its own (clients can lie, chunked uploads have none), so the
    streaming check below is the authoritative one.
    """
    if content_length is not None and content_length > max_bytes:
        raise _too_large(max_bytes)

    buffer = bytearray()
    while True:
        # Never ask for more than one byte past the limit, so an oversized
        # upload is detected after reading at most max_bytes + 1 bytes
        # instead of buffering a whole extra chunk.
        chunk = await upload.read(min(_CHUNK_BYTES, max_bytes - len(buffer) + 1))
        if not chunk:
            break
        buffer.extend(chunk)
        if len(buffer) > max_bytes:
            raise _too_large(max_bytes)
    return bytes(buffer)


def _open_image(data: bytes, settings: Settings) -> ImageFile.ImageFile:
    """Open (header only) with Pillow's safety limits applied.

    This is the single place that configures Pillow's process-wide guards:
    - MAX_IMAGE_PIXELS comes from settings. Pillow only *warns* above that
      limit (and raises above 2x), so the warning is escalated to an error
      here; otherwise a 1x-2x bomb would slip through.
    - LOAD_TRUNCATED_IMAGES is forced off so truncated files fail loudly
      instead of decoding to a half-grey picture.
    """
    with _PIL_GUARD, warnings.catch_warnings():
        Image.MAX_IMAGE_PIXELS = settings.max_image_pixels
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        return Image.open(io.BytesIO(data))


def _canonical_format(image: Image.Image) -> str:
    fmt = (image.format or "").upper()
    return _FORMAT_ALIASES.get(fmt, fmt)


def _oriented_size(image: Image.Image) -> tuple[int, int]:
    """Size the image will have once its EXIF orientation is applied."""
    width, height = image.size
    try:
        orientation = image.getexif().get(_EXIF_ORIENTATION_TAG, 1)
    except Exception:  # corrupt EXIF must not fail a decodable image
        logger.debug("unreadable EXIF while measuring image", exc_info=True)
        orientation = 1
    return (height, width) if orientation in _SWAPPED_ORIENTATIONS else (width, height)


def decode(data: bytes, settings: Settings) -> Image.Image:
    """Validate and fully decode untrusted bytes into a Pillow image.

    The returned image is loaded (pixels in memory) but still in its original
    orientation/mode: ``normalize`` does the rest. Raises:
    - ``ImageTooLargeDimensionsError`` (400) for decompression bombs,
    - ``UnsupportedMediaTypeError`` (415) when the *decoded* format is not allowed,
    - ``InvalidImageError`` (400) for corrupt/truncated/unidentifiable data,
    - ``ImageTooSmallError`` (400) when the upright image is below the minimum side.
    """
    allowed = {fmt.upper() for fmt in settings.allowed_image_formats}
    try:
        probe = _open_image(data, settings)
        # Trust the decoded format, never the client's Content-Type. Checked
        # before verify()/load() so a disallowed format never reaches its
        # (often less-hardened) pixel decoder.
        fmt = _canonical_format(probe)
        if fmt not in allowed:
            raise UnsupportedMediaTypeError(
                f"Image format {fmt or 'unknown'} is not supported. "
                f"Allowed formats: {', '.join(sorted(allowed))}."
            )
        # verify() invalidates the object (Pillow docs), so reopen to decode.
        # load() is what actually catches truncated data: verify() is a no-op
        # for JPEG and WEBP.
        probe.verify()
        image = _open_image(data, settings)
        image.load()
    except AppError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageTooLargeDimensionsError(
            f"Image dimensions exceed the {settings.max_image_pixels}-pixel limit."
        ) from exc
    except Exception as exc:
        # Pillow decoders raise OSError, SyntaxError, ValueError, struct.error,
        # EOFError, ... for hostile input; to a client they are all "not an image".
        logger.info("image decode failed: %s", type(exc).__name__)
        raise InvalidImageError("The uploaded file is not a valid image.") from exc

    # Measured on the upright size so a rotated phone photo is judged by how
    # it will actually be stored.
    width, height = _oriented_size(image)
    if min(width, height) < settings.min_image_side_px:
        raise ImageTooSmallError(
            f"Image must be at least {settings.min_image_side_px}x{settings.min_image_side_px} "
            f"pixels (got {width}x{height})."
        )
    return image


def _first_frame_upright(image: Image.Image) -> Image.Image:
    """First frame only (animated GIF/WEBP/APNG), with EXIF orientation applied."""
    # seek(0) is a no-op for still images and rewinds animated ones; EOFError
    # would only mean "no frames", which the decoder already ruled out.
    with contextlib.suppress(EOFError):
        image.seek(0)
    try:
        return ImageOps.exif_transpose(image)
    except Exception:  # corrupt EXIF must not fail the upload
        logger.debug("EXIF transpose failed; keeping stored orientation", exc_info=True)
        return image.copy()


def _icc_to_srgb(image: Image.Image, icc_profile: bytes, source_space: str) -> Image.Image | None:
    """Convert ``image`` from its embedded ICC profile to sRGB.

    Returns ``None`` when no conversion should/can happen (profile unusable,
    wrong colour space for the image, or already sRGB); the caller then keeps
    the pixels as they are. A bad profile must never fail an upload.
    """
    try:
        source = ImageCms.ImageCmsProfile(io.BytesIO(icc_profile))
        space = str(source.profile.xcolor_space).strip().upper()
        if space != source_space:
            logger.debug("ICC colour space %s does not match image (%s)", space, source_space)
            return None
        if source_space == "RGB" and "srgb" in ImageCms.getProfileDescription(source).lower():
            return None
        target = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
        # Relative colorimetric + black point compensation is what browsers
        # and Photoshop default to for photographic content.
        return ImageCms.profileToProfile(
            image,
            source,
            target,
            renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
            outputMode="RGB",
            flags=ImageCms.Flags.BLACKPOINTCOMPENSATION,
        )
    except Exception:  # corrupt/unsupported profile: LittleCMS raises many types
        logger.debug("ICC conversion failed; keeping pixels unchanged", exc_info=True)
        return None


def _scale_to_8bit(image: Image.Image) -> Image.Image:
    """Map 16-bit / 32-bit integer / float grayscale onto 8-bit 'L' without clipping.

    Pillow's own convert("L") clips these modes (a 16-bit 1000 becomes white),
    which turns real 16-bit grayscale PNGs into near-solid white.
    - I;16*: fixed full scale (0..65535 -> 0..255).
    - I: assumed 16-bit data when within 0..65535, otherwise stretched by its extrema.
    - F: 0..1 data is scaled by 255, otherwise stretched by its extrema.
    """
    if image.mode.startswith("I;16"):
        image = image.convert("I")
        low, high = 0.0, float(_FULL_SCALE_16_BIT)
    else:
        # Single-band modes only ever return one (min, max) pair.
        observed_low, observed_high = cast(tuple[float, float], image.getextrema())
        full_high = float(_FULL_SCALE_16_BIT) if image.mode == "I" else 1.0
        if observed_low >= 0 and observed_high <= full_high:
            low, high = 0.0, full_high
        else:
            low, high = observed_low, observed_high
    scale = 255.0 / (high - low) if high > low else 0.0
    # +0.5 turns Pillow's truncating cast into rounding; the 1e-6 keeps exact
    # halves from rounding down when floating-point error leaves them at .4999...
    offset = 0.5 + 1e-6 - low * scale
    return image.point(lambda value: value * scale + offset).convert("L")


def _composite_on_white(image: Image.Image) -> Image.Image:
    # White, not Pillow's default black: transparent leaf cut-outs would
    # otherwise turn into dark halos/backgrounds that bias the models.
    rgba = image.convert("RGBA")
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(white, rgba).convert("RGB")


def _has_alpha(image: Image.Image) -> bool:
    return image.mode in _ALPHA_MODES or "transparency" in image.info


def _to_rgb(image: Image.Image, icc_profile: bytes | None) -> tuple[Image.Image, bool]:
    """Convert any Pillow mode to RGB; second item says whether an RGB ICC profile is still to apply."""
    mode = image.mode
    if mode.startswith("I;16") or mode in {"I", "F"}:
        return _scale_to_8bit(image).convert("RGB"), False
    if mode == "CMYK":
        # CMYK has to be colour-managed from CMYK itself; a naive convert("RGB")
        # gives the dull, shifted colours typical of print-oriented JPEGs.
        converted = _icc_to_srgb(image, icc_profile, "CMYK") if icc_profile else None
        return (converted if converted is not None else image.convert("RGB")), False
    if _has_alpha(image):
        try:
            return _composite_on_white(image), icc_profile is not None
        except (ValueError, TypeError):
            # A malformed tRNS entry (e.g. a tuple on a palette image) makes Pillow
            # refuse to build the alpha channel. Better to keep the opaque colours
            # than to reject an otherwise decodable photo.
            logger.debug("ignoring unusable transparency on %s image", mode, exc_info=True)
    return image.convert("RGB"), icc_profile is not None


def _fit_within(image: Image.Image, max_side: int) -> Image.Image:
    """Downscale so the longest side is at most ``max_side``; never upscales."""
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image
    scale = max_side / longest
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def _encode_jpeg(image: Image.Image) -> bytes:
    # No exif=/icc_profile= arguments: Pillow only writes what is passed to
    # save(), so the stored file carries no GPS/device data (P-06). The pixels
    # are already sRGB, so no profile is needed.
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=_JPEG_QUALITY, optimize=True)
    return buffer.getvalue()


def _encode_thumbnail(image: Image.Image, side: int) -> bytes:
    buffer = io.BytesIO()
    _fit_within(image, side).save(buffer, format="WEBP", quality=_THUMBNAIL_QUALITY)
    return buffer.getvalue()


def encode_png(image: Image.Image) -> bytes:
    """Lossless PNG bytes, e.g. for result overlays/masks of Models 2 and 3."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def normalize(image: Image.Image, settings: Settings) -> NormalizedImage:
    """Turn a decoded image into the canonical upright, sRGB, size-capped RGB image.

    Steps: first frame -> EXIF orientation -> RGB (alpha on white, 16-bit and
    CMYK handled properly) -> downscale if larger than STORE_MAX_SIDE_PX ->
    ICC profile to sRGB -> encode the stored JPEG and the WEBP thumbnail.
    """
    # Read before any copy: the profile is plain `info` metadata.
    icc_value = image.info.get("icc_profile")
    icc_profile = icc_value if isinstance(icc_value, bytes) and icc_value else None

    try:
        upright = _first_frame_upright(image)
        rgb, icc_pending = _to_rgb(upright, icc_profile)
        # Downscale before the (per-pixel, comparatively slow) ICC transform.
        rgb = _fit_within(rgb, settings.store_max_side_px)
        if icc_pending and icc_profile is not None:
            converted = _icc_to_srgb(rgb, icc_profile, "RGB")
            if converted is not None:
                rgb = converted
    except (ValueError, OSError) as exc:
        logger.info("image normalisation failed: %s", type(exc).__name__)
        raise InvalidImageError("The uploaded image could not be processed.") from exc

    # Drop leftover EXIF/ICC/XMP so nothing downstream can re-save stale metadata.
    rgb.info = {}

    jpeg_bytes = _encode_jpeg(rgb)
    return NormalizedImage(
        image=rgb,
        width=rgb.width,
        height=rgb.height,
        jpeg_bytes=jpeg_bytes,
        thumb_bytes=_encode_thumbnail(rgb, settings.thumbnail_side_px),
        sha256=hashlib.sha256(jpeg_bytes).hexdigest(),
    )
