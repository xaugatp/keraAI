"""Model 2 visualisation: the photo with the leaf outlined and damage shaded.

VISUALISATION ONLY. All reported areas/percentages come from the full-resolution
masks in `postprocess`; this image is downscaled for fast delivery to the phone,
so never measure on it. Torch-free (numpy/OpenCV/Pillow).
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

# Colours are RGB triples (we draw on an RGB array; OpenCV does not care about order).
LEAF_COLOUR = (0, 200, 0)
# Thin white line under the green one keeps the outline legible on dark AND bright photos.
HALO_COLOUR = (255, 255, 255)
LESION_FILL_COLOUR = (230, 30, 30)
LESION_CONTOUR_COLOUR = (120, 0, 0)  # darker red than the fill so lesion edges stay visible
LESION_ALPHA = 0.45

_REFERENCE_SIDE = 1280  # line widths are specified for a 1280 px image and scale from there


def _resize_mask(mask: NDArray[np.bool_], size: tuple[int, int]) -> NDArray[np.uint8]:
    """Nearest-neighbour resize of a bool mask to (width, height) as uint8 0/1.

    NEAREST, not a smooth filter: a mask must stay binary, and the contour
    finder needs crisp edges.
    """
    as_u8 = mask.astype(np.uint8)
    if (as_u8.shape[1], as_u8.shape[0]) == size:
        return as_u8
    return np.asarray(cv2.resize(as_u8, size, interpolation=cv2.INTER_NEAREST), dtype=np.uint8)


def _draw_contours(
    canvas: NDArray[np.uint8], mask: NDArray[np.uint8], colour: tuple[int, int, int], thickness: int
) -> None:
    # RETR_LIST: every boundary, outer and inner (a lesion with a hole shows both).
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if contours:
        cv2.drawContours(canvas, contours, -1, colour, thickness, lineType=cv2.LINE_AA)


def render_overlay(
    image_rgb: Image.Image,
    leaf_mask: NDArray[np.bool_],
    affected_mask: NDArray[np.bool_],
    max_side: int = 1280,
) -> Image.Image:
    """RGB overlay at most `max_side` px on the longest side (never upscaled).

    Masks must be at the photo's own resolution, shape (height, width). Neither
    the input image nor the masks are modified.
    """
    if max_side <= 0:
        raise ValueError(f"max_side must be positive, got {max_side}")
    width, height = image_rgb.size
    if width <= 0 or height <= 0:
        raise ValueError(f"image has no pixels (size={image_rgb.size})")
    for name, mask in (("leaf_mask", leaf_mask), ("affected_mask", affected_mask)):
        if mask.shape != (height, width):
            raise ValueError(
                f"{name} has shape {mask.shape} but the image is {height}x{width} (HxW)"
            )

    scale = min(1.0, max_side / max(width, height))
    out_size = (max(1, round(width * scale)), max(1, round(height * scale)))

    # np.array (not asarray) makes a writable copy; the caller's image stays untouched.
    canvas = np.array(image_rgb.convert("RGB"), dtype=np.uint8)
    if out_size != (width, height):
        # INTER_AREA: the only sensible filter for shrinking a sharp photo.
        canvas = np.asarray(
            cv2.resize(canvas, out_size, interpolation=cv2.INTER_AREA), dtype=np.uint8
        )

    leaf = _resize_mask(leaf_mask, out_size)
    affected = _resize_mask(affected_mask, out_size)

    longest = max(out_size)
    leaf_thickness = max(2, round(2 * longest / _REFERENCE_SIDE))
    lesion_thickness = max(1, round(longest / _REFERENCE_SIDE))

    # 1) semi-transparent red fill over the damage (blend only the masked pixels).
    inside = affected.astype(np.bool_)
    if inside.any():
        blended = (
            canvas[inside].astype(np.float32) * (1.0 - LESION_ALPHA)
            + np.array(LESION_FILL_COLOUR, dtype=np.float32) * LESION_ALPHA
        )
        canvas[inside] = np.rint(blended).astype(np.uint8)

    # 2) thin dark-red lesion outline, 3) leaf outline on top (white halo first).
    _draw_contours(canvas, affected, LESION_CONTOUR_COLOUR, lesion_thickness)
    _draw_contours(canvas, leaf, HALO_COLOUR, leaf_thickness + 2)
    _draw_contours(canvas, leaf, LEAF_COLOUR, leaf_thickness)

    return Image.fromarray(canvas, mode="RGB")
