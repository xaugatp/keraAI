"""Model 2 input preparation: PIL RGB image -> float32 NCHW network input.

Must reproduce the training pipeline (`A.Resize(512, 512)` then
`A.Normalize(ImageNet mean/std)`), or the weights see a distribution they never
trained on. Torch-free on purpose (numpy/OpenCV only).

Why a direct stretch to SxS and NOT letterboxing
------------------------------------------------
The network was trained on square 640x640 photos squashed to 512x512 with no
padding. It has never seen a grey/black border, and a letterboxed input would
put large constant regions in front of a model with only 50 training images.
Stretching is what it was taught. The cost: very elongated photos (aspect
ratio beyond roughly 2:1) are distorted much more than anything in training, so
segmentation quality on those is a KNOWN RISK; the masks are mapped back to the
original geometry afterwards, so the distortion only affects what the network
"sees", not the reported areas' geometry.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

# Same constants as the training notebook (torchvision/ImageNet statistics).
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _resize(pixels: NDArray[np.uint8], imgsz: int) -> NDArray[np.uint8]:
    """Resize to imgsz x imgsz, choosing the interpolation PER AXIS.

    * Shrinking uses INTER_AREA: phone photos are stored up to 2048 px, a 4x
      downscale, and bilinear (INTER_LINEAR) only looks at 2x2 source pixels per
      output pixel, so it aliases thin lesion edges. INTER_AREA averages every
      source pixel it covers.
    * Enlarging uses INTER_LINEAR (INTER_AREA degenerates to blocky
      nearest-neighbour when zooming in).

    A photo can shrink along one axis and enlarge along the other (e.g.
    1600x300), so shrink first (area), then enlarge (linear): each axis gets the
    right filter and the result is still a direct stretch.
    """
    height, width = pixels.shape[:2]
    if width > imgsz or height > imgsz:
        pixels = np.asarray(
            cv2.resize(
                pixels, (min(width, imgsz), min(height, imgsz)), interpolation=cv2.INTER_AREA
            ),
            dtype=np.uint8,
        )
    height, width = pixels.shape[:2]
    if width != imgsz or height != imgsz:
        pixels = np.asarray(
            cv2.resize(pixels, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR), dtype=np.uint8
        )
    return pixels


def prepare_input(image_rgb: Image.Image, imgsz: int) -> NDArray[np.float32]:
    """Return a C-contiguous float32 array of shape (1, 3, imgsz, imgsz).

    Raises ValueError (a programming/contract error, wrapped into INFERENCE_FAILED
    by `Predictor.predict`) for non-RGB or empty images: the shared imaging layer
    already guarantees RGB, so anything else means a caller bug that should be
    loud rather than silently converted (RGBA -> RGB would composite on black).
    """
    if imgsz <= 0:
        raise ValueError(f"imgsz must be positive, got {imgsz}")
    if image_rgb.mode != "RGB":
        raise ValueError(f"expected an RGB image, got mode {image_rgb.mode!r}")
    width, height = image_rgb.size
    if width <= 0 or height <= 0:
        raise ValueError(f"image has no pixels (size={image_rgb.size})")

    pixels = np.asarray(image_rgb, dtype=np.uint8)
    pixels = _resize(pixels, imgsz)

    # 0-255 -> 0-1 -> (x - mean) / std, per channel, in float32 (HWC broadcasting).
    scaled = pixels.astype(np.float32) / np.float32(255.0)
    normalised = (scaled - IMAGENET_MEAN) / IMAGENET_STD
    # HWC -> CHW, add the batch axis, and make the memory contiguous: the
    # transpose is only a view, and torch.from_numpy needs real strides.
    return np.ascontiguousarray(normalised.transpose(2, 0, 1)[np.newaxis, ...], dtype=np.float32)
