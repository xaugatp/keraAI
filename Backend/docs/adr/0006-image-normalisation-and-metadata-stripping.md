# 0006 — Image normalisation: sRGB, white alpha, no metadata (P-06 + Phase 3 extensions)

**Status:** accepted (P-06 from the spec; colour/alpha rules added in Phase 3) · **Date:** 2026-10-05

## Context
Phone photos carry GPS/device EXIF, are often rotated via EXIF, and are frequently Display-P3. Transparent PNGs,
CMYK JPEGs and 16-bit grayscale exist in the wild. All three models should see consistent pixels.

## Decision
`app/imaging/processing.py` (shared by all models):
- trust the decoded format (MPO is treated as JPEG); escalate Pillow's decompression-bomb *warning* to an error;
  reject truncated files; judge minimum size after EXIF rotation; first frame only;
- convert embedded ICC profiles to sRGB (relative colorimetric + black-point compensation, safe fallback);
- composite alpha on **white** (black halos would bias the models);
- scale 16-bit/int/float grayscale instead of clipping; downscale-only to `STORE_MAX_SIDE_PX`;
- store JPEG q90 and a WEBP thumbnail with **no EXIF/ICC** (location lives only in explicit columns).
Models receive the lossless normalised RGB image, not the re-decoded JPEG.

## Consequences
+ Consistent, privacy-safe pixels.
− Colour conversion costs a few ms on large images; HEIC is still unsupported (O-04 default).
