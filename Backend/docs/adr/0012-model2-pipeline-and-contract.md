# 0012 — Model 2 (leaf segmentation): pipeline choices and API contract

**Status:** accepted (contract and the `segmentation-models-pytorch` dependency approved by the owner, 2026-10-05) ·
**Date:** 2026-10-05

## Context
Reference: notebook `02_banana_leaf_segmentation.ipynb` — `smp.Unet(resnet34)`, 2 sigmoid channels `[leaf, affected]`,
512×512, ImageNet normalisation, thresholds 0.5 / 0.85. Trained on 50 images (only 2 healthy; 8 in the test split).
The 0.85 threshold was tuned on the *test* split (optimistic). The notebook flags a leaf unhealthy if **any** pixel overlaps,
which is far too trigger-happy for a product.

## Decision
**Pre-processing:** direct 512×512 stretch (the net never saw letterbox padding), `INTER_AREA` when shrinking, ImageNet
normalisation, optional horizontal-flip TTA averaged in probability space (`LEAF_SEG_TTA_HFLIP`, 2× latency).
Very elongated photos (>2:1) are a known risk.

**Post-processing** (pure numpy/cv2, unit-tested on synthetic maps): upsample *probabilities* to the stored image size;
clean the leaf mask (close/open, fill holes, drop blobs <1 % of the image or <20 % of the largest); damage counts only
inside a small tolerance band around the leaf; drop lesions below `MIN_LESION_PCT`; metrics come from the final masks.
Labels: `no_leaf` if the leaf is <3 % of the image, `affected` if damage ≥0.5 % of the leaf, else `healthy`.
All thresholds are env-configurable and **unvalidated** until the model is retrained with more healthy leaves.

**Contract (D-06, no invented numbers):** `POST /api/v1/leaf-segmentation/predict`;
`prediction.label ∈ {affected, healthy, no_leaf}`, `confidence = null`; `details` = `LeafSegDetails` (leaf % of image,
damage % of leaf, lesion count, largest lesion %, mean probabilities, thresholds echoed); `image.result_url` = overlay PNG
(≤1280 px, green leaf contour, red damage); `GET /models` gains `is_placeholder`.

**Dependency:** `segmentation-models-pytorch` (plus timm, huggingface_hub, safetensors) so the notebook's bare state_dict
loads unchanged.

## Consequences
+ Real, explainable numbers; real weights drop in with no conversion.
− "Healthy" is not validated. Severity bands (low/moderate/severe) were deliberately **not** added: they need agronomic
  thresholds from the owner.

## Update — real training run finished (2026-10-07)
`experiment/02_leaf_segmentation.ipynb` retrained the model on 161 images (not the 50-image reference
notebook above) and fixed the test-split-tuning flaw this ADR flagged: the affected threshold is now
tuned on validation only (0.90, up from the reference notebook's optimistic 0.85) and touched against
test exactly once, afterward. Test-set results: leaf IoU 0.972 / Dice 0.986; affected IoU 0.514 / Dice
0.679 at threshold 0.90. Checkpoint promoted to `Backend/weights/leaf_seg_v1.pt`,
`LEAF_SEG_MODEL_VERSION=leaf_seg_real_v1`, `LEAF_SEG_IS_PLACEHOLDER=false` (ADR 0010's promotion
mechanism). The "unvalidated" caveat above no longer applies to the threshold itself, but the
underlying caveat stands in spirit: 161 training images is still not a lot of data for the affected
channel, and a "healthy" verdict still means "no damage above threshold found," not a clinical
guarantee.
