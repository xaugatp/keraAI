# 0019 — Model 3 (leaf disease identification): pipeline choices and API contract

**Status:** accepted (owner asked for Model 3 to be built out following the Model 2 pattern,
2026-10-07) · **Date:** 2026-10-07

## Context
CLAUDE.md lists Model 3 as "Healthy vs diseased leaf (+ disease type)", architecture
"U-Net-type", **Planned**, with no fixed contract — unlike Models 1 and 2, which both have
accepted ADRs. `experiment/03_leaf_disease_segmentation.ipynb` (Section 0) makes and documents an
explicit design choice: reuse Model 2's architecture pattern instead of a plain whole-image
classifier, reasoning that (a) the same `Leaf Segmentation.v1i.yolov8` export already carries
pixel-accurate disease-region polygons, so a segmentation head costs nothing extra in labels while
also localizing *where* the disease is, and (b) every image in that dataset carries exactly one
disease category, so an image-level diagnosis falls out of pooling the predicted masks with no
separate classification head needed. That notebook's own training run had not finished at the
time this ADR was written (see "Known limitations" below) — this ADR approves the **architecture
and API contract**, not yet a specific trained checkpoint.

## Decision
**Architecture:** `smp.Unet(resnet34, imagenet)`, same family as Model 2, but 3 sigmoid output
channels `[leaf, black_sigatoka, yellow_sigatoka]` instead of 2. Multi-label (not softmax): a
lesion pixel is legitimately both "part of the leaf" and "black_sigatoka". `cordana` (1 image in
the full dataset) and the 1 image with no disease/healthy polygon are excluded from training —
too sparse to learn or validate, not folded into another channel (unlike Model 2's treatment of
`cordana`, since Model 3's whole job is telling disease types apart).

**Pre-processing:** identical to Model 2 — direct 512x512 stretch, ImageNet normalisation,
optional horizontal-flip TTA.

**Post-processing:** identical pipeline shape to Model 2 (upsample probabilities, clean masks,
drop small blobs), run independently per disease channel, each with its own
validation-tuned threshold (rarer classes often need a lower bar than a shared one).

**Image-level diagnosis:** the leaf mask gates what counts as "on the leaf"; whichever disease
channel covers the largest leaf-relative area above its own threshold wins; no channel above
threshold means `healthy`.

**Contract (D-06, no invented numbers):** `POST /api/v1/leaf-disease/predict`;
`prediction.label ∈ {healthy, black_sigatoka, yellow_sigatoka, no_leaf}`, `confidence = null`;
`details` = `LeafDiseaseDetails` (leaf % of image, per-disease % of leaf, lesion counts, mean
probabilities, thresholds echoed); `image.result_url` = overlay PNG (green leaf outline, each
disease shaded a different colour); `GET /models` gains the real entry (replacing the "planned"
placeholder).

**Dummy weights until the real checkpoint lands:** `scripts/make_dummy_weights.py` gains a
`leaf_disease` target (same bare-state_dict pattern as Model 2's), and `LEAF_DISEASE_IS_PLACEHOLDER`
starts `true`, so the service, database and frontend are fully wired and testable before training
finishes — the same promotion path already used for Models 1 and 2 (ADR 0010).

## Consequences
+ Model 3 is architecturally consistent with Model 2 (same backbone family, same loss family, same
  post-processing shape, same promotion mechanics) instead of a fourth unrelated pattern.
+ Nothing in the service/storage/repository layer changes — adding a model is exactly "one new
  settings group, one new predictor class, one new schema, one new thin route" as designed.
− `yellow_sigatoka` has only 7 images in the whole dataset; any accuracy number from it should be
  read as directional, not reliable, until more data exists (flagged again in the training
  notebook's own "Known limitations").
− This is still an experimentation-stage design per the notebook's own Section 0; if the owner's
  real training run surfaces a reason to change the channel set or diagnosis rule, this ADR will
  need a follow-up before the dummy weights are replaced with real ones.

## Update — real training run finished (2026-10-07)
`experiment/03_leaf_disease_segmentation.ipynb` finished training (AdamW, early stop at epoch 50
of 80, mean validation Dice 0.4018). Test-set per-channel metrics: leaf IoU 0.914 / Dice 0.955
(precision 0.996, recall 0.917); black_sigatoka IoU 0.328 / Dice 0.494 at its validation-tuned
threshold of 0.85; yellow_sigatoka IoU 0.243 / Dice 0.391 at its validation-tuned threshold of
0.55. Despite the weak per-pixel disease scores, the image-level diagnosis rule (largest
above-threshold channel wins) reaches 96% accuracy on the 24-image test split (23/24 correct,
one yellow_sigatoka case misread) — the whole-leaf pooling in the decision rule is more forgiving
than pixel overlap, which is exactly the effect the architecture was chosen for. The checkpoint
is promoted to `Backend/weights/leaf_disease_v1.pt`, `LEAF_DISEASE_MODEL_VERSION=leaf_disease_real_v1`,
`LEAF_DISEASE_IS_PLACEHOLDER=false`, same mechanism as Models 1 and 2 (ADR 0010). The
`yellow_sigatoka` caveat above still applies: 7 training images is not enough to trust the lesion
area/coverage numbers that channel reports, only the presence/absence call.
