# 0013 — Model 1 stays a YOLOv8-cls classifier (the notebook's detector is not used)

**Status:** accepted · **Date:** 2026-10-05

## Context
`CLAUDE.md` and the spec define Model 1 as a YOLOv8-cls banana-tree vs non-banana classifier (`tree_cls_v1.pt`).
The reference repo's `01_banana_tree_detection.ipynb` is a single-class YOLOv8n *detector* (mAP50 0.195, recall 0.385,
no negative class yet).

## Decision
Follow the spec: a classifier with verdict `banana_tree | not_banana_tree | uncertain` (threshold 0.60) and no bounding
box in the contract. The detector is not wired; it could be added later as a separate predictor and route without
touching the service.

## Consequences
+ Matches the owner's recorded architecture and needs no box UI.
− The frontend's bounding-box mock is removed (see the frontend integration spec).
