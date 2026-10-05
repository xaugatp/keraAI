# 0010 — Dummy weights are allowed, but never pass as real (D-06)

**Status:** accepted (owner: "use dummy weights for now, replace later") · **Date:** 2026-10-05

## Context
No trained weights exist on this machine for Models 1 or 2 (the training notebook's checkpoint lives elsewhere).
The pipeline, API, storage and UI still need to be built and exercised end to end.

## Decision
`python -m scripts.make_dummy_weights` writes seeded, randomly initialised stand-ins in the **same file formats** as the
real ones (`weights/tree_cls_v1.pt` = Ultralytics classify checkpoint; `weights/leaf_seg_v1.pt` = bare `smp.Unet`
state_dict), so real files are a drop-in. They are git-ignored.
While they are in use, `.env` sets `*_MODEL_VERSION=*_dummy_v0` and `*_IS_PLACEHOLDER=true`: every DB row records the
dummy version, `GET /models` exposes `is_placeholder`, `check_env` warns, and the UI shows a "Demo weights" badge.

## Consequences
+ Everything is testable now. The swap is: copy the files, edit the two env vars, restart.
− Rows and samples produced with dummy weights are meaningless. Re-seed samples (`seed_samples --force`) after the swap and
  filter or delete rows with `model_version LIKE '%dummy%'`.
