# 0005 — Persist failed inferences; reject (do not persist) invalid uploads (P-05)

**Status:** accepted (spec P-05) · **Date:** 2026-10-05

## Context
We want observability of model failures without storing junk.

## Decision
Invalid uploads (corrupt, too small, wrong format, too large) are rejected with a 4xx and nothing is stored.
If the model itself fails (`InferenceError`), the service saves the already-validated image, writes a row with
`status='failed'` plus `error_code`, then returns `500 INFERENCE_FAILED`. The internal `error_message` is never
sent to clients. If the model is *unavailable*, the request fails with 503 before any work, so no row and no
files exist.

## Consequences
+ Failures are debuggable from the DB and disk.
− Failed rows exist and must be kept out of user-facing lists.
