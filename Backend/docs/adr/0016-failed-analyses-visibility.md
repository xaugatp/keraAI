# 0016 — Failed analyses: never listed, visible only to their owner, no internals

**Status:** accepted (refines ADR 0005) · **Date:** 2026-10-05

## Context
ADR 0005 persists a `status='failed'` row (and the image) when inference raises, so failures can be debugged. Such a row has no
prediction and carries an internal `error_message` (e.g. a Python exception text). The history UI must not show it, and
nothing internal may reach a client (spec §11).

## Decision
- `AnalysisService.list` always filters `status="completed"`: failed rows never appear in any list, whatever the `scope`.
- `_is_visible`: a failed row is visible only to its owner (`client_id` match). A failed *sample* has no client id, so nobody
  can see it, and `_is_public` is false for it (its image link needs a signature, ADR 0015).
- For the owner, `GET /analyses/{id}` maps the row with `prediction: null` and `details: null`. `error_code` and
  `error_message` are not part of any response schema; the client only ever saw `500 INFERENCE_FAILED` with a fixed message
  (a fresh `InferenceError`, not the original one).
- The owner may delete a failed row like any other of their own rows.

## Consequences
+ Users never see half-results or exception text; the operator keeps the row, the image and the `analysis_failed` log line
  (it carries `analysis_id` and `error_code`; the full message stays in the `error_message` column).
− The 500 response does not include the analysis id, so in practice failed rows are inspected by the operator (SSMS/logs),
  not by the client. Failed sample rows are invisible debris; `seed_samples` retires them when it re-seeds that image.
