# 0007 — Soft delete for user deletions (P-07)

**Status:** accepted (spec P-07) · **Date:** 2026-10-05

## Context
Users can delete their own analyses; we want recoverability and an audit trail.

## Decision
`DELETE /analyses/{id}` sets `deleted_at`; the repository's `get`/`list` exclude soft-deleted rows.
Files stay on disk (retention policy is open decision O-05, default: keep forever).
Samples cannot be deleted (`403 SAMPLE_IMMUTABLE`). `seed_samples --force` soft-deletes the old sample row
before re-seeding.

## Consequences
+ Recoverable.
− Storage is not reclaimed until a retention/cleanup job exists (v2).
