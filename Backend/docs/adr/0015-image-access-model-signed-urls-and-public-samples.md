# 0015 — Image access model: signed links for private images, public samples

**Status:** accepted (refines ADR 0002) · **Date:** 2026-10-05

## Context
`<img src>` cannot send `X-Client-Id`, so `GET /analyses/{id}/image` cannot identify the caller (ADR 0002). We still must not
let a prober learn which analysis ids exist, and samples (D-07) must be viewable by everyone without a signature.

## Decision
`AnalysisService.open_image` works as follows; the route takes no client id.
- A completed sample is public: no `exp`/`sig` needed, `Cache-Control: public, max-age=86400`, and its URLs in API responses
  are unsigned. Everything else needs a valid signature: private rows, failed rows (even failed samples), soft-deleted rows
  and ids that do not exist. Cache-Control for those is `private, max-age=3600`.
- The signature is checked BEFORE existence. An unknown id and a private id therefore both answer `403 INVALID_SIGNATURE`;
  only a link with a valid signature can reach `404` (row deleted, or file missing on disk).
- `exp` is taken as text: a non-numeric value is the same 403 as a missing or wrong signature, never a 422 that would tell
  a prober which part of the link was wrong.
- The signature covers `{id}:{variant}:{exp}` only. A valid link works for whoever holds it until it expires
  (`SIGNED_URL_TTL_SECONDS`, default 1 h); it is not bound to a client.

## Consequences
+ No existence oracle; plain `<img>` tags work; samples are cache-friendly.
− A leaked link is usable by anyone until expiry (`Referrer-Policy: no-referrer` limits accidental leaks).
− After a soft delete (ADR 0007) an old link answers 404 while still unexpired, 403 afterwards.
