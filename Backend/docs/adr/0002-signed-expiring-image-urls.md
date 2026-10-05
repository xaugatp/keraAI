# 0002 — HMAC-signed, expiring image URLs (P-02)

**Status:** accepted (spec P-02) · **Date:** 2026-10-05

## Context
`<img src>` cannot send the `X-Client-Id` header, so header-based access cannot protect image files.

## Decision
Image URLs are relative and carry `?variant=&exp=&sig=` where
`sig = base64url(HMAC-SHA256(SIGNING_SECRET, "{id}:{variant}:{exp}"))`, verified with a constant-time compare,
an expiry check and a variant allowlist (`app/core/signing.py`).
Samples are public (no signature). A bad or expired signature returns `403 INVALID_SIGNATURE`.

## Consequences
+ Works with plain `<img>`.
− Links stop working after `SIGNED_URL_TTL_SECONDS` (default 1 h): the frontend must not cache them and must
  refetch the detail on image error.
− Rotating `SIGNING_SECRET` invalidates all outstanding links (acceptable).
`X-Client-Id` is still not authentication (README limitation).
