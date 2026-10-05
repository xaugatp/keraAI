# 0017 — One shared predict rate limit per client IP; strict UUID v4 `X-Client-Id`

**Status:** accepted · **Date:** 2026-10-05

## Context
What needs protecting is the laptop's CPU/GPU, not any individual route (spec §12). `X-Client-Id` is an anonymous device id
(D-04), not authentication, and is the key of a person's history.

## Decision
- `app/api/predict.py` defines ONE `limiter.shared_limit(RATE_LIMIT_PREDICT, scope="predict")` and both predict routes
  (`/tree/predict`, `/leaf-segmentation/predict`) use it: a client IP has a single budget (default 10/minute) across all
  models, not 10 per route. The key is `CF-Connecting-IP` when `TRUST_CLOUDFLARE_HEADERS=true`, else the socket peer address.
  Counters are in memory (single worker, ADR 0003) and reset on restart.
- `RATE_LIMIT_DEFAULT` exists in Settings and `.env.example` but is currently unused: no route applies a default limit.
  It is reserved for the day read endpoints need one.
- `X-Client-Id` must parse as a UUID of version 4 (`get_client_id`): a missing header is `400 MISSING_CLIENT_ID`, anything
  else (not a UUID, nil UUID, other versions) is `400 INVALID_CLIENT_ID`. Version 4 rules out guessable ids such as the nil
  UUID or sequential ones, which would be trivial history keys.

## Consequences
+ A flood on one model cannot be sidestepped by calling the other.
− Behind the Cloudflare tunnel with `TRUST_CLOUDFLARE_HEADERS=false` every user appears as 127.0.0.1 and they all share one
  budget: set it to `true` when tunnelled (safe because the API only listens on 127.0.0.1).
− Users behind one NAT share a budget; a client can mint new ids freely, so the id is no basis for limiting.
