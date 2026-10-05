# 0004 — RFC 9457 problem+json with stable error codes (P-04)

**Status:** accepted (spec P-04) · **Date:** 2026-10-05

## Context
The frontend needs machine-switchable errors; clients must never see stack traces, SQL or file paths.

## Decision
Every error response (including 404/405/422/429 and unhandled exceptions) is `application/problem+json`
`{type, title, status, detail, instance, code, request_id}`, built from the `AppError` hierarchy in
`app/core/errors.py`. 4xx log at INFO/WARNING, 5xx at ERROR with traceback and `request_id`.
Internal messages are logged, never returned.

## Consequences
+ The frontend switches on `code`; support can find a request by `request_id`.
− Every new failure mode needs a class and a code (deliberate: it forces a decision about what the client may know).
