# 0014 — Refuse oversized uploads before the body is parsed (`UploadGuardRoute`)

**Status:** accepted · **Date:** 2026-10-05

## Context
Spec §6 asks for an early `Content-Length` check. A check inside the endpoint is too late: FastAPI parses the whole
multipart body (spooling the file to a temp file) before any dependency or endpoint code runs, so a 5 GB upload would
already be on disk when `read_upload_limited` finally said 413.

## Decision
The predict routers use `route_class=UploadGuardRoute` (`app/api/predict.py`). It wraps the route handler and raises
`PayloadTooLargeError` (413 `IMAGE_TOO_LARGE`) when the declared `Content-Length` exceeds `MAX_UPLOAD_MB` + 64 KiB.
The slack covers multipart boundaries, part headers and the other form fields (Content-Length counts the whole body, not
just the file). The exact limit on the file itself is still enforced while streaming it (`read_upload_limited`, 1 MB chunks).
Verified on a real uvicorn with a 30 MB body: 413 comes back without the body being spooled.

## Consequences
+ A declared-oversized upload is refused immediately, with the normal problem+json body, request id and CORS headers.
− A chunked upload without `Content-Length` bypasses the guard: Starlette still spools it to disk before the streaming
  check runs. Browsers and phones always send `Content-Length` for form uploads, and in production Cloudflare enforces its
  own request-body limit in front of the tunnel; a malicious local client is out of scope (the API binds to 127.0.0.1).
− A body between the limit and limit + 64 KiB passes the guard and is rejected by the exact streaming check instead.
