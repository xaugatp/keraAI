# 0018 — Catch-all error middleware, a truthful OpenAPI document, and write compensation

**Status:** accepted · **Date:** 2026-10-05

## Context
1. Starlette runs the `Exception` handler in its outermost layer, outside our middleware. Phase 1 therefore rendered unhandled
   500s with `request_id: null`, no `X-Request-ID` header and no CORS headers (so a browser hides the body from the frontend).
2. FastAPI documents errors as `application/json` and 422 as its own `HTTPValidationError`, and cannot make `X-Client-Id`
   both optional in code (to get our own 400 instead of a generic 422) and required in the schema.
3. An analysis is several files plus one row; the database and the disk cannot share a transaction.

## Decision
1. `CatchAllErrorsMiddleware` (innermost) renders any unexpected exception as the usual problem+json 500 *inside* the stack, so
   it gets the request id, CORS and security headers. The `Exception` handler stays as the last-resort net.
2. `install_problem_json_openapi` post-processes the schema once: `X-Client-Id` becomes `required` with `format: uuid`; every
   error response is `application/problem+json` referencing `ProblemDetail`; FastAPI's own validation schemas are dropped. The
   generated TypeScript client therefore matches what the API really returns.
3. `AnalysisService._store_and_commit` writes files, inserts, commits; on ANY failure it deletes every planned file (including
   the one whose write failed) and rolls back, so there are no orphan files and no rows pointing at missing files.

## Consequences
+ Every error, including bugs, has the same shape and is readable by the browser.
− If the connection dies exactly while the COMMIT is acknowledged, a row can exist whose files were just deleted. It is
  harmless: the image route answers 404 and logs `image_file_missing` for the operator.
− The OpenAPI patching is coupled to FastAPI's schema layout; `tests/integration/api/test_openapi.py` guards it.
