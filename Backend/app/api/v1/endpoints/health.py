from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# A readiness check reports (is_ok, detail). Phase 2 registers a DB ping here
# and Phase 4 registers model-registry status, both via
# request.app.state.readiness_checks — this endpoint never changes for them.
ReadinessCheck = Callable[[], Awaitable[tuple[bool, str | None]]]


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness only — no DB/model checks, must always be fast."""
    return {"status": "ok"}


@router.get("/health/ready")
async def health_ready(request: Request) -> JSONResponse:
    checks: dict[str, ReadinessCheck] = getattr(request.app.state, "readiness_checks", {})
    components: dict[str, dict[str, str | None]] = {}
    all_ok = True

    for name, check in checks.items():
        try:
            ok, detail = await check()
        except Exception as exc:  # a broken check must not crash readiness itself
            logger.exception("readiness_check_failed", extra={"component": name})
            ok, detail = False, str(exc)
        components[name] = {"status": "ok" if ok else "unavailable", "detail": detail}
        all_ok = all_ok and ok

    return JSONResponse(
        status_code=200 if all_ok else 503,
        content={"status": "ok" if all_ok else "degraded", "components": components},
    )
