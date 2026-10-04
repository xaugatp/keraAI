from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from slowapi.middleware import SlowAPIMiddleware
from sqlalchemy import Engine

from app.api.v1.endpoints import health
from app.api.v1.endpoints.health import ReadinessCheck
from app.api.v1.router import api_router
from app.core.config import Settings, get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging
from app.core.middleware import (
    AccessLogMiddleware,
    RequestIDMiddleware,
    SecurityHeadersMiddleware,
)
from app.core.rate_limit import limiter
from app.db.session import create_db_engine, create_session_factory, ping_database

logger = logging.getLogger(__name__)


def _database_readiness_check(engine: Engine) -> ReadinessCheck:
    async def check() -> tuple[bool, str | None]:
        # pyodbc is blocking: run the ping in the threadpool so a slow or hung
        # SQL Server cannot stall the event loop (and with it /health liveness).
        return await run_in_threadpool(ping_database, engine)

    return check


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    configure_logging(settings)

    # Readiness checks are populated by the subsystems that own them — the DB
    # engine (below) and the model registry (Phase 4) each add their own check
    # here during their slice of startup.
    app.state.readiness_checks = {}

    # --- database ---
    # An injected engine (tests) is owned by whoever created it; one we create
    # ourselves is ours to dispose on shutdown.
    injected_engine: Engine | None = app.state.engine_override
    engine = injected_engine if injected_engine is not None else create_db_engine(settings)
    app.state.engine = engine
    app.state.session_factory = create_session_factory(engine)
    app.state.readiness_checks["database"] = _database_readiness_check(engine)

    # Degraded mode, never a crash: if SQL Server is down (laptop just booted,
    # service still starting) the API still comes up, /health/ready reports 503,
    # and the pool reconnects on its own once the database is back. ping_database
    # logs the WARNING with the reason.
    db_ok, _ = await run_in_threadpool(ping_database, engine)
    if db_ok:
        logger.info("Database connection OK")
    else:
        logger.warning("Starting in degraded mode: database is unavailable")

    # Model registry: Phase 4 adds its build/warm-up and readiness check here.

    try:
        yield
    finally:
        if injected_engine is None:
            await run_in_threadpool(engine.dispose)


def create_app(engine: Engine | None = None) -> FastAPI:
    """Application factory.

    ``engine`` lets tests inject a SQLite engine so the suite never needs SQL
    Server; production passes nothing and the lifespan builds one from Settings.
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url="/redoc" if settings.enable_docs else None,
        openapi_url="/openapi.json" if settings.enable_docs else None,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.limiter = limiter
    app.state.engine_override = engine

    register_exception_handlers(app)

    # Added innermost-first: the LAST middleware added runs FIRST on the way
    # in (and last on the way out), so this order yields the spec's
    # RequestID -> access log -> rate limit -> security headers -> CORS ->
    # GZip -> route, and the exact reverse on the response.
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-Client-Id", "X-Request-ID"],
        expose_headers=["X-Request-ID", "Location"],
    )
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(SlowAPIMiddleware)
    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(RequestIDMiddleware)

    app.include_router(health.router)
    app.include_router(api_router, prefix=settings.api_v1_prefix)

    return app


app = create_app()
