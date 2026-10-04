from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings

logger = logging.getLogger(__name__)


def build_database_url(settings: Settings) -> URL:
    """Build the SQL Server URL from discrete settings.

    URL.create (rather than an f-string) quotes passwords containing ``@ : /``
    and keeps ``localhost\\SQLEXPRESS`` intact (spec §4). No secret is ever
    stored in a string we might log: callers should render with
    ``url.render_as_string(hide_password=True)``.
    """
    query: dict[str, str] = {
        "driver": settings.db_driver,
        # ODBC Driver 18 encrypts by default; a local dev instance only has a
        # self-signed cert, so without this the TLS handshake is rejected.
        "TrustServerCertificate": "yes" if settings.db_trust_server_cert else "no",
    }
    username: str | None = None
    password: str | None = None
    if settings.db_trusted_connection:
        query["Trusted_Connection"] = "yes"  # Windows auth; user/password ignored
    else:
        username = settings.db_user
        password = settings.db_password

    return URL.create(
        "mssql+pyodbc",
        username=username,
        password=password,
        host=settings.db_host,
        port=settings.db_port,
        database=settings.db_name,
        query=query,
    )


def create_db_engine(settings: Settings, url: URL | str | None = None) -> Engine:
    """Create the SQLAlchemy engine.

    ``url`` overrides the URL built from settings — used by tests (SQLite temp
    file) and Alembic. Creating an engine never connects, so this cannot fail
    just because the database is down.
    """
    resolved = make_url(url) if url is not None else build_database_url(settings)
    kwargs: dict[str, Any] = {
        # Probe a pooled connection before handing it out: after a laptop sleep or
        # a SQL Server restart the pool holds dead sockets, and without this the
        # first request afterwards would fail instead of transparently reconnecting.
        "pool_pre_ping": True,
        "echo": settings.db_echo,
    }
    if resolved.get_backend_name() == "sqlite":
        # Requests run in a threadpool, so a connection may be used from a thread
        # other than the one that opened it. SQLite also has no pool_size.
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs["pool_size"] = settings.db_pool_size
    return create_engine(resolved, **kwargs)


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    # expire_on_commit=False: the service commits and *then* maps the row to a
    # response. With the default (True) every attribute access after commit would
    # silently issue a fresh SELECT (and fail outright once the session is closed).
    return sessionmaker(bind=engine, expire_on_commit=False)


def get_db(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    """Yield one Session per request; commit is the *service's* job, not ours.

    The rollback in ``finally`` is a no-op after a successful commit and undoes
    anything half-done after an error, so a request can never leak an open
    transaction (and its locks) back into the connection pool.
    """
    session = session_factory()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


def ping_database(engine: Engine) -> tuple[bool, str | None]:
    """Blocking ``SELECT 1``; returns ``(ok, detail)`` and never raises.

    The real reason is logged (WARNING) for the operator. ``detail`` is
    deliberately generic because /health/ready is client-visible and driver
    errors contain server names and SQL (spec §11: never leak internals).
    """
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception as exc:  # any failure means "not ready", never "crash"
        logger.warning("Database ping failed: %s: %s", type(exc).__name__, exc)
        return False, "Database unreachable"
    return True, None
