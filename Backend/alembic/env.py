"""Alembic environment.

The database URL is never hard-coded and never lives in alembic.ini: it comes
from the app's Settings (DB_* variables / Backend/.env), exactly like the API.
Two escape hatches exist so the same env.py works on SQLite in tests and for
one-off CLI runs:

1. ``config.attributes["database_url"]`` - set by Python callers (tests).
2. ``alembic -x url=<sqlalchemy-url> ...`` - set on the command line.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool
from sqlalchemy.engine import URL, make_url

import app.db.models  # noqa: F401  (side effect: registers every model on Base.metadata)
from app.core.config import get_settings
from app.db.base import Base
from app.db.session import build_database_url

config = context.config

# `configure_logger` lets in-process callers (tests, the app) opt out: fileConfig
# would otherwise replace the app's own logging setup mid-run.
if config.config_file_name is not None and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def _database_url() -> URL:
    override = config.attributes.get("database_url") or context.get_x_argument(
        as_dictionary=True
    ).get("url")
    if override:
        return make_url(override)
    return build_database_url(get_settings())


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of connecting (`alembic upgrade head --sql`).

    Useful to review the exact T-SQL before it touches SQL Server. Only the dialect
    is taken from the URL; no connection is opened.
    """
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    # NullPool: a migration run is one short-lived process; there is nothing to pool.
    connectable = create_engine(_database_url(), poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # Detect column type changes in autogenerate (off by default).
            compare_type=True,
            # SQLite cannot ALTER most things in place; batch mode rebuilds the table.
            # Harmless on SQL Server, so only switched on where it is needed.
            render_as_batch=connection.dialect.name == "sqlite",
        )
        with context.begin_transaction():
            context.run_migrations()
    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
