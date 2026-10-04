from __future__ import annotations

import io
from pathlib import Path

from alembic.config import Config
from sqlalchemy.engine import URL

BACKEND_DIR = Path(__file__).resolve().parents[3]


def make_alembic_config(
    url: URL | str | None = None, *, output: io.StringIO | None = None
) -> Config:
    """Alembic Config for in-process runs (no subprocess, no cwd dependence).

    `url` is handed to env.py through `attributes` (see alembic/env.py); leave it
    None to exercise the real path where env.py builds the URL from Settings.
    `configure_logger=False` stops env.py's fileConfig from replacing the logging
    setup of the test process.
    """
    config = Config(str(BACKEND_DIR / "alembic.ini"), output_buffer=output)
    config.attributes["configure_logger"] = False
    if url is not None:
        config.attributes["database_url"] = url
    return config
