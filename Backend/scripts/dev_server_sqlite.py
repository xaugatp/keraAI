"""DEV-ONLY launcher: run the whole API on a throwaway SQLite database.

    python -m scripts.dev_server_sqlite [--port 8000] [--reset] [--seed]

Lets the frontend be developed end to end WITHOUT SQL Server. Everything lives under
``Backend/.dev-data/`` (git-ignored): ``dev.sqlite3`` and ``data/`` (the images). It never
touches the real DATA_ROOT (e.g. C:/kera-data) or SQL Server.

What it does, in order:
1. points DATA_ROOT (and the CORS origins) at the dev folder through environment variables,
   BEFORE anything reads Settings (real env vars outrank ``.env``),
2. creates the schema by running the real Alembic migrations (never ``create_all``),
3. loads the real model registry (the dummy weights in Backend/weights, unless real ones
   are installed),
4. with ``--seed``, seeds every ``samples/<model>/`` set through ``scripts.seed_samples``,
5. serves ``create_app(engine=...)`` with uvicorn on 127.0.0.1 only, in this single process.

``build_dev_app`` does steps 1-4 without starting uvicorn, so tests never open a port.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from collections.abc import MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, TextIO

from alembic import command
from alembic.config import Config
from pydantic import ValidationError
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings, get_settings
from app.db.session import create_db_engine, create_session_factory
from app.ml.registry import ModelRegistry, build_registry
from app.storage.local import LocalImageStorage
from scripts.seed_samples import (
    MANIFEST_NAME,
    SAMPLES_ROOT,
    SeedError,
    SeedReport,
    format_report,
    seed,
)

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

BACKEND_DIR: Final = Path(__file__).resolve().parent.parent
DEV_DIR: Final = BACKEND_DIR / ".dev-data"
HOST: Final = "127.0.0.1"  # never configurable: the dev server must not be reachable from the LAN
DEFAULT_PORT: Final = 8000
# Vite's dev server listens on port 3000 and can be opened as localhost OR 127.0.0.1;
# browsers treat those as different origins, so both are allowed.
DEV_CORS_ORIGINS: Final = ("http://localhost:3000", "http://127.0.0.1:3000")

# Written into a dev folder when we create it. --reset only deletes a folder that has it, so
# a mistyped path can never wipe real data.
_MARKER: Final = ".kera-dev-data"


class DevServerError(Exception):
    """A problem the developer must fix (message is shown as is)."""


def configure_environment(dev_dir: Path, environ: MutableMapping[str, str] | None = None) -> None:
    """Point the app at the dev folder. Must run before Settings is first read.

    DATA_ROOT is always forced: this is the safety guarantee that the real image store is
    never used. CORS_ORIGINS is set unless the shell already defines it explicitly.
    """
    env = os.environ if environ is None else environ
    env["DATA_ROOT"] = str(dev_dir / "data")
    if "CORS_ORIGINS" not in env:
        env["CORS_ORIGINS"] = json.dumps(list(DEV_CORS_ORIGINS))
    # Settings is cached; anything built earlier in this process saw the old values.
    get_settings.cache_clear()


def reset_dev_dir(dev_dir: Path) -> None:
    """Delete a previous dev folder (``--reset``), refusing anything we did not create."""
    if not dev_dir.exists():
        return
    if not (dev_dir / _MARKER).is_file():
        raise DevServerError(
            f"Refusing to delete {dev_dir}: it was not created by dev_server_sqlite "
            f"(no {_MARKER} marker). Remove it by hand if you are sure."
        )
    shutil.rmtree(dev_dir)


def sqlite_url(dev_dir: Path) -> str:
    return f"sqlite:///{(dev_dir / 'dev.sqlite3').as_posix()}"


def migrate(url: str) -> None:
    """Create/upgrade the schema with the real Alembic migrations (CLAUDE.md: never create_all)."""
    config = Config(str(BACKEND_DIR / "alembic.ini"))
    # In-process run: keep the app's logging setup and hand env.py the SQLite URL.
    config.attributes["configure_logger"] = False
    config.attributes["database_url"] = url
    command.upgrade(config, "head")


@dataclass
class DevApp:
    """Everything ``build_dev_app`` wired together."""

    app: FastAPI
    engine: Engine
    settings: Settings
    registry: ModelRegistry
    session_factory: sessionmaker[Session]
    storage: LocalImageStorage
    dev_dir: Path


def build_dev_app(
    dev_dir: Path = DEV_DIR, *, reset: bool = False, registry: ModelRegistry | None = None
) -> DevApp:
    """Prepare the dev folder, the migrated SQLite DB and the app; do NOT start serving.

    ``registry`` lets tests inject fakes; by default the real models are loaded once, up
    front, and handed to the app (so ``--seed`` and the server share one copy in memory).
    """
    if reset:
        reset_dev_dir(dev_dir)
    (dev_dir / "data").mkdir(parents=True, exist_ok=True)
    (dev_dir / _MARKER).write_text(
        "Created by scripts/dev_server_sqlite.py - throwaway dev data, safe to delete.\n",
        encoding="utf-8",
    )

    configure_environment(dev_dir)
    settings = get_settings()

    url = sqlite_url(dev_dir)
    migrate(url)
    engine = create_db_engine(settings, url=url)

    loaded = registry if registry is not None else build_registry(settings)

    # Imported only now, after the environment is set: app.main builds an `app` object (and
    # reads Settings) at import time, and that must see the dev DATA_ROOT, not .env's.
    from app.main import create_app

    return DevApp(
        app=create_app(engine=engine, registry=loaded),
        engine=engine,
        settings=settings,
        registry=loaded,
        session_factory=create_session_factory(engine),
        storage=LocalImageStorage(settings.data_root),
        dev_dir=dev_dir,
    )


def sample_model_keys(samples_root: Path = SAMPLES_ROOT) -> list[str]:
    """Models that have a samples/<model>/manifest.json."""
    if not samples_root.is_dir():
        return []
    return sorted(p.name for p in samples_root.iterdir() if (p / MANIFEST_NAME).is_file())


@dataclass(frozen=True)
class SeedOutcome:
    model_key: str
    report: SeedReport | None
    error: str | None = None


def seed_all(dev: DevApp, samples_root: Path = SAMPLES_ROOT) -> list[SeedOutcome]:
    """Seed every sample set. A set that cannot start is reported, not fatal."""
    outcomes: list[SeedOutcome] = []
    for model_key in sample_model_keys(samples_root):
        try:
            report = seed(
                model_key,
                session_factory=dev.session_factory,
                registry=dev.registry,
                storage=dev.storage,
                settings=dev.settings,
                samples_dir=samples_root / model_key,
            )
        except SeedError as exc:
            outcomes.append(SeedOutcome(model_key, None, str(exc)))
        else:
            outcomes.append(SeedOutcome(model_key, report))
    return outcomes


def banner(dev: DevApp, port: int) -> str:
    bar = "=" * 78
    placeholders = [
        i.key for i in dev.registry.all_info() if i.status == "ready" and i.is_placeholder
    ]
    lines = [
        bar,
        "  KeraAI DEV SERVER  -  DEV ONLY, NOT FOR PRODUCTION",
        bar,
        f"  Database : THROWAWAY SQLite file  {dev.dev_dir / 'dev.sqlite3'}",
        f"  Images   : {dev.settings.data_root}   (NOT your real DATA_ROOT)",
        f"  Listening: http://{HOST}:{port}   (this machine only)",
        f"  CORS     : {', '.join(dev.settings.cors_origins)}",
    ]
    if dev.settings.enable_docs:
        lines.append(f"  API docs : http://{HOST}:{port}/docs")
    if placeholders:
        lines += [
            "",
            "  !! DUMMY WEIGHTS in use for: " + ", ".join(placeholders),
            "  !! Results are random numbers - NOT real predictions.",
        ]
    lines += [
        "",
        "  Reset everything:  python -m scripts.dev_server_sqlite --reset --seed",
        "  Stop with Ctrl+C.",
        bar,
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DEV ONLY: serve the API on a throwaway SQLite database (no SQL Server needed)."
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"port on 127.0.0.1 (default {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--reset", action="store_true", help="delete .dev-data/ first (fresh start)"
    )
    parser.add_argument("--seed", action="store_true", help="seed the sample images before serving")
    return parser


def run(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    if not 1 <= args.port <= 65535:
        print(f"ERROR: --port must be between 1 and 65535 (got {args.port}).", file=err)
        return 2

    try:
        dev = build_dev_app(DEV_DIR, reset=args.reset)
    except DevServerError as exc:
        print(f"ERROR: {exc}", file=err)
        return 2
    except ValidationError as exc:
        # Field names + messages only: pydantic's str() would echo the offending values,
        # which can be secrets.
        detail = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or 'settings'}: {e['msg']}" for e in exc.errors()
        )
        print(
            f"ERROR: Settings are invalid ({detail}). Copy .env.example to Backend/.env and "
            "set SIGNING_SECRET (32+ characters).",
            file=err,
        )
        return 2

    try:
        print(banner(dev, args.port), file=out)
        if args.seed:
            for outcome in seed_all(dev):
                if outcome.report is not None:
                    print(format_report(outcome.report), file=out)
                else:
                    # Loud, but the server still starts: the frontend can work without samples.
                    print(
                        f"WARNING: samples for {outcome.model_key} NOT seeded: {outcome.error}",
                        file=err,
                    )
        # Imported here so tests that only build the app never need to import uvicorn.
        import uvicorn

        # One process, loopback only. `reload` is off on purpose: the reloader would start
        # a second process that re-imports everything (and loads the models again).
        uvicorn.run(dev.app, host=HOST, port=args.port, log_level=dev.settings.log_level.lower())
    finally:
        dev.engine.dispose()
    return 0


def main(argv: list[str] | None = None) -> int:
    # Windows consoles may default to a legacy code page; never crash on output.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    return run(build_parser().parse_args(argv), sys.stdout, sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
