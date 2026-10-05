"""Verify that this machine can run the KeraAI backend (spec §13).

    python -m scripts.check_env

Prints a table of checks, each marked OK / WARN / FAIL, with an actionable fix
under every failure. Exit code is 1 if any check FAILED (warnings do not fail).
Secrets are never printed (the database URL is shown with its password masked).
"""

from __future__ import annotations

import sys
import tempfile
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from app.core.config import Settings
from app.ml.base import ModelLoadError
from app.ml.registry import MODEL_SPECS, ModelSpec, import_factory
from app.ml.tree_classifier import (
    BACKEND_DIR,
    resolve_weights_path,
    stray_coco_folder_hint,
    validate_weights_file,
)

if TYPE_CHECKING:
    from sqlalchemy.engine import Connection

OK, WARN, FAIL = "ok", "warn", "fail"
_ICONS = {OK: "✅", WARN: "⚠️ ", FAIL: "❌"}

EXPECTED_PYTHON = (3, 11)
VENV_DIR = BACKEND_DIR / ".venv"
ODBC_DOWNLOAD = "https://aka.ms/downloadmsodbcsql"


@dataclass
class Check:
    status: str
    name: str
    detail: str = ""
    fix: str = ""


def _mask(text: str, secrets: tuple[str, ...]) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    return text


# --- individual checks -------------------------------------------------------


def check_python(
    version: tuple[int, int, int] | None = None, prefix: str | None = None
) -> list[Check]:
    """Python 3.11, running inside Backend/.venv. Arguments exist only for tests."""
    major, minor, micro = version or sys.version_info[:3]
    shown = f"{major}.{minor}.{micro}"
    checks = []
    if (major, minor) == EXPECTED_PYTHON:
        checks.append(Check(OK, "Python version", shown))
    else:
        checks.append(
            Check(
                FAIL,
                "Python version",
                f"{shown} (expected {EXPECTED_PYTHON[0]}.{EXPECTED_PYTHON[1]})",
                "Recreate the venv with Python 3.11:  py -3.11 -m venv .venv",
            )
        )

    active = Path(prefix or sys.prefix).resolve()
    if active == VENV_DIR.resolve():
        checks.append(Check(OK, "Virtual environment", str(active)))
    else:
        checks.append(
            Check(
                FAIL,
                "Virtual environment",
                f"running outside Backend/.venv (sys.prefix={active})",
                r"Activate it first:  .venv\Scripts\activate   (then re-run this script)",
            )
        )
    return checks


def check_odbc(settings: Settings) -> Check:
    try:
        import pyodbc  # type: ignore[import-not-found]
    except ImportError as exc:
        return Check(
            FAIL,
            "pyodbc",
            f"cannot import: {exc}",
            "Install dependencies inside the venv:  pip install -r requirements.txt",
        )
    drivers = list(pyodbc.drivers())
    wanted = settings.db_driver
    if wanted in drivers:
        return Check(OK, "ODBC driver", f"'{wanted}' found")
    return Check(
        FAIL,
        "ODBC driver",
        f"'{wanted}' not found. Installed: {drivers or 'none'}",
        f"Install 'ODBC Driver 18 for SQL Server' from {ODBC_DOWNLOAD} "
        "(or set DB_DRIVER to an installed driver).",
    )


def check_database(settings: Settings) -> list[Check]:
    secrets = (settings.db_password,)
    try:
        # Imported lazily so a missing/broken DB layer is a reported failure,
        # not a crash of the whole checker.
        from app.db.session import build_database_url, create_db_engine
    except ImportError as exc:
        return [
            Check(
                FAIL,
                "Database layer",
                f"database layer not present ({exc})",
                "The DB phase (app/db/session.py) has not been built or is broken.",
            )
        ]

    url = build_database_url(settings)
    shown = url.render_as_string(hide_password=True)
    engine = create_db_engine(settings)
    checks: list[Check] = []
    try:
        from sqlalchemy import text

        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            checks.append(Check(OK, "Database connection", f"SELECT 1 succeeded ({shown})"))
            checks.append(_check_alembic(connection))
    except Exception as exc:
        message = _mask(f"{type(exc).__name__}: {exc}", secrets).splitlines()[0][:300]
        checks.append(Check(FAIL, "Database connection", f"{message}  ({shown})", _db_fix(message)))
    finally:
        engine.dispose()
    return checks


def _db_fix(message: str) -> str:
    """Pick the likely remedy from the driver's error text."""
    lowered = message.lower()
    if "login failed" in lowered or "cannot open database" in lowered:
        return (
            "SQL Server is reachable but refused this login / the database is missing. Run "
            "db/sql/001_create_database_and_login.sql in SSMS (creates KeraAI + login), and for "
            "Windows auth (DB_TRUSTED_CONNECTION=true) grant your Windows account access to it."
        )
    if "data source name not found" in lowered or "can't open lib" in lowered:
        return "Check DB_DRIVER in .env against the installed ODBC drivers (see the check above)."
    return (
        "Is the SQL Server service running? Is TCP/IP enabled (SQL Server Configuration "
        "Manager)? Do DB_HOST / DB_PORT / DB_NAME in .env match the instance?"
    )


def _check_alembic(connection: Connection) -> Check:
    try:
        from alembic.config import Config
        from alembic.runtime.migration import MigrationContext
        from alembic.script import ScriptDirectory

        config = Config(str(BACKEND_DIR / "alembic.ini"))
        head = ScriptDirectory.from_config(config).get_current_head()
        current = MigrationContext.configure(connection).get_current_revision()
    except Exception as exc:
        return Check(
            FAIL,
            "Alembic revision",
            f"{type(exc).__name__}: {exc}",
            "Check Backend/alembic.ini and the alembic/ folder.",
        )
    if current == head:
        return Check(OK, "Alembic revision", f"at head ({head})")
    return Check(
        FAIL,
        "Alembic revision",
        f"database is at {current or 'no migration'}, code expects {head}",
        "Apply migrations:  alembic upgrade head",
    )


def check_data_root(settings: Settings) -> Check:
    root = Path(settings.data_root)
    try:
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=root, prefix=".kera-write-check-"):
            pass  # created, then deleted on close
    except OSError as exc:
        return Check(
            FAIL,
            "DATA_ROOT writable",
            f"'{root}': {type(exc).__name__}: {exc}",
            "Set DATA_ROOT in .env to a folder on an existing drive that you can write to "
            "(this machine has no D: drive — use C:/kera-data).",
        )
    return Check(OK, "DATA_ROOT writable", str(root))


def check_cuda(settings: Settings) -> Check:
    try:
        import torch
    except ImportError as exc:
        return Check(
            FAIL, "PyTorch", f"cannot import: {exc}", "Install torch (see README / pytorch.org)."
        )
    configured = settings.inference_device.strip().lower()
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return Check(OK, "CUDA", f"available ({name}); INFERENCE_DEVICE={configured}")
    if configured.startswith("cuda"):
        return Check(
            FAIL,
            "CUDA",
            f"INFERENCE_DEVICE={configured} but CUDA is not available (torch {torch.__version__})",
            "Set INFERENCE_DEVICE=auto (or cpu), or install a CUDA build of torch.",
        )
    return Check(
        WARN,
        "CUDA",
        f"not available - inference will run on CPU (torch {torch.__version__})",
        "Fine for development; install a CUDA build of torch for GPU speed.",
    )


def check_extracted_coco_folder() -> Check:
    """Spec §1: Backend/yolov8n/ is an unzipped COCO detector, not our model."""
    hint = stray_coco_folder_hint()
    if hint is None:
        return Check(OK, "Stray COCO folder", "Backend/yolov8n/ not present")
    return Check(
        WARN,
        "Stray COCO folder",
        hint,
        "Leave it alone (it is git-ignored) and put the REAL classifier at "
        "Backend/weights/tree_cls_v1.pt as the original .pt file.",
    )


def check_model(spec: ModelSpec, settings: Settings) -> list[Check]:
    """Weights present -> loads with the expected task/classes -> one warm-up inference."""
    group = spec.settings_group(settings)
    label = spec.key
    if not group.enabled:
        return [Check(WARN, f"{label}", "disabled by configuration (skipped)")]

    weights = resolve_weights_path(group.weights_path)
    checks: list[Check] = []
    try:
        if spec.key == "tree_classification":
            validate_weights_file(weights)  # exists + .pt + zip + extracted-folder detection
        elif not weights.is_file():
            raise ModelLoadError(f"Weights file not found: '{weights}'.")
        elif not zipfile.is_zipfile(weights):
            raise ModelLoadError(f"Weights file '{weights}' is not a valid torch archive (zip).")
    except ModelLoadError as exc:
        return [
            Check(
                FAIL,
                f"{label}: weights file",
                str(exc),
                "Put the real weights there, or create random stand-ins with "
                "`python -m scripts.make_dummy_weights`.",
            )
        ]
    size_mb = weights.stat().st_size / 1_000_000
    checks.append(Check(OK, f"{label}: weights file", f"{weights} ({size_mb:.1f} MB, zip)"))

    try:
        predictor = import_factory(spec.factory)(settings)
        predictor.load()
    except ModelLoadError as exc:
        checks.append(Check(FAIL, f"{label}: loads", str(exc), _LOAD_FIX))
        return checks
    except Exception as exc:  # ImportError of the model module, constructor error, ...
        checks.append(
            Check(FAIL, f"{label}: loads", f"{type(exc).__name__}: {exc}", _LOAD_FIX_CODE)
        )
        return checks
    checks.append(
        Check(
            OK,
            f"{label}: loads",
            f"task={predictor.task}, version={predictor.version}, classes={predictor.classes}",
        )
    )

    predictor.mark_ready()  # warmup() goes through predict(), which needs the ready flag
    try:
        started = time.perf_counter()
        predictor.warmup()
        elapsed_ms = round((time.perf_counter() - started) * 1000)
    except Exception as exc:
        checks.append(
            Check(
                FAIL,
                f"{label}: warm-up inference",
                f"{type(exc).__name__}: {exc}",
                "The weights loaded but cannot run; re-export/re-copy the checkpoint.",
            )
        )
        return checks
    checks.append(Check(OK, f"{label}: warm-up inference", f"ok in {elapsed_ms} ms"))

    if predictor.is_placeholder:
        checks.append(
            Check(
                WARN,
                f"{label}: placeholder",
                "dummy weights - results are not real (is_placeholder=true)",
                "Replace with the trained checkpoint, then set *_IS_PLACEHOLDER=false and "
                "update *_MODEL_VERSION in .env.",
            )
        )
    return checks


_LOAD_FIX = (
    "Check that the file is the right model (task and class names) and matches the "
    "TREE_*/LEAF_SEG_* settings in .env."
)
_LOAD_FIX_CODE = "The model's code failed to import/construct; see the error above."


# --- runner ------------------------------------------------------------------


def _print_table(checks: list[Check]) -> None:
    width = max(len(check.name) for check in checks)
    for check in checks:
        print(f"{_ICONS[check.status]} {check.name.ljust(width)}  {check.detail}")
        if check.status != OK and check.fix:
            print(f"   {' ' * width}  -> fix: {check.fix}")


def _run(label: str, fn: Callable[[], list[Check] | Check]) -> list[Check]:
    """Run one check; a bug in a check becomes a FAIL row, never a crash."""
    try:
        result = fn()
    except Exception as exc:
        return [Check(FAIL, label, f"check crashed: {type(exc).__name__}: {exc}")]
    return result if isinstance(result, list) else [result]


def main() -> int:
    # Windows consoles may default to a legacy code page that cannot print the status icons.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")

    print("KeraAI environment check\n")
    checks = check_python()

    try:
        from app.core.config import get_settings

        settings = get_settings()
    except Exception as exc:
        # Report field names + messages only: pydantic's str() would echo the
        # offending values, which can be secrets.
        errors = getattr(exc, "errors", None)
        if callable(errors):
            detail = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or 'settings'}: {e['msg']}" for e in errors()
            )
        else:
            detail = type(exc).__name__
        checks.append(
            Check(
                FAIL,
                "Settings (.env)",
                detail,
                "Copy .env.example to .env (in Backend/) and fill in SIGNING_SECRET and DATA_ROOT.",
            )
        )
        _print_table(checks)
        return 1

    checks.append(Check(OK, "Settings (.env)", f"APP_ENV={settings.app_env}"))
    checks += _run("ODBC driver", lambda: check_odbc(settings))
    checks += _run("Database", lambda: check_database(settings))
    checks += _run("DATA_ROOT writable", lambda: check_data_root(settings))
    checks += _run("CUDA", lambda: check_cuda(settings))
    checks += _run("Stray COCO folder", check_extracted_coco_folder)
    for spec in MODEL_SPECS:
        checks += _run(spec.key, partial(check_model, spec, settings))

    _print_table(checks)
    failed = sum(1 for c in checks if c.status == FAIL)
    warned = sum(1 for c in checks if c.status == WARN)
    print(f"\n{len(checks) - failed - warned} ok, {warned} warning(s), {failed} failure(s)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
