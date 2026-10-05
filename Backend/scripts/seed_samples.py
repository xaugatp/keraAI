"""Seed the public sample analyses of one model (spec 13, D-07).

    python -m scripts.seed_samples --model tree_classification [--force]

Reads ``samples/<model_key>/manifest.json`` (see samples/README.md) and runs every image
through the REAL ``AnalysisService`` with ``source="sample"``, ``is_sample=True`` and no
client id, so a sample row is indistinguishable from a user's row except for its flags.

Idempotent: an image whose normalised JPEG hash is already a live sample of this model is
skipped. ``--force`` re-analyses it and retires the old row (soft delete). Use ``--force``
after installing the real weights: samples made with dummy weights are not real results.

Exit codes: 0 = every file seeded or skipped, 1 = at least one file failed,
2 = nothing was attempted (bad manifest, model unavailable, database not ready).

The core is ``seed(...)`` (no I/O besides what it is handed, easy to test); ``main`` only
builds the real engine/registry/storage from Settings.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal, TextIO

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator
from sqlalchemy import Engine, inspect
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.db.models.analysis import Analysis
from app.db.session import create_db_engine, create_session_factory, ping_database
from app.imaging import processing
from app.ml.registry import MODEL_SPECS, ModelRegistry, build_registry
from app.repositories.analysis_repository import AnalysisFilter, AnalysisRepository
from app.schemas.analysis import CaptureMeta
from app.services.analysis_service import AnalysisService
from app.storage.base import ImageStorage
from app.storage.local import LocalImageStorage

logger = logging.getLogger(__name__)

BACKEND_DIR: Final = Path(__file__).resolve().parent.parent
SAMPLES_ROOT: Final = BACKEND_DIR / "samples"
MANIFEST_NAME: Final = "manifest.json"

EXIT_OK: Final = 0
EXIT_FILE_FAILED: Final = 1
EXIT_NOT_ATTEMPTED: Final = 2

# Column sizes of analyses.title / analyses.description (spec 5.2).
_TITLE_MAX: Final = 200
_DESCRIPTION_MAX: Final = 1000

_LIST_PAGE_SIZE: Final = 100


class SeedError(Exception):
    """Nothing was seeded because the run could not start (message is for the owner)."""


class ManifestError(SeedError):
    """samples/<model>/manifest.json is missing or wrong; every problem is listed."""


# --------------------------------------------------------------------------- manifest


class ManifestEntry(BaseModel):
    """One sample: ``{file, title, description?, latitude?, longitude?}``."""

    # A typo such as "lat" must be an error, not a silently ignored field.
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    file: str = Field(min_length=1, description="Image path relative to the manifest's folder")
    title: str = Field(min_length=1, max_length=_TITLE_MAX)
    description: str | None = Field(default=None, max_length=_DESCRIPTION_MAX)
    latitude: float | None = Field(default=None, ge=-90, le=90, allow_inf_nan=False)
    longitude: float | None = Field(default=None, ge=-180, le=180, allow_inf_nan=False)

    @model_validator(mode="after")
    def _coordinates_come_in_pairs(self) -> ManifestEntry:
        # Same rule as the predict form: half a pair would plot a marker on the equator.
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("latitude and longitude must be given together (or both left out)")
        return self


_ENTRIES: Final = TypeAdapter(list[ManifestEntry])


def _describe(location: Sequence[str | int], message: str) -> str:
    """'entry #2 -> latitude: message' (1-based, the way the owner counts)."""
    if location and isinstance(location[0], int):
        where = f"entry #{location[0] + 1}"
        field_path = ".".join(str(part) for part in location[1:])
        return f"{where} -> {field_path}: {message}" if field_path else f"{where}: {message}"
    return message


def _check_file(samples_dir: Path, relative: str) -> tuple[Path | None, str | None]:
    """Resolve ``relative`` inside ``samples_dir``. Returns (path, None) or (None, problem)."""
    candidate = Path(relative)
    # Windows-aware: "C:x", "\\x" and "/x" are not absolute for pathlib on every
    # platform but all of them escape the folder, so reject drive and root outright.
    if candidate.is_absolute() or candidate.drive or candidate.root:
        return None, f"'{relative}' must be a relative path inside the samples folder"
    if ".." in candidate.parts:
        return None, f"'{relative}' must not contain '..' (files must stay inside the folder)"
    root = samples_dir.resolve()
    resolved = (root / candidate).resolve()
    # Second line of defence: a symlink/junction inside the folder can still lead outside.
    if not resolved.is_relative_to(root):
        return None, f"'{relative}' resolves outside the samples folder"
    if not resolved.is_file():
        return None, f"file '{relative}' not found in {samples_dir}"
    return resolved, None


@dataclass(frozen=True)
class SampleFile:
    """A validated manifest entry with its image resolved to an existing file."""

    entry: ManifestEntry
    path: Path


def load_manifest(samples_dir: Path) -> list[SampleFile]:
    """Read and fully validate ``manifest.json``; raise ManifestError listing EVERY problem."""
    manifest = samples_dir / MANIFEST_NAME
    if not manifest.is_file():
        raise ManifestError(
            f"No manifest found at {manifest}. Create it as a JSON list of "
            '{"file", "title", "description"} objects; see samples/README.md.'
        )
    try:
        # utf-8-sig: Windows editors (Notepad) like to prepend a byte-order mark.
        raw = json.loads(manifest.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError) as exc:
        raise ManifestError(f"Cannot read {manifest}: {type(exc).__name__}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(
            f"{manifest} is not valid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})."
        ) from exc
    if not isinstance(raw, list):
        raise ManifestError(
            f"{manifest} must be a JSON list of sample objects, not {type(raw).__name__}."
        )
    if not raw:
        raise ManifestError(f"{manifest} is empty: add at least one sample entry.")

    try:
        entries = _ENTRIES.validate_python(raw)
    except ValidationError as exc:
        found = [
            _describe(error["loc"], error["msg"].removeprefix("Value error, "))
            for error in exc.errors()
        ]
        raise ManifestError(_problem_list(manifest, found)) from exc

    problems: list[str] = []
    samples: list[SampleFile] = []
    seen: set[str] = set()
    for number, entry in enumerate(entries, start=1):
        if entry.file.lower() in seen:
            problems.append(f"entry #{number} -> file: '{entry.file}' is listed more than once")
            continue
        seen.add(entry.file.lower())
        path, problem = _check_file(samples_dir, entry.file)
        if path is None:
            problems.append(f"entry #{number} -> file: {problem}")
        else:
            samples.append(SampleFile(entry, path))
    if problems:
        raise ManifestError(_problem_list(manifest, problems))
    return samples


def _problem_list(manifest: Path, problems: Sequence[str]) -> str:
    return f"{manifest} has {len(problems)} problem(s):\n" + "\n".join(f"  - {p}" for p in problems)


# --------------------------------------------------------------------------- report


SeedStatus = Literal["seeded", "skipped", "failed"]


@dataclass(frozen=True)
class SeedItem:
    file: str
    status: SeedStatus
    analysis_id: uuid.UUID | None = None
    label: str | None = None
    model_version: str | None = None
    detail: str | None = None  # why it was skipped / what went wrong


@dataclass
class SeedReport:
    model_key: str
    model_version: str
    is_placeholder: bool
    items: list[SeedItem] = field(default_factory=list)

    def count(self, status: SeedStatus) -> int:
        return sum(1 for item in self.items if item.status == status)

    @property
    def ok(self) -> bool:
        return self.count("failed") == 0


_BAR = "!" * 78
_PLACEHOLDER_WARNING = "\n".join(
    [
        _BAR,
        "!!  PLACEHOLDER WEIGHTS: {key} ({version})",
        "!!  These samples were analysed by DUMMY weights - the results are NOT real.",
        "!!  After the real weights are installed (and *_IS_PLACEHOLDER / *_MODEL_VERSION",
        "!!  are updated in .env) re-seed them:  python -m scripts.seed_samples --model {key} --force",
        _BAR,
    ]
)


def format_report(report: SeedReport) -> str:
    lines: list[str] = []
    if report.is_placeholder:
        lines.append(
            _PLACEHOLDER_WARNING.format(key=report.model_key, version=report.model_version)
        )
        lines.append("")
    lines.append(f"Samples for {report.model_key} (model version {report.model_version}):")
    width = max((len(item.file) for item in report.items), default=0)
    for item in report.items:
        parts = [f"  {item.status.upper():<7} {item.file.ljust(width)}"]
        if item.analysis_id is not None:
            parts.append(f"id={item.analysis_id}")
        if item.label is not None:
            parts.append(f'label="{item.label}"')
        if item.model_version is not None:
            parts.append(f"model_version={item.model_version}")
        lines.append("  ".join(parts))
        if item.detail:
            lines.append(f"      {item.detail}")
    lines.append(
        f"{report.count('seeded')} seeded, {report.count('skipped')} skipped, "
        f"{report.count('failed')} failed"
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- core


def _unavailable_reason(registry: ModelRegistry, model_key: str) -> str:
    info = next((i for i in registry.all_info() if i.key == model_key), None)
    return (info.reason if info is not None else None) or "unknown model"


def _live_samples(repo: AnalysisRepository, model_key: str, sha256: str) -> list[Analysis]:
    """Every live sample row of this model with this image hash, whatever its status.

    ``repo.get_sample_by_hash`` returns an arbitrary single row and ignores ``status``, but a
    failed sample row (the service keeps those for debugging) can share its hash with a
    good one. The sample set of a model is small, so paging the listing is cheap and exact.
    """
    found: list[Analysis] = []
    page = 1
    while True:
        rows, total = repo.list(
            AnalysisFilter(model_key=model_key, scope="samples"), page, _LIST_PAGE_SIZE
        )
        found.extend(row for row in rows if row.image_sha256 == sha256)
        if page * _LIST_PAGE_SIZE >= total:
            return found
        page += 1


def seed(
    model_key: str,
    *,
    session_factory: sessionmaker[Session],
    registry: ModelRegistry,
    storage: ImageStorage,
    settings: Settings,
    force: bool = False,
    samples_dir: Path | None = None,
) -> SeedReport:
    """Seed ``samples/<model_key>`` through the real pipeline; see the module docstring.

    Raises ``SeedError`` (a ``ManifestError`` for manifest problems) BEFORE creating any
    row when the run cannot start: model unavailable, manifest missing or invalid. Once
    the run has started, a problem with one image is reported for that image only and the
    remaining images are still processed.
    """
    # Fail clearly and early. `registry.get` is the production check (503 if unavailable).
    try:
        predictor = registry.get(model_key)
    except AppError as exc:
        raise SeedError(
            f"Model '{model_key}' is unavailable: {_unavailable_reason(registry, model_key)}. "
            "No samples were created. Run `python -m scripts.check_env` for details."
        ) from exc

    folder = samples_dir if samples_dir is not None else SAMPLES_ROOT / model_key
    samples = load_manifest(folder)

    report = SeedReport(
        model_key=model_key,
        model_version=predictor.version,
        is_placeholder=predictor.is_placeholder,
    )
    for sample in samples:
        report.items.append(
            _seed_one(
                sample,
                model_key=model_key,
                current_version=predictor.version,
                session_factory=session_factory,
                registry=registry,
                storage=storage,
                settings=settings,
                force=force,
            )
        )
    return report


def _seed_one(
    sample: SampleFile,
    *,
    model_key: str,
    current_version: str,
    session_factory: sessionmaker[Session],
    registry: ModelRegistry,
    storage: ImageStorage,
    settings: Settings,
    force: bool,
) -> SeedItem:
    name = sample.path.name
    entry = sample.entry
    try:
        data = sample.path.read_bytes()
        # The stored hash is of the NORMALISED jpeg (EXIF rotation, RGB, downscale,
        # re-encode), not of the file on disk, so the idempotency check must use the
        # production decode+normalize to compute the same value.
        sha256 = processing.normalize(processing.decode(data, settings), settings).sha256
    except OSError as exc:
        return SeedItem(entry.file, "failed", detail=f"cannot read {name}: {exc}")
    except AppError as exc:
        return SeedItem(entry.file, "failed", detail=f"{exc.code}: {exc.detail}")

    with session_factory() as session:
        try:
            repo = AnalysisRepository(session)
            live = _live_samples(repo, model_key, sha256)
            completed = [row for row in live if row.status == "completed"]
            if completed and not force:
                row = completed[0]
                hint = "already seeded; use --force to re-analyse"
                if row.model_version != current_version:
                    hint = (
                        f"already seeded with {row.model_version}, the model is now "
                        f"{current_version}; use --force to refresh"
                    )
                return SeedItem(
                    entry.file,
                    "skipped",
                    analysis_id=row.id,
                    label=row.display_label,
                    model_version=row.model_version,
                    detail=hint,
                )

            detail = AnalysisService(session, registry, storage, settings).analyze(
                model_key,
                data,
                name,
                CaptureMeta(latitude=entry.latitude, longitude=entry.longitude),
                None,  # samples belong to nobody
                "sample",
                is_sample=True,
                title=entry.title,
                description=entry.description,
            )
            # New row first, old rows retired afterwards: if the new analysis fails the old
            # sample stays visible, so a failed --force never leaves a gap in the gallery.
            # Old rows here are the replaced sample (--force) and/or failed-run leftovers.
            for old in live:
                repo.soft_delete(old)
            session.commit()
        except AppError as exc:
            return SeedItem(entry.file, "failed", detail=f"{exc.code}: {exc.detail}")
        except Exception as exc:  # one bad image must not stop the others (and is reported)
            logger.exception("seeding %s failed", name)
            return SeedItem(entry.file, "failed", detail=f"unexpected {type(exc).__name__}: {exc}")

    prediction = detail.prediction
    return SeedItem(
        entry.file,
        "seeded",
        analysis_id=detail.id,
        label=prediction.display_label if prediction is not None else None,
        model_version=detail.model_version,
        detail="replaced the previous sample"
        if any(row.status == "completed" for row in live)
        else None,
    )


# --------------------------------------------------------------------------- CLI


def database_problem(engine: Engine) -> str | None:
    """None if the database is reachable and migrated, else what to do about it."""
    ok, _ = ping_database(engine)  # logs the real reason at WARNING
    if not ok:
        return (
            "Cannot reach the database (the reason is logged above). Run "
            "`python -m scripts.check_env`. No SQL Server yet? "
            "`python -m scripts.dev_server_sqlite --seed` seeds a throwaway SQLite database."
        )
    if not inspect(engine).has_table("analyses"):
        return "The database has no 'analyses' table: run `alembic upgrade head` first."
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seed the public sample analyses of a model (idempotent)."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=[spec.key for spec in MODEL_SPECS],
        help="model whose samples/<model>/manifest.json is seeded",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-analyse samples that already exist and retire the old rows "
        "(use after replacing weights or editing the images)",
    )
    return parser


def run(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    settings = get_settings()
    engine = create_db_engine(settings)
    try:
        problem = database_problem(engine)
        if problem is not None:
            print(f"ERROR: {problem}", file=err)
            return EXIT_NOT_ATTEMPTED
        # Only the requested model is loaded (loading the other network is wasted time).
        registry = build_registry(settings, only=[args.model])
        report = seed(
            args.model,
            session_factory=create_session_factory(engine),
            registry=registry,
            storage=LocalImageStorage(settings.data_root),
            settings=settings,
            force=args.force,
        )
    except SeedError as exc:
        print(f"ERROR: {exc}", file=err)
        return EXIT_NOT_ATTEMPTED
    finally:
        engine.dispose()

    print(format_report(report), file=out)
    return EXIT_OK if report.ok else EXIT_FILE_FAILED


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
