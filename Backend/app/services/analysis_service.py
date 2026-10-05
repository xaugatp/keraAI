"""AnalysisService — THE pipeline shared by every model (spec 9).

Nothing in this module may know which model it is running: everything
model-specific (labels, details JSON, optional overlay image, timings) arrives
inside ``PredictionOutput``. If you ever need an ``if model_key == ...`` here,
stop — that belongs in the predictor (spec 8).

Layering: this module orchestrates ``ml`` (registry), ``imaging``, ``storage``
and ``repositories``; the API layer calls it and does nothing but HTTP.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from math import ceil
from pathlib import Path
from typing import Final

from sqlalchemy.orm import Session

from app.core import signing
from app.core.config import Settings
from app.core.errors import (
    AnalysisNotFoundError,
    InferenceError,
    InvalidSignatureError,
    SampleImmutableError,
)
from app.core.time import to_aware_utc
from app.db.models.analysis import Analysis
from app.db.types import utcnow_naive
from app.imaging import processing
from app.ml.registry import ModelRegistry
from app.repositories.analysis_repository import MAX_PAGE_SIZE, AnalysisFilter, AnalysisRepository
from app.schemas.analysis import AnalysisDetail, AnalysisSummary, CaptureMeta
from app.schemas.common import ImageVariant, Page
from app.services.analysis_mapper import to_detail, to_summary
from app.storage.base import ImageStorage, StoredFileNotFoundError, build_paths

# AnalysisFilter is part of this service's public signature (`list`); re-exporting
# it lets the API layer depend on the service only, not on the repository.
__all__ = ["AnalysisFilter", "AnalysisService", "StoredImage", "sanitize_filename"]

logger = logging.getLogger(__name__)

# Column sizes of analyses.original_filename / error_message.
_FILENAME_MAX_UTF16_UNITS: Final = 255
_ERROR_MESSAGE_MAX_CHARS: Final = 1000

# The media type follows from the variant, never from a client-supplied
# name: originals are re-encoded to JPEG, thumbnails to WEBP, overlays to PNG.
_MEDIA_TYPES: Final[dict[str, str]] = {
    "original": "image/jpeg",
    "thumbnail": "image/webp",
    "result": "image/png",
}

# (relative path, bytes) pairs to write for one analysis.
_Files = Sequence[tuple[str, bytes]]


@dataclass(frozen=True)
class StoredImage:
    """What the image route needs to answer: where the file is and how to cache it."""

    path: Path
    media_type: str
    is_public: bool  # sample images are public; everything else is private


def sanitize_filename(name: str | None) -> str | None:
    """Make a client-supplied filename safe to *display* (never to build a path from).

    - path components are dropped (browsers on Windows may send ``C:\\fakepath\\x.jpg``),
    - control, format and surrogate characters are dropped (this also removes the
      right-to-left override trick that makes ``evil\\u202egpj.exe`` look like a JPEG),
    - truncated to the column size. SQL Server's NVARCHAR counts UTF-16 code units,
      so an emoji costs 2: counting Python characters could overflow the column.
    """
    if not name:
        return None
    base = name.replace("\\", "/").rsplit("/", 1)[-1]
    cleaned = "".join(char for char in base if char.isprintable()).strip()
    if cleaned in {"", ".", ".."}:
        return None
    encoded = cleaned.encode("utf-16-le")[: _FILENAME_MAX_UTF16_UNITS * 2]
    # errors="ignore" drops a surrogate pair that the cut split in half.
    return encoded.decode("utf-16-le", errors="ignore") or None


def _is_visible(row: Analysis, client_id: uuid.UUID) -> bool:
    """The access rule (spec 5.3): visible iff it is a sample or the caller's own.

    A failed run has no result worth showing, so it is visible to its owner only —
    never as a public sample.
    """
    owned = row.client_id is not None and row.client_id == client_id
    if row.status == "failed":
        return owned
    return row.is_sample or owned


def _is_public(row: Analysis) -> bool:
    return row.is_sample and row.status == "completed"


def _round_coordinate(value: float | None) -> float | None:
    """6 decimals (~0.1 m) is what the DECIMAL(9,6) column keeps. Rounding here makes the
    POST response show exactly what a later GET will read back from SQL Server, which
    would otherwise round on its own (SQLite, used by the tests, would not)."""
    return None if value is None else round(value, 6)


def _to_naive_utc_ms(value: datetime | None) -> datetime | None:
    """Aware/naive UTC -> naive UTC truncated to ms (what DATETIME2(3) stores)."""
    if value is None:
        return None
    naive = to_aware_utc(value).replace(tzinfo=None)
    return naive.replace(microsecond=naive.microsecond // 1000 * 1000)


class AnalysisService:
    """One instance per request: it owns that request's DB ``Session``.

    The service commits (the ``get_db`` dependency only guarantees rollback and
    close), because the commit must happen *inside* the compensation scope that
    cleans up stored files when it fails.
    """

    def __init__(
        self,
        session: Session,
        registry: ModelRegistry,
        storage: ImageStorage,
        settings: Settings,
    ) -> None:
        self._session = session
        self._registry = registry
        self._storage = storage
        self._settings = settings
        self._repo = AnalysisRepository(session)

    # ------------------------------------------------------------------
    # analyze: the 9-step pipeline
    # ------------------------------------------------------------------
    def analyze(
        self,
        model_key: str,
        upload_bytes: bytes,
        original_filename: str | None,
        meta: CaptureMeta,
        client_id: uuid.UUID | None,
        source: str,
        *,
        is_sample: bool = False,
        title: str | None = None,
        description: str | None = None,
    ) -> AnalysisDetail:
        started = time.perf_counter()

        # Programmer-error guards (the HTTP layer can only produce valid combinations).
        if is_sample != (source == "sample"):
            raise ValueError("is_sample=True and source='sample' must be used together")
        if is_sample == (client_id is not None):
            raise ValueError("samples have no client_id; every other analysis needs one")

        # 1. Fail fast with 503 BEFORE spending any CPU/disk on the upload.
        predictor = self._registry.get(model_key)

        # 2. Validate + normalise (EXIF-rotate, RGB, downscale, strip metadata).
        normalized = processing.normalize(
            processing.decode(upload_bytes, self._settings), self._settings
        )

        # 3. Identity and storage locations. The folder date comes from the same
        #    timestamp as created_at, so a row and its files can be matched by eye.
        analysis_id = uuid.uuid4()
        now = utcnow_naive()
        paths = build_paths(model_key, analysis_id, now)
        stored_files: list[tuple[str, bytes]] = [
            (paths.original, normalized.jpeg_bytes),
            (paths.thumbnail, normalized.thumb_bytes),
        ]

        # Columns shared by completed and failed rows.
        common = {
            "id": analysis_id,
            "model_key": model_key,
            "model_version": predictor.version,
            "source": source,
            "is_sample": is_sample,
            "title": title,
            "description": description,
            "client_id": client_id,
            "latitude": _round_coordinate(meta.latitude),
            "longitude": _round_coordinate(meta.longitude),
            "gps_accuracy_m": meta.gps_accuracy_m,
            "captured_at": _to_naive_utc_ms(meta.captured_at),
            "original_image_path": paths.original,
            "thumbnail_path": paths.thumbnail,
            "original_filename": sanitize_filename(original_filename),
            "image_width": normalized.width,
            "image_height": normalized.height,
            "image_sha256": normalized.sha256,
            "created_at": now,
            "updated_at": now,
        }

        # 4. Inference. On failure keep the image (so it can be debugged), record a
        #    `failed` row, and tell the client only that inference failed (P-05).
        try:
            output = predictor.predict(normalized.image)
        except InferenceError as exc:
            self._record_failure(common, stored_files, exc, started)
            # A fresh error: the original's detail holds internals ("ValueError: ...")
            # that the problem+json handler would otherwise send to the client.
            raise InferenceError() from exc

        # 5. Result overlay (Models 2/3), encoded BEFORE touching the disk so an
        #    encoding bug cannot leave half an analysis behind.
        if output.result_image is not None:
            stored_files.append((paths.result, processing.encode_png(output.result_image)))

        # 6. Build the row. total_ms is the wall-clock of the analysis up to this
        #    point (decode, normalise, inference, encode); it excludes the DB insert
        #    and commit because the row has to carry the number itself.
        row = Analysis(
            **common,
            status="completed",
            predicted_label=output.label,
            display_label=output.display_label,
            confidence=output.confidence,
            is_uncertain=output.is_uncertain,
            result_image_path=paths.result if output.result_image is not None else None,
            preprocess_ms=output.timings.preprocess_ms,
            inference_ms=output.timings.inference_ms,
            postprocess_ms=output.timings.postprocess_ms,
            total_ms=round((time.perf_counter() - started) * 1000),
            # mode="json" turns UUIDs/enums/etc. into plain JSON types for the column.
            details=output.details.model_dump(mode="json"),
        )

        # 7. Write files + insert + commit, with compensation on any failure.
        self._store_and_commit(row, stored_files)

        # 8. One structured line. Coordinates are deliberately NOT logged (privacy).
        self._log_completed(row, client_id)

        # 9. Map to the response (signed image URLs included).
        return to_detail(row, self._settings)

    def _store_and_commit(self, row: Analysis, files: _Files) -> None:
        """Write every file, insert the row, commit — or undo all of it.

        Invariant: after this returns normally the row and ALL its files exist;
        after it raises, neither does. So there are never orphan files, and never a
        row pointing at a missing file.
        """
        try:
            for rel_path, data in files:
                self._storage.save(rel_path, data)
            self._repo.add(row)
            self._session.commit()
        except BaseException:
            # Delete every planned path, including the one whose write failed:
            # delete() is idempotent, and a partly written file must not survive.
            # (Residual risk: if the connection dies exactly while the COMMIT is
            # acknowledged, the row may exist although we deleted its files. That
            # window is tiny and a dangling row 404s on read rather than lying.)
            self._discard_files([rel_path for rel_path, _ in files])
            self._rollback_quietly()
            raise

    def _discard_files(self, rel_paths: Sequence[str]) -> None:
        for rel_path in rel_paths:
            try:
                self._storage.delete(rel_path)
            except Exception:
                # Cleanup must never mask the original error; leave a trace instead.
                logger.exception("compensation_delete_failed", extra={"path": rel_path})

    def _rollback_quietly(self) -> None:
        try:
            self._session.rollback()
        except Exception:
            logger.exception("rollback_failed")

    def _record_failure(
        self,
        common: dict[str, object],
        files: _Files,
        exc: InferenceError,
        started: float,
    ) -> None:
        row = Analysis(
            **common,
            status="failed",
            is_uncertain=False,
            error_code=exc.code,
            # Internal only: never mapped into a response (spec 5.2 / 11).
            error_message=exc.detail[:_ERROR_MESSAGE_MAX_CHARS],
            total_ms=round((time.perf_counter() - started) * 1000),
        )
        try:
            self._store_and_commit(row, files)
        except Exception:
            # Failing to record the failure must not hide the inference error the
            # client is about to get; it is loud in the log instead.
            logger.exception(
                "failed_analysis_not_persisted",
                extra={"analysis_id": str(row.id), "model_key": row.model_key},
            )
        else:
            logger.warning(
                "analysis_failed",
                extra={
                    "analysis_id": str(row.id),
                    "model_key": row.model_key,
                    "error_code": exc.code,
                },
            )

    def _log_completed(self, row: Analysis, client_id: uuid.UUID | None) -> None:
        fields = {
            "analysis_id": str(row.id),
            "model_key": row.model_key,
            "model_version": row.model_version,
            "label": row.predicted_label,
            "confidence": row.confidence,
            "total_ms": row.total_ms,
            # First 8 characters only: enough to correlate a device, not to impersonate it.
            "client_id": str(client_id)[:8] if client_id is not None else None,
        }
        # key=value text for the human console, the same keys as `extra` for JSON logs.
        logger.info(
            "analysis_completed " + " ".join(f"{key}={value}" for key, value in fields.items()),
            extra=fields,
        )

    # ------------------------------------------------------------------
    # reads
    # ------------------------------------------------------------------
    def _visible_row(self, analysis_id: uuid.UUID, client_id: uuid.UUID) -> Analysis:
        row = self._repo.get(analysis_id)
        # Same answer whether the row is missing, deleted or someone else's, so
        # the API never reveals that another user's analysis exists.
        if row is None or not _is_visible(row, client_id):
            raise AnalysisNotFoundError()
        return row

    def get_detail(self, analysis_id: uuid.UUID, client_id: uuid.UUID) -> AnalysisDetail:
        return to_detail(self._visible_row(analysis_id, client_id), self._settings)

    def list(
        self,
        filters: AnalysisFilter,
        page: int,
        page_size: int,
        client_id: uuid.UUID,
    ) -> Page[AnalysisSummary]:
        page = max(page, 1)
        page_size = min(max(page_size, 1), MAX_PAGE_SIZE)
        # The caller's identity and the "completed only" rule are imposed here, so
        # whatever the HTTP layer put in `filters` cannot widen what is visible.
        effective = replace(filters, client_id=client_id, status="completed")
        rows, total = self._repo.list(effective, page, page_size)
        return Page[AnalysisSummary](
            items=[to_summary(row, self._settings) for row in rows],
            page=page,
            page_size=page_size,
            total=total,
            total_pages=ceil(total / page_size),
        )

    # ------------------------------------------------------------------
    # delete
    # ------------------------------------------------------------------
    def delete(self, analysis_id: uuid.UUID, client_id: uuid.UUID) -> None:
        """Soft-delete (P-07): the row is hidden, the files stay on disk."""
        row = self._visible_row(analysis_id, client_id)
        if row.is_sample:
            raise SampleImmutableError()
        self._repo.soft_delete(row)
        try:
            self._session.commit()
        except BaseException:
            self._rollback_quietly()
            raise

    # ------------------------------------------------------------------
    # image bytes
    # ------------------------------------------------------------------
    def open_image(
        self,
        analysis_id: uuid.UUID,
        variant: ImageVariant,
        exp: int | None,
        sig: str | None,
    ) -> StoredImage:
        """Resolve one image for the image route.

        <img src> cannot send X-Client-Id, so the signed link IS the credential for
        private images (P-02): anyone holding a valid, unexpired link can load it.
        """
        row = self._repo.get(analysis_id)

        # Public samples need no signature. For everything else — INCLUDING ids that
        # do not exist — the signature is checked first, so a prober gets the same
        # 403 for "no such id" and "private id" and cannot test which ids exist.
        if row is None or not _is_public(row):
            if exp is None or sig is None:
                raise InvalidSignatureError()
            signing.verify(analysis_id, variant, exp, sig, self._settings)
        if row is None:
            raise AnalysisNotFoundError()

        rel_path = {
            "original": row.original_image_path,
            "thumbnail": row.thumbnail_path,
            "result": row.result_image_path,
        }[variant]
        if rel_path is None:
            raise AnalysisNotFoundError("This analysis has no image of that kind.")

        try:
            path = self._storage.open(rel_path)
        except StoredFileNotFoundError:
            # The row says the file exists but the disk disagrees (deleted by hand,
            # DATA_ROOT moved?). The client just sees 404; the operator gets this.
            logger.error(
                "image_file_missing",
                extra={"analysis_id": str(row.id), "variant": variant, "path": rel_path},
            )
            raise AnalysisNotFoundError("Image not found.") from None
        return StoredImage(path=path, media_type=_MEDIA_TYPES[variant], is_public=_is_public(row))
