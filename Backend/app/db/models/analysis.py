from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Final

from sqlalchemy import (
    CHAR,
    DECIMAL,
    JSON,
    Boolean,
    CheckConstraint,
    Float,
    Index,
    Integer,
    Unicode,
    Uuid,
    false,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.db.types import UTC_DATETIME, utcnow_naive

# --- Allowed value sets ------------------------------------------------------
# Single source of truth: the CHECK constraints below are built from these, and
# the API schemas (Phase 5) import them too, so the DB and the API can never
# disagree about what is valid. NOTE: an Alembic migration must NOT import these
# — it freezes the literal values at the time it was written (see 0001).
MODEL_KEYS: Final[tuple[str, ...]] = ("tree_classification", "leaf_segmentation", "leaf_disease")
ANALYSIS_STATUSES: Final[tuple[str, ...]] = ("completed", "failed")
ANALYSIS_SOURCES: Final[tuple[str, ...]] = ("upload", "camera", "sample")


def _in_list(column: str, values: tuple[str, ...]) -> str:
    # Values are compile-time constants from this module, never user input.
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


class Analysis(Base):
    """One model run on one image (spec §5.2).

    Hybrid schema (D-03): common columns here, model-specific output in the
    ``details`` JSON column. Datetimes are UTC-naive (see app/db/types.py).
    """

    __tablename__ = "analyses"
    __table_args__ = (
        # NULL passes a CHECK (the predicate is UNKNOWN, not FALSE), so the
        # nullable columns below need no explicit `IS NULL OR` clause.
        CheckConstraint(_in_list("model_key", MODEL_KEYS), name="model_key"),
        CheckConstraint(_in_list("status", ANALYSIS_STATUSES), name="status"),
        CheckConstraint(_in_list("source", ANALYSIS_SOURCES), name="source"),
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="confidence_range"),
        CheckConstraint("latitude >= -90 AND latitude <= 90", name="latitude_range"),
        CheckConstraint("longitude >= -180 AND longitude <= 180", name="longitude_range"),
        CheckConstraint("gps_accuracy_m >= 0", name="gps_accuracy_nonneg"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)

    # What produced the row. Unicode(...) => NVARCHAR on SQL Server (String => VARCHAR).
    model_key: Mapped[str] = mapped_column(Unicode(50), nullable=False)
    model_version: Mapped[str] = mapped_column(Unicode(100), nullable=False)
    status: Mapped[str] = mapped_column(Unicode(20), nullable=False)
    source: Mapped[str] = mapped_column(Unicode(20), nullable=False)

    # Samples (D-07) are public; `client_id` is NULL for them (D-04).
    # server_default as well as default so a hand-written INSERT in SSMS works too.
    is_sample: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=false()
    )
    title: Mapped[str | None] = mapped_column(Unicode(200), nullable=True)
    description: Mapped[str | None] = mapped_column(Unicode(1000), nullable=True)
    client_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, nullable=True)

    # Prediction summary. Null on failed rows.
    predicted_label: Mapped[str | None] = mapped_column(Unicode(100), nullable=True)
    display_label: Mapped[str | None] = mapped_column(Unicode(200), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_uncertain: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=false()
    )

    # Location. DECIMAL(9,6) on the wire (~0.11 m resolution) but plain floats in
    # Python (asdecimal=False): the API speaks JSON numbers, and it avoids
    # SQLite's "Decimal not natively supported" warnings in tests.
    latitude: Mapped[float | None] = mapped_column(DECIMAL(9, 6, asdecimal=False), nullable=True)
    longitude: Mapped[float | None] = mapped_column(DECIMAL(9, 6, asdecimal=False), nullable=True)
    gps_accuracy_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    captured_at: Mapped[datetime | None] = mapped_column(UTC_DATETIME, nullable=True)

    # Image files live on disk under DATA_ROOT (D-02): relative, forward slashes.
    original_image_path: Mapped[str] = mapped_column(Unicode(400), nullable=False)
    thumbnail_path: Mapped[str] = mapped_column(Unicode(400), nullable=False)
    result_image_path: Mapped[str | None] = mapped_column(Unicode(400), nullable=True)
    original_filename: Mapped[str | None] = mapped_column(Unicode(255), nullable=True)
    image_width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    image_height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # SHA-256 hex of the stored bytes; makes sample seeding idempotent.
    image_sha256: Mapped[str] = mapped_column(CHAR(64), nullable=False)

    preprocess_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    inference_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    postprocess_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # JSON => NVARCHAR(MAX) on SQL Server. none_as_null: Python None becomes SQL
    # NULL rather than the JSON text 'null', so "no details" is one thing, not two.
    details: Mapped[dict[str, Any] | None] = mapped_column(JSON(none_as_null=True), nullable=True)

    # Populated only when status='failed'. error_message is internal: the API
    # must never echo it to clients (spec §11).
    error_code: Mapped[str | None] = mapped_column(Unicode(50), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Unicode(1000), nullable=True)

    created_at: Mapped[datetime] = mapped_column(UTC_DATETIME, nullable=False, default=utcnow_naive)
    updated_at: Mapped[datetime] = mapped_column(
        UTC_DATETIME, nullable=False, default=utcnow_naive, onupdate=utcnow_naive
    )
    # Soft delete (P-07): the repository hides rows where this is set.
    deleted_at: Mapped[datetime | None] = mapped_column(UTC_DATETIME, nullable=True)


# Indexes are declared after the class so that `created_at.desc()` can be used —
# a DESC key in the Index is what makes Alembic emit `created_at DESC`, matching
# the "newest first" history queries (history = WHERE client_id=? ORDER BY created_at DESC).
Index("ix_analyses_client_created", Analysis.client_id, Analysis.created_at.desc())
Index("ix_analyses_model_created", Analysis.model_key, Analysis.created_at.desc())
Index("ix_analyses_sample", Analysis.is_sample, Analysis.model_key)
Index("ix_analyses_sha256", Analysis.image_sha256)
