"""create analyses

Revision ID: 0001
Revises:
Create Date: 2026-10-05

Creates the `analyses` table (spec §5.2). Hand-reviewed against the SQL Server
output of `alembic upgrade head --sql`:
  UNIQUEIDENTIFIER ids, NVARCHAR text, BIT flags, DATETIME2(3) timestamps,
  NVARCHAR(MAX) for the JSON details, CHAR(64) hash, DESC indexes.

Everything is written out literally on purpose: a migration is a frozen
snapshot, so it must not import app code (the model will keep evolving, the
migration must keep meaning what it meant). All constraint and index names are
explicit (`op.f(...)`) so they never depend on the naming convention.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mssql

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# DATETIME2(3) on SQL Server; a plain DateTime elsewhere (SQLite in tests).
_utc_datetime = sa.DateTime().with_variant(mssql.DATETIME2(precision=3), "mssql")


def upgrade() -> None:
    op.create_table(
        "analyses",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("model_key", sa.Unicode(length=50), nullable=False),
        sa.Column("model_version", sa.Unicode(length=100), nullable=False),
        sa.Column("status", sa.Unicode(length=20), nullable=False),
        sa.Column("source", sa.Unicode(length=20), nullable=False),
        sa.Column("is_sample", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("title", sa.Unicode(length=200), nullable=True),
        sa.Column("description", sa.Unicode(length=1000), nullable=True),
        sa.Column("client_id", sa.Uuid(), nullable=True),
        sa.Column("predicted_label", sa.Unicode(length=100), nullable=True),
        sa.Column("display_label", sa.Unicode(length=200), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("is_uncertain", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("latitude", sa.DECIMAL(precision=9, scale=6, asdecimal=False), nullable=True),
        sa.Column("longitude", sa.DECIMAL(precision=9, scale=6, asdecimal=False), nullable=True),
        sa.Column("gps_accuracy_m", sa.Float(), nullable=True),
        sa.Column("captured_at", _utc_datetime, nullable=True),
        sa.Column("original_image_path", sa.Unicode(length=400), nullable=False),
        sa.Column("thumbnail_path", sa.Unicode(length=400), nullable=False),
        sa.Column("result_image_path", sa.Unicode(length=400), nullable=True),
        sa.Column("original_filename", sa.Unicode(length=255), nullable=True),
        sa.Column("image_width", sa.Integer(), nullable=True),
        sa.Column("image_height", sa.Integer(), nullable=True),
        sa.Column("image_sha256", sa.CHAR(length=64), nullable=False),
        sa.Column("preprocess_ms", sa.Integer(), nullable=True),
        sa.Column("inference_ms", sa.Integer(), nullable=True),
        sa.Column("postprocess_ms", sa.Integer(), nullable=True),
        sa.Column("total_ms", sa.Integer(), nullable=True),
        sa.Column("details", sa.JSON(none_as_null=True), nullable=True),
        sa.Column("error_code", sa.Unicode(length=50), nullable=True),
        sa.Column("error_message", sa.Unicode(length=1000), nullable=True),
        sa.Column("created_at", _utc_datetime, nullable=False),
        sa.Column("updated_at", _utc_datetime, nullable=False),
        sa.Column("deleted_at", _utc_datetime, nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_analyses")),
        sa.CheckConstraint(
            "model_key IN ('tree_classification', 'leaf_segmentation', 'leaf_disease')",
            name=op.f("ck_analyses_model_key"),
        ),
        sa.CheckConstraint("status IN ('completed', 'failed')", name=op.f("ck_analyses_status")),
        sa.CheckConstraint(
            "source IN ('upload', 'camera', 'sample')", name=op.f("ck_analyses_source")
        ),
        sa.CheckConstraint(
            "confidence >= 0 AND confidence <= 1", name=op.f("ck_analyses_confidence_range")
        ),
        sa.CheckConstraint(
            "latitude >= -90 AND latitude <= 90", name=op.f("ck_analyses_latitude_range")
        ),
        sa.CheckConstraint(
            "longitude >= -180 AND longitude <= 180", name=op.f("ck_analyses_longitude_range")
        ),
        sa.CheckConstraint("gps_accuracy_m >= 0", name=op.f("ck_analyses_gps_accuracy_nonneg")),
    )
    # History list: WHERE client_id = ? ORDER BY created_at DESC. The DESC key lets
    # SQL Server read the index in order instead of sorting.
    op.create_index(
        op.f("ix_analyses_client_created"),
        "analyses",
        ["client_id", sa.text("created_at DESC")],
        unique=False,
    )
    op.create_index(
        op.f("ix_analyses_model_created"),
        "analyses",
        ["model_key", sa.text("created_at DESC")],
        unique=False,
    )
    op.create_index(
        op.f("ix_analyses_sample"), "analyses", ["is_sample", "model_key"], unique=False
    )
    op.create_index(op.f("ix_analyses_sha256"), "analyses", ["image_sha256"], unique=False)


def downgrade() -> None:
    # Indexes first, then the table (dropping the table would drop them anyway,
    # but explicit is clearer and mirrors upgrade() in reverse). The CHECK and PK
    # constraints and the is_sample/is_uncertain defaults go with the table.
    op.drop_index(op.f("ix_analyses_sha256"), table_name="analyses")
    op.drop_index(op.f("ix_analyses_sample"), table_name="analyses")
    op.drop_index(op.f("ix_analyses_model_created"), table_name="analyses")
    op.drop_index(op.f("ix_analyses_client_created"), table_name="analyses")
    op.drop_table("analyses")
