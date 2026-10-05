"""Analysis request/response payloads (spec 10).

``AnalysisDetail`` is what ``POST .../predict`` and ``GET /analyses/{id}`` return;
``AnalysisSummary`` is one row of the history table. Field names are snake_case
and every datetime is aware UTC (serialised with a "Z").
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field

from app.schemas.common import (
    AnalysisSource,
    AnalysisStatus,
    GeoLocation,
    ModelKey,
    UtcDatetime,
)
from app.schemas.leaf_seg import LeafSegDetails
from app.schemas.tree import TreeDetails

# A second model exists now, so `details` is a discriminated union on `kind`
# (spec 10): the JSON stored in analyses.details is validated against exactly one
# shape, and OpenAPI renders a `oneOf` with a discriminator so the frontend can
# narrow the TypeScript type on `kind`. Adding Model 3 = add its details class here.
AnalysisDetails = Annotated[TreeDetails | LeafSegDetails, Field(discriminator="kind")]


@dataclass(frozen=True)
class CaptureMeta:
    """Optional capture metadata the client sends with an upload.

    Already validated by the HTTP layer (ranges, both-or-neither coordinates,
    ``captured_at`` converted to UTC); the service only stores it.
    """

    latitude: float | None = None
    longitude: float | None = None
    gps_accuracy_m: float | None = None
    captured_at: datetime | None = None  # aware or naive-UTC; the service stores naive UTC


class Prediction(BaseModel):
    label: str | None = Field(description="Raw class key produced by the model")
    display_label: str = Field(description="Human-readable label, or 'Uncertain'")
    confidence: float | None = Field(
        ge=0, le=1, description="0-1; null when the model has no single honest number"
    )
    is_uncertain: bool


class ImageLinks(BaseModel):
    """Relative URLs (the frontend prefixes its API base). Private images are signed
    and expire; sample images are public and need no signature."""

    width: int | None = None
    height: int | None = None
    original_url: str
    thumbnail_url: str
    result_url: str | None = Field(
        default=None, description="Overlay/mask image; only models that produce one"
    )


class TimingsMs(BaseModel):
    preprocess: int | None = None
    inference: int | None = None
    postprocess: int | None = None
    total: int | None = Field(default=None, description="Wall-clock of the whole analysis")


class AnalysisDetail(BaseModel):
    id: uuid.UUID
    model_key: ModelKey
    model_version: str
    status: AnalysisStatus
    source: AnalysisSource
    is_sample: bool
    title: str | None = None
    description: str | None = None
    # Null only for status="failed" (the model raised): there is no result to show.
    # The internal error message is never exposed.
    prediction: Prediction | None
    details: AnalysisDetails | None
    location: GeoLocation | None = Field(
        description="Null when the client sent no coordinates, accuracy or capture time"
    )
    image: ImageLinks
    timings_ms: TimingsMs
    created_at: UtcDatetime


class AnalysisSummary(BaseModel):
    id: uuid.UUID
    model_key: ModelKey
    source: AnalysisSource
    is_sample: bool
    title: str | None = None
    display_label: str
    confidence: float | None = Field(ge=0, le=1)
    is_uncertain: bool
    thumbnail_url: str
    latitude: float | None = None
    longitude: float | None = None
    created_at: UtcDatetime
