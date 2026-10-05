"""ORM row -> response schema mapping, shared by every read path of the service.

Pure functions, no I/O. This is also where image URLs are minted: private
images get a fresh signed, expiring link on every response (P-02); samples are
public (D-07) so their link is a plain, cacheable URL without a signature.
"""

from __future__ import annotations

from urllib.parse import urlencode

from app.core.config import Settings
from app.core.signing import signed_image_path
from app.db.models.analysis import Analysis
from app.schemas.analysis import (
    AnalysisDetail,
    AnalysisSummary,
    ImageLinks,
    Prediction,
    TimingsMs,
)
from app.schemas.common import GeoLocation, ImageVariant


def image_url(row: Analysis, variant: ImageVariant, settings: Settings) -> str:
    if row.is_sample:
        query = urlencode({"variant": variant})
        return f"{settings.api_v1_prefix}/analyses/{row.id}/image?{query}"
    return signed_image_path(row.id, variant, settings)


def _location(row: Analysis) -> GeoLocation | None:
    if (
        row.latitude is None
        and row.longitude is None
        and row.gps_accuracy_m is None
        and row.captured_at is None
    ):
        return None
    return GeoLocation(
        latitude=row.latitude,
        longitude=row.longitude,
        accuracy_m=row.gps_accuracy_m,
        captured_at=row.captured_at,  # naive UTC in the DB; UtcDatetime makes it aware
    )


def to_detail(row: Analysis, settings: Settings) -> AnalysisDetail:
    failed = row.status == "failed"
    return AnalysisDetail(
        id=row.id,
        model_key=row.model_key,
        model_version=row.model_version,
        status=row.status,
        source=row.source,
        is_sample=row.is_sample,
        title=row.title,
        description=row.description,
        # A failed run has no result. error_message is internal and is never mapped.
        prediction=None
        if failed
        else Prediction(
            label=row.predicted_label,
            display_label=row.display_label or "",
            confidence=row.confidence,
            is_uncertain=row.is_uncertain,
        ),
        # The stored JSON is validated against exactly one details class, chosen
        # by its `kind` (discriminated union).
        details=None if failed else row.details,
        location=_location(row),
        image=ImageLinks(
            width=row.image_width,
            height=row.image_height,
            original_url=image_url(row, "original", settings),
            thumbnail_url=image_url(row, "thumbnail", settings),
            result_url=(
                image_url(row, "result", settings) if row.result_image_path is not None else None
            ),
        ),
        timings_ms=TimingsMs(
            preprocess=row.preprocess_ms,
            inference=row.inference_ms,
            postprocess=row.postprocess_ms,
            total=row.total_ms,
        ),
        created_at=row.created_at,
    )


def to_summary(row: Analysis, settings: Settings) -> AnalysisSummary:
    return AnalysisSummary(
        id=row.id,
        model_key=row.model_key,
        source=row.source,
        is_sample=row.is_sample,
        title=row.title,
        display_label=row.display_label or "",
        confidence=row.confidence,
        is_uncertain=row.is_uncertain,
        thumbnail_url=image_url(row, "thumbnail", settings),
        latitude=row.latitude,
        longitude=row.longitude,
        created_at=row.created_at,
    )
