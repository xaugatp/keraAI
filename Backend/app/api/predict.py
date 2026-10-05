"""Everything the ``POST .../predict`` routes have in common (spec 10).

Each model's route file is a few lines: pick the model key, apply the rate limit,
call ``run_predict``. The multipart form, its validation, the upload-size guard
and the call into the service live here exactly once, so Model 3 is "one more
thin route" and the two existing ones can never drift apart.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Any

from fastapi import File, Form, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute

from app.api.responses import ProblemResponses, problem_responses
from app.core.config import Settings, get_settings
from app.core.errors import PayloadTooLargeError
from app.core.rate_limit import limiter
from app.core.time import to_aware_utc
from app.imaging.processing import max_upload_bytes, read_upload_limited
from app.schemas.analysis import AnalysisDetail, CaptureMeta
from app.schemas.common import ClientSource
from app.services.analysis_service import AnalysisService

# The Content-Length header counts the whole multipart body: the file PLUS the
# boundaries, part headers and the other form fields. This slack stops a file just
# under the limit from being refused by the cheap pre-check; the exact limit on the
# file itself is enforced while streaming it (read_upload_limited).
_MULTIPART_OVERHEAD_BYTES = 64 * 1024


class UploadGuardRoute(APIRoute):
    """Route class that refuses an oversized upload BEFORE its body is parsed.

    FastAPI parses the whole multipart body (spooling the file to a temp file)
    before any dependency or endpoint code runs, so a size check written inside
    the endpoint would only fire after a 5 GB upload had already been written to
    disk. Checking the declared Content-Length here makes the 413 immediate.
    Uploads without a Content-Length (chunked) are still bounded afterwards by
    ``read_upload_limited`` (and in production by Cloudflare's own body limit).
    """

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        handler = super().get_route_handler()

        async def guarded(request: Request) -> Response:
            settings: Settings = request.app.state.settings
            declared = _declared_length(request)
            if declared is not None and declared > (
                max_upload_bytes(settings) + _MULTIPART_OVERHEAD_BYTES
            ):
                raise PayloadTooLargeError(
                    f"Image exceeds the {settings.max_upload_mb:g} MB limit."
                )
            return await handler(request)

        return guarded


def _declared_length(request: Request) -> int | None:
    raw = request.headers.get("content-length")
    if raw is None or not raw.isdigit():
        return None  # absent or malformed: the streaming check is the backstop
    return int(raw)


def predict_limit() -> str:
    # Looked up per request (not frozen at import) so a settings change - or a
    # test - takes effect without re-importing the module.
    return get_settings().rate_limit_predict


# ONE shared budget for every predict route (RATE_LIMIT_PREDICT per client IP):
# what is being protected is the laptop's CPU/GPU, not an individual route.
predict_rate_limit = limiter.shared_limit(predict_limit, scope="predict")


@dataclass(frozen=True)
class PredictInput:
    image: UploadFile
    source: ClientSource
    meta: CaptureMeta


def _invalid(field: str, message: str) -> RequestValidationError:
    """A 422 in the same shape FastAPI's own field validation produces."""
    return RequestValidationError([{"type": "value_error", "loc": ("body", field), "msg": message}])


async def get_predict_input(
    image: Annotated[UploadFile, File(description="Photo to analyse (JPEG, PNG or WEBP)")],
    latitude: Annotated[
        float | None, Form(ge=-90, le=90, allow_inf_nan=False, description="Degrees, -90..90")
    ] = None,
    longitude: Annotated[
        float | None, Form(ge=-180, le=180, allow_inf_nan=False, description="Degrees, -180..180")
    ] = None,
    gps_accuracy_m: Annotated[
        float | None, Form(ge=0, allow_inf_nan=False, description="GPS accuracy in metres")
    ] = None,
    captured_at: Annotated[
        datetime | None,
        Form(description="ISO-8601 capture time; a time without an offset is read as UTC"),
    ] = None,
    source: Annotated[ClientSource, Form(description="Where the photo came from")] = "upload",
) -> PredictInput:
    """The multipart form shared by every predict route.

    ``source=sample`` is deliberately not accepted: only the seed script creates samples.
    """
    # Both or neither: half a coordinate pair is meaningless and would plot a
    # marker at the equator/meridian.
    if (latitude is None) != (longitude is None):
        missing = "longitude" if longitude is None else "latitude"
        raise _invalid(missing, "latitude and longitude must be sent together")
    if captured_at is not None:
        try:
            captured_at = to_aware_utc(captured_at)
        except OverflowError:  # e.g. year 0001 with a positive UTC offset
            raise _invalid("captured_at", "captured_at is out of range") from None
    return PredictInput(
        image=image,
        source=source,
        meta=CaptureMeta(
            latitude=latitude,
            longitude=longitude,
            gps_accuracy_m=gps_accuracy_m,
            captured_at=captured_at,
        ),
    )


async def run_predict(
    *,
    model_key: str,
    response: Response,
    payload: PredictInput,
    client_id: uuid.UUID,
    service: AnalysisService,
    settings: Settings,
) -> AnalysisDetail:
    """Read the upload, run the shared pipeline in the threadpool, shape the 201."""
    data = await read_upload_limited(payload.image, max_upload_bytes(settings))
    # Inference, image encoding and pyodbc are all blocking: run them off the
    # event loop so health checks and other requests stay responsive (P-01).
    detail = await run_in_threadpool(
        service.analyze,
        model_key,
        data,
        payload.image.filename,
        payload.meta,
        client_id,
        payload.source,
    )
    # The 201 itself is declared on the route (so /docs shows it); Location points
    # at the new resource, relative like every other URL we hand out.
    response.headers["Location"] = f"{settings.api_v1_prefix}/analyses/{detail.id}"
    return detail


# What every predict route can answer, for /docs and the generated client.
PREDICT_RESPONSES: ProblemResponses = {
    201: {
        "description": "Analysis created and stored.",
        "headers": {
            "Location": {
                "description": "Relative URL of the new analysis (GET it for later).",
                "schema": {"type": "string"},
            }
        },
    },
    **problem_responses(400, 413, 415, 422, 429, 500, 503),
}
