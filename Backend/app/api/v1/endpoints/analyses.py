"""History, detail, image and delete routes for analyses (spec 10).

These are plain ``def`` endpoints on purpose: FastAPI runs them in its threadpool,
which is where blocking work (pyodbc, disk) belongs (P-01). All rules (who may see
what, soft delete, signature checks) live in ``AnalysisService``.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Query, Response
from fastapi.responses import FileResponse

from app.api.deps import AnalysisServiceDep, ClientId
from app.api.responses import problem_responses
from app.schemas.analysis import AnalysisDetail, AnalysisSummary
from app.schemas.common import ImageVariant, ModelKey, Page, Scope
from app.services.analysis_service import AnalysisFilter

router = APIRouter(prefix="/analyses", tags=["analyses"])

# Private images are tied to one device and expire with their signed link.
_CACHE_PRIVATE = "private, max-age=3600"
# Samples are public and their URLs are stable, so shared caches may keep them.
_CACHE_PUBLIC = "public, max-age=86400"


def _parse_expiry(raw: str | None) -> int | None:
    """``exp`` as an int, or None when absent/garbled. The service treats None like a
    missing signature: private images then answer 403, public samples ignore it."""
    try:
        return int(raw) if raw is not None else None
    except ValueError:
        return None


@router.get(
    "",
    response_model=Page[AnalysisSummary],
    summary="History: your analyses plus the public samples",
    responses=problem_responses(400, 422, 500, 503),
)
def list_analyses(
    client_id: ClientId,
    service: AnalysisServiceDep,
    model_key: Annotated[ModelKey | None, Query(description="Only this model")] = None,
    scope: Annotated[
        Scope,
        Query(description="mine = your own, samples = public samples, all_visible = both"),
    ] = "all_visible",
    page: Annotated[int, Query(ge=1, le=1_000_000, description="1-based page number")] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> Page[AnalysisSummary]:
    return service.list(
        AnalysisFilter(model_key=model_key, scope=scope), page, page_size, client_id
    )


@router.get(
    "/{analysis_id}",
    response_model=AnalysisDetail,
    summary="One analysis in full",
    responses=problem_responses(400, 404, 422, 500, 503),
)
def get_analysis(
    analysis_id: uuid.UUID, client_id: ClientId, service: AnalysisServiceDep
) -> AnalysisDetail:
    return service.get_detail(analysis_id, client_id)


@router.get(
    "/{analysis_id}/image",
    response_class=FileResponse,
    summary="Download one image of an analysis",
    responses={
        200: {
            "description": "The image file.",
            "content": {"image/jpeg": {}, "image/webp": {}, "image/png": {}},
        },
        **problem_responses(403, 404, 422, 500, 503),
    },
)
def get_analysis_image(
    analysis_id: uuid.UUID,
    service: AnalysisServiceDep,
    variant: Annotated[ImageVariant, Query(description="Which image")] = "original",
    # Taken as text and parsed below so that EVERY signature problem — including a
    # non-numeric `exp` — ends as the same 403 INVALID_SIGNATURE (the service decides),
    # not as a 422 that would tell a prober which part of the link it got wrong.
    exp: Annotated[str | None, Query(description="Unix expiry time, from the signed URL")] = None,
    sig: Annotated[str | None, Query(description="HMAC signature, from the signed URL")] = None,
) -> FileResponse:
    """No ``X-Client-Id`` here: an ``<img>`` tag cannot send headers, so the signed
    link is the credential. Samples are public and need no signature."""
    image = service.open_image(analysis_id, variant, _parse_expiry(exp), sig)
    return FileResponse(
        image.path,
        media_type=image.media_type,
        headers={"Cache-Control": _CACHE_PUBLIC if image.is_public else _CACHE_PRIVATE},
    )


@router.delete(
    "/{analysis_id}",
    status_code=204,
    response_class=Response,
    summary="Delete one of your analyses",
    responses=problem_responses(400, 403, 404, 422, 500, 503),
)
def delete_analysis(
    analysis_id: uuid.UUID, client_id: ClientId, service: AnalysisServiceDep
) -> Response:
    """Soft delete: hidden from every read afterwards; the files stay on disk (P-07)."""
    service.delete(analysis_id, client_id)
    return Response(status_code=204)
