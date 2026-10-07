"""GET /models — what the backend can run, and whether it is ready (spec 10)."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import Registry
from app.api.responses import problem_responses
from app.schemas.common import ModelInfoOut

router = APIRouter(tags=["models"])


@router.get(
    "/models",
    response_model=list[ModelInfoOut],
    summary="List models and their status",
    responses=problem_responses(500),
)
async def list_models(registry: Registry) -> list[ModelInfoOut]:
    """All three models are built; `status: unavailable` means one failed to load at startup
    (missing weights, bad checkpoint), not a future model.

    A pure in-memory read, so it runs on the event loop without a threadpool hop.
    """
    return [ModelInfoOut.model_validate(info) for info in registry.all_info()]
