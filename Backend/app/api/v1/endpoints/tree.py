"""Model 1 — banana tree classification route (spec 10)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response

from app.api.deps import AnalysisServiceDep, ClientId, SettingsDep
from app.api.predict import (
    PREDICT_RESPONSES,
    PredictInput,
    UploadGuardRoute,
    get_predict_input,
    predict_rate_limit,
    run_predict,
)
from app.schemas.analysis import AnalysisDetail

router = APIRouter(route_class=UploadGuardRoute, tags=["tree classification"])

MODEL_KEY = "tree_classification"


@router.post(
    "/tree/predict",
    response_model=AnalysisDetail,
    status_code=201,
    summary="Is this a banana tree?",
    responses=PREDICT_RESPONSES,
)
@predict_rate_limit
async def predict_tree(
    # `request` and `response` are required by the rate limiter (it reads the client
    # IP from one and writes X-RateLimit-* headers onto the other).
    request: Request,
    response: Response,
    payload: Annotated[PredictInput, Depends(get_predict_input)],
    client_id: ClientId,
    service: AnalysisServiceDep,
    settings: SettingsDep,
) -> AnalysisDetail:
    return await run_predict(
        model_key=MODEL_KEY,
        response=response,
        payload=payload,
        client_id=client_id,
        service=service,
        settings=settings,
    )
