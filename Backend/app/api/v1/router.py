from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.endpoints import analyses, leaf_segmentation, models, tree

# Versioned (/api/v1/...) endpoints. A new model's predict route is one more
# module in endpoints/ plus one include_router line here.
api_router = APIRouter()
api_router.include_router(models.router)
api_router.include_router(tree.router)
api_router.include_router(leaf_segmentation.router)
api_router.include_router(analyses.router)
