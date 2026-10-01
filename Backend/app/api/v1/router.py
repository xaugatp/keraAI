from __future__ import annotations

from fastapi import APIRouter

# Versioned (/api/v1/...) endpoints. Empty in Phase 1 — models.py, tree.py and
# analyses.py each add themselves here in later phases with no changes
# needed to this file beyond the include_router call.
api_router = APIRouter()
