from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from app.db.models.analysis import Analysis


def make_analysis(**overrides: Any) -> Analysis:
    """A valid `Analysis` with every NOT NULL column filled; override per test.

    The hash is unique per call so tests never trip over each other's rows.
    """
    analysis_id = overrides.pop("id", None) or uuid.uuid4()
    values: dict[str, Any] = {
        "id": analysis_id,
        "model_key": "tree_classification",
        "model_version": "tree_cls_v1",
        "status": "completed",
        "source": "upload",
        "client_id": uuid.uuid4(),
        "predicted_label": "banana_tree",
        "display_label": "Banana tree",
        "confidence": 0.947,
        "original_image_path": f"images/tree_classification/2026/10/{analysis_id}/original.jpg",
        "thumbnail_path": f"images/tree_classification/2026/10/{analysis_id}/thumb.webp",
        "image_sha256": uuid.uuid4().hex + uuid.uuid4().hex,  # 64 hex chars
    }
    values.update(overrides)
    return Analysis(**values)


def at(year: int, month: int, day: int, *, hour: int = 0, second: int = 0) -> datetime:
    """A UTC-naive datetime, the way the DB stores them."""
    return datetime(year, month, day, hour, 0, second)
