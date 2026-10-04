"""Storage contract: the ``ImageStorage`` protocol, its errors and the path layout.

The layout lives here (not in ``local.py``) because it is part of the contract:
the database stores these relative paths, so any future backend (S3, ...)
must use the same keys.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from app.core.time import to_aware_utc

# Mirrors the CHECK constraint on analyses.model_key (spec section 5.2).
ALLOWED_MODEL_KEYS: tuple[str, ...] = (
    "tree_classification",
    "leaf_segmentation",
    "leaf_disease",
)


class StorageError(Exception):
    """Base class for storage failures (internal; never shown to clients)."""


class InvalidStoragePathError(StorageError, ValueError):
    """A relative path was rejected by the path-traversal guard or the layout rules."""


class StoredFileNotFoundError(StorageError, FileNotFoundError):
    """The requested stored file does not exist."""


class ImageStorage(Protocol):
    """Where image bytes live. Paths are always relative, with forward slashes."""

    def save(self, rel_path: str, data: bytes) -> None: ...

    def open(self, rel_path: str) -> Path: ...

    def delete(self, rel_path: str) -> None: ...

    def exists(self, rel_path: str) -> bool: ...


@dataclass(frozen=True)
class ImagePaths:
    original: str
    thumbnail: str
    result: str


def build_paths(model_key: str, analysis_id: uuid.UUID | str, created_at: datetime) -> ImagePaths:
    """Relative storage paths for one analysis.

    ``images/{model_key}/{YYYY}/{MM}/{analysis_id}/original.jpg | thumb.webp | result.png``

    Every piece is validated or generated here (model key from an allowlist,
    UUID re-rendered canonically, date from our own clock) so a path can never
    contain client-supplied text.
    """
    if model_key not in ALLOWED_MODEL_KEYS:
        raise InvalidStoragePathError(f"Unknown model key {model_key!r}")
    if isinstance(analysis_id, uuid.UUID):
        parsed = analysis_id
    else:
        try:
            parsed = uuid.UUID(analysis_id)
        except (ValueError, AttributeError, TypeError) as exc:
            raise InvalidStoragePathError("analysis_id must be a UUID") from exc

    moment = to_aware_utc(created_at)
    folder = f"images/{model_key}/{moment.year:04d}/{moment.month:02d}/{parsed}"
    return ImagePaths(
        original=f"{folder}/original.jpg",
        thumbnail=f"{folder}/thumb.webp",
        result=f"{folder}/result.png",
    )
