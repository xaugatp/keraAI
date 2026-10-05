"""Shared helpers for the API integration tests (plain functions, no fixtures)."""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from app.db.models.analysis import Analysis
from app.storage.base import build_paths
from tests.fakes import make_tree_details
from tests.fixtures.image_factory import encode, gradient_image, jpeg_bytes, png_bytes
from tests.unit.db.factories import make_analysis

PROBLEM_JSON = "application/problem+json"
PREFIX = "/api/v1"
TREE_URL = f"{PREFIX}/tree/predict"
LEAF_URL = f"{PREFIX}/leaf-segmentation/predict"
ANALYSES_URL = f"{PREFIX}/analyses"


def headers_for(client_id: uuid.UUID | str) -> dict[str, str]:
    return {"X-Client-Id": str(client_id)}


def post_image(
    client: TestClient,
    url: str,
    client_id: uuid.UUID | str | None,
    *,
    content: bytes | None = None,
    filename: str = "banana.jpg",
    data: dict[str, Any] | None = None,
    extra_headers: dict[str, str] | None = None,
) -> httpx.Response:
    """POST a multipart predict request. ``client_id=None`` omits the header."""
    headers = {} if client_id is None else headers_for(client_id)
    headers.update(extra_headers or {})
    return client.post(
        url,
        files={"image": (filename, content if content is not None else jpeg_bytes(), "image/jpeg")},
        data=data or {},
        headers=headers,
    )


def assert_problem(response: httpx.Response, status: int, code: str) -> dict[str, Any]:
    """Every error must be RFC 9457 problem+json with a stable code and a request id."""
    assert response.status_code == status, response.text
    assert response.headers["content-type"] == PROBLEM_JSON
    body: dict[str, Any] = response.json()
    assert body["code"] == code
    assert body["status"] == status
    assert body["request_id"], "error body must carry the request id"
    assert body["request_id"] == response.headers["x-request-id"]
    return body


def files_under(root: Path) -> list[Path]:
    """Every regular file below ``root`` (empty if it does not exist)."""
    return sorted(path for path in root.rglob("*") if path.is_file()) if root.exists() else []


def relative_url(url: str) -> str:
    """Image URLs are relative (``/api/v1/...``); TestClient takes them as they are."""
    assert url.startswith(PREFIX), url
    return url


def seed_analysis(
    factory: sessionmaker[Session],
    data_root: Path,
    *,
    client_id: uuid.UUID | None,
    created_at: datetime,
    is_sample: bool = False,
    status: str = "completed",
    model_key: str = "tree_classification",
    with_files: bool = True,
    with_result: bool = False,
    **overrides: Any,
) -> uuid.UUID:
    """Insert one row (and, by default, real files for it) the way the service would."""
    analysis_id = uuid.uuid4()
    paths = build_paths(model_key, analysis_id, created_at)
    values: dict[str, Any] = {
        "id": analysis_id,
        "model_key": model_key,
        "client_id": client_id,
        "is_sample": is_sample,
        "source": "sample" if is_sample else "upload",
        "status": status,
        "created_at": created_at,
        "updated_at": created_at,
        "original_image_path": paths.original,
        "thumbnail_path": paths.thumbnail,
        "result_image_path": paths.result if with_result else None,
        "image_width": 200,
        "image_height": 150,
        "details": make_tree_details().model_dump(mode="json") if status == "completed" else None,
    }
    if status == "failed":
        values.update(
            predicted_label=None,
            display_label=None,
            confidence=None,
            error_code="INFERENCE_FAILED",
            error_message="RuntimeError: secret internal detail",
        )
    values.update(overrides)
    with factory() as session:
        session.add(make_analysis(**values))
        session.commit()

    if with_files:
        _write(data_root, paths.original, jpeg_bytes())
        _write(data_root, paths.thumbnail, encode(gradient_image(32, 24), "WEBP"))
        if with_result:
            _write(data_root, paths.result, png_bytes())
    return analysis_id


def _write(root: Path, rel_path: str, data: bytes) -> None:
    target = root.joinpath(*rel_path.split("/"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)


def fetch_row(factory: sessionmaker[Session], analysis_id: uuid.UUID | str) -> Analysis | None:
    """The row INCLUDING soft-deleted ones (the repository hides those)."""
    with factory() as session:
        return session.get(Analysis, uuid.UUID(str(analysis_id)))


def count_rows(factory: sessionmaker[Session]) -> int:
    with factory() as session:
        return len(session.query(Analysis).all())
