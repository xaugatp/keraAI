"""Hermetic ``Settings`` factory for unit tests.

Every field the imaging/storage/signing code reads is pinned explicitly, so a
developer's real environment variables or ``.env`` file cannot change a test's
outcome. Pass keyword overrides for the values a test cares about.
"""

from __future__ import annotations

from typing import Any

from app.core.config import Settings


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "signing_secret": "unit-test-signing-secret-0123456789abcdef",
        "data_root": "unused-in-this-test",
        "api_v1_prefix": "/api/v1",
        "signed_url_ttl_seconds": 3600,
        "max_upload_mb": 10,
        "min_image_side_px": 64,
        "max_image_pixels": 40_000_000,
        "store_max_side_px": 2048,
        "thumbnail_side_px": 320,
        "allowed_image_formats": ["JPEG", "PNG", "WEBP"],
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)
