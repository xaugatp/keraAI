from __future__ import annotations

from unittest.mock import Mock

import pytest

from app.core.config import get_settings
from app.core.rate_limit import rate_limit_key


def _request(client_host: str | None, headers: dict[str, str]) -> Mock:
    request = Mock()
    request.headers = headers
    request.client = Mock(host=client_host) if client_host else None
    return request


def test_uses_client_host_by_default() -> None:
    request = _request("10.0.0.5", {})
    assert rate_limit_key(request) == "10.0.0.5"


def test_falls_back_to_unknown_without_a_client() -> None:
    request = _request(None, {})
    assert rate_limit_key(request) == "unknown"


def test_ignores_cloudflare_header_when_not_trusted() -> None:
    request = _request("127.0.0.1", {"CF-Connecting-IP": "203.0.113.9"})
    assert rate_limit_key(request) == "127.0.0.1"


def test_uses_cloudflare_header_when_trusted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRUST_CLOUDFLARE_HEADERS", "true")
    get_settings.cache_clear()
    try:
        request = _request("127.0.0.1", {"CF-Connecting-IP": "203.0.113.9"})
        assert rate_limit_key(request) == "203.0.113.9"
    finally:
        monkeypatch.setenv("TRUST_CLOUDFLARE_HEADERS", "false")
        get_settings.cache_clear()
