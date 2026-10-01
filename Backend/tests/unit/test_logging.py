from __future__ import annotations

from app.core.config import Settings
from app.core.logging import _formatter_config


def _settings(**overrides: object) -> Settings:
    return Settings(signing_secret="x" * 40, data_root="C:/kera-data-test", **overrides)  # type: ignore[arg-type]


def test_plain_formatter_by_default() -> None:
    formatter = _formatter_config(_settings(log_json=False))
    assert "()" not in formatter
    assert "%(request_id)s" in formatter["format"]


def test_json_formatter_when_enabled() -> None:
    formatter = _formatter_config(_settings(log_json=True))
    assert formatter["()"] == "pythonjsonlogger.json.JsonFormatter"
