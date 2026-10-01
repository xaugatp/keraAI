from __future__ import annotations

import logging
import logging.config
from contextvars import ContextVar
from typing import Any

from app.core.config import Settings

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


def get_request_id() -> str | None:
    return request_id_var.get()


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id() or "-"
        return True


def _formatter_config(settings: Settings) -> dict[str, Any]:
    if settings.log_json:
        return {
            "()": "pythonjsonlogger.json.JsonFormatter",
            "format": "%(asctime)s %(levelname)s %(name)s %(request_id)s %(message)s",
        }
    return {
        "format": "%(asctime)s %(levelname)-8s %(name)s [%(request_id)s] %(message)s",
    }


def configure_logging(settings: Settings) -> None:
    formatter = _formatter_config(settings)

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "filters": {"request_id": {"()": RequestIdFilter}},
            "formatters": {"default": formatter},
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "filters": ["request_id"],
                }
            },
            "root": {"level": settings.log_level, "handlers": ["console"]},
            "loggers": {
                "uvicorn": {
                    "level": settings.log_level,
                    "handlers": ["console"],
                    "propagate": False,
                },
                "uvicorn.access": {
                    "level": settings.log_level,
                    "handlers": ["console"],
                    "propagate": False,
                },
            },
        }
    )
