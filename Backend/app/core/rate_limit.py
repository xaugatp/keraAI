from __future__ import annotations

from slowapi import Limiter
from starlette.requests import Request

from app.core.config import get_settings


def rate_limit_key(request: Request) -> str:
    """Cloudflare client IP when behind the tunnel, else the socket peer address."""
    settings = get_settings()
    if settings.trust_cloudflare_headers:
        cf_ip = request.headers.get("CF-Connecting-IP")
        if cf_ip:
            return cf_ip
    client = request.client
    return client.host if client else "unknown"


limiter = Limiter(key_func=rate_limit_key, headers_enabled=True)
