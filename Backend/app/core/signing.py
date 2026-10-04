"""HMAC-signed, expiring image URLs (spec P-02 / section 10).

``<img src>`` cannot send the ``X-Client-Id`` header, so image access is
protected the way S3 pre-signed URLs are: the API hands out a link carrying an
expiry and an HMAC over ``{id}:{variant}:{exp}``. Sample images are public and
do not use these helpers.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import uuid
from datetime import datetime
from urllib.parse import urlencode

from app.core.config import Settings
from app.core.errors import InvalidSignatureError
from app.core.time import utcnow

VARIANTS: tuple[str, ...] = ("original", "thumbnail", "result")


def _canonical_id(analysis_id: uuid.UUID | str) -> str:
    # Re-rendered so "ABC..." / braces / URN forms of the same UUID sign
    # identically; the path parameter is usually already a UUID object.
    return str(analysis_id if isinstance(analysis_id, uuid.UUID) else uuid.UUID(analysis_id))


def _compute(settings: Settings, analysis_id: str, variant: str, exp: int) -> str:
    message = f"{analysis_id}:{variant}:{exp}".encode()
    digest = hmac.new(settings.signing_secret.encode(), message, hashlib.sha256).digest()
    # base64url without padding: URL-safe as-is, no '=' to escape.
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def sign(analysis_id: uuid.UUID | str, variant: str, exp: int, settings: Settings) -> str:
    """Signature for one image link. ``exp`` is a Unix timestamp in seconds."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown image variant {variant!r}")
    return _compute(settings, _canonical_id(analysis_id), variant, exp)


def verify(
    analysis_id: uuid.UUID | str,
    variant: str,
    exp: int,
    sig: str,
    settings: Settings,
    *,
    now: datetime | None = None,
) -> None:
    """Raise ``InvalidSignatureError`` unless ``sig`` is valid and not expired.

    Every failure (bad id, unknown variant, wrong/tampered signature, expiry)
    produces the same error and message, so a caller learns nothing about
    which check failed.
    """
    try:
        canonical = _canonical_id(analysis_id)
    except ValueError:
        raise InvalidSignatureError() from None
    if variant not in VARIANTS:
        raise InvalidSignatureError()

    expected = _compute(settings, canonical, variant, exp)
    # compare_digest is constant-time; encode first because it rejects
    # non-ASCII str. The signature is checked before the expiry so timing never
    # reveals whether a forged link "would have been" expired.
    signature_ok = hmac.compare_digest(expected.encode(), sig.encode("utf-8", errors="replace"))
    current = int((now or utcnow()).timestamp())
    if not signature_ok or exp < current:
        raise InvalidSignatureError()


def signed_image_path(
    analysis_id: uuid.UUID | str,
    variant: str,
    settings: Settings,
    *,
    now: datetime | None = None,
) -> str:
    """Relative URL (the frontend prefixes its API base) for one image variant."""
    exp = int((now or utcnow()).timestamp()) + settings.signed_url_ttl_seconds
    query = urlencode(
        {"variant": variant, "exp": exp, "sig": sign(analysis_id, variant, exp, settings)}
    )
    return f"{settings.api_v1_prefix}/analyses/{_canonical_id(analysis_id)}/image?{query}"
