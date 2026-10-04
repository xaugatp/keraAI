from __future__ import annotations

import base64
import hashlib
import hmac
import re
import uuid
from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qs, urlsplit

import pytest

from app.core import signing
from app.core.errors import InvalidSignatureError
from app.core.signing import VARIANTS, sign, signed_image_path, verify
from tests.fixtures.settings import make_settings

ANALYSIS_ID = uuid.UUID("6f1c2a9e-1b2c-4d3e-8f90-a1b2c3d4e5f6")
NOW = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
EXP = int(NOW.timestamp()) + 3600
SETTINGS = make_settings()


def valid_sig(variant: str = "original", exp: int = EXP) -> str:
    return sign(ANALYSIS_ID, variant, exp, SETTINGS)


# --- signature format ----------------------------------------------------------------


def test_signature_is_unpadded_base64url_of_hmac_sha256() -> None:
    sig = valid_sig()
    assert re.fullmatch(r"[A-Za-z0-9_-]{43}", sig)  # 32 bytes -> 43 chars, no '='
    expected = hmac.new(
        SETTINGS.signing_secret.encode(),
        f"{ANALYSIS_ID}:original:{EXP}".encode(),
        hashlib.sha256,
    ).digest()
    assert base64.urlsafe_b64decode(sig + "=") == expected


def test_signing_is_deterministic() -> None:
    assert valid_sig() == valid_sig()


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_variant_signs_and_verifies(variant: str) -> None:
    verify(ANALYSIS_ID, variant, EXP, valid_sig(variant), SETTINGS, now=NOW)


def test_the_variant_allowlist_is_the_three_from_the_spec() -> None:
    assert set(VARIANTS) == {"original", "thumbnail", "result"}


def test_string_ids_are_equivalent_to_uuid_objects() -> None:
    sig = valid_sig()
    for text in (str(ANALYSIS_ID), str(ANALYSIS_ID).upper(), "{" + str(ANALYSIS_ID) + "}"):
        verify(text, "original", EXP, sig, SETTINGS, now=NOW)
        assert sign(text, "original", EXP, SETTINGS) == sig


# --- verification -------------------------------------------------------------------


def test_a_valid_unexpired_signature_passes() -> None:
    verify(ANALYSIS_ID, "original", EXP, valid_sig(), SETTINGS, now=NOW)


def test_verify_defaults_to_the_current_time() -> None:
    future = int((datetime.now(UTC) + timedelta(minutes=5)).timestamp())
    verify(ANALYSIS_ID, "thumbnail", future, valid_sig("thumbnail", future), SETTINGS)
    past = int((datetime.now(UTC) - timedelta(minutes=5)).timestamp())
    with pytest.raises(InvalidSignatureError):
        verify(ANALYSIS_ID, "thumbnail", past, valid_sig("thumbnail", past), SETTINGS)


def test_expiry_boundary_is_inclusive() -> None:
    expiry = int(NOW.timestamp())
    sig = valid_sig(exp=expiry)
    verify(ANALYSIS_ID, "original", expiry, sig, SETTINGS, now=NOW)  # exactly at exp: ok
    with pytest.raises(InvalidSignatureError):
        verify(ANALYSIS_ID, "original", expiry, sig, SETTINGS, now=NOW + timedelta(seconds=1))


def test_expired_signature_is_rejected_even_though_it_is_otherwise_valid() -> None:
    with pytest.raises(InvalidSignatureError):
        verify(ANALYSIS_ID, "original", EXP, valid_sig(), SETTINGS, now=NOW + timedelta(days=1))


def test_tampered_expiry_is_rejected() -> None:
    with pytest.raises(InvalidSignatureError):
        verify(ANALYSIS_ID, "original", EXP + 1, valid_sig(), SETTINGS, now=NOW)


def test_wrong_id_is_rejected() -> None:
    with pytest.raises(InvalidSignatureError):
        verify(uuid.uuid4(), "original", EXP, valid_sig(), SETTINGS, now=NOW)


def test_wrong_variant_is_rejected() -> None:
    with pytest.raises(InvalidSignatureError):
        verify(ANALYSIS_ID, "thumbnail", EXP, valid_sig("original"), SETTINGS, now=NOW)


def test_wrong_secret_is_rejected() -> None:
    other = make_settings(signing_secret="a-completely-different-secret-0123456789")
    with pytest.raises(InvalidSignatureError):
        verify(ANALYSIS_ID, "original", EXP, valid_sig(), other, now=NOW)


def _flip_first_char(sig: str) -> str:
    return ("A" if sig[0] != "A" else "B") + sig[1:]


@pytest.mark.parametrize(
    "tamper",
    [
        _flip_first_char,
        lambda sig: sig[:-1],
        lambda sig: sig + "A",
        lambda sig: sig + "=",
        lambda sig: sig.lower() if sig != sig.lower() else sig.upper(),
        lambda sig: " " + sig,
        lambda sig: sig + "\n",
        lambda sig: "",
        lambda sig: "é" * 43,
        lambda sig: "\ud800" * 43,  # lone surrogate: not even encodable
    ],
    ids=[
        "flipped-char",
        "truncated",
        "extended",
        "padded",
        "case-changed",
        "leading-space",
        "trailing-newline",
        "empty",
        "non-ascii",
        "lone-surrogate",
    ],
)
def test_tampered_signatures_are_rejected(tamper) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(InvalidSignatureError):
        verify(ANALYSIS_ID, "original", EXP, tamper(valid_sig()), SETTINGS, now=NOW)


@pytest.mark.parametrize("variant", ["", "ORIGINAL", "../original", "full", "original "])
def test_unknown_variants_fail_verification_rather_than_crash(variant: str) -> None:
    with pytest.raises(InvalidSignatureError):
        verify(ANALYSIS_ID, variant, EXP, valid_sig(), SETTINGS, now=NOW)


@pytest.mark.parametrize("analysis_id", ["", "not-a-uuid", "../../etc/passwd", "123"])
def test_malformed_ids_fail_verification_rather_than_crash(analysis_id: str) -> None:
    with pytest.raises(InvalidSignatureError):
        verify(analysis_id, "original", EXP, valid_sig(), SETTINGS, now=NOW)


def test_signing_an_unknown_variant_or_bad_id_is_a_programming_error() -> None:
    with pytest.raises(ValueError, match="variant"):
        sign(ANALYSIS_ID, "full", EXP, SETTINGS)
    with pytest.raises(ValueError):
        sign("not-a-uuid", "original", EXP, SETTINGS)


def test_every_failure_mode_looks_identical_to_the_caller() -> None:
    attempts = [
        lambda: verify(ANALYSIS_ID, "original", EXP, "bad", SETTINGS, now=NOW),
        lambda: verify(
            ANALYSIS_ID, "original", EXP, valid_sig(), SETTINGS, now=NOW + timedelta(days=9)
        ),
        lambda: verify(ANALYSIS_ID, "full", EXP, valid_sig(), SETTINGS, now=NOW),
        lambda: verify("junk", "original", EXP, valid_sig(), SETTINGS, now=NOW),
    ]
    seen = set()
    for attempt in attempts:
        with pytest.raises(InvalidSignatureError) as exc_info:
            attempt()
        error = exc_info.value
        assert (error.status_code, error.code) == (403, "INVALID_SIGNATURE")
        seen.add(error.detail)
    assert len(seen) == 1  # same message: no oracle for which check failed


def test_signature_comparison_is_constant_time(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[bytes, bytes]] = []
    real = hmac.compare_digest

    def spy(a: bytes, b: bytes) -> bool:
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(signing.hmac, "compare_digest", spy)
    sig = valid_sig()
    verify(ANALYSIS_ID, "original", EXP, sig, SETTINGS, now=NOW)
    with pytest.raises(InvalidSignatureError):
        verify(ANALYSIS_ID, "original", EXP, "x" * 43, SETTINGS, now=NOW)
    # The (expected, supplied) pair was compared with hmac.compare_digest both times,
    # also when the link was bad, and never with ==.
    assert len(calls) == 2
    assert calls[0] == (sig.encode(), sig.encode())
    assert all(isinstance(a, bytes) and isinstance(b, bytes) for a, b in calls)


def test_signature_is_checked_even_when_the_link_is_expired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    real = hmac.compare_digest

    def spy(a: bytes, b: bytes) -> bool:
        calls.append(1)
        return real(a, b)

    monkeypatch.setattr(signing.hmac, "compare_digest", spy)
    with pytest.raises(InvalidSignatureError):
        verify(ANALYSIS_ID, "original", EXP, valid_sig(), SETTINGS, now=NOW + timedelta(days=2))
    assert calls == [1]


# --- URL building ----------------------------------------------------------------------


def test_signed_path_has_the_spec_shape() -> None:
    path = signed_image_path(ANALYSIS_ID, "thumbnail", SETTINGS, now=NOW)
    parts = urlsplit(path)
    assert parts.scheme == "" and parts.netloc == ""  # relative: the frontend adds the base URL
    assert parts.path == f"/api/v1/analyses/{ANALYSIS_ID}/image"
    assert [key for key, _ in (pair.split("=") for pair in parts.query.split("&"))] == [
        "variant",
        "exp",
        "sig",
    ]


def test_signed_path_carries_a_verifiable_signature_and_the_configured_ttl() -> None:
    path = signed_image_path(ANALYSIS_ID, "original", SETTINGS, now=NOW)
    query = parse_qs(urlsplit(path).query)
    assert query["variant"] == ["original"]
    exp = int(query["exp"][0])
    assert exp == int(NOW.timestamp()) + SETTINGS.signed_url_ttl_seconds
    verify(ANALYSIS_ID, "original", exp, query["sig"][0], SETTINGS, now=NOW)
    verify(
        ANALYSIS_ID, "original", exp, query["sig"][0], SETTINGS, now=NOW + timedelta(seconds=3600)
    )
    with pytest.raises(InvalidSignatureError):
        verify(
            ANALYSIS_ID,
            "original",
            exp,
            query["sig"][0],
            SETTINGS,
            now=NOW + timedelta(seconds=3601),
        )


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_variant_gets_a_working_link(variant: str) -> None:
    query = parse_qs(urlsplit(signed_image_path(ANALYSIS_ID, variant, SETTINGS, now=NOW)).query)
    verify(ANALYSIS_ID, variant, int(query["exp"][0]), query["sig"][0], SETTINGS, now=NOW)


def test_signed_path_honours_the_api_prefix_and_ttl_settings() -> None:
    settings = make_settings(api_v1_prefix="/v2/", signed_url_ttl_seconds=60)
    path = signed_image_path(ANALYSIS_ID, "result", settings, now=NOW)
    assert path.startswith(f"/v2/analyses/{ANALYSIS_ID}/image?")
    assert parse_qs(urlsplit(path).query)["exp"] == [str(int(NOW.timestamp()) + 60)]


def test_signed_path_defaults_to_the_current_time() -> None:
    before = int(datetime.now(UTC).timestamp())
    path = signed_image_path(str(ANALYSIS_ID), "original", SETTINGS)
    exp = int(parse_qs(urlsplit(path).query)["exp"][0])
    assert before + 3600 <= exp <= before + 3602


def test_signed_path_rejects_unknown_variants() -> None:
    with pytest.raises(ValueError):
        signed_image_path(ANALYSIS_ID, "full", SETTINGS, now=NOW)


def test_the_signature_needs_no_url_escaping() -> None:
    path = signed_image_path(ANALYSIS_ID, "original", SETTINGS, now=NOW)
    assert "%" not in path  # base64url alphabet is URL-safe as-is
