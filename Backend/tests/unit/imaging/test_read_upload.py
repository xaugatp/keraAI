from __future__ import annotations

import asyncio
import io

import pytest
from starlette.datastructures import UploadFile

from app.core.errors import PayloadTooLargeError
from app.imaging.processing import max_upload_bytes, read_upload_limited
from tests.fixtures.settings import make_settings

MB = 1024 * 1024


class FakeUpload:
    """Async reader that records how much of the body was actually consumed."""

    def __init__(self, data: bytes) -> None:
        self._stream = io.BytesIO(data)
        self.bytes_served = 0
        self.read_sizes: list[int] = []

    async def read(self, size: int = -1, /) -> bytes:
        self.read_sizes.append(size)
        chunk = self._stream.read(size)
        self.bytes_served += len(chunk)
        return chunk


def _read(upload: FakeUpload, limit: int, content_length: int | None = None) -> bytes:
    return asyncio.run(read_upload_limited(upload, limit, content_length))


def test_returns_whole_body_under_the_limit_in_1mb_chunks() -> None:
    body = bytes(range(256)) * (3 * MB // 256) + b"tail"
    upload = FakeUpload(body)
    assert _read(upload, 10 * MB) == body
    assert max(upload.read_sizes) <= MB


def test_empty_body_is_returned_as_empty_bytes() -> None:
    assert _read(FakeUpload(b""), MB) == b""


def test_body_exactly_at_the_limit_is_accepted() -> None:
    body = b"x" * (2 * MB)
    assert _read(FakeUpload(body), 2 * MB) == body


def test_one_byte_over_the_limit_is_rejected() -> None:
    with pytest.raises(PayloadTooLargeError) as exc_info:
        _read(FakeUpload(b"x" * (2 * MB + 1)), 2 * MB)
    assert exc_info.value.status_code == 413
    assert exc_info.value.code == "IMAGE_TOO_LARGE"


def test_oversized_stream_is_aborted_mid_stream_not_read_to_the_end() -> None:
    upload = FakeUpload(b"x" * (5 * MB))
    with pytest.raises(PayloadTooLargeError):
        _read(upload, 2 * MB)
    # Detected after at most one byte past the limit: the other 3 MB were never read.
    assert upload.bytes_served == 2 * MB + 1


def test_content_length_over_the_limit_is_rejected_before_reading_anything() -> None:
    upload = FakeUpload(b"x" * 100)
    with pytest.raises(PayloadTooLargeError):
        _read(upload, MB, content_length=MB + 1)
    assert upload.read_sizes == []


def test_content_length_equal_to_the_limit_is_not_rejected_early() -> None:
    body = b"y" * MB
    assert _read(FakeUpload(body), MB, content_length=MB) == body


def test_a_lying_content_length_cannot_bypass_the_streaming_limit() -> None:
    upload = FakeUpload(b"x" * (3 * MB))
    with pytest.raises(PayloadTooLargeError):
        _read(upload, MB, content_length=10)


def test_error_message_states_the_limit_in_megabytes() -> None:
    with pytest.raises(PayloadTooLargeError) as exc_info:
        _read(FakeUpload(b"x" * (MB + 1)), MB)
    assert exc_info.value.detail == "Image exceeds the 1 MB limit."


def test_max_upload_bytes_is_derived_from_settings() -> None:
    assert max_upload_bytes(make_settings(max_upload_mb=10)) == 10 * MB


def test_works_with_a_real_starlette_upload_file() -> None:
    body = b"z" * (MB + 123)
    upload = UploadFile(file=io.BytesIO(body), filename="leaf.jpg")
    assert asyncio.run(read_upload_limited(upload, 2 * MB)) == body
