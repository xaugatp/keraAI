from __future__ import annotations

import pytest

from app.core import errors

APP_ERROR_SUBCLASSES = [
    (errors.InvalidImageError, 400, "INVALID_IMAGE"),
    (errors.ImageTooSmallError, 400, "IMAGE_TOO_SMALL"),
    (errors.ImageTooLargeDimensionsError, 400, "IMAGE_TOO_LARGE_DIMENSIONS"),
    (errors.MissingClientIdError, 400, "MISSING_CLIENT_ID"),
    (errors.InvalidClientIdError, 400, "INVALID_CLIENT_ID"),
    (errors.InvalidSignatureError, 403, "INVALID_SIGNATURE"),
    (errors.SampleImmutableError, 403, "SAMPLE_IMMUTABLE"),
    (errors.AnalysisNotFoundError, 404, "ANALYSIS_NOT_FOUND"),
    (errors.PayloadTooLargeError, 413, "IMAGE_TOO_LARGE"),
    (errors.UnsupportedMediaTypeError, 415, "UNSUPPORTED_MEDIA_TYPE"),
    (errors.InferenceError, 500, "INFERENCE_FAILED"),
    (errors.ModelUnavailableError, 503, "MODEL_UNAVAILABLE"),
    (errors.DatabaseUnavailableError, 503, "DATABASE_UNAVAILABLE"),
]


@pytest.mark.parametrize(("error_cls", "status_code", "code"), APP_ERROR_SUBCLASSES)
def test_app_error_subclass_shape(
    error_cls: type[errors.AppError], status_code: int, code: str
) -> None:
    assert error_cls.status_code == status_code
    assert error_cls.code == code
    assert error_cls().detail == error_cls.title


def test_app_error_detail_defaults_to_title() -> None:
    exc = errors.AnalysisNotFoundError()
    assert exc.detail == "Analysis not found"
    assert exc.extra == {}


def test_app_error_accepts_custom_detail_and_extra() -> None:
    exc = errors.InvalidImageError("corrupt JPEG", extra={"field": "image"})
    assert exc.detail == "corrupt JPEG"
    assert exc.extra == {"field": "image"}
