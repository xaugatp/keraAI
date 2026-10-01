from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

from app.core.time import to_aware_utc, utcnow


def test_utcnow_is_aware_and_utc() -> None:
    now = utcnow()
    assert now.tzinfo is not None
    assert now.utcoffset() == timedelta(0)


def test_to_aware_utc_treats_naive_as_already_utc() -> None:
    naive = datetime(2026, 10, 1, 12, 0, 0)
    converted = to_aware_utc(naive)
    assert converted.tzinfo is UTC
    assert converted.hour == 12


def test_to_aware_utc_converts_other_timezones() -> None:
    plus_five = datetime(2026, 10, 1, 17, 0, 0, tzinfo=timezone(timedelta(hours=5)))
    converted = to_aware_utc(plus_five)
    assert converted.tzinfo is UTC
    assert converted.hour == 12
