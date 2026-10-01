from __future__ import annotations

from datetime import UTC, datetime


def utcnow() -> datetime:
    """Current time as an aware UTC datetime."""
    return datetime.now(UTC)


def to_aware_utc(value: datetime) -> datetime:
    """Convert to aware UTC.

    Naive datetimes are assumed to already represent UTC — that's how we
    store them in SQL Server's naive DATETIME2 columns.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
