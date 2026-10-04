"""Portable column types and the clock they pair with.

Spec §5.2: datetimes are stored as UTC-*naive* ``DATETIME2(3)`` and converted to
aware UTC only at the schema boundary. Naive-in-the-DB keeps SQLite (tests) and
SQL Server behaving identically — SQL Server has no tz-aware ``DATETIME2``.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime
from sqlalchemy.dialects import mssql

from app.core.time import utcnow

# Plain DateTime everywhere, DATETIME2(3) on SQL Server (the legacy DATETIME type
# SQLAlchemy would pick otherwise only has ~3.33 ms resolution and a 1753 floor).
UTC_DATETIME = DateTime().with_variant(mssql.DATETIME2(precision=3), "mssql")


def utcnow_naive() -> datetime:
    """Now in UTC, tzinfo stripped, truncated to whole milliseconds.

    Truncating here (instead of letting SQL Server round) means the value we
    hold in Python is exactly what DATETIME2(3) will store, so a row read back
    compares equal on every backend.
    """
    now = utcnow().replace(tzinfo=None)
    return now.replace(microsecond=now.microsecond // 1000 * 1000)
