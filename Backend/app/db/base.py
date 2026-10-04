from __future__ import annotations

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

# Deterministic constraint/index names. Without a convention SQL Server invents
# names like PK__analyses__3213E83F..., which differ per database and make
# Alembic downgrades (DROP CONSTRAINT <name>) impossible to write portably.
# `ck_` relies on %(constraint_name)s, so every CheckConstraint must be given
# a short explicit `name=` (the table name is prefixed automatically).
NAMING_CONVENTION: dict[str, str] = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)
