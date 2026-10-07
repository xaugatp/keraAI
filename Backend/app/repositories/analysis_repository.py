from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Final, Literal

from sqlalchemy import ColumnElement, false, func, or_, select, true
from sqlalchemy.orm import Session

from app.db.models.analysis import Analysis
from app.db.types import utcnow_naive

MAX_PAGE_SIZE: Final = 100

# mine        -> rows owned by `client_id`
# samples     -> public sample rows
# all_visible -> samples plus the caller's own rows
AnalysisScope = Literal["mine", "samples", "all_visible"]

# Module-level alias: inside AnalysisRepository the method named `list` shadows the
# builtin, which would break `list[Analysis]` in later annotations (mypy).
AnalysisRows = list[Analysis]
Conditions = list[ColumnElement[bool]]


@dataclass(frozen=True)
class AnalysisFilter:
    model_key: str | None = None
    scope: AnalysisScope = "all_visible"
    client_id: uuid.UUID | None = None
    status: str | None = None


class AnalysisRepository:
    """Persistence for ``Analysis`` rows. Pure queries — no business rules.

    Who may *see* a given row (and the 404-instead-of-403 policy) is decided in
    the service (spec §5.3). Scope filtering below is just query construction:
    "which rows match this client_id / is_sample combination".

    Nothing here commits; the caller owns the transaction.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, analysis: Analysis) -> None:
        self._session.add(analysis)

    def get(self, analysis_id: uuid.UUID) -> Analysis | None:
        """Fetch by id; soft-deleted rows are treated as not found."""
        stmt = select(Analysis).where(Analysis.id == analysis_id, Analysis.deleted_at.is_(None))
        return self._session.scalars(stmt).first()

    def list(
        self, filters: AnalysisFilter, page: int = 1, page_size: int = 20
    ) -> tuple[AnalysisRows, int]:
        """One page of matching rows (newest first) and the total match count.

        ``page`` is 1-based. Out-of-range input is clamped rather than rejected:
        the HTTP layer validates and returns 422, this just stays safe.
        """
        page = max(page, 1)
        page_size = min(max(page_size, 1), MAX_PAGE_SIZE)
        conditions = self._conditions(filters)

        total = self._session.scalar(select(func.count()).select_from(Analysis).where(*conditions))

        # `id` is a tiebreaker: rows created in the same millisecond would otherwise
        # have no defined order, and OFFSET paging could repeat or skip them.
        stmt = (
            select(Analysis)
            .where(*conditions)
            .order_by(Analysis.created_at.desc(), Analysis.id.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        return list(self._session.scalars(stmt)), total or 0

    def soft_delete(self, analysis: Analysis) -> None:
        """Mark deleted (P-07). The row and its files stay; reads hide it."""
        analysis.deleted_at = utcnow_naive()

    def get_sample_by_hash(self, model_key: str, sha256: str) -> Analysis | None:
        """Find an existing, non-deleted sample for this image — seed idempotency."""
        stmt = select(Analysis).where(
            Analysis.is_sample == true(),
            Analysis.model_key == model_key,
            Analysis.image_sha256 == sha256,
            Analysis.deleted_at.is_(None),
        )
        return self._session.scalars(stmt).first()

    @staticmethod
    def _conditions(filters: AnalysisFilter) -> Conditions:
        conditions: Conditions = [Analysis.deleted_at.is_(None)]

        if filters.model_key is not None:
            conditions.append(Analysis.model_key == filters.model_key)
        if filters.status is not None:
            conditions.append(Analysis.status == filters.status)

        # `client_id == None` would compile to `IS NULL` — i.e. match every SAMPLE row.
        # So an unknown caller must never reach an equality test: "mine" for nobody is
        # empty, and "all_visible" for nobody degrades to samples only.
        client_id = filters.client_id
        is_sample = Analysis.is_sample == true()
        if filters.scope == "samples":
            conditions.append(is_sample)
        elif filters.scope == "mine":
            conditions.append(false() if client_id is None else Analysis.client_id == client_id)
        else:  # all_visible
            conditions.append(
                is_sample if client_id is None else or_(is_sample, Analysis.client_id == client_id)
            )
        return conditions
