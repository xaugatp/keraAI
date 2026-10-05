"""Schema pieces shared by several endpoints: errors, paging, location, enums.

The allowed-value Literals below are written out (a type checker cannot expand a
runtime tuple into a ``Literal``), but each is pinned to the single source of
truth — the constants in ``app.db`` / ``app.core.signing`` / the repository — by
a unit test (``tests/unit/schemas``), so the DB and the API cannot drift apart.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Generic, Literal, TypeVar

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from app.core.time import to_aware_utc
from app.repositories.analysis_repository import AnalysisScope

# --- enums -----------------------------------------------------------------

ModelKey = Literal["tree_classification", "leaf_segmentation", "leaf_disease"]
AnalysisStatus = Literal["completed", "failed"]
AnalysisSource = Literal["upload", "camera", "sample"]
# `sample` rows are created by the seed script only (D-07); a client may never
# claim that source, so the upload form accepts just these two.
ClientSource = Literal["upload", "camera"]
ImageVariant = Literal["original", "thumbnail", "result"]
# "mine" | "samples" | "all_visible" — defined next to the query that implements it.
Scope = AnalysisScope

# --- datetimes -------------------------------------------------------------

# The database holds UTC-*naive* datetimes (spec 5.2). Any datetime that enters
# a response schema is converted to aware UTC here, at the boundary, so JSON
# always carries an explicit "Z" and no caller can forget to do it.
UtcDatetime = Annotated[datetime, AfterValidator(to_aware_utc)]

# --- generics --------------------------------------------------------------

T = TypeVar("T")


class ProblemDetail(BaseModel):
    """RFC 9457 application/problem+json body shape (for OpenAPI docs)."""

    type: str
    title: str
    status: int
    detail: str
    instance: str
    code: str
    request_id: str | None = None
    errors: list[dict[str, str]] | None = None


class Page(BaseModel, Generic[T]):
    """One page of a listing (spec 10)."""

    items: list[T]
    page: int = Field(ge=1, description="1-based page number")
    page_size: int = Field(ge=1, le=100)
    total: int = Field(ge=0, description="Number of matching items over all pages")
    total_pages: int = Field(ge=0, description="0 when there are no matching items")


class GeoLocation(BaseModel):
    """Where and when the photo was taken, as reported by the client's device.

    Every field is optional because the client may send any subset; the whole
    block is ``null`` on an analysis when it sent none of them.
    """

    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    accuracy_m: float | None = Field(default=None, ge=0, description="GPS accuracy in metres")
    captured_at: UtcDatetime | None = Field(
        default=None, description="Client-reported capture time (UTC)"
    )


class ModelInfoOut(BaseModel):
    """One entry of ``GET /models`` — ``app.ml.base.ModelInfo`` plus ``is_placeholder``."""

    model_config = ConfigDict(from_attributes=True)

    key: ModelKey
    display_name: str
    version: str
    task: str
    classes: list[str]
    status: Literal["ready", "unavailable"]
    reason: str | None = Field(
        default=None, description="Why the model is unavailable (never contains file paths)"
    )
    is_placeholder: bool = Field(
        description="True while the weights are a randomly-initialised stand-in (not real results)"
    )
