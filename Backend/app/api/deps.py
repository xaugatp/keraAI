from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db import session as db_session

SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_db(request: Request) -> Iterator[Session]:
    """Per-request DB session (the factory is built in the app lifespan).

    The service commits; this dependency only guarantees rollback + close.
    """
    yield from db_session.get_db(request.app.state.session_factory)


DbSession = Annotated[Session, Depends(get_db)]
