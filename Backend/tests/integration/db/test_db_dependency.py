from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from app.api.deps import DbSession
from app.db.models.analysis import Analysis
from tests.unit.db.factories import make_analysis


def _count(factory: sessionmaker[Session]) -> int:
    with factory() as session:
        return session.scalar(select(func.count()).select_from(Analysis)) or 0


def test_dependency_provides_a_working_session(
    app: FastAPI, session_factory: sessionmaker[Session]
) -> None:
    @app.get("/__test__/db-count")
    def read_count(db: DbSession) -> dict[str, int]:
        return {"count": db.scalar(select(func.count()).select_from(Analysis)) or 0}

    with TestClient(app) as client:
        assert client.get("/__test__/db-count").json() == {"count": 0}


def test_dependency_leaves_commit_to_the_handler(
    app: FastAPI, session_factory: sessionmaker[Session]
) -> None:
    @app.post("/__test__/db-forgot-commit")
    def forgot_commit(db: DbSession) -> dict[str, str]:
        db.add(make_analysis())
        db.flush()
        return {"ok": "yes"}

    @app.post("/__test__/db-commit")
    def commit(db: DbSession) -> dict[str, str]:
        db.add(make_analysis())
        db.commit()
        return {"ok": "yes"}

    with TestClient(app) as client:
        assert client.post("/__test__/db-forgot-commit").status_code == 200
        assert _count(session_factory) == 0  # rolled back by the dependency

        assert client.post("/__test__/db-commit").status_code == 200
        assert _count(session_factory) == 1


def test_dependency_rolls_back_and_closes_when_the_handler_raises(
    app: FastAPI, session_factory: sessionmaker[Session]
) -> None:
    captured: list[Session] = []

    @app.post("/__test__/db-boom")
    def boom(db: DbSession) -> None:
        captured.append(db)
        db.add(make_analysis())
        db.flush()
        raise RuntimeError("handler failed")

    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.post("/__test__/db-boom").status_code == 500

    assert _count(session_factory) == 0
    assert not captured[0].in_transaction()  # connection went back to the pool


def test_each_request_gets_its_own_session(app: FastAPI) -> None:
    sessions: list[Session] = []

    @app.get("/__test__/db-objects")
    def objects(db: DbSession) -> dict[str, bool]:
        sessions.append(db)
        return {"ok": True}

    with TestClient(app) as client:
        client.get("/__test__/db-objects")
        client.get("/__test__/db-objects")

    assert len(sessions) == 2
    assert sessions[0] is not sessions[1]
