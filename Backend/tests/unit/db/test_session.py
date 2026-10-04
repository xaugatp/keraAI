from __future__ import annotations

import logging
import threading
from pathlib import Path

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.db.session import (
    build_database_url,
    create_db_engine,
    create_session_factory,
    get_db,
    ping_database,
)
from tests.unit.db.factories import make_analysis


def _settings(**overrides: object) -> Settings:
    kwargs: dict[str, object] = {
        "signing_secret": "x" * 40,
        "data_root": "C:/kera-data-test",
        # Pin every DB field so a developer's real .env cannot change the result.
        "db_host": "localhost",
        "db_port": None,
        "db_name": "KeraAI",
        "db_driver": "ODBC Driver 18 for SQL Server",
        "db_trusted_connection": False,
        "db_user": "kera_app",
        "db_password": "p@ss:w/ord",
        "db_trust_server_cert": True,
        "db_pool_size": 7,
        "db_echo": False,
    }
    kwargs.update(overrides)
    return Settings(**kwargs)  # type: ignore[arg-type]


# --- build_database_url ------------------------------------------------------


def test_sql_login_url_has_credentials_and_escapes_password() -> None:
    url = build_database_url(_settings())
    assert url.drivername == "mssql+pyodbc"
    assert url.username == "kera_app"
    assert url.password == "p@ss:w/ord"
    assert url.database == "KeraAI"
    # A naive f-string URL would break on '@' ':' '/' in the password.
    rendered = url.render_as_string(hide_password=False)
    assert "p%40ss%3Aw%2Ford" in rendered
    assert "p@ss:w/ord" not in url.render_as_string(hide_password=True)
    assert "Trusted_Connection" not in url.query


def test_windows_auth_url_drops_credentials() -> None:
    url = build_database_url(_settings(db_trusted_connection=True))
    assert url.username is None
    assert url.password is None
    assert url.query["Trusted_Connection"] == "yes"


def test_trust_server_certificate_follows_setting() -> None:
    assert build_database_url(_settings()).query["TrustServerCertificate"] == "yes"
    untrusting = build_database_url(_settings(db_trust_server_cert=False))
    assert untrusting.query["TrustServerCertificate"] == "no"


def test_pyodbc_connect_string_for_named_instance_and_port() -> None:
    # Prove the URL turns into the ODBC string we expect, without connecting.
    named = build_database_url(
        _settings(db_host="localhost\\SQLEXPRESS", db_trusted_connection=True)
    )
    engine = create_db_engine(_settings(), url=named)
    args, _ = engine.dialect.create_connect_args(named)
    connection_string = args[0]
    assert "DRIVER={ODBC Driver 18 for SQL Server}" in connection_string
    assert "Server=localhost\\SQLEXPRESS" in connection_string
    assert "Database=KeraAI" in connection_string
    assert "Trusted_Connection=yes" in connection_string

    with_port = build_database_url(_settings(db_port=1444))
    ported_args, _ = engine.dialect.create_connect_args(with_port)
    ported = ported_args[0]
    assert "Server=localhost,1444" in ported
    engine.dispose()


# --- create_db_engine / session factory --------------------------------------


def test_engine_for_sql_server_uses_pre_ping_and_pool_size_without_connecting() -> None:
    # Creating the engine must not touch the network — the server may be down.
    engine = create_db_engine(_settings())
    try:
        assert engine.pool._pre_ping is True
        assert engine.pool.size() == 7  # type: ignore[attr-defined]
        assert engine.dialect.name == "mssql"
    finally:
        engine.dispose()


def test_sqlite_override_works_across_threads(sqlite_engine: Engine) -> None:
    result: list[int] = []

    def worker() -> None:
        with sqlite_engine.connect() as connection:
            result.append(connection.execute(text("SELECT 1")).scalar_one())

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert result == [1]
    assert sqlite_engine.pool._pre_ping is True


def test_session_factory_does_not_expire_on_commit(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        row = make_analysis()
        session.add(row)
        session.commit()
        # No refresh SELECT needed: attributes stay loaded after commit.
        assert "model_key" in row.__dict__


def test_create_session_factory_binds_engine(sqlite_engine: Engine) -> None:
    factory = create_session_factory(sqlite_engine)
    with factory() as session:
        assert session.get_bind() is sqlite_engine


# --- get_db ------------------------------------------------------------------


def test_get_db_rolls_back_uncommitted_work_and_closes(
    session_factory: sessionmaker[Session],
) -> None:
    generator = get_db(session_factory)
    session = next(generator)
    session.add(make_analysis())
    session.flush()
    assert session.in_transaction()

    generator.close()  # what FastAPI does when the request ends
    assert not session.in_transaction()

    with session_factory() as other:
        assert other.execute(text("SELECT COUNT(*) FROM analyses")).scalar_one() == 0


def test_get_db_rolls_back_when_the_request_raises(
    session_factory: sessionmaker[Session],
) -> None:
    generator = get_db(session_factory)
    session = next(generator)
    session.add(make_analysis())
    session.flush()
    with pytest.raises(RuntimeError):
        generator.throw(RuntimeError("handler blew up"))
    assert not session.in_transaction()

    with session_factory() as other:
        assert other.execute(text("SELECT COUNT(*) FROM analyses")).scalar_one() == 0


def test_get_db_keeps_work_the_service_committed(
    session_factory: sessionmaker[Session],
) -> None:
    generator = get_db(session_factory)
    session = next(generator)
    session.add(make_analysis())
    session.commit()  # the service's job
    generator.close()

    with session_factory() as other:
        assert other.execute(text("SELECT COUNT(*) FROM analyses")).scalar_one() == 1


# --- ping_database -----------------------------------------------------------


def test_ping_ok(sqlite_engine: Engine) -> None:
    assert ping_database(sqlite_engine) == (True, None)


def test_ping_failure_is_reported_not_raised_and_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    unreachable = create_db_engine(
        _settings(), url=f"sqlite:///{(tmp_path / 'missing-dir' / 'x.db').as_posix()}"
    )
    try:
        with caplog.at_level(logging.WARNING, logger="app.db.session"):
            ok, detail = ping_database(unreachable)
    finally:
        unreachable.dispose()

    assert ok is False
    # The client-visible detail is generic; the real reason goes to the log only.
    assert detail == "Database unreachable"
    assert "missing-dir" not in (detail or "")
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "OperationalError" in warnings[0].getMessage()
