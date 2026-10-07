"""scripts/dev_server_sqlite.py: builds the app + migrated SQLite DB without opening a port."""

from __future__ import annotations

import io
import json
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from scripts import dev_server_sqlite
from scripts.dev_server_sqlite import (
    DevServerError,
    build_dev_app,
    configure_environment,
    sample_model_keys,
)
from scripts.seed_samples import SAMPLES_ROOT, load_manifest
from sqlalchemy import inspect, text

from app.core.config import get_settings
from app.ml.registry import ModelRegistry
from tests.fakes import FakePredictor

CLIENT_ID = "7c3f1b0e-2a44-4f6e-9d3b-0d7f6a9b1c21"

# Not hard-coded: the committed manifests (and which model keys even have one) change
# independently of this test file (e.g. a new model, or placeholder photos swapped for
# real ones, spec 13).
TOTAL_COMMITTED_SAMPLES = sum(len(load_manifest(SAMPLES_ROOT / key)) for key in sample_model_keys())


@pytest.fixture
def dev_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A throwaway dev folder, with the process environment protected.

    ``build_dev_app`` writes DATA_ROOT/CORS_ORIGINS into os.environ (that is its job);
    registering them with monkeypatch first makes pytest restore them afterwards.
    """
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "unused"))
    # setenv registers the variable for restoration at teardown (whether or not it was set
    # before); delenv then leaves it unset so the dev server has to fill it in itself.
    monkeypatch.setenv("CORS_ORIGINS", "[]")
    monkeypatch.delenv("CORS_ORIGINS")
    get_settings.cache_clear()
    yield tmp_path / "dev"
    get_settings.cache_clear()


@pytest.fixture
def dev_registry(fake_registry: ModelRegistry) -> ModelRegistry:
    return fake_registry


class TestConfigureEnvironment:
    def test_data_root_is_always_forced_into_the_dev_folder(self, tmp_path: Path) -> None:
        env = {"DATA_ROOT": "C:/kera-data"}
        configure_environment(tmp_path / "dev", env)
        assert env["DATA_ROOT"] == str(tmp_path / "dev" / "data")

    def test_cors_defaults_allow_both_spellings_of_the_vite_origin(self, tmp_path: Path) -> None:
        env: dict[str, str] = {}
        configure_environment(tmp_path / "dev", env)
        assert json.loads(env["CORS_ORIGINS"]) == [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]

    def test_an_explicit_cors_setting_in_the_shell_wins(self, tmp_path: Path) -> None:
        env = {"CORS_ORIGINS": '["http://localhost:5173"]'}
        configure_environment(tmp_path / "dev", env)
        assert env["CORS_ORIGINS"] == '["http://localhost:5173"]'

    def test_the_settings_cache_is_dropped(self, tmp_path: Path) -> None:
        before = get_settings()
        configure_environment(tmp_path / "dev", {})
        assert get_settings() is not before


class TestBuildDevApp:
    def test_migrations_are_applied_and_the_api_serves(
        self, dev_dir: Path, dev_registry: ModelRegistry
    ) -> None:
        dev = build_dev_app(dev_dir, registry=dev_registry)
        try:
            # Real Alembic migrations, not create_all: the version table proves it.
            names = set(inspect(dev.engine).get_table_names())
            assert {"analyses", "alembic_version"} <= names
            with dev.engine.connect() as connection:
                version = connection.execute(text("SELECT version_num FROM alembic_version"))
                assert version.scalar_one() == "0001"
            assert (dev_dir / "dev.sqlite3").is_file()

            with TestClient(dev.app) as client:
                assert client.get("/health").json() == {"status": "ok"}
                response = client.get("/api/v1/models")
                assert response.status_code == 200
                keys = {m["key"] for m in response.json()}
                assert {"tree_classification", "leaf_segmentation", "leaf_disease"} <= keys
                assert client.get("/health/ready").status_code == 200
        finally:
            dev.engine.dispose()

    def test_images_go_under_the_dev_folder_never_the_real_data_root(
        self, dev_dir: Path, dev_registry: ModelRegistry
    ) -> None:
        dev = build_dev_app(dev_dir, registry=dev_registry)
        try:
            assert Path(dev.settings.data_root) == dev_dir / "data"
            assert dev.storage.root == (dev_dir / "data").resolve()
            assert (dev_dir / "data").is_dir()
        finally:
            dev.engine.dispose()

    def test_cors_allows_the_frontend_dev_origins_only(
        self, dev_dir: Path, dev_registry: ModelRegistry
    ) -> None:
        dev = build_dev_app(dev_dir, registry=dev_registry)
        try:
            with TestClient(dev.app) as client:

                def preflight(origin: str) -> Any:
                    return client.options(
                        "/api/v1/analyses",
                        headers={
                            "Origin": origin,
                            "Access-Control-Request-Method": "GET",
                            "Access-Control-Request-Headers": "X-Client-Id",
                        },
                    )

                for origin in ("http://localhost:3000", "http://127.0.0.1:3000"):
                    assert preflight(origin).headers["access-control-allow-origin"] == origin
                assert "access-control-allow-origin" not in preflight("http://evil.example").headers
        finally:
            dev.engine.dispose()

    def test_rebuilding_an_existing_dev_folder_keeps_its_data(
        self, dev_dir: Path, dev_registry: ModelRegistry
    ) -> None:
        first = build_dev_app(dev_dir, registry=dev_registry)
        dev_server_sqlite.seed_all(first)
        first.engine.dispose()

        second = build_dev_app(dev_dir, registry=dev_registry)  # migrations are a no-op now
        try:
            with TestClient(second.app) as client:
                page = client.get(
                    "/api/v1/analyses",
                    params={"scope": "samples"},
                    headers={"X-Client-Id": CLIENT_ID},
                ).json()
            assert page["total"] == TOTAL_COMMITTED_SAMPLES
        finally:
            second.engine.dispose()

    def test_reset_starts_from_scratch(self, dev_dir: Path, dev_registry: ModelRegistry) -> None:
        first = build_dev_app(dev_dir, registry=dev_registry)
        dev_server_sqlite.seed_all(first)
        first.engine.dispose()
        leftover = dev_dir / "data" / "leftover.txt"
        leftover.write_text("old")

        second = build_dev_app(dev_dir, reset=True, registry=dev_registry)
        try:
            assert not leftover.exists()
            with TestClient(second.app) as client:
                page = client.get(
                    "/api/v1/analyses",
                    params={"scope": "samples"},
                    headers={"X-Client-Id": CLIENT_ID},
                ).json()
            assert page["total"] == 0
        finally:
            second.engine.dispose()

    def test_reset_refuses_a_folder_it_did_not_create(self, tmp_path: Path) -> None:
        precious = tmp_path / "precious"
        precious.mkdir()
        (precious / "important.txt").write_text("keep me")

        with pytest.raises(DevServerError, match="Refusing to delete"):
            dev_server_sqlite.reset_dev_dir(precious)

        assert (precious / "important.txt").read_text() == "keep me"

    def test_reset_of_a_missing_folder_is_fine(self, tmp_path: Path) -> None:
        dev_server_sqlite.reset_dev_dir(tmp_path / "never-existed")


class TestSeedAll:
    def test_seeds_every_committed_sample_set_through_the_real_pipeline(
        self, dev_dir: Path, dev_registry: ModelRegistry
    ) -> None:
        dev = build_dev_app(dev_dir, registry=dev_registry)
        try:
            outcomes = dev_server_sqlite.seed_all(dev)

            assert [o.model_key for o in outcomes] == sample_model_keys()
            assert all(o.report is not None and o.report.ok for o in outcomes)
            with TestClient(dev.app) as client:
                page = client.get(
                    "/api/v1/analyses",
                    params={"scope": "samples"},
                    headers={"X-Client-Id": CLIENT_ID},
                ).json()
                assert page["total"] == TOTAL_COMMITTED_SAMPLES
                for item in page["items"]:
                    assert item["is_sample"] is True
                    thumbnail = client.get(item["thumbnail_url"])
                    assert thumbnail.status_code == 200
                    assert thumbnail.headers["content-type"] == "image/webp"
        finally:
            dev.engine.dispose()

    def test_a_set_that_cannot_start_is_reported_not_fatal(
        self, dev_dir: Path, tmp_path: Path
    ) -> None:
        only_tree = ModelRegistry.from_predictors([FakePredictor("tree_classification")])
        dev = build_dev_app(dev_dir, registry=only_tree)
        try:
            outcomes = {o.model_key: o for o in dev_server_sqlite.seed_all(dev)}
            assert outcomes["tree_classification"].report is not None
            broken = outcomes["leaf_segmentation"]
            assert broken.report is None and "unavailable" in (broken.error or "")
        finally:
            dev.engine.dispose()

    def test_sample_model_keys_ignores_folders_without_a_manifest(self, tmp_path: Path) -> None:
        (tmp_path / "with").mkdir()
        (tmp_path / "with" / "manifest.json").write_text("[]")
        (tmp_path / "without").mkdir()
        assert dev_server_sqlite.sample_model_keys(tmp_path) == ["with"]
        assert dev_server_sqlite.sample_model_keys(tmp_path / "missing") == []


class TestBanner:
    def test_says_dev_only_and_shows_where_things_live(
        self, dev_dir: Path, dev_registry: ModelRegistry
    ) -> None:
        dev = build_dev_app(dev_dir, registry=dev_registry)
        try:
            text_ = dev_server_sqlite.banner(dev, 8123)
        finally:
            dev.engine.dispose()
        assert "DEV ONLY" in text_ and "THROWAWAY SQLite" in text_
        assert "http://127.0.0.1:8123" in text_ and "/docs" in text_
        assert "http://localhost:3000" in text_ and "http://127.0.0.1:3000" in text_
        assert str(dev_dir) in text_
        assert "DUMMY WEIGHTS" not in text_  # the fakes are not placeholders

    def test_dummy_weights_are_called_out(self, dev_dir: Path) -> None:
        dummy = ModelRegistry.from_predictors(
            [FakePredictor("tree_classification", is_placeholder=True)]
        )
        dev = build_dev_app(dev_dir, registry=dummy)
        try:
            text_ = dev_server_sqlite.banner(dev, 8000)
        finally:
            dev.engine.dispose()
        assert "DUMMY WEIGHTS in use for: tree_classification" in text_
        assert "NOT real predictions" in text_


class TestRunAndMain:
    @pytest.fixture
    def serve(
        self,
        dev_dir: Path,
        dev_registry: ModelRegistry,
        monkeypatch: pytest.MonkeyPatch,
    ) -> dict[str, Any]:
        """Replace uvicorn and the model loader so ``run`` can be exercised without a port."""
        calls: dict[str, Any] = {}

        def fake_run(app: Any, **kwargs: Any) -> None:
            calls["app"] = app
            calls["kwargs"] = kwargs

        monkeypatch.setitem(sys.modules, "uvicorn", types.SimpleNamespace(run=fake_run))
        monkeypatch.setattr(dev_server_sqlite, "build_registry", lambda settings: dev_registry)
        monkeypatch.setattr(dev_server_sqlite, "DEV_DIR", dev_dir)
        return calls

    def run(self, *argv: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        args = dev_server_sqlite.build_parser().parse_args(list(argv))
        code = dev_server_sqlite.run(args, out, err)
        return code, out.getvalue(), err.getvalue()

    def test_serves_on_loopback_only_with_one_process(self, serve: dict[str, Any]) -> None:
        code, out, err = self.run("--port", "8123")

        assert code == 0 and err == ""
        assert serve["kwargs"]["host"] == "127.0.0.1"
        assert serve["kwargs"]["port"] == 8123
        assert "reload" not in serve["kwargs"] and "workers" not in serve["kwargs"]
        assert "DEV ONLY" in out
        assert "seeded" not in out  # no --seed

    def test_seed_runs_before_serving_and_prints_the_reports(self, serve: dict[str, Any]) -> None:
        code, out, _ = self.run("--seed")

        assert code == 0
        assert "Samples for tree_classification" in out
        assert "Samples for leaf_segmentation" in out
        assert out.count("SEEDED") == TOTAL_COMMITTED_SAMPLES
        assert serve["kwargs"]["port"] == 8000

    def test_seed_problems_are_loud_but_the_server_still_starts(
        self,
        dev_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        serve: dict[str, Any],
    ) -> None:
        down = ModelRegistry.from_predictors([FakePredictor("tree_classification")])
        monkeypatch.setattr(dev_server_sqlite, "build_registry", lambda settings: down)

        code, _, err = self.run("--seed")

        assert code == 0 and "kwargs" in serve
        assert "WARNING: samples for leaf_segmentation NOT seeded" in err

    def test_reset_flag_wipes_the_dev_folder(self, dev_dir: Path, serve: dict[str, Any]) -> None:
        dev_dir.mkdir(parents=True)
        (dev_dir / dev_server_sqlite._MARKER).write_text("x")
        (dev_dir / "stale.txt").write_text("old")

        code, _, _ = self.run("--reset")

        assert code == 0 and not (dev_dir / "stale.txt").exists()

    def test_reset_of_a_foreign_folder_is_an_error_exit(
        self, dev_dir: Path, serve: dict[str, Any]
    ) -> None:
        dev_dir.mkdir(parents=True)
        (dev_dir / "foreign.txt").write_text("not ours")

        code, _, err = self.run("--reset")

        assert code == 2 and "Refusing to delete" in err
        assert (dev_dir / "foreign.txt").exists() and "kwargs" not in serve

    @pytest.mark.parametrize("port", ["0", "70000", "-1"])
    def test_a_bad_port_is_rejected_before_anything_is_created(
        self, dev_dir: Path, serve: dict[str, Any], port: str
    ) -> None:
        code, _, err = self.run("--port", port)
        assert code == 2 and "--port must be between 1 and 65535" in err
        assert not dev_dir.exists()

    def test_invalid_settings_give_an_actionable_message_without_echoing_values(
        self,
        serve: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
        dev_dir: Path,
    ) -> None:
        monkeypatch.setenv("SIGNING_SECRET", "too-short-secret-value")
        code, _, err = self.run()
        assert code == 2
        assert "signing_secret" in err and ".env.example" in err
        assert "too-short-secret-value" not in err

    def test_main_parses_argv(
        self, serve: dict[str, Any], capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert dev_server_sqlite.main(["--port", "8222"]) == 0
        assert serve["kwargs"]["port"] == 8222
        assert "DEV ONLY" in capsys.readouterr().out
