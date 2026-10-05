"""scripts/check_env.py logic, with the environment faked (no torch, no SQL Server)."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pytest
from scripts import check_env
from scripts.check_env import FAIL, OK, WARN, Check
from sqlalchemy import create_engine, text

from app.ml.base import ModelLoadError
from tests.fakes import FakePredictor
from tests.fixtures.settings import make_settings


def write_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("a/data.pkl", b"x")


class TestSmallHelpers:
    def test_mask_hides_secrets(self):
        assert check_env._mask("login failed for pw=hunter2!", ("hunter2",)) == (
            "login failed for pw=***!"
        )

    def test_mask_ignores_empty_secret(self):
        assert check_env._mask("abc", ("",)) == "abc"

    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            ("Login failed for user 'x'. (18456)", "001_create_database_and_login.sql"),
            ('Cannot open database "KeraAI" requested', "001_create_database_and_login.sql"),
            ("Data source name not found", "DB_DRIVER"),
            ("Named Pipes Provider: Could not open a connection", "TCP/IP"),
        ],
    )
    def test_db_fix_picks_a_relevant_remedy(self, message, expected):
        assert expected in check_env._db_fix(message)


class TestPython:
    def test_reports_version_and_venv_rows(self):
        names = [c.name for c in check_env.check_python()]
        assert names == ["Python version", "Virtual environment"]

    def test_expected_version_and_venv_pass(self):
        rows = check_env.check_python(version=(3, 11, 9), prefix=str(check_env.VENV_DIR))
        assert [r.status for r in rows] == [OK, OK]

    def test_wrong_python_version_is_a_failure_with_a_fix(self):
        version = check_env.check_python(version=(3, 14, 0))[0]
        assert version.status == FAIL and "3.11" in version.fix and "3.14.0" in version.detail

    def test_running_outside_the_venv_is_a_failure(self, tmp_path):
        venv = check_env.check_python(prefix=str(tmp_path))[1]
        assert venv.status == FAIL and "activate" in venv.fix


class TestOdbc:
    def test_driver_present(self, monkeypatch):
        import pyodbc

        monkeypatch.setattr(pyodbc, "drivers", lambda: ["ODBC Driver 18 for SQL Server"])
        assert check_env.check_odbc(make_settings()).status == OK

    def test_driver_missing_lists_what_is_installed(self, monkeypatch):
        import pyodbc

        monkeypatch.setattr(pyodbc, "drivers", lambda: ["SQL Server"])
        result = check_env.check_odbc(make_settings())
        assert result.status == FAIL
        assert "SQL Server" in result.detail
        assert "aka.ms/downloadmsodbcsql" in result.fix


class TestDatabase:
    def test_missing_database_layer_is_a_failure_not_a_crash(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "app.db.session", None)  # makes the import fail
        (result,) = check_env.check_database(make_settings())
        assert result.status == FAIL
        assert "database layer not present" in result.detail

    def _use_sqlite(self, monkeypatch, tmp_path) -> object:
        import app.db.session as session

        engine = create_engine(f"sqlite:///{(tmp_path / 'check.db').as_posix()}")
        monkeypatch.setattr(session, "create_db_engine", lambda settings: engine)
        return engine

    def _head(self) -> str:
        from alembic.config import Config
        from alembic.script import ScriptDirectory

        config = Config(str(check_env.BACKEND_DIR / "alembic.ini"))
        head = ScriptDirectory.from_config(config).get_current_head()
        assert head is not None
        return head

    def test_connection_ok_and_alembic_at_head(self, monkeypatch, tmp_path):
        engine = self._use_sqlite(monkeypatch, tmp_path)
        with engine.begin() as connection:  # type: ignore[attr-defined]
            connection.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32))"))
            connection.execute(text("INSERT INTO alembic_version VALUES (:v)"), {"v": self._head()})
        checks = check_env.check_database(make_settings())
        assert [c.status for c in checks] == [OK, OK]
        assert "at head" in checks[1].detail

    def test_unmigrated_database_is_a_failure_with_the_upgrade_command(self, monkeypatch, tmp_path):
        self._use_sqlite(monkeypatch, tmp_path)
        checks = check_env.check_database(make_settings())
        assert checks[0].status == OK
        assert checks[1].status == FAIL
        assert "no migration" in checks[1].detail
        assert "alembic upgrade head" in checks[1].fix

    def test_database_behind_head_is_a_failure(self, monkeypatch, tmp_path):
        engine = self._use_sqlite(monkeypatch, tmp_path)
        with engine.begin() as connection:  # type: ignore[attr-defined]
            connection.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32))"))
            connection.execute(text("INSERT INTO alembic_version VALUES ('0000_ancient')"))
        checks = check_env.check_database(make_settings())
        assert checks[1].status == FAIL
        assert "0000_ancient" in checks[1].detail

    def test_unreachable_database_masks_the_password(self, monkeypatch, tmp_path):
        import app.db.session as session

        broken = create_engine(
            f"sqlite:///{(tmp_path / 'no' / 'such' / 'dir' / 'x.db').as_posix()}"
        )
        monkeypatch.setattr(session, "create_db_engine", lambda settings: broken)
        settings = make_settings(db_trusted_connection=False, db_password="s3cret-pw")
        (result,) = check_env.check_database(settings)
        assert result.status == FAIL and result.fix
        assert "s3cret-pw" not in result.detail
        assert "s3cret-pw" not in result.fix


class TestDataRoot:
    def test_writable_root_passes_and_leaves_no_files_behind(self, tmp_path):
        root = tmp_path / "kera-data"
        result = check_env.check_data_root(make_settings(data_root=str(root)))
        assert result.status == OK
        assert list(root.iterdir()) == []  # the probe file was removed

    def test_unusable_root_is_a_failure_with_a_fix(self, tmp_path):
        blocker = tmp_path / "a-file"
        blocker.write_text("x")
        result = check_env.check_data_root(make_settings(data_root=str(blocker / "sub")))
        assert result.status == FAIL
        assert "DATA_ROOT" in result.fix


class TestStrayCocoFolder:
    def test_absent_is_ok(self, monkeypatch):
        monkeypatch.setattr(check_env, "stray_coco_folder_hint", lambda: None)
        assert check_env.check_extracted_coco_folder().status == OK

    def test_present_is_a_warning_with_the_spec_message(self, monkeypatch):
        monkeypatch.setattr(check_env, "stray_coco_folder_hint", lambda: "COCO detector message")
        result = check_env.check_extracted_coco_folder()
        assert result.status == WARN
        assert result.detail == "COCO detector message"
        assert "Backend/weights/tree_cls_v1.pt" in result.fix


class TestModelChecks:
    TREE = next(spec for spec in check_env.MODEL_SPECS if spec.key == "tree_classification")
    LEAF = next(spec for spec in check_env.MODEL_SPECS if spec.key == "leaf_segmentation")

    def use_fake(self, monkeypatch, fake: FakePredictor) -> None:
        monkeypatch.setattr(check_env, "import_factory", lambda path: lambda settings: fake)

    def test_disabled_model_is_skipped_with_a_warning(self):
        (row,) = check_env.check_model(self.TREE, make_settings(tree_enabled=False))
        assert row.status == WARN and "disabled" in row.detail

    def test_missing_tree_weights_fail_with_path_and_fix(self, tmp_path):
        missing = tmp_path / "tree.pt"
        (row,) = check_env.check_model(self.TREE, make_settings(tree_weights_path=str(missing)))
        assert row.status == FAIL
        assert str(missing) in row.detail
        assert "make_dummy_weights" in row.fix

    def test_tree_weights_that_are_not_a_zip_fail(self, tmp_path):
        bad = tmp_path / "tree.pt"
        bad.write_bytes(b"not a zip")
        (row,) = check_env.check_model(self.TREE, make_settings(tree_weights_path=str(bad)))
        assert row.status == FAIL and "not a valid .pt archive" in row.detail

    def test_missing_leaf_weights_fail(self, tmp_path):
        (row,) = check_env.check_model(
            self.LEAF, make_settings(leaf_seg_weights_path=str(tmp_path / "leaf.pt"))
        )
        assert row.status == FAIL and "not found" in row.detail

    def test_leaf_weights_that_are_not_a_zip_fail(self, tmp_path):
        bad = tmp_path / "leaf.pt"
        bad.write_bytes(b"nope")
        (row,) = check_env.check_model(self.LEAF, make_settings(leaf_seg_weights_path=str(bad)))
        assert row.status == FAIL and "not a valid torch archive" in row.detail

    def test_healthy_model_reports_weights_load_and_warmup(self, tmp_path, monkeypatch):
        weights = tmp_path / "tree.pt"
        write_zip(weights)
        self.use_fake(monkeypatch, FakePredictor("tree_classification", ready=False))
        rows = check_env.check_model(self.TREE, make_settings(tree_weights_path=str(weights)))
        assert [r.status for r in rows] == [OK, OK, OK]
        assert [r.name.split(": ")[1] for r in rows] == [
            "weights file",
            "loads",
            "warm-up inference",
        ]
        assert "classes=['banana_tree', 'non_banana']" in rows[1].detail

    def test_placeholder_weights_produce_a_warning(self, tmp_path, monkeypatch):
        weights = tmp_path / "tree.pt"
        write_zip(weights)
        fake = FakePredictor("tree_classification", ready=False, is_placeholder=True)
        self.use_fake(monkeypatch, fake)
        rows = check_env.check_model(self.TREE, make_settings(tree_weights_path=str(weights)))
        assert rows[-1].status == WARN
        assert "dummy weights - results are not real" in rows[-1].detail

    def test_load_error_is_reported_with_its_message(self, tmp_path, monkeypatch):
        weights = tmp_path / "tree.pt"
        write_zip(weights)
        fake = FakePredictor(
            "tree_classification",
            ready=False,
            load_error=ModelLoadError("expected classify, got detect — wrong weights?"),
        )
        self.use_fake(monkeypatch, fake)
        rows = check_env.check_model(self.TREE, make_settings(tree_weights_path=str(weights)))
        assert rows[-1].status == FAIL
        assert "expected classify, got detect" in rows[-1].detail
        assert rows[-1].name.endswith("loads")

    def test_unexpected_load_exception_is_a_failure_not_a_crash(self, tmp_path, monkeypatch):
        weights = tmp_path / "tree.pt"
        write_zip(weights)
        fake = FakePredictor("tree_classification", ready=False, load_error=RuntimeError("boom"))
        self.use_fake(monkeypatch, fake)
        rows = check_env.check_model(self.TREE, make_settings(tree_weights_path=str(weights)))
        assert rows[-1].status == FAIL and "RuntimeError: boom" in rows[-1].detail

    def test_model_module_that_cannot_be_imported_is_a_failure(self, tmp_path, monkeypatch):
        weights = tmp_path / "tree.pt"
        write_zip(weights)

        def broken_import(path):
            raise ModuleNotFoundError("No module named 'ultralytics'")

        monkeypatch.setattr(check_env, "import_factory", broken_import)
        rows = check_env.check_model(self.TREE, make_settings(tree_weights_path=str(weights)))
        assert rows[-1].status == FAIL and "ultralytics" in rows[-1].detail

    def test_warmup_failure_is_reported(self, tmp_path, monkeypatch):
        weights = tmp_path / "tree.pt"
        write_zip(weights)
        fake = FakePredictor("tree_classification", ready=False, warmup_error=ValueError("shape"))
        self.use_fake(monkeypatch, fake)
        rows = check_env.check_model(self.TREE, make_settings(tree_weights_path=str(weights)))
        assert rows[-1].status == FAIL and rows[-1].name.endswith("warm-up inference")


class TestRunner:
    def test_run_converts_a_crashing_check_into_a_failure_row(self):
        def crashing():
            raise RuntimeError("oops")

        (row,) = check_env._run("My check", crashing)
        assert row.status == FAIL and "check crashed: RuntimeError: oops" in row.detail

    def test_run_wraps_a_single_check_in_a_list(self):
        assert check_env._run("x", lambda: Check(OK, "x")) == [Check(OK, "x")]

    def test_table_shows_a_fix_under_every_non_ok_row(self, capsys):
        check_env._print_table(
            [
                Check(OK, "a", "fine"),
                Check(FAIL, "b", "broken", "do this"),
                Check(WARN, "c", "meh", "or this"),
            ]
        )
        out = capsys.readouterr().out
        assert "✅" in out and "❌" in out and "⚠" in out
        assert "-> fix: do this" in out and "-> fix: or this" in out
        assert out.count("-> fix") == 2

    def _stub_all_checks(self, monkeypatch, *, failing: bool) -> None:
        monkeypatch.setattr("app.core.config.get_settings", lambda: make_settings())
        row = Check(FAIL if failing else OK, "stub", "x", "fix")
        for name in ("check_odbc", "check_data_root", "check_cuda", "check_extracted_coco_folder"):
            monkeypatch.setattr(check_env, name, lambda *a, _row=row: _row)
        monkeypatch.setattr(check_env, "check_database", lambda *a, _row=row: [_row])
        monkeypatch.setattr(check_env, "check_model", lambda *a, _row=row: [_row])

    def test_exit_code_zero_when_nothing_failed(self, monkeypatch, capsys):
        self._stub_all_checks(monkeypatch, failing=False)
        monkeypatch.setattr(check_env, "check_python", lambda: [Check(OK, "py")])
        assert check_env.main() == 0
        assert "0 failure(s)" in capsys.readouterr().out

    def test_exit_code_one_when_anything_failed(self, monkeypatch, capsys):
        self._stub_all_checks(monkeypatch, failing=True)
        monkeypatch.setattr(check_env, "check_python", lambda: [Check(OK, "py")])
        assert check_env.main() == 1
        assert "failure(s)" in capsys.readouterr().out

    def test_invalid_settings_are_reported_without_echoing_secrets(self, monkeypatch, capsys):
        def broken():
            make_settings(signing_secret="too-short-secret-value")  # raises ValidationError

        monkeypatch.setattr("app.core.config.get_settings", broken)
        assert check_env.main() == 1
        out = capsys.readouterr().out
        assert "Settings (.env)" in out
        assert "signing_secret" in out
        assert "too-short-secret-value" not in out
