"""scripts/seed_samples.py against SQLite + a temp DATA_ROOT + FakePredictors (no torch).

The REAL AnalysisService, imaging pipeline, repository and storage run; only the model is fake.
"""

from __future__ import annotations

import io
import json
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from scripts import seed_samples
from scripts.seed_samples import ManifestError, SeedError, SeedReport, seed
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.db.models.analysis import Analysis
from app.ml.registry import ModelRegistry
from app.services.analysis_service import AnalysisFilter, AnalysisService
from app.storage.local import LocalImageStorage
from tests.fakes import FakePredictor
from tests.fixtures.image_factory import jpeg_bytes
from tests.fixtures.settings import make_settings

TREE = "tree_classification"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(data_root=str(tmp_path / "data"))


@pytest.fixture
def storage(tmp_path: Path) -> LocalImageStorage:
    return LocalImageStorage(tmp_path / "data")


@pytest.fixture
def predictor() -> FakePredictor:
    return FakePredictor(TREE, version="tree_fake_v1")


@pytest.fixture
def registry(predictor: FakePredictor) -> ModelRegistry:
    return ModelRegistry.from_predictors([predictor])


def make_samples(
    tmp_path: Path, entries: list[dict[str, Any]] | None = None, *, count: int = 2
) -> Path:
    """A samples folder with ``count`` DIFFERENT images (different size => different hash)."""
    folder = tmp_path / "samples" / TREE
    folder.mkdir(parents=True)
    names = [f"leaf{i}.jpg" for i in range(1, count + 1)]
    for index, name in enumerate(names):
        (folder / name).write_bytes(jpeg_bytes(200 + 20 * index, 150 + 10 * index))
    manifest = entries or [
        {
            "file": name,
            "title": f"Leaf {index}",
            "description": f"Description {index}",
            **({"latitude": 27.7172, "longitude": 85.324} if index == 1 else {}),
        }
        for index, name in enumerate(names, start=1)
    ]
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return folder


Seeder = Callable[..., SeedReport]


@pytest.fixture
def run_seed(
    session_factory: sessionmaker[Session],
    registry: ModelRegistry,
    storage: LocalImageStorage,
    settings: Settings,
) -> Seeder:
    def _run(folder: Path, *, force: bool = False, reg: ModelRegistry | None = None) -> SeedReport:
        return seed(
            TREE,
            session_factory=session_factory,
            registry=reg or registry,
            storage=storage,
            settings=settings,
            force=force,
            samples_dir=folder,
        )

    return _run


def all_rows(factory: sessionmaker[Session]) -> list[Analysis]:
    """Every row INCLUDING soft-deleted and failed ones, oldest first."""
    with factory() as session:
        return list(session.scalars(select(Analysis).order_by(Analysis.created_at, Analysis.id)))


def live_completed(factory: sessionmaker[Session]) -> list[Analysis]:
    return [r for r in all_rows(factory) if r.deleted_at is None and r.status == "completed"]


def files_under(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file()) if root.exists() else []


class TestSeeding:
    def test_runs_the_real_pipeline_and_stores_rows_and_files(
        self,
        tmp_path: Path,
        run_seed: Seeder,
        session_factory: sessionmaker[Session],
        storage: LocalImageStorage,
    ) -> None:
        report = run_seed(make_samples(tmp_path))

        assert [i.status for i in report.items] == ["seeded", "seeded"]
        assert report.ok and report.model_version == "tree_fake_v1"
        assert report.is_placeholder is False
        rows = all_rows(session_factory)
        assert len(rows) == 2
        first, second = rows

        for row in rows:
            assert row.is_sample is True
            assert row.client_id is None
            assert row.source == "sample"
            assert row.status == "completed"
            assert row.model_key == TREE and row.model_version == "tree_fake_v1"
            assert row.deleted_at is None
            assert row.display_label == "Banana tree"
            assert storage.exists(row.original_image_path)
            assert storage.exists(row.thumbnail_path)
        assert (first.title, first.description) == ("Leaf 1", "Description 1")
        assert (second.title, second.description) == ("Leaf 2", "Description 2")
        assert (first.latitude, first.longitude) == (27.7172, 85.324)
        assert (second.latitude, second.longitude) == (None, None)
        assert first.original_filename == "leaf1.jpg"
        # Both report items point at the stored rows.
        assert {i.analysis_id for i in report.items} == {first.id, second.id}
        assert all(
            i.label == "Banana tree" and i.model_version == "tree_fake_v1" for i in report.items
        )

    def test_seeded_samples_are_public_through_the_normal_service(
        self,
        tmp_path: Path,
        run_seed: Seeder,
        session_factory: sessionmaker[Session],
        registry: ModelRegistry,
        storage: LocalImageStorage,
        settings: Settings,
    ) -> None:
        run_seed(make_samples(tmp_path))
        with session_factory() as session:
            service = AnalysisService(session, registry, storage, settings)
            page = service.list(AnalysisFilter(scope="samples"), 1, 20, uuid.uuid4())

        assert page.total == 2
        assert {item.title for item in page.items} == {"Leaf 1", "Leaf 2"}
        assert all(item.is_sample for item in page.items)
        assert all("sig=" not in item.thumbnail_url for item in page.items)  # public, unsigned

    def test_the_stored_hash_is_the_hash_of_the_normalised_jpeg(
        self,
        tmp_path: Path,
        run_seed: Seeder,
        session_factory: sessionmaker[Session],
        storage: LocalImageStorage,
    ) -> None:
        import hashlib

        run_seed(make_samples(tmp_path, count=1))
        (row,) = all_rows(session_factory)
        stored = storage.open(row.original_image_path).read_bytes()
        assert row.image_sha256 == hashlib.sha256(stored).hexdigest()

    def test_the_placeholder_flag_of_the_model_is_reported(
        self, tmp_path: Path, run_seed: Seeder
    ) -> None:
        dummy = FakePredictor(TREE, version="tree_cls_dummy_v0", is_placeholder=True)
        report = run_seed(
            make_samples(tmp_path, count=1), reg=ModelRegistry.from_predictors([dummy])
        )
        assert report.is_placeholder is True and report.model_version == "tree_cls_dummy_v0"


class TestIdempotency:
    def test_a_second_run_skips_everything_and_writes_nothing(
        self,
        tmp_path: Path,
        run_seed: Seeder,
        session_factory: sessionmaker[Session],
    ) -> None:
        folder = make_samples(tmp_path)
        first = run_seed(folder)
        files_before = files_under(tmp_path / "data")

        second = run_seed(folder)

        assert [i.status for i in second.items] == ["skipped", "skipped"]
        assert [i.analysis_id for i in second.items] == [i.analysis_id for i in first.items]
        assert all("--force" in (i.detail or "") for i in second.items)
        assert len(all_rows(session_factory)) == 2
        assert files_under(tmp_path / "data") == files_before
        assert second.ok

    def test_a_new_image_added_later_is_seeded_alone(
        self, tmp_path: Path, run_seed: Seeder, session_factory: sessionmaker[Session]
    ) -> None:
        folder = make_samples(tmp_path, count=1)
        run_seed(folder)
        (folder / "extra.jpg").write_bytes(jpeg_bytes(300, 200))
        manifest = json.loads((folder / "manifest.json").read_text())
        manifest.append({"file": "extra.jpg", "title": "Extra"})
        (folder / "manifest.json").write_text(json.dumps(manifest))

        report = run_seed(folder)

        assert [(i.file, i.status) for i in report.items] == [
            ("leaf1.jpg", "skipped"),
            ("extra.jpg", "seeded"),
        ]
        assert len(live_completed(session_factory)) == 2

    def test_skipping_after_a_model_change_says_so(self, tmp_path: Path, run_seed: Seeder) -> None:
        folder = make_samples(tmp_path, count=1)
        run_seed(folder)
        newer = ModelRegistry.from_predictors([FakePredictor(TREE, version="tree_v2")])

        (item,) = run_seed(folder, reg=newer).items

        assert item.status == "skipped" and item.model_version == "tree_fake_v1"
        assert "tree_fake_v1" in (item.detail or "") and "tree_v2" in (item.detail or "")
        assert "--force" in (item.detail or "")

    def test_the_same_picture_for_another_model_is_not_a_duplicate(
        self,
        tmp_path: Path,
        session_factory: sessionmaker[Session],
        storage: LocalImageStorage,
        settings: Settings,
    ) -> None:
        both = ModelRegistry.from_predictors(
            [FakePredictor(TREE), FakePredictor("leaf_segmentation", task="segment")]
        )
        folder = make_samples(tmp_path, count=1)
        for key in (TREE, "leaf_segmentation"):
            report = seed(
                key,
                session_factory=session_factory,
                registry=both,
                storage=storage,
                settings=settings,
                samples_dir=folder,
            )
            assert report.items[0].status == "seeded"
        assert {r.model_key for r in live_completed(session_factory)} == {TREE, "leaf_segmentation"}


class TestForce:
    def test_force_reanalyses_and_retires_the_old_row(
        self,
        tmp_path: Path,
        run_seed: Seeder,
        session_factory: sessionmaker[Session],
        storage: LocalImageStorage,
    ) -> None:
        folder = make_samples(tmp_path, count=1)
        (old_item,) = run_seed(folder).items
        newer = ModelRegistry.from_predictors([FakePredictor(TREE, version="tree_v2")])

        (item,) = run_seed(folder, force=True, reg=newer).items

        assert item.status == "seeded" and item.model_version == "tree_v2"
        assert item.analysis_id != old_item.analysis_id
        assert "replaced" in (item.detail or "")
        rows = {r.id: r for r in all_rows(session_factory)}
        assert rows[old_item.analysis_id].deleted_at is not None  # soft-deleted (P-07)
        assert storage.exists(rows[old_item.analysis_id].original_image_path)  # files stay
        (live,) = live_completed(session_factory)
        assert live.id == item.analysis_id and live.model_version == "tree_v2"

    def test_force_on_a_fresh_database_just_seeds(
        self, tmp_path: Path, run_seed: Seeder, session_factory: sessionmaker[Session]
    ) -> None:
        (item,) = run_seed(make_samples(tmp_path, count=1), force=True).items
        assert item.status == "seeded" and item.detail is None
        assert len(live_completed(session_factory)) == 1

    def test_a_failed_force_keeps_the_old_sample_visible(
        self,
        tmp_path: Path,
        run_seed: Seeder,
        session_factory: sessionmaker[Session],
    ) -> None:
        folder = make_samples(tmp_path, count=1)
        (old_item,) = run_seed(folder).items
        broken = ModelRegistry.from_predictors([FakePredictor(TREE, fail=True)])

        report = run_seed(folder, force=True, reg=broken)

        (item,) = report.items
        assert item.status == "failed" and "INFERENCE_FAILED" in (item.detail or "")
        assert not report.ok
        (live,) = live_completed(session_factory)
        assert live.id == old_item.analysis_id  # nothing was lost

        # ...and the failed attempt left no visible duplicate once the model works again.
        (again,) = run_seed(folder).items
        assert again.status == "skipped" and again.analysis_id == old_item.analysis_id

    def test_a_failed_first_run_is_retried_and_the_leftover_retired(
        self, tmp_path: Path, run_seed: Seeder, session_factory: sessionmaker[Session]
    ) -> None:
        folder = make_samples(tmp_path, count=1)
        broken = ModelRegistry.from_predictors([FakePredictor(TREE, fail=True)])
        (failed,) = run_seed(folder, reg=broken).items
        assert failed.status == "failed"
        (debris,) = all_rows(session_factory)
        assert debris.status == "failed" and debris.is_sample is True

        (item,) = run_seed(folder).items  # NO --force: a failed row is not "already seeded"

        assert item.status == "seeded" and item.detail is None
        rows = {r.id: r for r in all_rows(session_factory)}
        assert rows[debris.id].deleted_at is not None
        (live,) = live_completed(session_factory)
        assert live.id == item.analysis_id


class TestRunCannotStart:
    def test_unavailable_model_fails_clearly_and_creates_nothing(
        self,
        tmp_path: Path,
        run_seed: Seeder,
        session_factory: sessionmaker[Session],
    ) -> None:
        down = ModelRegistry.from_predictors([FakePredictor(TREE, ready=False)])

        with pytest.raises(SeedError, match=r"Model 'tree_classification' is unavailable"):
            run_seed(make_samples(tmp_path), reg=down)

        assert all_rows(session_factory) == []
        assert files_under(tmp_path / "data") == []

    def test_unknown_model_key(self, tmp_path: Path, run_seed: Seeder) -> None:
        with pytest.raises(SeedError, match="unknown model"):
            seed(
                "no_such_model",
                session_factory=None,  # type: ignore[arg-type]
                registry=ModelRegistry.from_predictors([FakePredictor(TREE)]),
                storage=None,  # type: ignore[arg-type]
                settings=make_settings(),
            )

    @pytest.mark.parametrize(
        ("entries", "text", "fragment"),
        [
            ([{"file": "missing.jpg", "title": "T"}], None, "not found"),
            ([{"file": "../x.jpg", "title": "T"}], None, "'..'"),
            (
                [{"file": "leaf1.jpg", "title": "T", "latitude": 27.7}],
                None,
                "must be given together",
            ),
            (None, '[{"file": "leaf1.jpg", "title": ', "not valid JSON"),
        ],
    )
    def test_a_bad_manifest_creates_nothing(
        self,
        tmp_path: Path,
        run_seed: Seeder,
        session_factory: sessionmaker[Session],
        entries: list[dict[str, Any]] | None,
        text: str | None,
        fragment: str,
    ) -> None:
        folder = make_samples(tmp_path, count=1)
        if text is not None:
            (folder / "manifest.json").write_text(text)
        else:
            (folder / "manifest.json").write_text(json.dumps(entries))

        with pytest.raises(ManifestError, match=fragment):
            run_seed(folder)

        assert all_rows(session_factory) == []
        assert files_under(tmp_path / "data") == []

    def test_a_missing_manifest_for_the_default_folder(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        session_factory: sessionmaker[Session],
        registry: ModelRegistry,
        storage: LocalImageStorage,
        settings: Settings,
    ) -> None:
        monkeypatch.setattr(seed_samples, "SAMPLES_ROOT", tmp_path / "nowhere")
        with pytest.raises(ManifestError, match="No manifest found"):
            seed(
                TREE,
                session_factory=session_factory,
                registry=registry,
                storage=storage,
                settings=settings,
            )


class TestOneBadFileDoesNotStopTheOthers:
    def folder_with_bad_file(self, tmp_path: Path) -> Path:
        folder = make_samples(tmp_path, count=2)
        (folder / "bad.jpg").write_bytes(b"this is not an image")
        manifest = json.loads((folder / "manifest.json").read_text())
        manifest.insert(1, {"file": "bad.jpg", "title": "Bad"})
        (folder / "manifest.json").write_text(json.dumps(manifest))
        return folder

    def test_a_corrupt_image_is_reported_and_the_rest_is_seeded(
        self, tmp_path: Path, run_seed: Seeder, session_factory: sessionmaker[Session]
    ) -> None:
        report = run_seed(self.folder_with_bad_file(tmp_path))

        assert [(i.file, i.status) for i in report.items] == [
            ("leaf1.jpg", "seeded"),
            ("bad.jpg", "failed"),
            ("leaf2.jpg", "seeded"),
        ]
        assert "INVALID_IMAGE" in (report.items[1].detail or "")
        assert not report.ok
        assert len(live_completed(session_factory)) == 2
        assert len(all_rows(session_factory)) == 2  # an invalid image is not persisted (P-05)

    def test_a_too_small_image_is_reported(self, tmp_path: Path, run_seed: Seeder) -> None:
        folder = make_samples(tmp_path, count=1)
        buffer = io.BytesIO()
        Image.new("RGB", (10, 10)).save(buffer, format="JPEG")
        (folder / "tiny.jpg").write_bytes(buffer.getvalue())
        manifest = json.loads((folder / "manifest.json").read_text())
        manifest.append({"file": "tiny.jpg", "title": "Tiny"})
        (folder / "manifest.json").write_text(json.dumps(manifest))

        report = run_seed(folder)

        assert [i.status for i in report.items] == ["seeded", "failed"]
        assert "IMAGE_TOO_SMALL" in (report.items[1].detail or "")

    def test_an_unreadable_file_is_reported(
        self, tmp_path: Path, run_seed: Seeder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        folder = make_samples(tmp_path, count=2)
        real = Path.read_bytes

        def read_bytes(self: Path) -> bytes:
            if self.name == "leaf1.jpg":
                raise PermissionError("locked by another program")
            return real(self)

        monkeypatch.setattr(Path, "read_bytes", read_bytes)

        report = run_seed(folder)

        assert [i.status for i in report.items] == ["failed", "seeded"]
        assert "cannot read leaf1.jpg" in (report.items[0].detail or "")

    def test_an_unexpected_error_is_reported_with_its_type(
        self, tmp_path: Path, run_seed: Seeder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        folder = make_samples(tmp_path, count=2)
        real = AnalysisService.analyze

        def analyze(self: AnalysisService, *args: Any, **kwargs: Any) -> Any:
            if kwargs.get("title") == "Leaf 1":
                raise RuntimeError("disk exploded")
            return real(self, *args, **kwargs)

        monkeypatch.setattr(AnalysisService, "analyze", analyze)

        report = run_seed(folder)

        assert [i.status for i in report.items] == ["failed", "seeded"]
        assert "unexpected RuntimeError: disk exploded" in (report.items[0].detail or "")


# --------------------------------------------------------------------------- the CLI


@pytest.fixture
def cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sqlite_engine: Engine,
    registry: ModelRegistry,
    settings: Settings,
) -> Callable[..., tuple[int, str, str]]:
    """Run ``seed_samples.main`` against the SQLite engine/fake registry; returns (code, out, err)."""
    folder_root = tmp_path / "samples"
    monkeypatch.setattr(seed_samples, "SAMPLES_ROOT", folder_root)
    monkeypatch.setattr(seed_samples, "get_settings", lambda: settings)
    monkeypatch.setattr(seed_samples, "create_db_engine", lambda s: sqlite_engine)
    state = {"registry": registry}
    monkeypatch.setattr(seed_samples, "build_registry", lambda s, only=None: state["registry"])

    def _run(*argv: str, reg: ModelRegistry | None = None) -> tuple[int, str, str]:
        if reg is not None:
            state["registry"] = reg
        # capsys is not needed: run() writes to the streams it is given.
        out, err = io.StringIO(), io.StringIO()
        code = seed_samples.run(seed_samples.build_parser().parse_args(list(argv)), out, err)
        return code, out.getvalue(), err.getvalue()

    return _run


class TestCli:
    def test_seeds_and_prints_a_per_file_summary(
        self, tmp_path: Path, cli: Callable[..., tuple[int, str, str]]
    ) -> None:
        make_samples(tmp_path)
        code, out, err = cli("--model", TREE)

        assert code == 0 and err == ""
        assert "SEEDED" in out and "leaf1.jpg" in out and 'label="Banana tree"' in out
        assert "model_version=tree_fake_v1" in out
        assert "2 seeded, 0 skipped, 0 failed" in out
        assert "PLACEHOLDER" not in out

    def test_a_second_run_is_all_skips_and_exits_zero(
        self, tmp_path: Path, cli: Callable[..., tuple[int, str, str]]
    ) -> None:
        make_samples(tmp_path)
        cli("--model", TREE)
        code, out, _ = cli("--model", TREE)
        assert code == 0 and "0 seeded, 2 skipped, 0 failed" in out

    def test_force_flag_is_passed_through(
        self, tmp_path: Path, cli: Callable[..., tuple[int, str, str]]
    ) -> None:
        make_samples(tmp_path, count=1)
        cli("--model", TREE)
        code, out, _ = cli("--model", TREE, "--force")
        assert code == 0 and "1 seeded, 0 skipped, 0 failed" in out and "replaced" in out

    def test_dummy_weights_print_the_loud_warning(
        self, tmp_path: Path, cli: Callable[..., tuple[int, str, str]]
    ) -> None:
        make_samples(tmp_path, count=1)
        dummy = ModelRegistry.from_predictors(
            [FakePredictor(TREE, version="tree_cls_dummy_v0", is_placeholder=True)]
        )
        code, out, _ = cli("--model", TREE, reg=dummy)
        assert code == 0
        assert "PLACEHOLDER WEIGHTS: tree_classification (tree_cls_dummy_v0)" in out
        assert "seed_samples --model tree_classification --force" in out

    def test_exit_code_1_when_a_file_failed(
        self, tmp_path: Path, cli: Callable[..., tuple[int, str, str]]
    ) -> None:
        folder = make_samples(tmp_path, count=1)
        (folder / "bad.jpg").write_bytes(b"nope")
        manifest = json.loads((folder / "manifest.json").read_text())
        manifest.append({"file": "bad.jpg", "title": "Bad"})
        (folder / "manifest.json").write_text(json.dumps(manifest))

        code, out, _ = cli("--model", TREE)

        assert code == 1
        assert "FAILED" in out and "1 seeded, 0 skipped, 1 failed" in out

    def test_exit_code_2_for_a_bad_manifest(
        self, tmp_path: Path, cli: Callable[..., tuple[int, str, str]]
    ) -> None:
        folder = make_samples(tmp_path, count=1)
        (folder / "manifest.json").write_text("[not json")
        code, out, err = cli("--model", TREE)
        assert code == 2 and out == "" and "not valid JSON" in err

    def test_exit_code_2_when_the_model_is_unavailable(
        self, tmp_path: Path, cli: Callable[..., tuple[int, str, str]]
    ) -> None:
        make_samples(tmp_path)
        down = ModelRegistry.from_predictors([FakePredictor(TREE, ready=False)])
        code, out, err = cli("--model", TREE, reg=down)
        assert code == 2 and out == ""
        assert "unavailable" in err and "No samples were created" in err

    def test_exit_code_2_when_the_database_is_not_migrated(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        cli: Callable[..., tuple[int, str, str]],
    ) -> None:
        make_samples(tmp_path)
        empty = create_engine(f"sqlite:///{(tmp_path / 'empty.db').as_posix()}")
        monkeypatch.setattr(seed_samples, "create_db_engine", lambda s: empty)
        code, out, err = cli("--model", TREE)
        assert code == 2 and out == "" and "alembic upgrade head" in err

    def test_the_model_option_is_required_and_restricted(self) -> None:
        parser = seed_samples.build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])
        with pytest.raises(SystemExit):
            parser.parse_args(["--model", "../../etc"])
        assert parser.parse_args(["--model", "leaf_segmentation"]).force is False

    def test_main_wires_argv_to_run(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        cli: Callable[..., tuple[int, str, str]],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        make_samples(tmp_path, count=1)
        assert seed_samples.main(["--model", TREE]) == 0
        assert "1 seeded" in capsys.readouterr().out


# --------------------------------------------------------------------------- committed samples


def test_the_committed_samples_seed_cleanly_and_stay_small(
    fake_registry: ModelRegistry,
    session_factory: sessionmaker[Session],
    storage: LocalImageStorage,
    settings: Settings,
) -> None:
    keys = sorted(p.name for p in seed_samples.SAMPLES_ROOT.iterdir() if p.is_dir())
    assert keys, "samples/ must hold at least one model folder"
    for key in keys:
        report = seed(
            key,
            session_factory=session_factory,
            registry=fake_registry,
            storage=storage,
            settings=settings,
        )
        assert report.ok and report.count("seeded") == len(report.items) > 0, key

        for sample in seed_samples.load_manifest(seed_samples.SAMPLES_ROOT / key):
            assert sample.path.stat().st_size < 400_000, f"{sample.path.name} is too big for git"
            with Image.open(sample.path) as image:
                assert max(image.size) <= 1024, f"{sample.path.name} should be <= 1024 px"
                assert len(image.getexif()) == 0, f"{sample.path.name} still carries EXIF"
