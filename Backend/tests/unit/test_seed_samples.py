"""scripts/seed_samples.py: manifest validation, report formatting, DB preflight.

The seeding core itself (rows, files, idempotency, --force) is in
tests/integration/test_seed_samples.py.
"""

from __future__ import annotations

import json
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any

import pytest
from scripts import seed_samples
from scripts.seed_samples import (
    ManifestError,
    SeedItem,
    SeedReport,
    database_problem,
    format_report,
    load_manifest,
)
from sqlalchemy import create_engine

from tests.fixtures.image_factory import jpeg_bytes


def make_folder(tmp_path: Path, entries: Any, *, images: tuple[str, ...] = ("a.jpg",)) -> Path:
    folder = tmp_path / "samples"
    folder.mkdir()
    for name in images:
        target = folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(jpeg_bytes())
    text = entries if isinstance(entries, str) else json.dumps(entries)
    (folder / "manifest.json").write_text(text, encoding="utf-8")
    return folder


def problems_of(folder: Path) -> str:
    with pytest.raises(ManifestError) as caught:
        load_manifest(folder)
    return str(caught.value)


class TestValidManifest:
    def test_minimal_and_full_entries(self, tmp_path: Path) -> None:
        folder = make_folder(
            tmp_path,
            [
                {"file": "a.jpg", "title": "  Leaf A  "},
                {
                    "file": "sub/b.jpg",
                    "title": "Leaf B",
                    "description": "Taken near the river",
                    "latitude": 27.7172,
                    "longitude": 85.324,
                },
            ],
            images=("a.jpg", "sub/b.jpg"),
        )
        first, second = load_manifest(folder)

        assert first.entry.title == "Leaf A"  # whitespace stripped
        assert first.entry.description is None
        assert (first.entry.latitude, first.entry.longitude) == (None, None)
        assert first.path == (folder / "a.jpg").resolve()
        assert second.entry.description == "Taken near the river"
        assert (second.entry.latitude, second.entry.longitude) == (27.7172, 85.324)
        assert second.path == (folder / "sub" / "b.jpg").resolve()

    def test_a_byte_order_mark_from_notepad_is_tolerated(self, tmp_path: Path) -> None:
        folder = make_folder(tmp_path, [])
        (folder / "manifest.json").write_bytes(
            b"\xef\xbb\xbf" + json.dumps([{"file": "a.jpg", "title": "T"}]).encode()
        )
        assert len(load_manifest(folder)) == 1

    def test_the_committed_manifests_are_valid(self) -> None:
        keys = sorted(p.name for p in seed_samples.SAMPLES_ROOT.iterdir() if p.is_dir())
        assert {"tree_classification", "leaf_segmentation"} <= set(keys)
        for key in keys:
            samples = load_manifest(seed_samples.SAMPLES_ROOT / key)
            assert samples, key
            for sample in samples:
                assert "placeholder" in sample.entry.title.lower()


class TestInvalidManifest:
    def test_missing_manifest_names_the_expected_path(self, tmp_path: Path) -> None:
        message = problems_of(tmp_path)
        assert "No manifest found" in message
        assert "manifest.json" in message and "samples/README.md" in message

    def test_malformed_json_reports_the_position(self, tmp_path: Path) -> None:
        message = problems_of(make_folder(tmp_path, '[{"file": "a.jpg", "title": '))
        assert "not valid JSON" in message and "line 1" in message

    def test_the_top_level_must_be_a_list(self, tmp_path: Path) -> None:
        assert "must be a JSON list" in problems_of(make_folder(tmp_path, {"file": "a.jpg"}))

    def test_an_empty_list_is_an_error(self, tmp_path: Path) -> None:
        assert "is empty" in problems_of(make_folder(tmp_path, []))

    def test_an_unreadable_manifest_is_reported_not_crashed(self, tmp_path: Path) -> None:
        folder = make_folder(tmp_path, [])
        (folder / "manifest.json").write_bytes(b"\xff\xfe\x00 not utf-8 \xff")
        assert "Cannot read" in problems_of(folder)

    def test_missing_required_fields(self, tmp_path: Path) -> None:
        message = problems_of(make_folder(tmp_path, [{"title": "No file"}, {"file": "a.jpg"}]))
        assert "entry #1 -> file: Field required" in message
        assert "entry #2 -> title: Field required" in message

    def test_unknown_fields_are_typos_not_ignored(self, tmp_path: Path) -> None:
        message = problems_of(
            make_folder(tmp_path, [{"file": "a.jpg", "title": "T", "lat": 1, "lon": 2}])
        )
        assert "entry #1 -> lat: Extra inputs are not permitted" in message

    @pytest.mark.parametrize(
        ("coordinates", "fragment"),
        [
            ({"latitude": 27.7}, "must be given together"),
            ({"longitude": 85.3}, "must be given together"),
            ({"latitude": 91, "longitude": 0}, "latitude"),
            ({"latitude": 0, "longitude": 181}, "longitude"),
        ],
    )
    def test_coordinates_are_both_or_neither_and_in_range(
        self, tmp_path: Path, coordinates: dict[str, float], fragment: str
    ) -> None:
        message = problems_of(
            make_folder(tmp_path, [{"file": "a.jpg", "title": "T", **coordinates}])
        )
        assert fragment in message
        assert "Value error," not in message  # pydantic's prefix is noise for the owner

    def test_title_and_description_respect_the_column_sizes(self, tmp_path: Path) -> None:
        message = problems_of(
            make_folder(
                tmp_path, [{"file": "a.jpg", "title": "t" * 201, "description": "d" * 1001}]
            )
        )
        assert "title" in message and "description" in message

    @pytest.mark.parametrize(
        "name",
        ["../x.jpg", "sub/../../x.jpg", "sub/../a.jpg", "..\\x.jpg", "/etc/x.jpg", "C:\\x.jpg"],
    )
    def test_path_traversal_and_absolute_paths_are_rejected(
        self, tmp_path: Path, name: str
    ) -> None:
        (tmp_path / "x.jpg").write_bytes(jpeg_bytes())  # exists, but outside the folder
        message = problems_of(make_folder(tmp_path, [{"file": name, "title": "T"}]))
        assert "entry #1 -> file:" in message
        assert "relative path" in message or "'..'" in message

    def test_a_link_pointing_outside_the_folder_is_rejected(self, tmp_path: Path) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "x.jpg").write_bytes(jpeg_bytes())
        folder = make_folder(tmp_path, [{"file": "escape/x.jpg", "title": "T"}], images=())
        link = folder / "escape"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            # Creating symlinks needs a privilege on Windows; a directory junction does not.
            junction = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(link), str(outside)],
                capture_output=True,
                check=False,
            )
            if junction.returncode != 0:
                pytest.skip("cannot create a symlink or a junction here")
        assert "resolves outside" in problems_of(folder)

    def test_a_missing_file_is_named(self, tmp_path: Path) -> None:
        message = problems_of(make_folder(tmp_path, [{"file": "nope.jpg", "title": "T"}]))
        assert "file 'nope.jpg' not found" in message

    def test_a_directory_is_not_an_image(self, tmp_path: Path) -> None:
        folder = make_folder(tmp_path, [{"file": "photos", "title": "T"}])
        (folder / "photos").mkdir()
        assert "not found" in problems_of(folder)

    def test_the_same_file_twice_is_an_error(self, tmp_path: Path) -> None:
        message = problems_of(
            make_folder(
                tmp_path, [{"file": "a.jpg", "title": "1"}, {"file": "A.JPG", "title": "2"}]
            )
        )
        assert "entry #2 -> file: 'A.JPG' is listed more than once" in message

    def test_every_problem_is_listed_at_once(self, tmp_path: Path) -> None:
        message = problems_of(
            make_folder(
                tmp_path,
                [{"file": "gone.jpg", "title": "T"}, {"file": "../x.jpg", "title": "U"}],
            )
        )
        assert "2 problem(s)" in message
        assert "entry #1" in message and "entry #2" in message


class TestFormatReport:
    def report(self, *, placeholder: bool) -> SeedReport:
        analysis_id = uuid.uuid4()
        return SeedReport(
            model_key="tree_classification",
            model_version="tree_cls_v1",
            is_placeholder=placeholder,
            items=[
                SeedItem("a.jpg", "seeded", analysis_id, "Banana tree", "tree_cls_v1"),
                SeedItem("b.jpg", "skipped", None, None, None, "already seeded"),
                SeedItem("c.jpg", "failed", detail="INVALID_IMAGE: not an image"),
            ],
        )

    def test_lists_every_file_with_id_label_and_version(self) -> None:
        report = self.report(placeholder=False)
        text = format_report(report)

        assert re.search(r"SEEDED\s+a\.jpg", text) and 'label="Banana tree"' in text
        assert "model_version=tree_cls_v1" in text
        assert re.search(r"SKIPPED\s+b\.jpg", text) and "already seeded" in text
        assert re.search(r"FAILED\s+c\.jpg", text) and "INVALID_IMAGE: not an image" in text
        assert "1 seeded, 1 skipped, 1 failed" in text
        assert "PLACEHOLDER" not in text
        assert report.ok is False

    def test_dummy_weights_get_a_loud_warning_with_the_force_hint(self) -> None:
        text = format_report(self.report(placeholder=True))
        assert "PLACEHOLDER WEIGHTS: tree_classification (tree_cls_v1)" in text
        assert "NOT real" in text
        assert "seed_samples --model tree_classification --force" in text

    def test_a_report_without_failures_is_ok(self) -> None:
        report = SeedReport("m", "v", False, [SeedItem("a.jpg", "seeded")])
        assert report.ok and report.count("seeded") == 1


class TestDatabaseProblem:
    def test_an_unreachable_database_points_at_check_env_and_the_sqlite_dev_server(
        self, tmp_path: Path
    ) -> None:
        engine = create_engine(f"sqlite:///{(tmp_path / 'missing-dir' / 'x.db').as_posix()}")
        try:
            message = database_problem(engine)
        finally:
            engine.dispose()
        assert message is not None
        assert "check_env" in message and "dev_server_sqlite" in message

    def test_an_unmigrated_database_says_to_run_alembic(self, tmp_path: Path) -> None:
        engine = create_engine(f"sqlite:///{(tmp_path / 'empty.db').as_posix()}")
        try:
            message = database_problem(engine)
        finally:
            engine.dispose()
        assert message is not None and "alembic upgrade head" in message

    def test_a_migrated_database_has_no_problem(self, sqlite_engine: Any) -> None:
        assert database_problem(sqlite_engine) is None
