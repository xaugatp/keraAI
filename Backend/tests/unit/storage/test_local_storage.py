from __future__ import annotations

import os
import subprocess
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.storage import local
from app.storage.base import (
    ImageStorage,
    InvalidStoragePathError,
    StorageError,
    StoredFileNotFoundError,
    build_paths,
)
from app.storage.local import LocalImageStorage

REL = "images/tree_classification/2026/10/6f1c2a9e-1b2c-4d3e-8f90-a1b2c3d4e5f6/original.jpg"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    data_root = tmp_path / "data-root"
    data_root.mkdir()
    return data_root


@pytest.fixture
def storage(root: Path) -> LocalImageStorage:
    return LocalImageStorage(root)


def files_under(path: Path) -> list[Path]:
    return sorted(p for p in path.rglob("*") if p.is_file())


def entries_under(path: Path) -> list[Path]:
    return sorted(path.rglob("*"))


# --- basics --------------------------------------------------------------------------


def test_local_storage_satisfies_the_protocol(storage: LocalImageStorage) -> None:
    protocol_view: ImageStorage = storage
    assert protocol_view is storage


def test_save_then_open_returns_the_file_inside_the_root(
    storage: LocalImageStorage, root: Path
) -> None:
    storage.save(REL, b"jpeg-bytes")
    opened = storage.open(REL)
    assert opened.read_bytes() == b"jpeg-bytes"
    assert opened.is_relative_to(root.resolve())
    assert opened == root.resolve() / Path(*REL.split("/"))


def test_save_creates_the_directory_tree(storage: LocalImageStorage, root: Path) -> None:
    storage.save(REL, b"x")
    assert (root / "images" / "tree_classification" / "2026" / "10").is_dir()


def test_save_creates_the_root_itself_if_it_does_not_exist_yet(tmp_path: Path) -> None:
    storage = LocalImageStorage(tmp_path / "not-yet" / "created")
    storage.save(REL, b"x")
    assert storage.exists(REL)


def test_root_may_be_given_as_a_string(root: Path) -> None:
    storage = LocalImageStorage(str(root))
    assert storage.root == root.resolve()


def test_save_overwrites_an_existing_file(storage: LocalImageStorage) -> None:
    storage.save(REL, b"old")
    storage.save(REL, b"new")
    assert storage.open(REL).read_bytes() == b"new"


def test_save_leaves_no_temp_files_behind(storage: LocalImageStorage, root: Path) -> None:
    storage.save(REL, b"x")
    assert [p.name for p in files_under(root)] == ["original.jpg"]


def test_exists_reports_files_only(storage: LocalImageStorage) -> None:
    assert not storage.exists(REL)
    storage.save(REL, b"x")
    assert storage.exists(REL)
    assert not storage.exists("images/tree_classification")  # a directory, not a file


def test_open_of_a_missing_file_raises_a_typed_not_found_error(
    storage: LocalImageStorage,
) -> None:
    with pytest.raises(StoredFileNotFoundError) as exc_info:
        storage.open(REL)
    assert isinstance(exc_info.value, StorageError)
    assert isinstance(exc_info.value, FileNotFoundError)


def test_open_of_a_directory_is_not_found(storage: LocalImageStorage) -> None:
    storage.save(REL, b"x")
    with pytest.raises(StoredFileNotFoundError):
        storage.open("images/tree_classification/2026")


def test_empty_files_are_stored(storage: LocalImageStorage) -> None:
    storage.save(REL, b"")
    assert storage.open(REL).read_bytes() == b""


# --- path-traversal guard -----------------------------------------------------------

BAD_PATHS = [
    pytest.param("", id="empty"),
    pytest.param("../outside.jpg", id="dotdot"),
    pytest.param("images/../../outside.jpg", id="dotdot-in-middle"),
    pytest.param("images/../../../../../windows/win.ini", id="dotdot-deep"),
    pytest.param("images/tree_classification/..", id="trailing-dotdot"),
    pytest.param("./original.jpg", id="dot-segment"),
    pytest.param("images/./original.jpg", id="dot-segment-middle"),
    pytest.param("/etc/passwd", id="posix-absolute"),
    pytest.param("/images/original.jpg", id="leading-slash"),
    pytest.param("C:/Windows/win.ini", id="drive-letter"),
    pytest.param("c:\\Windows\\win.ini", id="drive-letter-backslash"),
    pytest.param("C:win.ini", id="drive-relative"),
    pytest.param("\\\\server\\share\\x.jpg", id="unc"),
    pytest.param("//server/share/x.jpg", id="unc-forward-slashes"),
    pytest.param("\\\\?\\C:\\Windows\\x", id="extended-length-prefix"),
    pytest.param("images\\tree\\original.jpg", id="backslashes"),
    pytest.param("images/..\\outside.jpg", id="backslash-dotdot"),
    pytest.param("images/original.jpg:stream", id="alternate-data-stream"),
    pytest.param("images/original.jpg::$DATA", id="ads-default-stream"),
    pytest.param("images/CON", id="reserved-con"),
    pytest.param("images/con.jpg", id="reserved-con-with-extension"),
    pytest.param("images/NUL.txt", id="reserved-nul"),
    pytest.param("images/aux", id="reserved-aux"),
    pytest.param("images/PRN.jpg", id="reserved-prn"),
    pytest.param("images/COM1", id="reserved-com1"),
    pytest.param("images/lpt9.jpg", id="reserved-lpt9"),
    pytest.param("CON/original.jpg", id="reserved-first-segment"),
    pytest.param("images/trailing.", id="trailing-dot"),
    pytest.param("images/trailing ", id="trailing-space"),
    pytest.param("images/...", id="dots-only"),
    pytest.param("images/has space.jpg", id="space"),
    pytest.param("images//original.jpg", id="double-slash"),
    pytest.param("images/original.jpg/", id="trailing-slash"),
    pytest.param(".hidden", id="leading-dot"),
    pytest.param("images/.original.jpg.abc.tmp", id="our-own-temp-file-name"),
    pytest.param("images/%2e%2e/outside.jpg", id="percent-encoded-dotdot"),
    pytest.param("images/PROGRA~1/x.jpg", id="short-8.3-name"),
    pytest.param("images/orig\x00inal.jpg", id="nul-byte"),
    pytest.param("images/orig\ninal.jpg", id="newline"),
    pytest.param("images/\u00e9.jpg", id="non-ascii"),
    pytest.param("images/\uff0e\uff0e/x.jpg", id="fullwidth-dots"),
    pytest.param("images/<x>.jpg", id="angle-brackets"),
    pytest.param('images/"x".jpg', id="quotes"),
    pytest.param("images/x?.jpg", id="question-mark"),
    pytest.param("images/*.jpg", id="wildcard"),
    pytest.param("images/a|b.jpg", id="pipe"),
    pytest.param("images/" + "a" * 500, id="too-long"),
]


@pytest.mark.parametrize("bad_path", BAD_PATHS)
def test_hostile_paths_are_rejected_by_every_operation(
    storage: LocalImageStorage, root: Path, bad_path: str
) -> None:
    with pytest.raises(InvalidStoragePathError):
        storage.save(bad_path, b"x")
    with pytest.raises(InvalidStoragePathError):
        storage.open(bad_path)
    with pytest.raises(InvalidStoragePathError):
        storage.exists(bad_path)
    with pytest.raises(InvalidStoragePathError):
        storage.delete(bad_path)
    assert entries_under(root) == []  # nothing was created, anywhere


def test_a_traversal_attempt_never_touches_the_file_it_points_at(
    storage: LocalImageStorage, root: Path
) -> None:
    victim = root.parent / "victim.txt"
    victim.write_text("precious")
    with pytest.raises(InvalidStoragePathError):
        storage.save("../victim.txt", b"overwritten")
    with pytest.raises(InvalidStoragePathError):
        storage.delete("../victim.txt")
    assert victim.read_text() == "precious"


def test_the_root_itself_is_not_addressable(storage: LocalImageStorage) -> None:
    for path in ("", ".", "images/.."):
        with pytest.raises(InvalidStoragePathError):
            storage.delete(path)


def test_invalid_path_error_is_also_a_value_error() -> None:
    assert issubclass(InvalidStoragePathError, ValueError)
    assert issubclass(InvalidStoragePathError, StorageError)


def _link_directory(link: Path, target: Path) -> None:
    """Create a symlink, or on Windows a junction, or skip the test if neither is allowed."""
    try:
        os.symlink(target, link, target_is_directory=True)
        return
    except (OSError, NotImplementedError):
        pass
    if os.name == "nt":
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
            check=False,
        )
        if completed.returncode == 0:
            return
    pytest.skip("this account may not create symlinks or junctions")


def test_a_directory_link_pointing_outside_the_root_is_rejected(
    storage: LocalImageStorage, root: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.jpg").write_bytes(b"secret")
    _link_directory(root / "images", outside)

    for operation in (storage.open, storage.exists, storage.delete):
        with pytest.raises(InvalidStoragePathError):
            operation("images/secret.jpg")
    with pytest.raises(InvalidStoragePathError):
        storage.save("images/new.jpg", b"leak")
    assert [p.name for p in outside.iterdir()] == ["secret.jpg"]
    assert (outside / "secret.jpg").read_bytes() == b"secret"


def test_a_link_that_resolves_to_the_root_itself_is_rejected(
    storage: LocalImageStorage, root: Path
) -> None:
    _link_directory(root / "images", root)
    for operation in (storage.open, storage.exists, storage.delete):
        with pytest.raises(InvalidStoragePathError):
            operation("images")


def test_a_link_nested_deeper_in_the_tree_is_rejected_too(
    storage: LocalImageStorage, root: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "images" / "tree_classification").mkdir(parents=True)
    _link_directory(root / "images" / "tree_classification" / "2026", outside)
    with pytest.raises(InvalidStoragePathError):
        storage.save("images/tree_classification/2026/10/x/original.jpg", b"leak")
    assert list(outside.iterdir()) == []


def test_a_root_that_is_itself_a_link_still_works(tmp_path: Path) -> None:
    real = tmp_path / "real-root"
    real.mkdir()
    link = tmp_path / "linked-root"
    _link_directory(link, real)
    storage = LocalImageStorage(link)
    storage.save(REL, b"x")
    assert storage.open(REL).read_bytes() == b"x"
    assert (real / "images").is_dir()


# --- atomic writes ---------------------------------------------------------------------


def test_a_failure_before_the_swap_leaves_no_file_and_no_directories(
    storage: LocalImageStorage, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_replace(*_: object, **__: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(local.os, "replace", failing_replace)
    with pytest.raises(OSError, match="disk full"):
        storage.save(REL, b"partial?")
    assert entries_under(root) == []


def test_a_failure_during_write_leaves_nothing(
    storage: LocalImageStorage, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_fsync(_: int) -> None:
        raise OSError("I/O error")

    monkeypatch.setattr(local.os, "fsync", failing_fsync)
    with pytest.raises(OSError, match="I/O error"):
        storage.save(REL, b"x" * 100_000)
    assert entries_under(root) == []


def test_unwritable_data_leaves_nothing(storage: LocalImageStorage, root: Path) -> None:
    with pytest.raises(TypeError):
        storage.save(REL, "text, not bytes")  # type: ignore[arg-type]
    assert entries_under(root) == []


def test_a_keyboard_interrupt_mid_write_still_cleans_up(
    storage: LocalImageStorage, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupted(_: int) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(local.os, "fsync", interrupted)
    with pytest.raises(KeyboardInterrupt):
        storage.save(REL, b"x")
    assert entries_under(root) == []


def test_a_failed_overwrite_keeps_the_previous_file_intact(
    storage: LocalImageStorage, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage.save(REL, b"original contents")

    def failing_replace(*_: object, **__: object) -> None:
        raise OSError("boom")

    monkeypatch.setattr(local.os, "replace", failing_replace)
    with pytest.raises(OSError):
        storage.save(REL, b"replacement")
    monkeypatch.undo()

    assert storage.open(REL).read_bytes() == b"original contents"
    assert [p.name for p in files_under(root)] == ["original.jpg"]  # no temp file


def test_a_failed_save_keeps_directories_that_still_hold_other_files(
    storage: LocalImageStorage, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sibling = "images/tree_classification/2026/10/aaaaaaaa-1b2c-4d3e-8f90-a1b2c3d4e5f6/original.jpg"
    storage.save(sibling, b"keep me")

    def failing_replace(*_: object, **__: object) -> None:
        raise OSError("boom")

    monkeypatch.setattr(local.os, "replace", failing_replace)
    with pytest.raises(OSError):
        storage.save(REL, b"x")
    monkeypatch.undo()
    assert storage.open(sibling).read_bytes() == b"keep me"


# --- delete and pruning ----------------------------------------------------------------


def test_delete_removes_the_file_and_prunes_empty_parents_but_not_the_root(
    storage: LocalImageStorage, root: Path
) -> None:
    storage.save(REL, b"x")
    storage.delete(REL)
    assert not storage.exists(REL)
    assert root.is_dir()
    assert entries_under(root) == []  # images/, model, year, month, analysis dirs are all gone


def test_delete_is_idempotent(storage: LocalImageStorage) -> None:
    storage.save(REL, b"x")
    storage.delete(REL)
    storage.delete(REL)  # already gone, including its directories
    storage.delete("images/leaf_disease/1999/01/never-existed/result.png")


def test_delete_keeps_directories_that_still_contain_other_files(
    storage: LocalImageStorage, root: Path
) -> None:
    folder = "images/tree_classification/2026/10/6f1c2a9e-1b2c-4d3e-8f90-a1b2c3d4e5f6"
    other = "images/tree_classification/2026/10/bbbbbbbb-1b2c-4d3e-8f90-a1b2c3d4e5f6/original.jpg"
    storage.save(f"{folder}/original.jpg", b"1")
    storage.save(f"{folder}/thumb.webp", b"2")
    storage.save(other, b"3")

    storage.delete(f"{folder}/original.jpg")
    assert storage.exists(f"{folder}/thumb.webp")  # same folder, still occupied

    storage.delete(f"{folder}/thumb.webp")
    assert not (root / Path(*folder.split("/"))).exists()  # emptied folder is pruned
    assert storage.exists(other)  # sibling and shared parents survive
    assert (root / "images" / "tree_classification" / "2026" / "10").is_dir()


def test_delete_never_removes_the_root_even_when_it_holds_nothing_else(
    storage: LocalImageStorage, root: Path
) -> None:
    storage.save(
        "images/leaf_disease/2026/01/cccccccc-1b2c-4d3e-8f90-a1b2c3d4e5f6/result.png", b"x"
    )
    storage.delete("images/leaf_disease/2026/01/cccccccc-1b2c-4d3e-8f90-a1b2c3d4e5f6/result.png")
    assert root.exists()


def test_delete_refuses_to_delete_a_directory(storage: LocalImageStorage, root: Path) -> None:
    storage.save(REL, b"x")
    with pytest.raises(InvalidStoragePathError):
        storage.delete("images/tree_classification")
    assert storage.exists(REL)


def test_pruning_stops_quietly_when_a_directory_cannot_be_removed(
    storage: LocalImageStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage.save(REL, b"x")

    def locked(self: Path) -> None:
        raise PermissionError("in use")

    monkeypatch.setattr(Path, "rmdir", locked)
    storage.delete(REL)  # the file goes; the directory stays; no exception
    monkeypatch.undo()
    assert not storage.exists(REL)


# --- concurrency -----------------------------------------------------------------------


def test_concurrent_saves_and_deletes_in_the_same_month_do_not_race(
    storage: LocalImageStorage, root: Path
) -> None:
    # Requests run in a threadpool; deleting the last file of a month while
    # another request is creating its folder must not fail either of them.
    created = datetime(2026, 10, 1, tzinfo=UTC)
    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def worker() -> None:
        try:
            barrier.wait()
            for _ in range(25):
                path = build_paths("tree_classification", uuid.uuid4(), created).original
                storage.save(path, b"x")
                assert storage.exists(path)
                storage.delete(path)
        except BaseException as exc:
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=8) as pool:
        for _ in range(8):
            pool.submit(worker)
    assert errors == []
    assert entries_under(root) == []
