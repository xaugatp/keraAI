"""Local-disk implementation of ``ImageStorage`` rooted at DATA_ROOT."""

from __future__ import annotations

import contextlib
import logging
import os
import re
import threading
import uuid
from pathlib import Path

from app.storage.base import InvalidStoragePathError, StoredFileNotFoundError

logger = logging.getLogger(__name__)

# Matches the NVARCHAR(400) path columns in the database; also bounds the
# work done on a hostile string.
_MAX_REL_PATH_CHARS = 400

# Deliberately an allowlist, not a blocklist. Every path we ever build is
# `images/<model_key>/<YYYY>/<MM>/<uuid>/<file>`, so a segment must start and
# end with an ASCII letter/digit and may contain only `._-` in between. That one
# rule excludes the whole family of Windows tricks at once: spaces and trailing
# dots (silently stripped by Win32), `:` (drive letters, NTFS alternate data
# streams), `\` (UNC / separators), `~` (8.3 short names), `<>"|?*`, control
# characters, look-alike Unicode and hidden/temporary dot-files.
_SEGMENT_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?")

# Win32 treats these as devices in every directory, with or without an extension
# ("con.txt" is still CON).
_RESERVED_DEVICE_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{n}" for n in range(1, 10)}
    | {f"lpt{n}" for n in range(1, 10)}
)


def _validate_rel_path(rel_path: str) -> None:
    """Reject anything that is not a plain, relative, forward-slash path."""
    if not rel_path:
        raise InvalidStoragePathError("Storage path is empty")
    if len(rel_path) > _MAX_REL_PATH_CHARS:
        raise InvalidStoragePathError("Storage path is too long")
    if rel_path.startswith("/"):
        raise InvalidStoragePathError(f"Absolute storage path rejected: {rel_path!r}")
    if "\\" in rel_path:
        raise InvalidStoragePathError(f"Backslash in storage path rejected: {rel_path!r}")
    if ":" in rel_path:
        raise InvalidStoragePathError(f"Drive letter / stream marker rejected: {rel_path!r}")
    for segment in rel_path.split("/"):
        if segment in {"", ".", ".."}:
            raise InvalidStoragePathError(f"Empty or relative segment rejected: {rel_path!r}")
        if not _SEGMENT_RE.fullmatch(segment):
            raise InvalidStoragePathError(f"Illegal characters in storage path: {rel_path!r}")
        if segment.split(".", 1)[0].lower() in _RESERVED_DEVICE_NAMES:
            raise InvalidStoragePathError(f"Reserved device name rejected: {rel_path!r}")


class LocalImageStorage:
    """Stores files under ``root`` (DATA_ROOT) using atomic writes.

    Relative paths come from database rows or ``build_paths``, never from
    clients, but are still validated and resolved on every call so a corrupted
    row (or a future bug) cannot read, write or delete outside ``root``.
    """

    def __init__(self, root: Path | str) -> None:
        # Resolved once so comparisons are against the real location even if
        # DATA_ROOT itself is a symlink/junction or has odd casing.
        self._root = Path(root).resolve()
        # Serialises directory creation against empty-directory pruning:
        # inference requests run in a threadpool, and without this a prune
        # could remove a month folder between mkdir() creating it and the
        # analysis folder inside it.
        self._dir_lock = threading.Lock()

    @property
    def root(self) -> Path:
        return self._root

    def _resolve(self, rel_path: str) -> Path:
        _validate_rel_path(rel_path)
        resolved = self._root.joinpath(*rel_path.split("/")).resolve()
        # Final containment check after following any symlinks/junctions on
        # the way. (A link swapped in between this check and the I/O is not
        # defended against: the app is the only writer under DATA_ROOT.)
        if resolved == self._root or not resolved.is_relative_to(self._root):
            raise InvalidStoragePathError(f"Storage path escapes DATA_ROOT: {rel_path!r}")
        return resolved

    def _prune_empty_dirs(self, start: Path) -> None:
        """Remove ``start`` and its now-empty ancestors, stopping below DATA_ROOT."""
        current = start
        with self._dir_lock:
            while current != self._root and current.is_relative_to(self._root):
                try:
                    current.rmdir()
                except FileNotFoundError:
                    pass  # already gone; keep climbing
                except OSError:
                    return  # not empty (or in use): stop here
                current = current.parent

    def save(self, rel_path: str, data: bytes) -> None:
        """Write atomically: temp file next to the target, then ``os.replace``.

        A reader (or a crash) therefore sees either the old file or the new
        one, never a partial write. On any failure the temp file is removed
        and so are directories this call created.
        """
        target = self._resolve(rel_path)
        # Same directory => same volume, which is what makes os.replace atomic.
        tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            with self._dir_lock:
                target.parent.mkdir(parents=True, exist_ok=True)
                handle = tmp.open("xb")
            with handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, target)
        except BaseException:
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
            self._prune_empty_dirs(target.parent)
            raise
        logger.debug("stored %d bytes at %s", len(data), rel_path)

    def open(self, rel_path: str) -> Path:
        """Return the absolute path of a stored file (for ``FileResponse``)."""
        target = self._resolve(rel_path)
        if not target.is_file():
            raise StoredFileNotFoundError(f"No stored file at {rel_path!r}")
        return target

    def exists(self, rel_path: str) -> bool:
        return self._resolve(rel_path).is_file()

    def delete(self, rel_path: str) -> None:
        """Delete a file if present, then prune empty parent directories.

        Idempotent: deleting something that is already gone is not an error,
        which keeps compensation/cleanup code simple.
        """
        target = self._resolve(rel_path)
        if target.is_dir():
            raise InvalidStoragePathError(f"Refusing to delete a directory: {rel_path!r}")
        target.unlink(missing_ok=True)
        self._prune_empty_dirs(target.parent)
