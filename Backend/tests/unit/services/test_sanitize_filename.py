from __future__ import annotations

import pytest

from app.services.analysis_service import sanitize_filename


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("banana.jpg", "banana.jpg"),
        ("my photo (1).JPG", "my photo (1).JPG"),
        ("../../etc/passwd", "passwd"),  # path components stripped
        ("C:\\fakepath\\IMG_0042.jpg", "IMG_0042.jpg"),  # what browsers send on Windows
        ("a/b\\c/d.png", "d.png"),  # mixed separators
        ("/absolute/path/leaf.webp", "leaf.webp"),
        ("tab\tin\nname\x00.jpg", "tabinname.jpg"),  # control characters dropped
        ("evil\u202egpj.exe", "evilgpj.exe"),  # right-to-left override is a format character
        ("zero\u200bwidth.jpg", "zerowidth.jpg"),
        ("  padded.jpg  ", "padded.jpg"),
        ("naïve-фото-香蕉.jpg", "naïve-фото-香蕉.jpg"),  # ordinary Unicode is kept
    ],
)
def test_filename_is_cleaned_for_display(raw: str, expected: str) -> None:
    assert sanitize_filename(raw) == expected


@pytest.mark.parametrize("raw", [None, "", " ", "/", "..", ".", "dir/", "\x00\x01", "../"])
def test_names_with_nothing_displayable_become_none(raw: str | None) -> None:
    assert sanitize_filename(raw) is None


def test_long_names_are_truncated_to_the_column_size() -> None:
    cleaned = sanitize_filename("x" * 1000 + ".jpg")
    assert cleaned is not None and len(cleaned) == 255


def test_truncation_counts_utf16_units_like_sql_server() -> None:
    # An emoji is 2 UTF-16 code units; 300 of them must still fit NVARCHAR(255).
    cleaned = sanitize_filename("🍌" * 300)
    assert cleaned is not None
    assert len(cleaned.encode("utf-16-le")) // 2 <= 255
    assert cleaned == "🍌" * 127  # 127 whole emoji = 254 units; no half surrogate left over


def test_the_result_is_never_used_for_storage_paths_so_traversal_text_is_harmless() -> None:
    cleaned = sanitize_filename("..\\..\\windows\\system32\\config")
    assert cleaned == "config"
    assert "/" not in (cleaned or "") and "\\" not in (cleaned or "")
