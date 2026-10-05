"""scripts/make_dummy_weights.py safety rules, with the (torch) builders faked."""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts import make_dummy_weights as script

from tests.fixtures.settings import make_settings


@pytest.fixture
def targets(tmp_path: Path, monkeypatch):
    """Point both weights paths at tmp files and replace the torch builders with stubs."""
    tree, leaf = tmp_path / "tree.pt", tmp_path / "leaf.pt"
    monkeypatch.setattr(
        script,
        "get_settings",
        lambda: make_settings(tree_weights_path=str(tree), leaf_seg_weights_path=str(leaf)),
    )
    built: list[str] = []

    def fake_tree(path: Path) -> None:
        built.append("tree")
        path.write_bytes(b"DUMMY-TREE")

    def fake_leaf(path: Path, encoder: str) -> None:
        built.append(f"leaf:{encoder}")
        path.write_bytes(b"DUMMY-LEAF")

    monkeypatch.setattr(script, "make_tree_dummy", fake_tree)
    monkeypatch.setattr(script, "make_leaf_seg_dummy", fake_leaf)
    return tree, leaf, built


def test_creates_both_files_and_prints_the_loud_banner_and_env_lines(targets, capsys):
    tree, leaf, built = targets
    assert script.main([]) == 0
    out = capsys.readouterr().out

    assert tree.read_bytes() == b"DUMMY-TREE" and leaf.read_bytes() == b"DUMMY-LEAF"
    assert built == ["tree", "leaf:resnet34"]  # encoder comes from settings
    assert "DUMMY WEIGHTS" in out and "MEANINGLESS" in out
    for line in (
        "TREE_MODEL_VERSION=tree_cls_dummy_v0",
        "TREE_IS_PLACEHOLDER=true",
        "LEAF_SEG_MODEL_VERSION=leaf_seg_dummy_v0",
        "LEAF_SEG_IS_PLACEHOLDER=true",
    ):
        assert line in out
    assert "did not modify .env" in out


def test_refuses_to_overwrite_existing_weights(targets, capsys):
    tree, leaf, built = targets
    tree.write_bytes(b"REAL-WEIGHTS")

    assert script.main(["--model", "tree_classification"]) == 1

    captured = capsys.readouterr()
    assert tree.read_bytes() == b"REAL-WEIGHTS"  # untouched
    assert built == []
    assert "REFUSING to overwrite" in captured.err and "--force" in captured.err
    assert "DUMMY WEIGHTS" not in captured.out  # nothing was written, so no banner


def test_refusal_does_not_block_the_other_model(targets, capsys):
    tree, leaf, built = targets
    tree.write_bytes(b"REAL-WEIGHTS")
    assert script.main([]) == 1  # non-zero because one was refused...
    assert tree.read_bytes() == b"REAL-WEIGHTS"
    assert leaf.read_bytes() == b"DUMMY-LEAF"  # ...but the missing one was still created


def test_force_overwrites(targets):
    tree, leaf, built = targets
    tree.write_bytes(b"OLD")
    assert script.main(["--model", "tree_classification", "--force"]) == 0
    assert tree.read_bytes() == b"DUMMY-TREE"


def test_model_selection_only_builds_the_requested_one(targets):
    tree, leaf, built = targets
    assert script.main(["--model", "leaf_segmentation"]) == 0
    assert built == ["leaf:resnet34"]
    assert not tree.exists()


def test_documented_env_lines_match_the_spec():
    assert script.ENV_LINES == (
        "TREE_MODEL_VERSION=tree_cls_dummy_v0",
        "TREE_IS_PLACEHOLDER=true",
        "LEAF_SEG_MODEL_VERSION=leaf_seg_dummy_v0",
        "LEAF_SEG_IS_PLACEHOLDER=true",
    )
    assert script.TREE_CLASS_NAMES == {0: "banana_tree", 1: "non_banana"}
