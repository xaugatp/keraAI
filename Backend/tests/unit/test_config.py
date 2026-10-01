from __future__ import annotations

import pytest
from pydantic import ValidationError
from pydantic_core import PydanticUndefined

from app.core.config import Settings, get_settings


def _base_kwargs(**overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "signing_secret": "x" * 40,
        "data_root": "C:/kera-data-test",
    }
    kwargs.update(overrides)
    return kwargs


def test_signing_secret_and_data_root_are_required() -> None:
    assert Settings.model_fields["signing_secret"].default is PydanticUndefined
    assert Settings.model_fields["data_root"].default is PydanticUndefined


def test_signing_secret_must_be_at_least_32_chars() -> None:
    with pytest.raises(ValidationError):
        Settings(**_base_kwargs(signing_secret="too-short"))


def test_blank_db_port_becomes_none() -> None:
    settings = Settings(**_base_kwargs(db_port=""))
    assert settings.db_port is None


def test_db_port_still_parses_a_real_port() -> None:
    settings = Settings(**_base_kwargs(db_port="1433"))
    assert settings.db_port == 1433


def test_api_v1_prefix_must_start_with_slash() -> None:
    with pytest.raises(ValidationError):
        Settings(**_base_kwargs(api_v1_prefix="api/v1"))


def test_api_v1_prefix_strips_trailing_slash() -> None:
    settings = Settings(**_base_kwargs(api_v1_prefix="/api/v1/"))
    assert settings.api_v1_prefix == "/api/v1"


def test_sql_login_requires_db_user() -> None:
    with pytest.raises(ValidationError):
        Settings(**_base_kwargs(db_trusted_connection=False, db_user=""))


def test_windows_auth_does_not_require_db_user() -> None:
    settings = Settings(**_base_kwargs(db_trusted_connection=True, db_user=""))
    assert settings.db_trusted_connection is True


def test_tree_settings_grouping() -> None:
    settings = Settings(
        **_base_kwargs(
            tree_enabled=True,
            tree_weights_path="weights/tree_cls_v1.pt",
            tree_model_version="tree_cls_v1",
            tree_imgsz=224,
            tree_positive_class="banana_tree",
            tree_uncertain_threshold=0.6,
            tree_display_names={"banana_tree": "Banana tree"},
        )
    )
    tree = settings.tree
    assert tree.enabled is True
    assert tree.positive_class == "banana_tree"
    # pydantic-settings deep-merges dict-typed fields across sources (init
    # kwargs here, TREE_DISPLAY_NAMES in the real Backend/.env this test run
    # also picks up) rather than replacing wholesale — assert the explicit
    # key landed, not that nothing else did.
    assert tree.display_names["banana_tree"] == "Banana tree"


def test_get_settings_is_cached_until_cleared(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIGNING_SECRET", "x" * 40)
    monkeypatch.setenv("DATA_ROOT", "C:/kera-data-test")
    get_settings.cache_clear()
    try:
        first = get_settings()
        second = get_settings()
        assert first is second
        get_settings.cache_clear()
        third = get_settings()
        assert third is not first
    finally:
        get_settings.cache_clear()
