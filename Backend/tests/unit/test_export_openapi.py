from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts import export_openapi


def test_export_writes_a_stable_schema_with_the_documented_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "openapi.json"
    monkeypatch.setattr(export_openapi, "OUTPUT_PATH", out)

    assert export_openapi.main() == 0
    first = out.read_text(encoding="utf-8")
    schema = json.loads(first)

    paths = set(schema["paths"])
    assert "/api/v1/tree/predict" in paths
    assert "/api/v1/leaf-segmentation/predict" in paths
    assert "/api/v1/analyses" in paths
    assert "/api/v1/analyses/{analysis_id}" in paths or any(
        p.startswith("/api/v1/analyses/{") for p in paths
    )
    assert "/api/v1/models" in paths

    # Deterministic output: exporting twice must not produce a diff.
    export_openapi.main()
    assert out.read_text(encoding="utf-8") == first
    assert first.endswith("\n")


def test_details_union_is_discriminated_on_required_kind() -> None:
    schema = export_openapi.build_schema()
    components = schema["components"]
    assert isinstance(components, dict)
    schemas = components["schemas"]
    assert isinstance(schemas, dict)

    # `kind` must be REQUIRED on both detail models, otherwise the generated
    # TypeScript cannot narrow the union.
    assert "kind" in schemas["TreeDetails"]["required"]
    assert "kind" in schemas["LeafSegDetails"]["required"]
