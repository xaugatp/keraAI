"""The generated OpenAPI document is what the frontend builds its types from."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.integration.api.helpers import PREFIX

PROBLEM_JSON = "application/problem+json"
PROBLEM_REF = "#/components/schemas/ProblemDetail"


@pytest.fixture
def schema(api: TestClient) -> dict[str, Any]:
    response = api.get("/openapi.json")
    assert response.status_code == 200
    document: dict[str, Any] = response.json()
    return document


def _operations(schema: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    return [
        (method.upper(), path, operation)
        for path, item in schema["paths"].items()
        if path.startswith(PREFIX)
        for method, operation in item.items()
    ]


def test_all_spec_endpoints_exist(schema: dict[str, Any]) -> None:
    assert {(method, path) for method, path, _ in _operations(schema)} == {
        ("GET", f"{PREFIX}/models"),
        ("POST", f"{PREFIX}/tree/predict"),
        ("POST", f"{PREFIX}/leaf-segmentation/predict"),
        ("POST", f"{PREFIX}/leaf-disease/predict"),
        ("GET", f"{PREFIX}/analyses"),
        ("GET", f"{PREFIX}/analyses/{{analysis_id}}"),
        ("GET", f"{PREFIX}/analyses/{{analysis_id}}/image"),
        ("DELETE", f"{PREFIX}/analyses/{{analysis_id}}"),
    }
    assert "/health" in schema["paths"] and "/health/ready" in schema["paths"]


def test_details_is_a_discriminated_union_on_kind(schema: dict[str, Any]) -> None:
    details = schema["components"]["schemas"]["AnalysisDetail"]["properties"]["details"]
    # `details` is nullable (a failed analysis has none), so pydantic wraps the union in anyOf.
    union = next(option for option in details["anyOf"] if "oneOf" in option)

    assert {option["$ref"] for option in union["oneOf"]} == {
        "#/components/schemas/TreeDetails",
        "#/components/schemas/LeafSegDetails",
        "#/components/schemas/LeafDiseaseDetails",
    }
    assert union["discriminator"]["propertyName"] == "kind"
    assert union["discriminator"]["mapping"] == {
        "tree_classification": "#/components/schemas/TreeDetails",
        "leaf_segmentation": "#/components/schemas/LeafSegDetails",
        "leaf_disease": "#/components/schemas/LeafDiseaseDetails",
    }


def test_every_operation_documents_problem_json_errors(schema: dict[str, Any]) -> None:
    for method, path, operation in _operations(schema):
        errors = {
            code: response for code, response in operation["responses"].items() if int(code) >= 400
        }
        assert errors, f"{method} {path} documents no error responses"
        for code, response in errors.items():
            assert list(response["content"]) == [PROBLEM_JSON], (method, path, code)
            assert response["content"][PROBLEM_JSON]["schema"] == {"$ref": PROBLEM_REF}


@pytest.mark.parametrize(
    ("method", "path", "expected"),
    [
        ("POST", "/tree/predict", {"400", "413", "415", "422", "429", "500", "503"}),
        ("POST", "/leaf-segmentation/predict", {"400", "413", "415", "422", "429", "500", "503"}),
        ("POST", "/leaf-disease/predict", {"400", "413", "415", "422", "429", "500", "503"}),
        ("GET", "/analyses", {"400", "422"}),
        ("GET", "/analyses/{analysis_id}", {"400", "404", "422"}),
        ("GET", "/analyses/{analysis_id}/image", {"403", "404", "422"}),
        ("DELETE", "/analyses/{analysis_id}", {"400", "403", "404", "422"}),
    ],
)
def test_error_codes_documented_per_route(
    schema: dict[str, Any], method: str, path: str, expected: set[str]
) -> None:
    documented = set(schema["paths"][f"{PREFIX}{path}"][method.lower()]["responses"])
    assert expected <= documented


def test_problem_detail_component_matches_the_real_error_body(schema: dict[str, Any]) -> None:
    problem = schema["components"]["schemas"]["ProblemDetail"]
    assert {"type", "title", "status", "detail", "instance", "code", "request_id"} <= set(
        problem["properties"]
    )
    # FastAPI's own (never-returned) validation schemas are gone.
    assert "HTTPValidationError" not in schema["components"]["schemas"]
    assert "ValidationError" not in schema["components"]["schemas"]


def test_predict_documents_201_with_a_location_header(schema: dict[str, Any]) -> None:
    for path in ("/tree/predict", "/leaf-segmentation/predict", "/leaf-disease/predict"):
        created = schema["paths"][f"{PREFIX}{path}"]["post"]["responses"]["201"]
        assert created["content"]["application/json"]["schema"] == {
            "$ref": "#/components/schemas/AnalysisDetail"
        }
        assert "Location" in created["headers"]


def test_predict_form_fields_and_constraints_are_documented(schema: dict[str, Any]) -> None:
    body = schema["components"]["schemas"]["Body_predict_tree_api_v1_tree_predict_post"]
    props = body["properties"]
    assert body["required"] == ["image"]
    assert props["image"]["contentMediaType"] == "application/octet-stream"  # OpenAPI 3.1 "file"
    assert set(props) == {
        "image",
        "latitude",
        "longitude",
        "gps_accuracy_m",
        "captured_at",
        "source",
    }
    latitude = next(o for o in props["latitude"]["anyOf"] if o.get("type") == "number")
    assert (latitude["minimum"], latitude["maximum"]) == (-90, 90)
    assert props["source"]["enum"] == ["upload", "camera"]  # `sample` is not offered to clients


def test_client_id_header_is_documented_as_required_where_it_is_needed(
    schema: dict[str, Any],
) -> None:
    needs_header = {
        ("POST", "/tree/predict"),
        ("POST", "/leaf-segmentation/predict"),
        ("POST", "/leaf-disease/predict"),
        ("GET", "/analyses"),
        ("GET", "/analyses/{analysis_id}"),
        ("DELETE", "/analyses/{analysis_id}"),
    }
    for method, path, operation in _operations(schema):
        headers = [p for p in operation.get("parameters", []) if p["in"] == "header"]
        key = (method, path.removeprefix(PREFIX))
        if key in needs_header:
            (header,) = headers
            assert header["name"] == "X-Client-Id" and header["required"] is True
            assert header["schema"]["format"] == "uuid"
        else:  # models + signed image URLs need no client id
            assert headers == [], key


def test_image_route_documents_the_media_types_and_query(schema: dict[str, Any]) -> None:
    operation = schema["paths"][f"{PREFIX}/analyses/{{analysis_id}}/image"]["get"]
    assert set(operation["responses"]["200"]["content"]) == {
        "image/jpeg",
        "image/webp",
        "image/png",
    }
    query = {p["name"]: p for p in operation["parameters"] if p["in"] == "query"}
    assert set(query) == {"variant", "exp", "sig"}
    assert query["variant"]["schema"]["enum"] == ["original", "thumbnail", "result"]


def test_history_query_parameters_are_documented(schema: dict[str, Any]) -> None:
    operation = schema["paths"][f"{PREFIX}/analyses"]["get"]
    query = {p["name"]: p["schema"] for p in operation["parameters"] if p["in"] == "query"}
    assert set(query) == {"model_key", "scope", "page", "page_size"}
    assert query["scope"]["enum"] == ["mine", "samples", "all_visible"]
    assert (query["page_size"]["minimum"], query["page_size"]["maximum"]) == (1, 100)
    assert query["page"]["minimum"] == 1


def test_models_response_includes_is_placeholder(schema: dict[str, Any]) -> None:
    model_info = schema["components"]["schemas"]["ModelInfoOut"]
    assert "is_placeholder" in model_info["required"]


def test_response_schemas_use_snake_case_and_spec_field_names(schema: dict[str, Any]) -> None:
    components = schema["components"]["schemas"]
    assert list(components["AnalysisDetail"]["properties"]) == [
        "id",
        "model_key",
        "model_version",
        "status",
        "source",
        "is_sample",
        "title",
        "description",
        "prediction",
        "details",
        "location",
        "image",
        "timings_ms",
        "created_at",
    ]
    assert set(components["AnalysisSummary"]["properties"]) == {
        "id",
        "model_key",
        "source",
        "is_sample",
        "title",
        "display_label",
        "confidence",
        "is_uncertain",
        "thumbnail_url",
        "latitude",
        "longitude",
        "created_at",
    }
    assert set(components["Page_AnalysisSummary_"]["properties"]) == {
        "items",
        "page",
        "page_size",
        "total",
        "total_pages",
    }
    assert set(components["Prediction"]["properties"]) == {
        "label",
        "display_label",
        "confidence",
        "is_uncertain",
    }
    assert set(components["ImageLinks"]["properties"]) == {
        "width",
        "height",
        "original_url",
        "thumbnail_url",
        "result_url",
    }
    assert set(components["TimingsMs"]["properties"]) == {
        "preprocess",
        "inference",
        "postprocess",
        "total",
    }
    assert set(components["GeoLocation"]["properties"]) == {
        "latitude",
        "longitude",
        "accuracy_m",
        "captured_at",
    }


def test_docs_are_served_when_enabled(api: TestClient) -> None:
    assert api.get("/docs").status_code == 200
    assert api.get("/redoc").status_code == 200
