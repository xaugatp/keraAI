"""Make the generated OpenAPI document tell the truth about the API contract.

Two things FastAPI cannot express by itself are corrected once, after it has built
the schema:

* Errors. FastAPI documents them as ``application/json`` and always advertises its
  own ``HTTPValidationError`` for 422. We answer every error with
  ``application/problem+json`` and the ``ProblemDetail`` shape (spec 11).
* ``X-Client-Id``. It is required (a missing header is our own 400
  ``MISSING_CLIENT_ID``), but it has to be *optional in the code* so FastAPI does not
  turn a missing header into its generic 422. The generated TypeScript client should
  nevertheless make callers pass it.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI

PROBLEM_JSON = "application/problem+json"
_PROBLEM_REF = {"$ref": "#/components/schemas/ProblemDetail"}
_FASTAPI_VALIDATION_SCHEMAS = ("HTTPValidationError", "ValidationError")
_CLIENT_ID_HEADER = "X-Client-Id"


def _require_client_id(operation: dict[str, Any]) -> None:
    for parameter in operation.get("parameters", []):
        if parameter.get("in") == "header" and parameter.get("name") == _CLIENT_ID_HEADER:
            parameter["required"] = True
            parameter["schema"] = {"type": "string", "format": "uuid", "title": _CLIENT_ID_HEADER}


def _correct_schema(schema: dict[str, Any]) -> None:
    for path_item in schema.get("paths", {}).values():
        for operation in path_item.values():
            if not isinstance(operation, dict):
                continue
            _require_client_id(operation)
            for status_code, response in operation.get("responses", {}).items():
                if not (status_code.isdigit() and int(status_code) >= 400):
                    continue
                content = response.setdefault("content", {})
                if status_code == "422":
                    # Our handler returns ProblemDetail + `errors`, not FastAPI's default body.
                    response["description"] = "Validation error"
                    content.clear()
                    content[PROBLEM_JSON] = {"schema": _PROBLEM_REF}
                elif "application/json" in content:
                    content[PROBLEM_JSON] = content.pop("application/json")

    # Drop FastAPI's own validation-error schemas once nothing refers to them.
    components = schema.get("components", {}).get("schemas", {})
    serialised = json.dumps(schema.get("paths", {}))
    for name in _FASTAPI_VALIDATION_SCHEMAS:
        if f"#/components/schemas/{name}" not in serialised:
            components.pop(name, None)


def install_problem_json_openapi(app: FastAPI) -> None:
    """Wrap ``app.openapi`` so the schema is corrected exactly once, then cached."""
    build = app.openapi

    def openapi() -> dict[str, Any]:
        schema = app.openapi_schema
        if schema is None:
            # build() caches the very dict it returns on app.openapi_schema, so
            # correcting it in place corrects every later call too.
            schema = build()
            _correct_schema(schema)
        return schema

    app.openapi = openapi  # type: ignore[method-assign]
