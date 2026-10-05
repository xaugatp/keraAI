"""Write Backend/openapi.json so the frontend can generate its TypeScript types.

    python -m scripts.export_openapi

The schema is built from the app object only: the lifespan never runs, so no
database connection is made and no model is loaded. `openapi.json` is committed;
re-run this after any API/schema change and then regenerate the frontend types
(`npm run gen:api` in Frontend/).
"""

from __future__ import annotations

import json
from pathlib import Path

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "openapi.json"


def build_schema() -> dict[str, object]:
    from app.main import create_app

    return create_app().openapi()


def main() -> int:
    schema = build_schema()
    # sort_keys + trailing newline keep the committed file diff-friendly: a real
    # API change shows up as a real diff, not as key reordering noise.
    OUTPUT_PATH.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths = schema.get("paths")
    count = len(paths) if isinstance(paths, dict) else 0
    print(f"Wrote {OUTPUT_PATH} ({count} paths)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
