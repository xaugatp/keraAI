# KeraAI — Project Guide for Claude Code

## What this project is
KeraAI is a full-stack computer-vision web app for banana farming. A user photographs a
banana plant or leaf (camera or upload) and sends it with its GPS location. The backend runs a
deep-learning model, saves the image to local disk and the result to SQL Server, and the
frontend shows the result plus a browsable history.

| # | Model key            | Purpose                                         | Arch          | Status       |
|---|----------------------|-------------------------------------------------|---------------|--------------|
| 1 | `tree_classification`| Banana tree vs non-banana tree                  | YOLOv8-cls    | **Building** |
| 2 | `leaf_segmentation`  | Separate healthy green tissue vs damaged tissue | U-Net-type    | Planned      |
| 3 | `leaf_disease`       | Healthy vs diseased leaf (+ disease type)       | U-Net-type    | Planned      |

Runtime topology:
```
Browser/phone → Netlify (React app) → Cloudflare Tunnel → FastAPI on owner's Windows laptop
                                                          ├─ SQL Server (localhost only)
                                                          └─ Images on local disk (DATA_ROOT)
```

## Repository layout
```
/Frontend   React 19 + Vite + Tailwind (currently uses mock data in src/data/mockData.ts)
/Backend    FastAPI service — FULL BUILD SPEC: Backend/docs/BACKEND_SPEC.md
CLAUDE.md   this file
```

## Working agreement — read before doing anything
1. **The owner (Saugat) makes all product, design and architecture decisions. You implement.**
2. STOP and ask (present 2–3 options with trade-offs and a clear recommendation) before you:
   - add a dependency not listed in the spec,
   - change the database schema or the API contract,
   - deviate from the spec in any way, or resolve an "Open decision" on your own.
3. Work **phase by phase** exactly as defined in the spec. At every **STOP gate**: summarise what
   you did, list files changed, tell the owner how to verify it, and wait for approval.
4. Small, focused commits using Conventional Commits (`feat(backend): …`, `fix:`, `test:`,
   `refactor:`, `chore:`, `docs:`). Run lint + tests before every commit. Never commit failing tests.
5. Never commit: `.env`, model weights (`*.pt`, `*.pth`, `*.onnx`), anything under `DATA_ROOT`,
   `.venv/`, `node_modules/`, `__pycache__/`.
6. Never silently work around a problem (missing ODBC driver, wrong weights, failing test,
   unclear requirement). Report it with the exact error and your suggested fix.
7. The owner is an ML engineer growing into full-stack. When you make a non-obvious engineering
   choice, explain the *why* in 1–3 sentences (commit body or a short code comment).
8. Record every significant decision as a short ADR in `Backend/docs/adr/NNNN-title.md`
   (Context → Decision → Consequences).

## Environment
- Windows laptop, PowerShell. Python **3.11** virtual env at `Backend/.venv`.
- SQL Server (local instance), managed with SSMS. Python connects via **ODBC Driver 18**.
- Inference device: auto-detected (CUDA if available, else CPU) — configurable.

## Backend commands (run from `Backend/`)
```powershell
.venv\Scripts\activate
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000   # dev server (localhost only!)
pytest                                                        # fast tests (no torch, SQLite)
pytest -m "mssql or model"                                    # slow tests: real SQL Server / real weights
ruff check . ; ruff format --check .                          # lint + format check
mypy app                                                      # type check
alembic upgrade head                                          # apply migrations
alembic revision --autogenerate -m "describe change"          # new migration (review it!)
python -m scripts.check_env                                   # verify python/ODBC/DB/weights
python -m scripts.predict_cli --model tree_classification path\to\image.jpg
python -m scripts.seed_samples --model tree_classification
python -m scripts.export_openapi                              # writes Backend/openapi.json
```

## Backend conventions (summary — details in the spec)
- Layering: `api` (HTTP only) → `services` (orchestration) → `ml` / `storage` / `repositories` → `db`.
  Routers contain no business logic. Repositories contain no business rules.
- Python 3.11, full type hints, Pydantic v2, SQLAlchemy 2.0 typed ORM (`Mapped[...]`).
- Config only via `app/core/config.py` (pydantic-settings, `.env`). No hard-coded paths/secrets.
- Errors: raise `AppError` subclasses; global handlers return RFC 9457 `application/problem+json`.
- Logging via stdlib `logging` (configured once); never `print`. Every log line carries `request_id`.
- Schema changes only through Alembic migrations. Never `Base.metadata.create_all()` in the app.
- Tests never import `torch`/`ultralytics` unless marked `@pytest.mark.model`.
