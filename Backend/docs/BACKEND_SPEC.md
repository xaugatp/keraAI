# KeraAI Backend — Build Specification (v1: Model 1 end-to-end)

> Audience: Claude Code. Owner: Saugat. Read `/CLAUDE.md` first — its working agreement applies.
> Goal of v1: Model 1 (banana tree classification) working end-to-end — upload → preprocess →
> predict → postprocess → save image + DB row → return result → history — built so that
> Model 2 and Model 3 plug in by **adding a predictor class + a route**, nothing else.

---

## 0. Decisions log

### Decided by the owner (do not change without asking)
| ID   | Decision |
|------|----------|
| D-01 | FastAPI backend on the owner's laptop, exposed via Cloudflare Tunnel; frontend on Netlify. |
| D-02 | SQL Server stores results; images stored on local disk under `DATA_ROOT` (DB stores relative paths, never blobs). |
| D-03 | **Hybrid schema**: one `analyses` table with common columns + a JSON `details` column for model-specific output. |
| D-04 | **Per-device history**: frontend sends an anonymous UUID in `X-Client-Id`; users see only their own analyses + public samples. No login in v1. |
| D-05 | **Per-model routes** (`POST /api/v1/tree/predict`, later `/leaf-segmentation/predict`, `/leaf-disease/predict`) sharing one internal pipeline. |
| D-06 | **Real data only** — responses contain only what the model/backend actually computes. No invented metrics. |
| D-07 | **Sample images** with pre-computed results (run through the real pipeline by a seed script), flagged `is_sample`, visible to everyone. |
| D-08 | Results table + detail view in the UI → backend provides paginated list, detail, and image endpoints; a thumbnail is generated for every image. |
D-09 Create virtual environment and work onit and activate it after first creation, keep requirements.txt will all packages and versions 

### Proposed by the spec (owner may override)
| ID   | Proposal | Why |
|------|----------|-----|
| P-01 | Sync SQLAlchemy + pyodbc; blocking work (inference, DB, disk) runs in FastAPI's threadpool. | pyodbc is synchronous and mature; async MSSQL drivers are less mature. Event loop stays free. |
| P-02 | Image URLs are **HMAC-signed, expiring** links (`?exp=&sig=`), like S3 pre-signed URLs. Samples are public. | `<img src>` cannot send the `X-Client-Id` header, so header-based access can't protect images. |
| P-03 | Single uvicorn worker; a per-model lock serialises `predict()`. | One model copy in (GPU) memory; Ultralytics models are not guaranteed thread-safe. |
| P-04 | Errors use RFC 9457 Problem Details (`application/problem+json`) + stable `code`. | Industry standard; frontend can switch on `code`. |
| P-05 | Failed inferences are persisted with `status='failed'` + `error_code`; invalid uploads are rejected and **not** persisted. | Observability of model failures without storing junk. |
| P-06 | Uploaded images are re-encoded (EXIF-rotated, **EXIF stripped**) before saving. | Phone EXIF contains GPS/device data; we store location only in explicit columns. |
| P-07 | Soft delete (`deleted_at`) for user deletions. | Recoverable; keeps audit trail. |

### Open decisions — ASK the owner when you reach them (defaults in brackets)
| ID   | Question | Default if owner says "use default" |
|------|----------|------|
| O-01 | SQL Server auth: Windows auth or SQL login `kera_app`? | SQL login `kera_app` (least privilege) |
| O-02 | `DATA_ROOT` location? | `D:/kera-data` (or `C:/kera-data` if no D:) |
| O-03 | "Uncertain" confidence threshold for Model 1? | `0.60` |
| O-04 | Support iPhone HEIC uploads (`pillow-heif` dependency)? | No (browsers usually send JPEG) |
| O-05 | Image retention policy (delete user images after N days)? | Keep forever in v1 |
| O-06 | Production domain names for CORS (`app.<domain>`, `api.<domain>`)? | Placeholder in `.env.example` |
| O-07 | Ultralytics is **AGPL-3.0**. Serving it publicly means the source must be available to users. Open-source the repo, or not? | Flag only; owner decides |

---

## 1. Known issue with the current weights — handle in Phase 0

`Backend/yolov8n/` is **not** the banana tree model. It is an *unzipped* copy of Ultralytics'
stock COCO checkpoint `yolov8n.pt`: `task=detect`, 80 COCO classes, `data=coco.yaml`,
created 2022-12-30 (Ultralytics 8.0.0.dev0). Its `banana` class means banana *fruit*.

- Do **not** use or re-zip this folder for the app. Add `Backend/yolov8n/` to `.gitignore`.
- The app expects the trained classifier at `Backend/weights/tree_cls_v1.pt`, as the original
  `.pt` file (do not extract it). It must report `model.task == "classify"`.
- `scripts/check_env.py` must detect this situation and print a clear message.
- The app must still **start** if weights are missing/invalid: `/health/ready` reports the model as
  `unavailable` and predict returns `503 MODEL_UNAVAILABLE` (degraded mode, not a crash).

---

## 2. Tech stack

Runtime: Python 3.11, `fastapi`, `uvicorn[standard]`, `pydantic` v2, `pydantic-settings`,
`sqlalchemy` 2.x, `pyodbc`, `alembic`, `python-multipart`, `pillow`, `ultralytics`
(+ `torch`, installed first from pytorch.org with the correct CUDA/CPU build), `slowapi`
(rate limiting), `python-json-logger` (JSON logs in production).

Dev: `pytest`, `pytest-cov`, `httpx`, `ruff`, `mypy`.

Packaging: `pyproject.toml` holds tool config (ruff, mypy, pytest markers, coverage).
`requirements.txt` (runtime) and `requirements-dev.txt` (`-r requirements.txt` + dev tools) with
**exact pins** taken from the installed versions. `torch` is NOT pinned in requirements.txt —
document its install command in the README instead (it's machine-specific).

Non-goals for v1: authentication, Docker, async DB driver, background job queue, cloud storage.
Design interfaces so each can be added later without rewrites.

---

## 3. Folder structure

```
Backend/
├── app/
│   ├── __init__.py
│   ├── main.py                    # create_app() factory + lifespan (startup/shutdown)
│   ├── core/
│   │   ├── config.py              # Settings (pydantic-settings) + get_settings() (lru_cache)
│   │   ├── logging.py             # dictConfig, request_id contextvar + filter
│   │   ├── errors.py              # AppError hierarchy + exception handlers (problem+json)
│   │   ├── middleware.py          # RequestID, access log/timing, security headers
│   │   ├── rate_limit.py          # slowapi limiter + Cloudflare-aware key function
│   │   ├── signing.py             # HMAC sign/verify for image URLs
│   │   └── time.py                # utcnow(), to_aware_utc() helpers
│   ├── api/
│   │   ├── deps.py                # Depends providers: settings, db session, client_id, services
│   │   └── v1/
│   │       ├── router.py          # includes all endpoint routers
│   │       └── endpoints/
│   │           ├── health.py      # /health, /health/ready
│   │           ├── models.py      # GET /models
│   │           ├── tree.py        # POST /tree/predict
│   │           └── analyses.py    # list, detail, image, delete
│   ├── ml/
│   │   ├── base.py                # Predictor ABC, PredictionOutput, ModelInfo, Timings
│   │   ├── registry.py            # ModelRegistry: key → Predictor; load/warmup/status
│   │   └── tree_classifier.py     # Model 1: YOLOv8-cls wrapper
│   ├── imaging/
│   │   └── processing.py          # decode/validate, normalise (EXIF, RGB, downscale), thumbnail, encode
│   ├── storage/
│   │   ├── base.py                # ImageStorage Protocol (save/open/delete/exists)
│   │   └── local.py               # LocalImageStorage (atomic writes under DATA_ROOT)
│   ├── db/
│   │   ├── base.py                # DeclarativeBase with naming convention
│   │   ├── session.py             # engine (pool_pre_ping), SessionLocal, get_db
│   │   └── models/
│   │       └── analysis.py        # Analysis ORM model
│   ├── repositories/
│   │   └── analysis_repository.py
│   ├── services/
│   │   └── analysis_service.py    # THE pipeline, shared by all models
│   └── schemas/
│       ├── common.py              # ProblemDetail, Page[T], GeoLocation, enums
│       ├── analysis.py            # AnalysisSummary, AnalysisDetail, ImageLinks, Timings
│       └── tree.py                # TreeDetails, TreeProbability, TreePredictForm
├── alembic/                       # env.py reads URL from Settings; versions/
├── alembic.ini
├── db/sql/001_create_database_and_login.sql   # owner runs this once in SSMS
├── scripts/
│   ├── check_env.py
│   ├── predict_cli.py
│   ├── seed_samples.py
│   └── export_openapi.py
├── samples/tree_classification/   # sample images + manifest.json (committed, keep small)
├── weights/                       # *.pt (git-ignored) + README.md explaining expected files
├── docs/
│   ├── BACKEND_SPEC.md            # this file
│   └── adr/
├── tests/
│   ├── conftest.py                # app fixture, SQLite engine, tmp DATA_ROOT, FakePredictor
│   ├── fixtures/images/           # tiny JPEG/PNG, rotated-EXIF JPEG, corrupt file, huge-dims PNG
│   ├── unit/
│   └── integration/
├── .env.example
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

Dependency rule: `api → services → (ml, imaging, storage, repositories) → db`. Lower layers
never import from `api`. `ml` never imports from `db`.

---

## 4. Configuration (`app/core/config.py` + `.env.example`)

Use `BaseSettings` with `model_config = SettingsConfigDict(env_file=".env", extra="ignore")`.
Build the DB URL with `sqlalchemy.engine.URL.create(...)` from discrete fields (avoids escaping
problems with passwords and `localhost\SQLEXPRESS`). Validate on startup; fail fast on bad config.

```dotenv
# --- app ---
APP_ENV=development            # development | production | test
APP_NAME=KeraAI API
API_V1_PREFIX=/api/v1
ENABLE_DOCS=true               # false in production
LOG_LEVEL=INFO
LOG_JSON=false
CORS_ORIGINS=["http://localhost:3000"]   # add https://app.<domain> in prod
TRUST_CLOUDFLARE_HEADERS=false           # true when behind the tunnel
SIGNING_SECRET=change-me-32+chars        # HMAC key for image URLs
SIGNED_URL_TTL_SECONDS=3600

# --- database ---
DB_HOST=localhost              # or localhost\SQLEXPRESS
DB_PORT=                       # optional
DB_NAME=KeraAI
DB_DRIVER=ODBC Driver 18 for SQL Server
DB_TRUSTED_CONNECTION=false    # true = Windows auth (then user/password ignored)
DB_USER=kera_app
DB_PASSWORD=
DB_TRUST_SERVER_CERT=true      # local dev self-signed cert
DB_POOL_SIZE=5
DB_ECHO=false

# --- storage & uploads ---
DATA_ROOT=D:/kera-data
MAX_UPLOAD_MB=10
MIN_IMAGE_SIDE_PX=64
MAX_IMAGE_PIXELS=40000000      # decompression-bomb guard
STORE_MAX_SIDE_PX=2048         # downscale stored original
THUMBNAIL_SIDE_PX=320
ALLOWED_IMAGE_FORMATS=["JPEG","PNG","WEBP"]

# --- inference ---
INFERENCE_DEVICE=auto          # auto | cpu | cuda:0
RATE_LIMIT_PREDICT=10/minute
RATE_LIMIT_DEFAULT=120/minute

# --- model 1: tree classification ---
TREE_ENABLED=true
TREE_WEIGHTS_PATH=weights/tree_cls_v1.pt
TREE_MODEL_VERSION=tree_cls_v1
TREE_IMGSZ=224
TREE_POSITIVE_CLASS=banana_tree          # must exist in model.names
TREE_UNCERTAIN_THRESHOLD=0.60
TREE_DISPLAY_NAMES={"banana_tree":"Banana tree","non_banana":"Not a banana tree"}
```
Group per-model settings into a nested `TreeModelSettings` object; Model 2/3 will add
`LeafSegSettings` etc. the same way. Startup must verify `TREE_POSITIVE_CLASS` and every key of
`TREE_DISPLAY_NAMES` exist in the loaded `model.names`. If they don't, mark the model unavailable
with a clear reason.

---

## 5. Database design

### 5.1 One-time setup script (`db/sql/001_create_database_and_login.sql`, owner runs in SSMS)
Create databases `KeraAI` and `KeraAI_Test`, login `kera_app` (strong password placeholder),
a user in each DB with `db_datareader`, `db_datawriter`, `db_ddladmin` (ddladmin needed for
Alembic in dev; note in a comment that production would split migration vs app logins).
Add comments explaining: enable **SQL Server and Windows Authentication mode** (server
properties → Security) + restart service if using SQL login; enable TCP/IP in SQL Server
Configuration Manager if the connection fails.

### 5.2 Table `analyses` (created by Alembic migration `0001_create_analyses`)
Use SQLAlchemy portable types so the same model runs on SQL Server and SQLite (tests).
Store datetimes as **UTC naive `DATETIME2`**; convert to aware UTC at the schema boundary.

| Column | Type (SQL Server) | Null | Notes |
|---|---|---|---|
| id | UNIQUEIDENTIFIER | PK | uuid4 generated in Python |
| model_key | NVARCHAR(50) | no | CHECK IN ('tree_classification','leaf_segmentation','leaf_disease') |
| model_version | NVARCHAR(100) | no | e.g. `tree_cls_v1` — every row records which model produced it |
| status | NVARCHAR(20) | no | CHECK IN ('completed','failed') |
| source | NVARCHAR(20) | no | CHECK IN ('upload','camera','sample') |
| is_sample | BIT | no | default 0 |
| title | NVARCHAR(200) | yes | samples only (from manifest) |
| description | NVARCHAR(1000) | yes | samples only |
| client_id | UNIQUEIDENTIFIER | yes | NULL for samples |
| predicted_label | NVARCHAR(100) | yes | raw class key from model |
| display_label | NVARCHAR(200) | yes | human label (or "Uncertain") |
| confidence | FLOAT | yes | 0–1, CHECK range |
| is_uncertain | BIT | no | default 0 |
| latitude | DECIMAL(9,6) | yes | CHECK −90..90 |
| longitude | DECIMAL(9,6) | yes | CHECK −180..180 |
| gps_accuracy_m | FLOAT | yes | ≥ 0 |
| captured_at | DATETIME2(3) | yes | client-reported capture time (UTC) |
| original_image_path | NVARCHAR(400) | no | relative to DATA_ROOT, forward slashes |
| thumbnail_path | NVARCHAR(400) | no | |
| result_image_path | NVARCHAR(400) | yes | overlays/masks (Model 2/3) |
| original_filename | NVARCHAR(255) | yes | sanitised, display only |
| image_width / image_height | INT | yes | of stored original |
| image_sha256 | CHAR(64) | no | of stored bytes; used for sample idempotency |
| preprocess_ms / inference_ms / postprocess_ms / total_ms | INT | yes | |
| details | NVARCHAR(MAX) (SQLAlchemy `JSON`) | yes | model-specific payload |
| error_code | NVARCHAR(50) | yes | when status='failed' |
| error_message | NVARCHAR(1000) | yes | internal message (never sent raw to client) |
| created_at | DATETIME2(3) | no | UTC |
| updated_at | DATETIME2(3) | no | UTC |
| deleted_at | DATETIME2(3) | yes | soft delete |

Indexes: `ix_analyses_client_created (client_id, created_at DESC)`,
`ix_analyses_model_created (model_key, created_at DESC)`,
`ix_analyses_sample (is_sample, model_key)`, `ix_analyses_sha256 (image_sha256)`.

`DeclarativeBase.metadata` uses a naming convention (`pk_%(table_name)s`,
`ix_%(column_0_label)s`, `ck_%(table_name)s_%(constraint_name)s`, `fk_…`, `uq_…`) so migrations
are deterministic. Review every autogenerated migration by hand. Make sure `downgrade()` works.

### 5.3 Repository (`AnalysisRepository`, takes a `Session`)
`add(analysis)`, `get(id) -> Analysis | None` (excludes soft-deleted),
`list(filters: AnalysisFilter, page, page_size) -> tuple[list[Analysis], int]`,
`soft_delete(analysis)`, `get_sample_by_hash(model_key, sha256)`.
`AnalysisFilter`: `model_key`, `scope` (`mine` | `samples` | `all_visible`), `client_id`, `status`.
Default sort `created_at DESC`; `page_size` max 100.
**Access rule** lives in the service: a row is visible if `is_sample` or `client_id == caller`.
Anything else is reported as **404** (never reveal that another user's record exists).

---

## 6. Image handling (`app/imaging/processing.py`) — shared by all models

1. **Read with limit**: stream `UploadFile` in 1 MB chunks; abort past `MAX_UPLOAD_MB` → `413`.
   Also reject early if the `Content-Length` header exceeds the limit.
2. **Decode & validate**: set `Image.MAX_IMAGE_PIXELS`; `Image.open` → `verify()` → reopen.
   Trust the decoded format, not the client's content type. Format not in
   `ALLOWED_IMAGE_FORMATS` → `415`. Undecodable → `400 INVALID_IMAGE`. Too small → `400 IMAGE_TOO_SMALL`.
   Decompression bomb → `400 IMAGE_TOO_LARGE_DIMENSIONS`.
3. **Normalise**: `ImageOps.exif_transpose` (phone photos are often rotated), convert to RGB,
   downscale so the longest side ≤ `STORE_MAX_SIDE_PX` (LANCZOS).
4. **Encode for storage**: JPEG quality 90, optimised, **no EXIF**. Thumbnail: WEBP, longest side
   `THUMBNAIL_SIDE_PX`.
5. Return a small dataclass: `NormalizedImage(image, width, height, jpeg_bytes, thumb_bytes, sha256)`.

Model-specific preprocessing (resize/letterbox/normalise to tensor) belongs **inside each
predictor**, not here.

## 7. Storage (`app/storage/`)
`ImageStorage` Protocol: `save(rel_path, data) -> None`, `open(rel_path) -> Path`,
`delete(rel_path)`, `exists(rel_path)`.
`LocalImageStorage(root)`: path layout
`images/{model_key}/{YYYY}/{MM}/{analysis_id}/original.jpg | thumb.webp | result.png`.
- Write atomically (temp file in the same dir + `os.replace`).
- Resolve and assert every path stays inside `DATA_ROOT` (path-traversal guard).
- Never build a path from client input. Paths come only from DB rows.
- Logs go to `DATA_ROOT/logs/` (rotating file handler, 10 MB × 5).

---

## 8. ML layer (`app/ml/`)

```python
@dataclass(frozen=True)
class ModelInfo:
    key: str; display_name: str; version: str; task: str
    classes: list[str]; status: Literal["ready", "unavailable"]; reason: str | None

@dataclass
class PredictionOutput:
    label: str | None             # raw class key
    display_label: str            # human label or "Uncertain"
    confidence: float | None      # 0..1
    is_uncertain: bool
    details: BaseModel            # model-specific pydantic model (e.g. TreeDetails)
    result_image: Image.Image | None = None   # Model 2/3 overlays
    timings: Timings = ...        # preprocess_ms, inference_ms, postprocess_ms

class Predictor(ABC):
    key: ClassVar[str]
    def load(self) -> None: ...                    # load weights, validate, raise ModelLoadError
    def warmup(self) -> None: ...                  # one dummy inference
    @abstractmethod
    def preprocess(self, image: Image.Image) -> Any: ...
    @abstractmethod
    def infer(self, x: Any) -> Any: ...
    @abstractmethod
    def postprocess(self, raw: Any) -> PredictionOutput: ...
    def predict(self, image: Image.Image) -> PredictionOutput:
        # template method: lock → time each stage → wrap unexpected errors in InferenceError
    @property
    def info(self) -> ModelInfo: ...
```

**`ModelRegistry`**: built in the app lifespan from settings. It loads and warms up each
*enabled* predictor. A load failure is logged with its reason and the model is marked
`unavailable`; the app keeps starting. `get(key)` raises `ModelUnavailableError` (503).
`all_info()` feeds `GET /models` and `/health/ready`. Tests replace the registry with a
`FakePredictor`.

**`TreeClassifier` (Model 1)**:
- `load`: `YOLO(weights_path)`. Assert `model.task == "classify"`; otherwise raise
  `ModelLoadError` with a message like "expected classify, got detect — wrong weights?".
  Validate configured class names against `model.names`. Resolve the device
  (`auto` → `cuda:0` if `torch.cuda.is_available()` else `cpu`).
- `preprocess`: pass the RGB PIL image; Ultralytics resizes to `TREE_IMGSZ`.
- `infer`: `model.predict(img, imgsz=..., device=..., verbose=False)[0].probs`.
- `postprocess`: full probability distribution sorted desc → `top1`, `top1conf`.
  `is_uncertain = top1conf < threshold`.
  `verdict = "uncertain" | "banana_tree" | "not_banana_tree"` (positive class vs anything else).
  `details = TreeDetails(kind="tree_classification", verdict, threshold, probabilities=[...])`.

Adding Model 2/3 later = new `Predictor` subclass + settings group + `XxxDetails` schema +
route file. The service, storage, repository, table and history endpoints stay unchanged.
Keep this true; if you find yourself editing the service for a model-specific reason, stop and
discuss it with the owner.

---

## 9. Service — the shared pipeline (`AnalysisService`)

`analyze(model_key, upload_bytes, original_filename, meta: CaptureMeta, client_id, source, *, is_sample=False, title=None, description=None) -> AnalysisDetail`

1. `predictor = registry.get(model_key)` (503 if unavailable), checked **before** any work.
2. `normalized = processing.normalize(processing.decode(upload_bytes))`.
3. `analysis_id = uuid4()`; build the storage paths.
4. `output = predictor.predict(normalized.image)`. On `InferenceError`, persist a `failed` row
   (image saved so it can be debugged) and then raise `500 INFERENCE_FAILED`.
5. Save original + thumbnail (+ result image) via storage.
6. Build the ORM row; `repo.add`; `session.commit()`.
7. **Compensation**: if step 5 or 6 fails, delete any files already written, roll back, re-raise.
   No orphan files, no rows pointing at missing files.
8. Log one structured line: `analysis_id, model_key, model_version, label, confidence,
   total_ms, client_id(first 8 chars)`. Do not log raw coordinates.
9. Map to `AnalysisDetail` (including signed image URLs) and return it.

Also: `get_detail(id, client_id)`, `list(filter, page, page_size, client_id)`,
`delete(id, client_id)` (samples → `403 SAMPLE_IMMUTABLE`), and
`open_image(id, variant, exp, sig)`.

The route is `async def`. It reads the upload, then calls the service via
`await run_in_threadpool(...)` so inference and DB I/O never block the event loop.
Each request gets its own `Session` from the `get_db` dependency (commit in the service,
rollback + close in the dependency's `finally`).

---

## 10. API contract (`/api/v1`)

Common headers. Request: `X-Client-Id: <uuid4>` is required on all `/analyses*` (except signed
image GETs) and predict routes; missing → `400 MISSING_CLIENT_ID`, malformed →
`400 INVALID_CLIENT_ID`. Response: `X-Request-ID` (echo the incoming one or generate).
Timestamps are ISO-8601 UTC with `Z`. Field naming is `snake_case`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness → `{"status":"ok"}` (no DB/model checks) |
| GET | `/health/ready` | DB ping + model statuses → 200 or 503 with component detail |
| GET | `/api/v1/models` | List `ModelInfo` (for About page / disable unavailable features) |
| POST | `/api/v1/tree/predict` | Model 1 inference → **201** + `Location` header |
| GET | `/api/v1/analyses` | Paginated history. Query: `model_key`, `scope=mine\|samples\|all_visible` (default `all_visible`), `page`, `page_size` |
| GET | `/api/v1/analyses/{id}` | Full detail (visible only) |
| GET | `/api/v1/analyses/{id}/image` | Query `variant=original\|thumbnail\|result`, `exp`, `sig`. Samples need no signature. `FileResponse` + `Cache-Control: private, max-age=3600` (samples: `public, max-age=86400`) |
| DELETE | `/api/v1/analyses/{id}` | Soft delete own analysis → 204 |

**`POST /api/v1/tree/predict`** — `multipart/form-data`:
`image` (file, required), `latitude` (−90..90, optional), `longitude` (−180..180, optional,
both-or-neither), `gps_accuracy_m` (≥0, optional), `captured_at` (ISO datetime, optional),
`source` (`upload|camera`, default `upload`). Rate limited by `RATE_LIMIT_PREDICT` per client IP.

**`AnalysisDetail`** (predict response and `GET /analyses/{id}`):
```json
{
  "id": "6f1c2a9e-…",
  "model_key": "tree_classification",
  "model_version": "tree_cls_v1",
  "status": "completed",
  "source": "camera",
  "is_sample": false,
  "title": null,
  "description": null,
  "prediction": {
    "label": "banana_tree",
    "display_label": "Banana tree",
    "confidence": 0.947,
    "is_uncertain": false
  },
  "details": {
    "kind": "tree_classification",
    "verdict": "banana_tree",
    "threshold": 0.6,
    "probabilities": [
      {"class_key": "banana_tree", "display_name": "Banana tree", "probability": 0.947},
      {"class_key": "non_banana",  "display_name": "Not a banana tree", "probability": 0.053}
    ]
  },
  "location": {"latitude": 27.71724, "longitude": 85.32402, "accuracy_m": 4.2,
               "captured_at": "2026-10-01T01:12:15Z"},
  "image": {
    "width": 1920, "height": 1440,
    "original_url":  "/api/v1/analyses/6f1c…/image?variant=original&exp=1790000000&sig=…",
    "thumbnail_url": "/api/v1/analyses/6f1c…/image?variant=thumbnail&exp=…&sig=…",
    "result_url": null
  },
  "timings_ms": {"preprocess": 14, "inference": 41, "postprocess": 1, "total": 102},
  "created_at": "2026-10-01T01:12:16Z"
}
```
`details` is typed per model. Implement it as a Pydantic **discriminated union on `kind`** once a
second model exists. Until then, type it directly as `TreeDetails`, with a comment.

**`AnalysisSummary`** (list items, for the results table): `id, model_key, source, is_sample,
title, display_label, confidence, is_uncertain, thumbnail_url, latitude, longitude, created_at`.

**`Page[T]`**: `{"items": [...], "page": 1, "page_size": 20, "total": 57, "total_pages": 3}`.

Image URLs are **relative**. The frontend prefixes its API base URL.
Signature: `HMAC-SHA256(SIGNING_SECRET, f"{id}:{variant}:{exp}")`, base64url, constant-time
compare. A bad or expired signature returns `403 INVALID_SIGNATURE`.

---

## 11. Error handling

Hierarchy in `app/core/errors.py`:
```python
class AppError(Exception):
    status_code: int = 500; code: str = "INTERNAL_ERROR"; title: str = "Internal error"
    def __init__(self, detail: str | None = None, *, extra: dict | None = None): ...
```
Subclasses (status / code):
| Class | Status | Code |
|---|---|---|
| InvalidImageError | 400 | `INVALID_IMAGE`, `IMAGE_TOO_SMALL`, `IMAGE_TOO_LARGE_DIMENSIONS` |
| MissingClientIdError / InvalidClientIdError | 400 | `MISSING_CLIENT_ID` / `INVALID_CLIENT_ID` |
| InvalidSignatureError | 403 | `INVALID_SIGNATURE` |
| SampleImmutableError | 403 | `SAMPLE_IMMUTABLE` |
| AnalysisNotFoundError | 404 | `ANALYSIS_NOT_FOUND` |
| PayloadTooLargeError | 413 | `IMAGE_TOO_LARGE` |
| UnsupportedMediaTypeError | 415 | `UNSUPPORTED_MEDIA_TYPE` |
| (RequestValidationError) | 422 | `VALIDATION_ERROR` with `errors: [{field, message}]` |
| (slowapi RateLimitExceeded) | 429 | `RATE_LIMITED` + `Retry-After` header |
| InferenceError | 500 | `INFERENCE_FAILED` |
| ModelUnavailableError | 503 | `MODEL_UNAVAILABLE` |
| DatabaseUnavailableError (from SQLAlchemy `OperationalError`/`InterfaceError`) | 503 | `DATABASE_UNAVAILABLE` |
| Anything else | 500 | `INTERNAL_ERROR` |

Response body (`Content-Type: application/problem+json`):
```json
{
  "type": "/errors/image-too-large",
  "title": "Image too large",
  "status": 413,
  "detail": "Image exceeds the 10 MB limit.",
  "instance": "/api/v1/tree/predict",
  "code": "IMAGE_TOO_LARGE",
  "request_id": "b7e3…"
}
```
Rules: never leak stack traces, SQL, file paths or internal messages to clients. 4xx errors are
logged at INFO/WARNING; 5xx at ERROR with traceback and `request_id`. Register handlers for
`AppError`, `RequestValidationError`, `StarletteHTTPException` (so 404/405 also use
problem+json), `RateLimitExceeded`, `SQLAlchemyError`, and `Exception`.

---

## 12. Cross-cutting concerns

- **Middleware order**: RequestID → access log/timing → security headers
  (`X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`,
  `X-Frame-Options: DENY`) → CORS → GZip (≥1 KB, JSON only).
- **CORS**: explicit `CORS_ORIGINS` (never `*`), methods `GET, POST, DELETE, OPTIONS`,
  allow headers `Content-Type, X-Client-Id, X-Request-ID`, expose `X-Request-ID, Location`.
- **Rate limiting**: key = `CF-Connecting-IP` when `TRUST_CLOUDFLARE_HEADERS=true`, otherwise
  `request.client.host`.
- **Bind to `127.0.0.1` only.** The Cloudflare tunnel connects locally; SQL Server is never exposed.
- `/docs` and `/redoc` only when `ENABLE_DOCS=true`.
- **Lifespan**: configure logging → create engine → DB ping (warn, don't crash) → build and warm
  up the registry → store on `app.state`. Shutdown: dispose engine.
- Run with **1 worker** (see P-03); document why in the README.
- Known v1 limitation (document in README): `X-Client-Id` is not authentication. Anyone with the
  UUID can read that history. Acceptable for v1; real auth can be added later and linked to
  `client_id`.

---

## 13. Scripts

- `check_env.py`: Python version; `pyodbc.drivers()` contains Driver 18; DB connect + `SELECT 1`
  + Alembic current revision; `DATA_ROOT` writable; for each enabled model, the weights file
  exists, is a zip (`zipfile.is_zipfile`), and loads with the expected `task` + classes; CUDA
  availability. Print a ✅/❌ table with an actionable fix for each ❌. Must detect the extracted
  COCO folder situation (§1).
- `predict_cli.py`: run a predictor on one or more images **without** DB or storage; print
  label, confidence and the full distribution. Used by the owner to sanity-check the model.
- `seed_samples.py --model <key> [--force]`: reads
  `samples/<model_key>/manifest.json` (`[{file, title, description, latitude?, longitude?}]`).
  Runs each image through **the real `AnalysisService`** with `source="sample"`,
  `is_sample=True`, `client_id=None`. Idempotent: skips if a sample with the same `image_sha256`
  exists, unless `--force` (which soft-deletes the old one).
- `export_openapi.py`: writes `Backend/openapi.json` so the frontend can generate TypeScript
  types with `openapi-typescript`.

---

## 14. Testing strategy

- `pytest` default run: **fast**, no torch, uses SQLite (temporary file DB, schema via
  `Base.metadata.create_all` *in tests only*), a temporary `DATA_ROOT`, and a `FakePredictor`
  (deterministic probabilities, configurable to raise).
- Markers: `mssql` (runs against `KeraAI_Test`, applies Alembic migrations up and down),
  `model` (real weights; skip if the file is missing).
- **Unit tests**: image decode/validate (corrupt, too small, bomb, wrong format, EXIF rotation
  applied, EXIF stripped); tree postprocess (threshold edges 0.59/0.60/0.61, verdict mapping);
  signing (valid, tampered, expired); storage path layout + traversal guard + atomic write;
  settings validation; error handler response shape.
- **Integration tests** (TestClient): predict happy path returns 201 + `Location` + DB row +
  files; history shows only the caller's rows + samples; another client's id → 404; delete own
  → 204, then 404; delete sample → 403; 413 / 415 / 422 / missing client id; 503 when model
  unavailable; DB commit failure leaves no orphan files; problem+json content type on all errors.
- Coverage ≥ **80%** on `app/` (exclude `ml/tree_classifier.py` from the fast run).

---

## 15. Execution plan for Claude Code — phases & STOP gates

Run lint + tests at the end of each phase. Commit per logical unit. Report at each **STOP**.

**Phase 0 — Environment & repo audit (no app code)**
- If `/` is not a git repo: `git init`, root `.gitignore` (Python, Node, `.env`, `*.pt`,
  `Backend/yolov8n/`, `Backend/weights/*` except README, `.venv`, `DATA_ROOT`). Initial commit
  of the existing frontend as a baseline.
- Inspect the environment: Python version, ODBC drivers, whether SQL Server is reachable,
  GPU/CUDA, and the weights situation (§1).
- **STOP** → report findings plus the Open decisions O-01…O-07. Wait for answers.

**Phase 1 — Skeleton & cross-cutting**
- `pyproject.toml`, requirements files, `.env.example`, `app/main.py` factory + lifespan,
  `core/` (config, logging, errors, middleware, rate_limit, time), `/health`, `/health/ready`
  (stubbed components), versioned router, test harness + tests for errors/middleware/config.
- **STOP** → owner runs `uvicorn` and opens `/docs` and `/health`.

**Phase 2 — Database**
- `db/sql/001_create_database_and_login.sql`, `db/base.py`, `session.py`, `Analysis` model,
  Alembic setup + `0001_create_analyses`, repository + tests (SQLite + `mssql` marker).
- **STOP** → owner runs the SQL script in SSMS, then `alembic upgrade head`, and reviews the
  table, constraints and indexes in SSMS.

**Phase 3 — Imaging & storage**
- `imaging/processing.py`, `storage/` + tests with fixture images.
- **STOP** (short) → summary only.

**Phase 4 — ML layer**
- `ml/base.py`, `registry.py`, `tree_classifier.py`, `scripts/check_env.py`,
  `scripts/predict_cli.py`, `FakePredictor` in tests, a `model`-marked test.
- **STOP** → owner places the real `weights/tree_cls_v1.pt`, runs `check_env` and
  `predict_cli` on ~5 known images, and confirms the predictions look right.

**Phase 5 — Service & API**
- `AnalysisService`, `signing.py`, all endpoints in §10, rate limiting, integration tests.
- **STOP** → owner tests via `/docs`: predict, list, detail, open the image URL, delete, and
  checks rows in SSMS and files under `DATA_ROOT`.

**Phase 6 — Samples, docs, hand-off to frontend**
- `seed_samples.py` + an example `manifest.json`, `export_openapi.py`, `Backend/README.md`
  (setup from zero on Windows, torch install, env vars, commands, architecture diagram,
  how to add a new model in 5 steps, known limitations), ADRs for P-01…P-07.
- **STOP** → v1 backend done. The next work item, frontend integration, gets its own spec.

## 16. Definition of Done (v1 backend)
- [ ] All phases approved by the owner; `ruff`, `mypy`, and `pytest` pass; coverage ≥ 80%.
- [ ] `alembic upgrade head` and `downgrade base` both work on SQL Server.
- [ ] Predict → row in SSMS + `original.jpg` + `thumb.webp` on disk, all sharing the same id.
- [ ] App starts in degraded mode without weights; `/health/ready` explains why.
- [ ] Every error response is problem+json with a stable `code` and a `request_id`.
- [ ] README lets someone set it up from scratch; "add a new model" guide is accurate.
