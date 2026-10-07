# KeraAI Backend

FastAPI service behind the KeraAI web app for banana farming. A user photographs a banana plant or leaf and sends it with
its GPS location. The backend runs a deep-learning model, stores the image on local disk and the result in SQL Server, and
serves the result plus a browsable history.

| # | Model key             | What it does                                         | Architecture | Status |
|---|-----------------------|------------------------------------------------------|--------------|--------|
| 1 | `tree_classification` | Banana tree vs not a banana tree                     | YOLOv8-cls   | built  |
| 2 | `leaf_segmentation`   | Measures healthy leaf tissue vs damaged tissue       | U-Net        | built  |
| 3 | `leaf_disease`        | Healthy vs diseased leaf (+ disease type)            | U-Net-type   | built  |

The authoritative design is `docs/BACKEND_SPEC.md`; decisions are recorded as ADRs in `docs/adr/`. The working agreement
for contributors (and for Claude Code) is the repo-root `CLAUDE.md`.

## Architecture

Runtime topology, everything behind the Cloudflare Tunnel runs on the owner's Windows laptop:

```
 Browser / phone
       |  https
       v
 Netlify  (React app, static files)
       |  https, header X-Client-Id
       v
 Cloudflare Tunnel  (cloudflared -> http://127.0.0.1:8000)
       |
       v
 FastAPI + uvicorn  (ONE worker, listens on 127.0.0.1 only)
       |-----------------------------> SQL Server  (localhost only, via ODBC Driver 18)
       '-----------------------------> DATA_ROOT   (images on local disk, e.g. C:/kera-data)
```

Inside the service, dependencies only point downwards (lower layers never import from higher ones):

```
 api            HTTP only: routing, validation, status codes   (app/api)
  |
  v
 services       orchestration: THE pipeline for every model    (app/services/analysis_service.py)
  |
  +--> ml            Predictor classes + model registry         (app/ml)
  +--> imaging       decode, validate, normalise, thumbnails    (app/imaging)
  +--> storage       image files under DATA_ROOT                (app/storage)
  +--> repositories  SQL queries, no business rules             (app/repositories)
          |
          v
        db           SQLAlchemy models, engine, sessions        (app/db)
```

Cross-cutting: `app/core` (settings, errors, logging, middleware, rate limit, URL signing), `app/schemas` (Pydantic
request/response models). `ml` never imports `db`. Routers contain no business logic.

## Setup from zero (Windows, PowerShell)

Everything below runs from the `Backend/` folder. You need: Python **3.11**, Git, a local SQL Server (SSMS to manage it)
and the **ODBC Driver 18 for SQL Server** (<https://aka.ms/downloadmsodbcsql>).

### 1. Python virtual environment

```powershell
cd Backend
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python --version        # must say 3.11.x
```

If PowerShell refuses to run the activation script ("running scripts is disabled"), allow it for this window only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

then activate again. You never have to activate: calling the venv's interpreter directly always works and is the safest habit, because the
machine's default `python` may be a different version:

```powershell
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m scripts.check_env
```

Every command below assumes the venv is active (or that you prefix `python` with `.venv\Scripts\`).

### 2. Install the dependencies (torch FIRST)

`torch` is machine-specific (CPU or a particular CUDA build), so it is not in `requirements.txt`. Install it before the rest:

```powershell
# CPU only (a laptop without an NVIDIA GPU):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# NVIDIA GPU: pick the command for your CUDA version at https://pytorch.org/get-started/locally/

pip install -r requirements-dev.txt        # requirements.txt (runtime, exact pins) + pytest/ruff/mypy
```

The project was developed with torch 2.14.1 and torchvision 0.29.1 (CPU builds).

### 3. Configuration (`.env`)

```powershell
Copy-Item .env.example .env
```

Edit `.env` (it is git-ignored; never commit it). Settings are read by `app/core/config.py`; an invalid value stops the app
at startup with a clear message. At minimum set `SIGNING_SECRET` (32+ random characters, e.g.
`python -c "import secrets; print(secrets.token_urlsafe(48))"`), the database block and `DATA_ROOT`.

| Variable | Default | Meaning |
|----------|---------|---------|
| **App** | | |
| `APP_ENV` | `development` | `development`, `production` or `test`. |
| `APP_NAME` | `KeraAI API` | Title shown in `/docs`. |
| `API_V1_PREFIX` | `/api/v1` | URL prefix of the versioned API. Must start with `/`. |
| `ENABLE_DOCS` | `true` | Serves `/docs`, `/redoc`, `/openapi.json`. Set `false` in production. |
| `LOG_LEVEL` | `INFO` | Standard logging level. |
| `LOG_JSON` | `false` | JSON log lines instead of text (useful in production). |
| `CORS_ORIGINS` | `["http://localhost:3000"]` | JSON list of allowed browser origins. Never `*`. Add the Netlify URL in production. |
| `TRUST_CLOUDFLARE_HEADERS` | `false` | `true` behind the tunnel: the rate limiter then keys on `CF-Connecting-IP`. |
| `SIGNING_SECRET` | none, **required** | HMAC key for image URLs, 32+ characters. Rotating it invalidates all outstanding links. |
| `SIGNED_URL_TTL_SECONDS` | `3600` | Lifetime of a signed image link. |
| **Database** | | |
| `DB_HOST` | `localhost` | Server, e.g. `localhost` or `localhost\SQLEXPRESS`. |
| `DB_PORT` | empty | Optional TCP port. |
| `DB_NAME` | `KeraAI` | Database name. |
| `DB_DRIVER` | `ODBC Driver 18 for SQL Server` | Must match an installed driver (`check_env` lists them). |
| `DB_TRUSTED_CONNECTION` | `true` | `true` = Windows authentication (user/password ignored); `false` = SQL login. |
| `DB_USER`, `DB_PASSWORD` | `kera_app`, empty | SQL login, used only when `DB_TRUSTED_CONNECTION=false`. |
| `DB_TRUST_SERVER_CERT` | `true` | Accept the self-signed certificate of a local server (Driver 18 encrypts by default). |
| `DB_POOL_SIZE` | `5` | Connection pool size. |
| `DB_ECHO` | `false` | Log every SQL statement (debugging only). |
| **Storage and uploads** | | |
| `DATA_ROOT` | none, **required** | Folder for images, e.g. `C:/kera-data` (this laptop has no D: drive). The database stores relative paths only. |
| `MAX_UPLOAD_MB` | `10` | Maximum upload size; larger gets 413. |
| `MIN_IMAGE_SIDE_PX` | `64` | Smaller images get 400 `IMAGE_TOO_SMALL`. |
| `MAX_IMAGE_PIXELS` | `40000000` | Decompression-bomb guard. |
| `STORE_MAX_SIDE_PX` | `2048` | The stored original is downscaled to this longest side. |
| `THUMBNAIL_SIDE_PX` | `320` | Longest side of the WEBP thumbnail. |
| `ALLOWED_IMAGE_FORMATS` | `["JPEG","PNG","WEBP"]` | Decoded formats accepted (the content decides, not the client's content type). |
| **Inference** | | |
| `INFERENCE_DEVICE` | `auto` | `auto` (CUDA if available, else CPU), `cpu` or `cuda:0`. |
| `RATE_LIMIT_PREDICT` | `10/minute` | ONE budget per client IP shared by all predict routes (ADR 0017). |
| `RATE_LIMIT_DEFAULT` | `120/minute` | Currently unused (reserved, ADR 0017). |
| **Model 1: tree classification** | | |
| `TREE_ENABLED` | `true` | Load this model at startup. |
| `TREE_WEIGHTS_PATH` | `weights/tree_cls_v1.pt` | Relative paths resolve against `Backend/`, not the working directory. |
| `TREE_MODEL_VERSION` | `tree_cls_v1` | Stored on every row. Use `tree_cls_dummy_v0` with dummy weights. |
| `TREE_IMGSZ` | `224` | Classifier input size. |
| `TREE_POSITIVE_CLASS` | `banana_tree` | Must exist in the model's class names. |
| `TREE_UNCERTAIN_THRESHOLD` | `0.60` | Top-1 confidence below this gives the verdict `uncertain`. |
| `TREE_DISPLAY_NAMES` | `{"banana_tree":"Banana tree","non_banana":"Not a banana tree"}` | JSON map class key to label; its keys must exist in the model. |
| `TREE_IS_PLACEHOLDER` | `false` | `true` while the weights are the dummy stand-in. |
| **Model 2: leaf segmentation** | | |
| `LEAF_SEG_ENABLED` | `true` | Load this model at startup. |
| `LEAF_SEG_WEIGHTS_PATH` | `weights/leaf_seg_v1.pt` | Bare `state_dict` from the training notebook. |
| `LEAF_SEG_MODEL_VERSION` | `leaf_seg_v1` | Use `leaf_seg_dummy_v0` with dummy weights. |
| `LEAF_SEG_ENCODER` | `resnet34` | Must match the checkpoint. |
| `LEAF_SEG_IMGSZ` | `512` | Network input side; must be a multiple of 32. |
| `LEAF_SEG_LEAF_THRESHOLD` | `0.50` | Sigmoid threshold of the leaf channel. |
| `LEAF_SEG_AFFECTED_THRESHOLD` | `0.85` | Sigmoid threshold of the damage channel (tuned on the test split: optimistic). |
| `LEAF_SEG_TTA_HFLIP` | `true` | Average with a horizontally flipped pass (doubles latency). |
| `LEAF_SEG_MIN_LEAF_PCT` | `3.0` | Leaf smaller than this % of the image gives `no_leaf`. |
| `LEAF_SEG_MIN_LESION_PCT` | `0.05` | Lesion blobs smaller than this % of the leaf are dropped. |
| `LEAF_SEG_MIN_AFFECTED_PCT` | `0.5` | Damage below this % of the leaf gives `healthy`. |
| `LEAF_SEG_IS_PLACEHOLDER` | `false` | `true` while the weights are the dummy stand-in. |
| **Model 3: leaf disease identification** | | |
| `LEAF_DISEASE_ENABLED` | `true` | Load this model at startup. |
| `LEAF_DISEASE_WEIGHTS_PATH` | `weights/leaf_disease_v1.pt` | Bare `state_dict` from the training notebook. |
| `LEAF_DISEASE_MODEL_VERSION` | `leaf_disease_v1` | Use `leaf_disease_dummy_v0` with dummy weights. |
| `LEAF_DISEASE_ENCODER` | `resnet34` | Must match the checkpoint. |
| `LEAF_DISEASE_IMGSZ` | `512` | Network input side; must be a multiple of 32. |
| `LEAF_DISEASE_LEAF_THRESHOLD` | `0.50` | Sigmoid threshold of the leaf channel. |
| `LEAF_DISEASE_BLACK_SIGATOKA_THRESHOLD` | `0.85` | Sigmoid threshold of the black Sigatoka channel (validation-tuned). |
| `LEAF_DISEASE_YELLOW_SIGATOKA_THRESHOLD` | `0.55` | Sigmoid threshold of the yellow Sigatoka channel (validation-tuned). |
| `LEAF_DISEASE_TTA_HFLIP` | `true` | Average with a horizontally flipped pass (doubles latency). |
| `LEAF_DISEASE_MIN_LEAF_PCT` | `3.0` | Leaf smaller than this % of the image gives `no_leaf`. |
| `LEAF_DISEASE_MIN_LESION_PCT` | `0.05` | Lesion blobs smaller than this % of the leaf are dropped. |
| `LEAF_DISEASE_MIN_DISEASE_PCT` | `0.5` | A disease below this % of the leaf is ignored. |
| `LEAF_DISEASE_IS_PLACEHOLDER` | `false` | `true` while the weights are the dummy stand-in. |

### 4. SQL Server

1. Open `db/sql/001_create_database_and_login.sql` in SSMS, connected as a sysadmin (your Windows login is fine).
2. Edit the CONFIG block at the top: `@WindowsLogin` (your `DOMAIN\user`, or `NULL` to skip) and, if you want a SQL login,
   `@SqlLoginPassword` (while it still starts with `CHANGE_ME` the SQL-login part is skipped on purpose).
3. Press F5. It creates the databases `KeraAI` and `KeraAI_Test` and grants the roles `db_datareader`, `db_datawriter`
   and `db_ddladmin`. It is safe to re-run.
4. Choose how the app logs in, in `.env`:
   * **Windows authentication** (`DB_TRUSTED_CONNECTION=true`, the default): nothing else to set up.
   * **SQL login** `kera_app` (`DB_TRUSTED_CONNECTION=false`, `DB_USER`, `DB_PASSWORD`): the server must accept it. In SSMS,
     server Properties > Security > "SQL Server and Windows Authentication mode", then restart the SQL Server service.
5. If the app cannot reach the server at all, enable **TCP/IP** in SQL Server Configuration Manager (SQL Server Network
   Configuration > Protocols) and restart the service.
6. Create the tables with Alembic (the app never creates tables itself):

   ```powershell
   alembic upgrade head
   ```

`db_ddladmin` is a development convenience so one login can migrate and serve; production would use two logins.

### 5. Model weights

Weights are never committed (`*.pt` is git-ignored). They live in `Backend/weights/` (see `weights/README.md`).

**Dummy weights** (random numbers, for plumbing only; see ADR 0010):

```powershell
python -m scripts.make_dummy_weights      # refuses to overwrite existing files; --force to overwrite
```

While they are in use, `.env` must say so, so every database row is identifiable and the UI can badge results:

```dotenv
TREE_MODEL_VERSION=tree_cls_dummy_v0
TREE_IS_PLACEHOLDER=true
LEAF_SEG_MODEL_VERSION=leaf_seg_dummy_v0
LEAF_SEG_IS_PLACEHOLDER=true
LEAF_DISEASE_MODEL_VERSION=leaf_disease_dummy_v0
LEAF_DISEASE_IS_PLACEHOLDER=true
```

**Switching to the real weights:**

1. Copy the trained files over the dummies, by hand: `weights/tree_cls_v1.pt` (the original Ultralytics `.pt`, a zip; never
   unzip it, and `Backend/yolov8n/` is the stock COCO detector, not your model), `weights/leaf_seg_v1.pt` and
   `weights/leaf_disease_v1.pt` (both bare `state_dict`s).
2. In `.env`, EDIT the existing lines (do not add duplicates): `TREE_MODEL_VERSION=tree_cls_v1`, `TREE_IS_PLACEHOLDER=false`,
   `LEAF_SEG_MODEL_VERSION=leaf_seg_v1`, `LEAF_SEG_IS_PLACEHOLDER=false`, `LEAF_DISEASE_MODEL_VERSION=leaf_disease_v1`,
   `LEAF_DISEASE_IS_PLACEHOLDER=false`. Check that `TREE_POSITIVE_CLASS`, `TREE_DISPLAY_NAMES`, `LEAF_SEG_ENCODER` and
   `LEAF_DISEASE_ENCODER` match the checkpoints, and that each model's thresholds reflect the training notebook's
   validation-tuned values.
3. `python -m scripts.check_env` must show all three models loading with no placeholder warning, and
   `python -m scripts.predict_cli --model tree_classification path\to\known.jpg` should give sensible answers.
4. Restart the server, then refresh the samples: `python -m scripts.seed_samples --model tree_classification --force`,
   `--model leaf_segmentation --force` and `--model leaf_disease --force`.
5. Rows made during the dummy era keep their `*_dummy_v0` version; filter or delete them
   (`WHERE model_version LIKE '%dummy%'`).

### 6. Check the environment

```powershell
python -m scripts.check_env
```

Prints one line per check (Python 3.11 in `Backend/.venv`, ODBC driver, database connection and Alembic revision, `DATA_ROOT`
writable, CUDA, weights present and loadable with a warm-up inference, dummy-weights warning) with a fix under every failure.
Exit code 1 if anything failed. Note that it creates `DATA_ROOT` if it does not exist.

### 7. Run

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/docs> (when `ENABLE_DOCS=true`) and `/health/ready`. Add `--reload` while developing.

* **Bind to 127.0.0.1 only.** The Cloudflare tunnel connects locally; the API and SQL Server are never exposed directly.
* **One worker (do not pass `--workers`).** One copy of each model lives in memory and a per-model lock serialises
  `predict()` (ADR 0003); several workers would load the networks several times and break the in-memory rate limiter.
* **The app starts even when something is missing.** No weights or no database gives degraded mode: `/health/ready` returns
  503 with the reason, and predict answers `503 MODEL_UNAVAILABLE` / `DATABASE_UNAVAILABLE`. `/health` is liveness only.

## No SQL Server? Use the SQLite dev server

For frontend work, or to try everything before SQL Server is set up:

```powershell
python -m scripts.dev_server_sqlite --reset --seed          # --port 8000 by default
```

* **Dev only.** It uses a throwaway SQLite database `Backend/.dev-data/dev.sqlite3` and images under `Backend/.dev-data/data`
  (git-ignored). It never touches SQL Server or your real `DATA_ROOT`: `DATA_ROOT` is forced to the dev folder.
* The schema comes from the real Alembic migrations, and the models are the real ones (the dummy weights in `weights/`
  unless you installed real ones), so the banner warns when results are not real.
* `--reset` deletes `.dev-data` first (only a folder this script created); `--seed` loads the sample images through the same
  code as `seed_samples`.
* It listens on 127.0.0.1 only. CORS allows `http://localhost:3000` and `http://127.0.0.1:3000` (the Vite dev server); set
  `CORS_ORIGINS` in your shell to override.
* The predict limit is still `RATE_LIMIT_PREDICT` (10/minute): for heavy UI work, `$env:RATE_LIMIT_PREDICT = "100/minute"`
  before starting it.

## Sample images

Samples are public, pre-computed example analyses (D-07) made by running photos through the real pipeline. Each model has
`samples/<model_key>/manifest.json` plus its images; see `samples/README.md` for the format.

```powershell
python -m scripts.seed_samples --model tree_classification
python -m scripts.seed_samples --model leaf_segmentation
python -m scripts.seed_samples --model leaf_segmentation --force      # re-analyse, e.g. after swapping weights
```

It is idempotent (an image already seeded for that model is skipped), prints `SEEDED` / `SKIPPED` / `FAILED` per file, and
exits 1 if a file failed. It prints a loud warning when the model uses placeholder weights. **The two images committed now are
development placeholders** (the project's own leaf photo): replace them with your real photos.

## Going public: Cloudflare Tunnel and Netlify (high level)

1. Install `cloudflared` and publish the local API through a tunnel, e.g. a named tunnel mapping `api.<your-domain>` to
   `http://127.0.0.1:8000` (for a quick test: `cloudflared tunnel --url http://127.0.0.1:8000`). Cloudflare terminates HTTPS and
   enforces its own request-body limit in front of the tunnel (ADR 0014).
2. Set in `Backend/.env`: `APP_ENV=production`, `ENABLE_DOCS=false`, `LOG_JSON=true`, a long random `SIGNING_SECRET`,
   `TRUST_CLOUDFLARE_HEADERS=true` (without it every visitor looks like 127.0.0.1 and they all share one rate-limit
   budget), and `CORS_ORIGINS=["https://app.<your-domain>"]` (the exact Netlify origin; add the Netlify preview URL while testing).
3. Deploy the React app on Netlify with the build-time variable `VITE_API_BASE_URL=https://api.<your-domain>`.
4. **No secrets in the frontend.** Anything in a `VITE_*` variable ships to every browser. The frontend only needs the API base
   URL; it sends an anonymous per-device UUID in `X-Client-Id`, which is not a secret and not authentication.
5. The laptop must be awake with SQL Server and the API running for the app to work.

## API

All JSON, snake_case, timestamps in UTC with `Z`. Errors are `application/problem+json` with a stable `code` and a
`request_id`. `X-Client-Id: <uuid v4>` is **required** on predict and `/analyses` routes except the image route
(missing: 400 `MISSING_CLIENT_ID`; malformed or not v4: 400 `INVALID_CLIENT_ID`). Image URLs in responses are relative.
The generated schema is committed as `openapi.json`; the frontend builds its TypeScript types from it.

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness: `{"status":"ok"}`, no DB or model checks. |
| GET | `/health/ready` | DB ping and model status: 200, or 503 with the reason per component. |
| GET | `/api/v1/models` | Every model with `status`, `is_placeholder`, classes. |
| POST | `/api/v1/tree/predict` | Model 1. Multipart: `image` (required), `latitude`, `longitude` (both or neither), `gps_accuracy_m`, `captured_at`, `source` (`upload`/`camera`). 201 + `Location`. |
| POST | `/api/v1/leaf-segmentation/predict` | Model 2, same form. 201 + `Location`; `image.result_url` is the overlay PNG. |
| POST | `/api/v1/leaf-disease/predict` | Model 3 (ADR 0019), same form. 201 + `Location`; `image.result_url` is the overlay PNG. |
| GET | `/api/v1/analyses` | History. Query: `model_key`, `scope` (`mine` / `samples` / `all_visible`), `page`, `page_size` (max 100). Completed analyses only. |
| GET | `/api/v1/analyses/{id}` | One analysis in full (yours or a sample, otherwise 404). |
| GET | `/api/v1/analyses/{id}/image` | Query `variant` (`original`/`thumbnail`/`result`), `exp`, `sig`. No client id: the signed link is the credential; samples need none (ADR 0015). |
| DELETE | `/api/v1/analyses/{id}` | Soft-delete your own analysis: 204. Samples: 403 `SAMPLE_IMMUTABLE`. |

Stored layout: `DATA_ROOT/images/{model_key}/{YYYY}/{MM}/{analysis_id}/original.jpg | thumb.webp | result.png`.

To refresh the schema file after an API change: `python -m scripts.export_openapi` (rewrites `openapi.json`).

## Tests, lint and types

```powershell
pytest                                   # fast: no torch, SQLite temp DB, tmp DATA_ROOT, fake predictors
pytest -m "mssql or model"               # slow, opt-in: real SQL Server (KeraAI_Test) / real weights
pytest --cov=app --cov-report=term-missing   # coverage; the project requires at least 80 %
ruff check .                             # lint
ruff format --check .                    # formatting (drop --check to fix)
mypy app                                 # types
```

The `mssql` tests skip themselves, with the reason, until you have run the SQL setup script. They apply the Alembic
migrations up and down on `KeraAI_Test` (never on `KeraAI`). Fast tests never import torch or ultralytics.

## Adding a new model in 5 steps

Adding a model means a settings group, a details schema, a predictor class, a registry entry and a route. The service, storage,
repository, table and history endpoints do not change; if you find yourself editing `analysis_service.py` for a
model-specific reason, stop and discuss it. **Model 3 (`leaf_disease`) is a finished, real example of all five steps** (ADR
0019) — read its files alongside this if you're adding a fourth model:

1. **Settings group** (`app/core/config.py`): a `LeafDiseaseModelSettings(BaseModel)` (`enabled`, `weights_path`,
   `model_version`, `is_placeholder`, plus its own knobs), flat `leaf_disease_*` fields on `Settings` with defaults, and a
   `leaf_disease` property returning the group, like `tree` and `leaf_seg`. The group is in the return type of
   `ModelSpec.settings_group` in `app/ml/registry.py` and the variables are in `.env.example`.
2. **Details schema** (`app/schemas/leaf_disease.py`): `LeafDiseaseDetails` with `kind: Literal["leaf_disease"]` (required, no
   default) and the model's real outputs only (D-06). It's in the discriminated union `AnalysisDetails` in
   `app/schemas/analysis.py`.
3. **Predictor** (`app/ml/leaf_disease_identifier.py` + the pure pre/post-processing package `app/ml/leaf_disease/`):
   subclasses `Predictor` (`app/ml/base.py`), `key: ClassVar[str] = "leaf_disease"`, calls
   `super().__init__(display_name=..., version=..., task=..., is_placeholder=...)` from the settings, implements `load`
   (imports torch lazily; raises `ModelLoadError` with an actionable message), `preprocess`, `infer` and `postprocess`
   (returns `PredictionOutput`, with a `result_image` overlay). `predict()` already adds the lock, timings and
   error wrapping. Never import torch at module top level (fast tests and degraded startup depend on it).
4. **Registry** (`app/ml/registry.py`): one `ModelSpec(key="leaf_disease",
   factory="app.ml.leaf_disease_identifier:LeafDiseaseIdentifier", display_name=..., task=...,
   settings_group=lambda s: s.leaf_disease)` in `MODEL_SPECS`. `GET /models`, `/health/ready`, `scripts.check_env` and
   `scripts.predict_cli` pick it up from there with no further changes.
5. **Route** (`app/api/v1/endpoints/leaf_disease.py`): copied from `leaf_segmentation.py`'s shape
   (`APIRouter(route_class=UploadGuardRoute)`, `POST /leaf-disease/predict` calling
   `run_predict(model_key="leaf_disease", ...)`), registered via `include_router` in `app/api/v1/router.py`. The `leaf_disease`
   key already existed in `MODEL_KEYS` (`app/db/models/analysis.py`) and the `ModelKey` Literal (`app/schemas/common.py`)
   before this model was built, so no migration was needed this time — a genuinely new key still needs an Alembic
   migration (`alembic revision -m "allow <key>"`) that drops and recreates the `ck_analyses_model_key` CHECK with the
   literal list of keys (a migration must not import app code).

Finish with: tests using `FakePredictor("leaf_disease", ...)` from `tests/fakes.py`, `python -m scripts.export_openapi`,
`samples/leaf_disease/` with a manifest, and an ADR for any non-obvious choice.

## Known limitations

* **`X-Client-Id` is not authentication.** It is an anonymous device UUID: anyone who knows it can read that history. Real login
  can be added later and linked to the id (D-04).
* **Single worker** (ADR 0003): requests for one model are serialised. Fine for one owner's laptop; scaling needs a separate
  inference worker.
* **Soft delete keeps the files** on disk (ADR 0007), and `--force` re-seeding leaves the old sample's files too.
* **No HEIC.** iPhone HEIC uploads are not supported (browsers normally send JPEG); that needs the `pillow-heif` dependency
  (open decision O-04).
* **Ultralytics is AGPL-3.0.** Serving it publicly implies the source must be available to users. Whether the repository is
  open-sourced is the owner's decision (O-07), still pending.
* **All three models are on real, trained weights** (`*_IS_PLACEHOLDER=false`). Model 1 (`tree_cls_real_v1`, 94.3% test
  accuracy — see `experiment/01_tree_classification.ipynb`), Model 2 (`leaf_seg_real_v1`, affected-channel test Dice
  0.68 at its validation-tuned threshold of 0.90, leaf-channel Dice 0.986 — see
  `experiment/02_leaf_segmentation.ipynb`) and Model 3 (`leaf_disease_real_v1`, per-image diagnosis accuracy 96% (23/24
  correct), leaf-channel Dice 0.955, black-Sigatoka Dice 0.49 and yellow-Sigatoka Dice 0.39 at their validation-tuned
  thresholds of 0.85/0.55 — see `experiment/03_leaf_disease_segmentation.ipynb`). Models 2 and 3's disease/damage
  channels are the hardest part of each pipeline: small, imbalanced lesions on under 200 training images (Model 3's
  yellow Sigatoka class has only 7 examples). Read a "healthy" verdict as "no damage above the configured threshold
  was found," not a clinical guarantee, and treat the per-channel coverage numbers as a useful signal rather than a
  precise area measurement.
* **Rate limiting** is per client IP and in memory (resets on restart); a signed image link works for whoever holds it until it
  expires (ADRs 0015, 0017).
* **SQL Server** code paths are exercised against SQLite in the fast suite; the SQL Server behaviour (migrations up and down,
  constraints) is only tested by the opt-in `mssql` tests, which can run once the owner has run the setup script.
