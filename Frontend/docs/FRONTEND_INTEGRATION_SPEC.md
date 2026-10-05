# KeraAI Frontend — Backend Integration Specification (v1)

> Audience: Claude Code. Owner: Saugat. `/CLAUDE.md` working agreement applies (owner decides, you
> implement; STOP and ask before adding dependencies or changing the API contract).
> Backend contract: `Backend/docs/BACKEND_SPEC.md` §10 (+ the Model 2 contract in §A below).
> Basis: a full read of `Frontend/src` (inventory done 2026-10-05). Visual design is kept; data sources change.

## 0. Goal and non-goals
Replace every mock with the real backend so that, for **Model 1 (tree)** and **Model 2 (leaf segmentation)**:
1. **Provided samples** → shown instantly with results pre-computed by the real pipeline (no inference).
2. **Camera / upload** → `POST` to the backend → real preprocess → model → postprocess → real result → saved in history.
3. **History** is real, per device (`X-Client-Id`), plus public samples.
4. **Model 3** is a visible "coming soon" placeholder (no fake results).

Non-goals (v1): login, offline mode, a client-side router, service workers, image editing.
Principle **D-06**: render only values the backend returns. Where a mock field has no real source it is
**removed** (or replaced by a real field) — never kept "for looks".

## A. Model 2 contract (owner-approved 2026-10-05)
`POST /api/v1/leaf-segmentation/predict`, same multipart fields as `/tree/predict`. Response `AnalysisDetail`
with `model_key="leaf_segmentation"`, `prediction.label ∈ {affected, healthy, no_leaf}`,
`prediction.confidence = null`, `prediction.is_uncertain = false`, `image.result_url` = overlay PNG
(leaf = green outline, damage = red), and:
```
details: { kind:"leaf_segmentation", leaf_area_pct_of_image, affected_area_pct_of_leaf, lesion_count,
           largest_lesion_pct_of_leaf, mean_leaf_probability|null, mean_affected_probability|null,
           thresholds:{leaf, affected, min_leaf_pct, min_lesion_pct, min_affected_pct, tta_hflip} }
```
`GET /models` items gain `is_placeholder: boolean` → the UI shows a **"Demo weights — results are not real"**
badge on every result produced by a placeholder model (`model_version` contains the same hint, e.g. `*_dummy_v0`).

## 1. Architecture (no new routing dependency)
- Keep the state-based navigation in `App.tsx` (a router would be a new dependency). Remove the 5 legacy `ViewTab`
  aliases and the dead `LeafDetectView.tsx`.
- Add app-level state: `{ file: File | null, tree: AnalysisDetail | null, leaf: AnalysisDetail | null }` plus
  `lastAnalysisId` in `sessionStorage` so a refresh can re-fetch `GET /analyses/{id}`.
- Stage flow stays: Detect (Model 1) → result → "Analyse the leaf" (re-POSTs the **same `File`** to the Model 2
  route — the stages are independent, spec README) → leaf result. Model 3 tab → "coming soon".
- New folder `src/api/`: `client.ts` (fetch wrapper: base URL from `VITE_API_BASE_URL`, `X-Client-Id`, problem+json
  parsing → typed `ApiError{code,status,detail,requestId,retryAfter}`, `AbortController`), `upload.ts` (XHR with
  `upload.onprogress`; do **not** set `Content-Type` on `FormData`), `clientId.ts` (`crypto.randomUUID()` +
  `localStorage`, try/catch-safe fallback), `analyses.ts`, `models.ts`, `types.ts` (generated, see §6),
  `urls.ts` (`absoluteImageUrl(rel)`; image URLs are relative and signed for 1 h — never cache them; refetch the
  detail on `<img>` error).
- Hooks: `useModels()`, `useSamples(modelKey)`, `useHistory(filter,page)`, `useAnalysis(id)`, `useGeolocation()`,
  `useClientId()`. Plain `useState/useEffect` — no new state library.

## 2. View-by-view changes
| View | Change |
|---|---|
| Header | Replace the fake "FastAPI: Ready (42ms)" pill with `GET /health/ready` (green/amber/red) + a "Demo weights" chip when `/models` says so. Replace remote logo with a bundled asset. |
| HomeView | Keep marketing, but delete fabricated HUD numbers (Dice, "Black Sigatoka 93.8%", 97.4%, "<0.6s", "12,000+"). Use real facts or none. Card 3 → "coming soon". |
| DetectionWorkspaceView (Model 1 input) | Real `File` (no `readAsDataURL`; use `URL.createObjectURL`), client-side type/size check (JPEG/PNG/WEBP ≤ 10 MB, friendly HEIC message), real drag-and-drop, GPS via hook. Samples = `useSamples("tree_classification")` buttons (thumbnail+title) → straight to result with that `AnalysisDetail` (no ProcessingModal). |
| Stage1ResultView | Drive from `AnalysisDetail`: verdict banner (3 states: banana tree / not banana tree / **uncertain** with `details.threshold`), `confidence`, real `probabilities` bars (2 classes), image, resolution `image.width×height`, `created_at`, `model_version`, `timings_ms`, location (or "—"). **Remove**: bounding box, foliage/pseudostem tiles, 4-class distribution, taxonomy/"herbarium" text, hard-coded "° N/° E" (use the sign of the number). |
| LeafAnalysisView (Model 2 result) | Drive from the Model 2 `AnalysisDetail`: banner from `prediction.label` (incl. `no_leaf`), `affected_area_pct_of_leaf`, healthy % = 100 − affected only when `label ≠ no_leaf`, leaf coverage, lesion count, largest lesion. Replace hard-coded SVG polygons with `<img src=result_url>` overlay; keep the split slider (switch to pointer events + keyboard arrows) and opacity. Drop healthy-only/unhealthy-only toggles, "Dice 0.924", "spectral points", Critical/Moderate buckets. Rename "Model 4" → "Model 2". |
| DetectDiseaseView (Model 3) | Replace with a "Model 3 — coming soon" panel; delete PSI index, recommendations, endpoint cards, JSON modal, fake lesion SVG. |
| AnalysisHistoryView | Real table from `GET /analyses` (thumbnail, label, confidence, date, GPS), `scope` tabs (Mine / Samples), pagination, delete (own only, confirm), row → fetch detail → open the matching result view. CSV export from the real rows. |
| AboutModelsView / Footer | Rewrite from `GET /models` + owner-supplied training metrics. Remove fabricated benchmarks, DOIs, BibTeX, ResNet-50/EfficientNet claims, TensorRT/Redis/T4/A100, fake cURL host, fake GitHub/MIT licence. Real facts: Model 1 YOLOv8-cls 224 px, uncertain < 0.60; Model 2 U-Net/resnet34 512 px, thresholds 0.50/0.85, h-flip TTA, no-leaf < 3 %, healthy < 0.5 %; Model 3 planned. Keep the contact name/email. **Licence line: ask owner (Ultralytics is AGPL-3.0, spec O-07).** |
| ProcessingModal | Props `{phase: 'preparing'\|'uploading'\|'analysing'\|'error', progress?, error?, onCancel, onRetry}`; no timers, no fake endpoints/latency. Phases: preparing → uploading (real %) → analysing (indeterminate). Error mapping by `code`: `RATE_LIMITED` (show Retry-After), `IMAGE_TOO_LARGE`, `UNSUPPORTED_MEDIA_TYPE`, `INVALID_IMAGE`/`IMAGE_TOO_SMALL`/`IMAGE_TOO_LARGE_DIMENSIONS`, `VALIDATION_ERROR`, `INFERENCE_FAILED`, `MODEL_UNAVAILABLE`/`DATABASE_UNAVAILABLE`, network failure ("server offline"). Always show `request_id` on errors. Cancel = `AbortController`. |
| CameraModal | Fix real defects: capture with `canvas.toBlob` at the video's **native** size (no fixed 1600×1200 distortion), return a `File` + `captured_at` stamped at shutter, render `streamErrorNotice`, no blank-navy fallback frame (disable shutter until a frame exists), remove fake lens/exposure/"60 FPS" or label them honestly, `dvh` instead of `vh`, close `AudioContext`. |
| GPS | Initial state **null** (no fake Kathmandu / "Locked ±4.2 m"); request on user action; send lat+lon together or not at all; `captured_at` from `position.timestamp`; show real accuracy. |

## 3. Tooling cleanup (Phase F0)
- `package.json`: rename to `kera-ai-frontend`; **remove** unused template leftovers (`@google/genai`, `express`,
  `@types/express`, `dotenv`, `tsx`, `esbuild`, `autoprefixer`, the `clean` script that deletes `server.js`) and the
  unused `motion`/`lucide-react` if confirmed unused. Removing is allowed; **adding** needs owner approval
  (the only expected additions are listed in §6).
- `.env.example`: replace `GEMINI_API_KEY`/`APP_URL` with `VITE_API_BASE_URL=http://127.0.0.1:8000`.
- `vite.config.ts`: drop the AI Studio `DISABLE_HMR` toggles; no secrets via `define`; optional dev proxy not needed (CORS is configured).
- `tsconfig.json`: enable `strict`, `noUnusedLocals/Parameters` once the dead code is gone.
- Fonts/icons: keep Google Fonts but document the CSP allowance; bundle the logo; drop `lh3.googleusercontent.com` URLs
  (they are mock assets and may expire).
- `netlify.toml`: build command, publish dir, SPA fallback, security headers incl. CSP with
  `connect-src`/`img-src` = the API origin (`api.<domain>` via Cloudflare Tunnel).

## 4. Phases and STOP gates
- **F0** tooling cleanup + `npm install` + baseline `npm run lint` and `npm run build` → STOP (short).
- **F1** `src/api/*`, types, `useModels`/`useClientId`, Header health pill, Model-3 "coming soon" → STOP.
- **F2** Model 1 end-to-end (input → real POST → result), samples, ProcessingModal real states → STOP (owner tries camera + upload + a sample on phone over HTTPS).
- **F3** Model 2 result view + stage-2 hand-off → STOP.
- **F4** History (list/detail/delete/CSV) + About/Footer rewrite + a11y pass (dialog roles, focus trap, Esc, labels, ≥12 px text) → STOP.
- **F5** Netlify/Cloudflare docs, CSP, final `build`; end-to-end checklist → STOP.

## 5. Verification (every phase)
`npm run lint` (tsc) and `npm run build` pass; with the backend running (`uvicorn … --port 8000`, dummy or real
weights) the owner can: open a sample (no network call to `/predict`), upload a JPEG, take a camera photo, see a
real history row, delete it, and see the correct error message for each forced failure (oversized file, wrong
type, backend stopped, 429). Camera/GPS need HTTPS or `localhost` — on a phone use the Netlify deploy or the tunnel.

## 6. Dependencies
Allowed to add after owner approval at F1: `openapi-typescript` (dev) — to generate `src/api/types.ts` from
`Backend/openapi.json` (backend spec §13). No router, no state library, no UI kit.
