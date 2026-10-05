# KeraAI Frontend

React 19 + Vite + Tailwind CSS 4 + TypeScript web app for KeraAI, a computer-vision tool for banana
farming. A user photographs a banana plant or leaf (camera or upload), the backend runs a deep-learning
model on it, and the app shows the result plus a browsable history.

> **Status:** the UI currently runs on **mock data** (`src/data/mockData.ts`). It is wired to the real
> FastAPI backend in the integration phases described in
> [docs/FRONTEND_INTEGRATION_SPEC.md](docs/FRONTEND_INTEGRATION_SPEC.md).

## Prerequisites

- Node.js 24 (developed with v24.15; any current LTS or newer should work) and npm 11.
- The backend (`../Backend`) is only needed once the app is wired to the API.

## Run locally

```powershell
npm install
npm run dev          # http://localhost:3000 (also reachable on your LAN for phone testing)
```

Camera and GPS only work on `localhost` or HTTPS, so test those on a phone via the Netlify deploy or
the Cloudflare Tunnel rather than the LAN address.

## Environment

Copy `.env.example` to `.env.local` and adjust:

| Variable            | Meaning                                  | Default                 |
|---------------------|------------------------------------------|-------------------------|
| `VITE_API_BASE_URL` | Base URL of the KeraAI backend (FastAPI) | `http://127.0.0.1:8000` |

`VITE_*` values are bundled into the browser build, so they are public: never put secrets in them.
On Netlify, set the same variable in the site's environment settings.

## Scripts

| Command           | What it does                                          |
|-------------------|-------------------------------------------------------|
| `npm run dev`     | Vite dev server on port 3000                          |
| `npm run lint`    | Type check (`tsc --noEmit`)                           |
| `npm run build`   | Production build into `dist/`                         |
| `npm run preview` | Serve the production build locally                    |
