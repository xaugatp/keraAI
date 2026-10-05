/**
 * THE single source of truth for where the API lives. Every other module imports
 * `API_BASE_URL` / `API_PREFIX` from here; the parsing/validation rules live in the pure
 * `resolveApiBaseUrl()` (baseUrl.ts) only so plain-Node unit tests can run them (`import.meta.env`
 * exists only under Vite).
 *
 * WHY env-driven: the backend sits on the owner's laptop behind a Cloudflare Tunnel, so its origin
 * differs per deployment (localhost in dev, `https://api.<domain>` on Netlify). `VITE_*` values
 * are baked into the bundle at build time and are therefore PUBLIC — never put secrets here.
 *
 * Throws at import time in a production build when `VITE_API_BASE_URL` is unset: a blank page
 * with a clear console error beats silently calling the wrong origin.
 */
import { resolveApiBaseUrl } from './baseUrl.ts';

/** Origin of the FastAPI backend, no trailing slash. */
export const API_BASE_URL: string = resolveApiBaseUrl(
  import.meta.env.VITE_API_BASE_URL,
  import.meta.env.DEV,
);

/** Prefix of every versioned route. `/health/*` is NOT under it. */
export const API_PREFIX = '/api/v1';
