/**
 * Pure helper that turns the raw `VITE_API_BASE_URL` into a clean base URL.
 *
 * WHY a separate file from `config.ts`: `config.ts` reads `import.meta.env`, which only exists
 * under Vite. Keeping the logic here (no `import.meta`) lets plain Node unit tests exercise it.
 */

/** Used only by the dev server when `VITE_API_BASE_URL` is unset (the backend binds to 127.0.0.1:8000). */
export const DEV_FALLBACK_BASE_URL = 'http://127.0.0.1:8000';

/**
 * @param raw   the raw env value (may be undefined/empty/padded/with trailing slashes)
 * @param isDev true under `vite` (dev server), false in a production build
 * @returns an absolute http(s) URL with no trailing slash
 * @throws  in a production build when the value is missing or not an absolute http(s) URL —
 *          shipping a build that silently calls the wrong origin is worse than a loud failure.
 */
export function resolveApiBaseUrl(raw: string | undefined, isDev: boolean): string {
  const trimmed = (raw ?? '').trim().replace(/\/+$/, '');

  if (trimmed === '') {
    if (isDev) return DEV_FALLBACK_BASE_URL;
    throw new Error(
      'VITE_API_BASE_URL is not set. Set it (e.g. https://api.<your-domain>) in the environment ' +
        'of the build (Netlify: Site settings > Environment variables) and rebuild.',
    );
  }

  let url: URL;
  try {
    url = new URL(trimmed);
  } catch {
    throw new Error(
      `VITE_API_BASE_URL must be an absolute http(s) URL such as https://api.example.com, got "${raw}".`,
    );
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    throw new Error(`VITE_API_BASE_URL must use http or https, got "${raw}".`);
  }
  return trimmed;
}
