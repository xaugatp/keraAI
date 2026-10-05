/**
 * Turn the relative image URLs the API returns into absolute ones.
 *
 * WHY: the backend returns RELATIVE URLs (`/api/v1/analyses/{id}/image?variant=...&exp=...&sig=...`)
 * so the same JSON works for any deployment; the browser must prefix the API origin.
 *
 * IMPORTANT — never cache these: private image URLs are HMAC-signed and expire after ~1 hour
 * (`exp`/`sig`). Do not store them in localStorage/sessionStorage or keep them across sessions.
 * When an `<img>` fails to load (`onError`), re-fetch the analysis (`getAnalysis(id)` /
 * `useAnalysis(id).reload()`) to get fresh links instead of retrying the stale URL. Sample
 * images are public and do not expire.
 */
import { API_BASE_URL } from './config.ts';

/** `null`/`undefined` pass through as `null` (e.g. `image.result_url` for models without an overlay). */
export function absoluteImageUrl(rel: string): string;
export function absoluteImageUrl(rel: string | null | undefined): string | null;
export function absoluteImageUrl(rel: string | null | undefined): string | null {
  if (rel == null) return null;
  if (/^https?:\/\//i.test(rel)) return rel; // already absolute
  return `${API_BASE_URL}${rel.startsWith('/') ? '' : '/'}${rel}`;
}
