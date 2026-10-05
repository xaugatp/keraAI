/**
 * Readiness probe for the header pill.
 *
 * WHY separate from `apiFetch`: `/health/ready` answers 503 on purpose when a component (DB,
 * model) is down, and the 503 BODY says which one — that is data for the UI, not an exception.
 * The route lives at the origin root (NOT under `/api/v1`), needs no `X-Client-Id`, and must
 * never be served from cache.
 */
import { apiRequest, toApiError } from './client.ts';
import { ApiError } from './errors.ts';

export interface ReadyComponent {
  status: 'ok' | 'unavailable';
  /** Server-written note. May contain operational detail — don't show it to end users. */
  detail: string | null;
}

export interface ReadyResult {
  /** True only when the server answered 200 with `status: "ok"`. */
  ok: boolean;
  /** e.g. `{ database: {...}, models: {...} }`. */
  components: Record<string, ReadyComponent>;
}

function parseComponents(value: unknown): Record<string, ReadyComponent> | null {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return null;
  const out: Record<string, ReadyComponent> = {};
  for (const [name, raw] of Object.entries(value)) {
    const rec = typeof raw === 'object' && raw !== null ? (raw as Record<string, unknown>) : {};
    out[name] = {
      status: rec.status === 'ok' ? 'ok' : 'unavailable',
      detail: typeof rec.detail === 'string' ? rec.detail : null,
    };
  }
  return out;
}

/**
 * Resolves `{ok, components}` for HTTP 200 and for HTTP 503 with a readiness body (degraded).
 * Rejects with `NetworkError` (server/tunnel unreachable or slow), `AbortedError`, or `ApiError`
 * (any other status, or a 503 that is not a readiness body, e.g. a proxy error page).
 */
export function getReady(signal?: AbortSignal): Promise<ReadyResult> {
  return apiRequest<ReadyResult>(
    '/health/ready',
    { signal, withClientId: false, timeoutMs: 8_000, cache: 'no-store' },
    async (response) => {
      if (response.status === 200 || response.status === 503) {
        const text = await response.text();
        try {
          const body: unknown = JSON.parse(text);
          const components =
            typeof body === 'object' && body !== null
              ? parseComponents((body as Record<string, unknown>).components)
              : null;
          if (components) {
            const status = (body as Record<string, unknown>).status;
            return { ok: response.status === 200 && status === 'ok', components };
          }
        } catch {
          // not JSON — handled below
        }
        if (response.status === 200) {
          throw new ApiError({
            code: 'INVALID_RESPONSE',
            status: 200,
            title: 'Invalid response',
            detail: 'The readiness endpoint returned an unexpected body.',
          });
        }
        // 503 without a readiness body: a proxy/tunnel error page, not "degraded".
        throw new ApiError({
          code: 'SERVER_UNREACHABLE',
          status: 503,
          title: 'Service unavailable',
          detail: 'The server is not reachable.',
          requestId: response.headers.get('X-Request-ID'),
        });
      }
      throw await toApiError(response);
    },
  );
}
