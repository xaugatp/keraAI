/**
 * The one place that talks to the backend with `fetch` (uploads with progress use XHR instead,
 * see `upload.ts`).
 *
 * WHY a wrapper: every call needs the same things — the base URL, the `X-Client-Id` header,
 * JSON/problem+json handling, cancellation, a timeout (a hung Cloudflare tunnel would otherwise
 * leave a spinner forever) and a uniform error type. Components never call `fetch` directly.
 */
import { API_BASE_URL } from './config.ts';
import { getClientId } from './clientId.ts';
import { AbortedError, ApiError, NetworkError, parseApiError } from './errors.ts';

export type QueryValue = string | number | boolean | null | undefined;

export interface ApiFetchOptions {
  method?: 'GET' | 'POST' | 'DELETE';
  /** Appended as a query string; `null`/`undefined` values are skipped. */
  query?: Record<string, QueryValue>;
  /**
   * `FormData` is sent as-is and the browser sets the multipart `Content-Type` itself (setting it
   * by hand would drop the boundary and break the upload). A plain object is sent as JSON.
   */
  body?: FormData | Record<string, unknown>;
  /** Cancels the request; the promise then rejects with `AbortedError`. */
  signal?: AbortSignal;
  /** Send `X-Client-Id` (default true). Turn off for routes that don't need it: it forces a CORS preflight. */
  withClientId?: boolean;
  /** Abort after this many ms with `NetworkError('timeout')` (default 30 000; 0 disables). */
  timeoutMs?: number;
  cache?: RequestCache;
}

export const DEFAULT_TIMEOUT_MS = 30_000;

/** `API_BASE_URL + path + ?query`. `path` is origin-relative and starts with `/` (e.g. `/api/v1/models`). */
export function buildUrl(path: string, query?: Record<string, QueryValue>): string {
  const url = `${API_BASE_URL}${path}`;
  if (!query) return url;
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    if (value !== null && value !== undefined) params.set(key, String(value));
  }
  const qs = params.toString();
  return qs ? `${url}?${qs}` : url;
}

function isAbortLike(error: unknown): boolean {
  return error instanceof Error && error.name === 'AbortError';
}

/**
 * Low-level request: handles URL/headers/body, the timeout, cancellation and network failures,
 * then hands the raw `Response` (ANY status) to `handle`, which runs inside the same timeout and
 * abort scope (so reading a slow body is covered too). Use `apiFetch` unless you need to treat
 * some error statuses as data (see `health.ts`).
 */
export async function apiRequest<T>(
  path: string,
  options: ApiFetchOptions,
  handle: (response: Response) => Promise<T>,
): Promise<T> {
  const { method = 'GET', query, body, signal, withClientId = true, timeoutMs = DEFAULT_TIMEOUT_MS, cache } = options;

  if (signal?.aborted) throw new AbortedError();

  const headers: Record<string, string> = { Accept: 'application/json' };
  if (withClientId) headers['X-Client-Id'] = getClientId();

  let payload: BodyInit | undefined;
  if (body instanceof FormData) {
    payload = body; // no Content-Type: the browser adds the multipart boundary
  } else if (body !== undefined) {
    headers['Content-Type'] = 'application/json';
    payload = JSON.stringify(body);
  }

  // One controller merges the caller's signal with our timeout.
  const controller = new AbortController();
  let timedOut = false;
  const timer =
    timeoutMs > 0
      ? setTimeout(() => {
          timedOut = true;
          controller.abort();
        }, timeoutMs)
      : undefined;
  const onCallerAbort = () => controller.abort();
  signal?.addEventListener('abort', onCallerAbort, { once: true });

  try {
    const response = await fetch(buildUrl(path, query), {
      method,
      headers,
      body: payload,
      signal: controller.signal,
      cache,
    });
    return await handle(response);
  } catch (error) {
    if (error instanceof ApiError || error instanceof NetworkError || error instanceof AbortedError) throw error;
    if (signal?.aborted) throw new AbortedError();
    if (timedOut) throw new NetworkError('timeout', { cause: error });
    if (isAbortLike(error)) throw new AbortedError();
    // fetch() rejects with a bare TypeError when the server is unreachable or CORS blocks the reply.
    throw new NetworkError('offline', { cause: error });
  } finally {
    if (timer !== undefined) clearTimeout(timer);
    signal?.removeEventListener('abort', onCallerAbort);
  }
}

/** Read an error response into the transport-neutral shape `parseApiError` expects. */
export async function toApiError(response: Response): Promise<ApiError> {
  let text = '';
  try {
    text = await response.text();
  } catch {
    text = '';
  }
  return parseApiError({
    status: response.status,
    body: text,
    requestIdHeader: response.headers.get('X-Request-ID'),
    retryAfterHeader: response.headers.get('Retry-After'),
  });
}

/**
 * JSON request. Resolves with the parsed body (`undefined` for 204); rejects with `ApiError`
 * (server error status), `NetworkError` (unreachable / timeout) or `AbortedError`.
 */
export function apiFetch<T>(path: string, options: ApiFetchOptions = {}): Promise<T> {
  return apiRequest<T>(path, options, async (response) => {
    if (!response.ok) throw await toApiError(response);
    if (response.status === 204) return undefined as T;
    try {
      return (await response.json()) as T;
    } catch (cause) {
      if (isAbortLike(cause)) throw cause;
      throw new ApiError({
        code: 'INVALID_RESPONSE',
        status: response.status,
        title: 'Invalid response',
        detail: 'The server returned a body that is not valid JSON.',
        requestId: response.headers.get('X-Request-ID'),
      });
    }
  });
}
