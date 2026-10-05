/**
 * Typed errors for every way an API call can fail, plus friendly user-facing messages.
 *
 * WHY three classes instead of one: the UI reacts differently to each —
 *   - `ApiError`     the server answered with an error (RFC 9457 `application/problem+json`, stable
 *                    `code`); show `userMessageFor()` + the `requestId` so the owner can find the log line.
 *   - `NetworkError` no usable answer (backend/tunnel down, DNS, CORS-blocked proxy page, timeout);
 *                    "server offline", usually worth a Retry.
 *   - `AbortedError` the caller cancelled (unmount / Cancel button); never show it as a failure.
 * This file has no imports on purpose so it stays trivially unit-testable in plain Node.
 */

export interface FieldError {
  field: string;
  message: string;
}

export interface ApiErrorInit {
  code: string;
  status: number;
  title: string;
  detail: string;
  requestId?: string | null;
  retryAfterSeconds?: number;
  fieldErrors?: FieldError[];
}

/** The server replied with an error status. */
export class ApiError extends Error {
  /** Stable machine code (`RATE_LIMITED`, `IMAGE_TOO_LARGE`, ...). Branch on this, never on `detail`. */
  readonly code: string;
  readonly status: number;
  readonly title: string;
  /** Server-written explanation. Fine for logs; prefer `userMessageFor(error)` in the UI. */
  readonly detail: string;
  /** Correlates with the backend log line (body `request_id`, else the `X-Request-ID` header). */
  readonly requestId: string | null;
  /** Seconds to wait before retrying (429). Only set when the `Retry-After` header was readable. */
  readonly retryAfterSeconds?: number;
  /** Per-field problems for `VALIDATION_ERROR` (422). */
  readonly fieldErrors?: FieldError[];

  constructor(init: ApiErrorInit) {
    super(init.detail || init.title);
    this.name = 'ApiError';
    this.code = init.code;
    this.status = init.status;
    this.title = init.title;
    this.detail = init.detail;
    this.requestId = init.requestId ?? null;
    this.retryAfterSeconds = init.retryAfterSeconds;
    this.fieldErrors = init.fieldErrors;
  }
}

/** No usable HTTP answer: server offline / tunnel down / blocked by CORS, or the request timed out. */
export class NetworkError extends Error {
  readonly reason: 'offline' | 'timeout';

  constructor(reason: 'offline' | 'timeout' = 'offline', options?: { cause?: unknown }) {
    super(
      reason === 'timeout' ? 'The server took too long to respond.' : 'Could not reach the server.',
      options,
    );
    this.name = 'NetworkError';
    this.reason = reason;
  }
}

/** The caller cancelled the request (AbortController). Not a failure — don't show an error. */
export class AbortedError extends Error {
  constructor() {
    super('The request was cancelled.');
    this.name = 'AbortedError';
  }
}

// --- Parsing ---------------------------------------------------------------------------------

/** Transport-neutral description of an error response (shared by the fetch and XHR code paths). */
export interface RawErrorResponse {
  status: number;
  /** Response body as text ('' when none). */
  body: string;
  /** `X-Request-ID` header (exposed by the backend's CORS config). */
  requestIdHeader?: string | null;
  /** `Retry-After` header (only readable cross-origin if the backend exposes it). */
  retryAfterHeader?: string | null;
}

/** Retry-After is either delta-seconds or an HTTP date. Returns whole seconds >= 0, or undefined. */
export function parseRetryAfter(header: string | null | undefined, now: number = Date.now()): number | undefined {
  if (header == null) return undefined;
  const value = header.trim();
  if (value === '') return undefined;
  if (/^\d+$/.test(value)) return Number(value);
  const when = Date.parse(value);
  if (Number.isNaN(when)) return undefined;
  return Math.max(0, Math.ceil((when - now) / 1000));
}

/** Codes for responses that are not a backend problem+json (proxy pages, empty bodies). */
function fallbackCode(status: number): string {
  if (status === 429) return 'RATE_LIMITED';
  if (status === 413) return 'IMAGE_TOO_LARGE';
  if (status === 415) return 'UNSUPPORTED_MEDIA_TYPE';
  if (status === 404) return 'NOT_FOUND';
  // Cloudflare Tunnel / reverse-proxy "origin is down" pages (HTML, no problem+json).
  if (status === 502 || status === 503 || status === 504 || (status >= 520 && status <= 530)) {
    return 'SERVER_UNREACHABLE';
  }
  return 'HTTP_ERROR';
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function str(value: unknown): string | undefined {
  return typeof value === 'string' && value !== '' ? value : undefined;
}

function parseFieldErrors(value: unknown): FieldError[] | undefined {
  if (!Array.isArray(value)) return undefined;
  const out: FieldError[] = [];
  for (const item of value) {
    const rec = asRecord(item);
    const message = str(rec?.message);
    if (rec && message) out.push({ field: str(rec.field) ?? '', message });
  }
  return out.length > 0 ? out : undefined;
}

/**
 * Build an `ApiError` from an error response. Never throws: a non-JSON body (an HTML proxy page,
 * an empty body) still yields a sensible ApiError with a status-derived `code`.
 */
export function parseApiError(raw: RawErrorResponse): ApiError {
  let json: Record<string, unknown> | null = null;
  const text = raw.body.trim();
  if (text.startsWith('{')) {
    try {
      json = asRecord(JSON.parse(text));
    } catch {
      json = null;
    }
  }

  const code = str(json?.code) ?? fallbackCode(raw.status);
  const title = str(json?.title) ?? `HTTP ${raw.status}`;
  const detail = str(json?.detail) ?? title;
  return new ApiError({
    code,
    status: raw.status,
    title,
    detail,
    requestId: str(json?.request_id) ?? str(raw.requestIdHeader) ?? null,
    retryAfterSeconds: parseRetryAfter(raw.retryAfterHeader),
    fieldErrors: parseFieldErrors(json?.errors),
  });
}

// --- Friendly messages -----------------------------------------------------------------------

/** Every `code` the backend can send (BACKEND_SPEC §11) plus the ones this client synthesises. */
export const KNOWN_ERROR_CODES = [
  'RATE_LIMITED',
  'IMAGE_TOO_LARGE',
  'UNSUPPORTED_MEDIA_TYPE',
  'INVALID_IMAGE',
  'IMAGE_TOO_SMALL',
  'IMAGE_TOO_LARGE_DIMENSIONS',
  'VALIDATION_ERROR',
  'MISSING_CLIENT_ID',
  'INVALID_CLIENT_ID',
  'INFERENCE_FAILED',
  'MODEL_UNAVAILABLE',
  'DATABASE_UNAVAILABLE',
  'ANALYSIS_NOT_FOUND',
  'SAMPLE_IMMUTABLE',
  'INVALID_SIGNATURE',
  'NOT_FOUND',
  'INTERNAL_ERROR',
  'HTTP_ERROR',
  // synthesised client-side
  'SERVER_UNREACHABLE',
  'INVALID_RESPONSE',
] as const;

export type KnownErrorCode = (typeof KNOWN_ERROR_CODES)[number];

const OFFLINE_MESSAGE =
  "Can't reach the KeraAI server. It may be offline or your connection may be down. Please try again in a moment.";
const GENERIC_MESSAGE = 'Something went wrong. Please try again.';

const MESSAGES: Record<KnownErrorCode, string | ((error: ApiError) => string)> = {
  RATE_LIMITED: (e) =>
    e.retryAfterSeconds !== undefined
      ? `Too many requests. Please wait ${e.retryAfterSeconds} second${e.retryAfterSeconds === 1 ? '' : 's'} and try again.`
      : 'Too many requests. Please wait a minute and try again.',
  IMAGE_TOO_LARGE: 'That photo is too large. The maximum size is 10 MB.',
  UNSUPPORTED_MEDIA_TYPE: "That file type isn't supported. Please use a JPEG, PNG or WEBP photo.",
  INVALID_IMAGE: "We couldn't read that image. Please try a different photo.",
  IMAGE_TOO_SMALL: 'That photo is too small to analyse. Please use a larger one.',
  IMAGE_TOO_LARGE_DIMENSIONS: 'That photo has too many pixels. Please use a smaller image.',
  VALIDATION_ERROR: "Some of the details sent weren't valid. Please check the photo and location and try again.",
  MISSING_CLIENT_ID: "This device couldn't be identified. Please reload the page and try again.",
  INVALID_CLIENT_ID: "This device couldn't be identified. Please reload the page and try again.",
  INFERENCE_FAILED: "The model couldn't analyse that photo. Please try again or use a different photo.",
  MODEL_UNAVAILABLE: "The analysis model isn't available right now. Please try again later.",
  DATABASE_UNAVAILABLE: "The server's database is temporarily unavailable. Please try again shortly.",
  ANALYSIS_NOT_FOUND: 'That analysis no longer exists.',
  SAMPLE_IMMUTABLE: "Sample analyses are read-only and can't be deleted.",
  INVALID_SIGNATURE: 'This image link has expired. Reload to get a fresh one.',
  NOT_FOUND: "We couldn't find what you asked for.",
  INTERNAL_ERROR: 'Something went wrong on the server. Please try again.',
  HTTP_ERROR: GENERIC_MESSAGE,
  SERVER_UNREACHABLE: OFFLINE_MESSAGE,
  INVALID_RESPONSE: "The server sent a reply we couldn't understand. Please try again.",
};

function isKnownCode(code: string): code is KnownErrorCode {
  return Object.prototype.hasOwnProperty.call(MESSAGES, code);
}

/**
 * Short, friendly English for any error thrown by this layer (or anything else).
 * Does NOT include the request id — the UI should always show `requestIdFor(error)` next to it.
 */
export function userMessageFor(error: unknown): string {
  if (error instanceof AbortedError) return 'Cancelled.';
  if (error instanceof NetworkError) {
    return error.reason === 'timeout' ? 'The server took too long to respond. Please try again.' : OFFLINE_MESSAGE;
  }
  if (error instanceof ApiError) {
    if (!isKnownCode(error.code)) return GENERIC_MESSAGE;
    const message = MESSAGES[error.code];
    return typeof message === 'function' ? message(error) : message;
  }
  return GENERIC_MESSAGE;
}

/** The backend request id to show next to an error (null for network errors / unknown errors). */
export function requestIdFor(error: unknown): string | null {
  return error instanceof ApiError ? error.requestId : null;
}

/** Whether the error means "the server could not be reached" (offline / tunnel down / timeout). */
export function isOfflineError(error: unknown): boolean {
  return (
    error instanceof NetworkError || (error instanceof ApiError && error.code === 'SERVER_UNREACHABLE')
  );
}
