/**
 * Anonymous per-device id sent as `X-Client-Id` (backend decision D-04).
 *
 * WHY: there is no login in v1; the backend scopes "my history" to this UUID. It is NOT
 * authentication. The id lives in localStorage so history survives reloads. Storage can be
 * unavailable or throw (private mode, blocked site data, sandboxed iframe) and
 * `crypto.randomUUID` only exists in secure contexts (HTTPS/localhost) — so every step is
 * guarded and the function ALWAYS returns a valid UUID v4 (the backend answers anything else with
 * 400 INVALID_CLIENT_ID). When storage fails the id is held in memory: history then lasts only
 * until the page is reloaded, which is the best we can do.
 */

export const CLIENT_ID_STORAGE_KEY = 'keraai.clientId.v1';

const UUID_V4 = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

/** True for a canonical RFC 4122 version-4 UUID string. */
export function isUuidV4(value: unknown): value is string {
  return typeof value === 'string' && UUID_V4.test(value);
}

/** Format 16 random bytes as a v4 UUID (sets the version and variant bits). */
function formatV4(bytes: ArrayLike<number>): string {
  const b = Array.from(bytes, (x) => x & 0xff);
  b[6] = (b[6] & 0x0f) | 0x40; // version 4
  b[8] = (b[8] & 0x3f) | 0x80; // variant 10xx
  const hex = b.map((x) => x.toString(16).padStart(2, '0'));
  return (
    hex.slice(0, 4).join('') + '-' + hex.slice(4, 6).join('') + '-' + hex.slice(6, 8).join('') +
    '-' + hex.slice(8, 10).join('') + '-' + hex.slice(10, 16).join('')
  );
}

/** A fresh v4 UUID: `crypto.randomUUID()` → `crypto.getRandomValues` → `Math.random` (last resort). */
export function generateUuidV4(): string {
  const c = globalThis.crypto as Crypto | undefined;
  try {
    if (typeof c?.randomUUID === 'function') {
      const id = c.randomUUID();
      if (isUuidV4(id)) return id;
    }
  } catch {
    // fall through
  }
  const bytes = new Uint8Array(16);
  try {
    if (typeof c?.getRandomValues === 'function') {
      c.getRandomValues(bytes);
      return formatV4(bytes);
    }
  } catch {
    // fall through
  }
  for (let i = 0; i < 16; i++) bytes[i] = Math.floor(Math.random() * 256);
  return formatV4(bytes);
}

let memoryId: string | null = null;

/** The id of this device/browser; stable across calls, always a valid v4 UUID. Never throws. */
export function getClientId(): string {
  if (memoryId) return memoryId;

  try {
    const stored = globalThis.localStorage?.getItem(CLIENT_ID_STORAGE_KEY);
    if (isUuidV4(stored)) {
      memoryId = stored.toLowerCase();
      return memoryId;
    }
  } catch {
    // storage unavailable — fall back to a generated id below
  }

  memoryId = generateUuidV4();
  try {
    globalThis.localStorage?.setItem(CLIENT_ID_STORAGE_KEY, memoryId);
  } catch {
    // could not persist; the in-memory id still works for this page load
  }
  return memoryId;
}
