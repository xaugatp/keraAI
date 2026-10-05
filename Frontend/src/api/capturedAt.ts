/**
 * Resolves the single `capturedAt` timestamp sent to `predict()` (see `upload.ts`).
 *
 * WHY a camera shutter timestamp wins over a GPS fix's own timestamp: `useGeolocation()`'s
 * `position.capturedAt` is `position.timestamp` — when the BROWSER produced that fix, which can be
 * a cached/stale reading (`maximumAge`) taken well before or after the photo. A camera capture in
 * this app always stamps the exact moment of the shutter press, so for a camera photo it is the
 * more correct answer to "when was this photo taken". For an uploaded file (no capture instant of
 * our own — the file could be old), the GPS fix time is the best honest proxy we have, which
 * matches `PredictInput.capturedAt`'s own doc ("e.g. `position.timestamp`, or the shutter time for
 * camera shots"). With neither available, `capturedAt` is omitted entirely rather than guessed.
 */
export interface CapturedAtInputs {
  /** ISO-8601 (or `Date`) shutter-press time, only present for a camera-sourced photo. */
  cameraCapturedAt?: string | Date | null;
  /** `useGeolocation()`'s `position.capturedAt` (the fix's own timestamp), if a fix exists. */
  gpsCapturedAt?: Date | null;
}

function toIsoOrNull(value: string | Date | null | undefined): string | null {
  if (value == null) return null;
  const date = value instanceof Date ? value : new Date(value);
  return Number.isNaN(date.getTime()) ? null : date.toISOString();
}

/** The `capturedAt` to send with a prediction, or `null` when nothing usable is known. */
export function resolveCapturedAt({ cameraCapturedAt, gpsCapturedAt }: CapturedAtInputs): string | null {
  return toIsoOrNull(cameraCapturedAt) ?? toIsoOrNull(gpsCapturedAt ?? null);
}
