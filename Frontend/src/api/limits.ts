/**
 * Upload constraints, mirrored from the backend so the UI can reject a bad file BEFORE spending
 * a slow upload on it. The server stays authoritative (it re-checks and answers with
 * IMAGE_TOO_LARGE / UNSUPPORTED_MEDIA_TYPE), so if the backend limit changes, change it here too.
 */

/** Backend `MAX_UPLOAD_MB` (default 10). */
export const MAX_UPLOAD_MB = 10;
export const MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024;

/** The only image types the backend accepts (HEIC/HEIF from some phones is NOT supported). */
export const ACCEPTED_IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/webp'] as const;
