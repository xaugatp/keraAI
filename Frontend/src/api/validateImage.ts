/**
 * Client-side mirror of the backend's upload constraints (`limits.ts`), so a bad file is rejected
 * instantly instead of after a slow upload over the tunnel. The server re-checks and stays
 * authoritative (`IMAGE_TOO_LARGE` / `UNSUPPORTED_MEDIA_TYPE`).
 *
 * WHY a plain `{name, type, size}` shape instead of `File`: this stays trivially unit-testable in
 * plain Node (no DOM `File` construction needed) and works the same for a `File` or any object
 * with those three fields.
 */
import { ACCEPTED_IMAGE_TYPES, MAX_UPLOAD_BYTES, MAX_UPLOAD_MB } from './limits.ts';

export interface PickedFileLike {
  name: string;
  type: string;
  size: number;
}

const HEIC_EXTENSION = /\.(heic|heif)$/i;

/**
 * Returns a friendly, ready-to-show error message, or `null` when the file passes every
 * client-side check. HEIC/HEIF gets its own message (common on iPhones; the backend does not
 * support it) rather than the generic "unsupported type" one.
 */
export function validateImageFile(file: PickedFileLike): string | null {
  const isHeic =
    file.type === 'image/heic' || file.type === 'image/heif' || HEIC_EXTENSION.test(file.name);
  if (isHeic) {
    return (
      'HEIC photos are not supported. Please use a JPEG, PNG or WEBP photo ' +
      '(on iPhone: Settings > Camera > Formats > Most Compatible).'
    );
  }
  if (!(ACCEPTED_IMAGE_TYPES as readonly string[]).includes(file.type)) {
    return "That file type isn't supported. Please use a JPEG, PNG or WEBP photo.";
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    return `That photo is too large. The maximum size is ${MAX_UPLOAD_MB} MB.`;
  }
  return null;
}
