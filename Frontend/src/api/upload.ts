/**
 * `predict()` — upload a photo to a model route and get the stored `AnalysisDetail` back.
 *
 * WHY XMLHttpRequest and not `fetch`: fetch cannot report UPLOAD progress, and a phone photo over
 * a tunnel can take many seconds. XHR's `upload.onprogress` drives the real percentage in the
 * processing dialog. Everything else in the app uses `fetch` (client.ts).
 *
 * Phases reported through `onProgress`:  { phase: 'uploading', percent: 0..100 }  while the
 * bytes go out, then  { phase: 'analysing' }  once the upload is complete and the server is
 * running the model (indeterminate — the backend gives no progress for inference).
 */
import { API_BASE_URL, API_PREFIX } from './config.ts';
import { getClientId } from './clientId.ts';
import { AbortedError, ApiError, NetworkError, parseApiError } from './errors.ts';
import type { AnalysisDetail, PredictModelKey, UploadSource } from './types.ts';

/** Predict route per model (relative to `API_PREFIX`). Model 3 has no route yet. */
export const PREDICT_ROUTES: Record<PredictModelKey, string> = {
  tree_classification: '/tree/predict',
  leaf_segmentation: '/leaf-segmentation/predict',
};

export interface PredictInput {
  /** The photo. A camera capture (`Blob` from `canvas.toBlob`) works too — wrap it in a `File` to name it. */
  file: Blob | File;
  /** Where the photo came from. */
  source: UploadSource;
  /** Degrees. Sent only together with `longitude`; if either is missing/invalid, NEITHER is sent. */
  latitude?: number | null;
  longitude?: number | null;
  /** GPS accuracy in metres. Only sent when coordinates are sent. */
  accuracyM?: number | null;
  /** When the photo was taken (e.g. `position.timestamp`, or the shutter time for camera shots). */
  capturedAt?: Date | string | null;
}

export type UploadProgress =
  | { phase: 'uploading'; percent: number }
  | { phase: 'analysing' };

export interface PredictOptions {
  onProgress?: (progress: UploadProgress) => void;
  /** Aborting rejects with `AbortedError` and cancels the in-flight upload. */
  signal?: AbortSignal;
  /** Hard cap for the whole upload + inference (default 120 000 ms) → `NetworkError('timeout')`. */
  timeoutMs?: number;
}

const DEFAULT_PREDICT_TIMEOUT_MS = 120_000;

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

/**
 * Build the multipart body. The backend accepts latitude/longitude only as a pair, so an
 * incomplete or out-of-range pair is dropped entirely rather than sent half-way (which the
 * server would reject with a 422).
 */
export function buildPredictFormData(input: PredictInput): FormData {
  const form = new FormData();
  const { file } = input;
  // Blobs have no name; the server only needs the bytes, but multipart wants a filename.
  form.append('image', file, file instanceof File && file.name ? file.name : 'photo.jpg');
  form.append('source', input.source);

  const { latitude, longitude } = input;
  const hasCoords =
    isFiniteNumber(latitude) &&
    isFiniteNumber(longitude) &&
    latitude >= -90 &&
    latitude <= 90 &&
    longitude >= -180 &&
    longitude <= 180;
  if (hasCoords) {
    form.append('latitude', String(latitude));
    form.append('longitude', String(longitude));
    if (isFiniteNumber(input.accuracyM) && input.accuracyM >= 0) {
      form.append('gps_accuracy_m', String(input.accuracyM));
    }
  }

  const captured = input.capturedAt;
  if (captured != null) {
    const date = captured instanceof Date ? captured : new Date(captured);
    if (!Number.isNaN(date.getTime())) form.append('captured_at', date.toISOString());
  }
  return form;
}

/** Run Model `modelKey` on a photo. Resolves with the stored analysis (HTTP 201 body). */
export function predict(
  modelKey: PredictModelKey,
  input: PredictInput,
  options: PredictOptions = {},
): Promise<AnalysisDetail> {
  const route = PREDICT_ROUTES[modelKey];
  if (!route) return Promise.reject(new Error(`Model "${modelKey}" has no predict route.`));

  const { onProgress, signal, timeoutMs = DEFAULT_PREDICT_TIMEOUT_MS } = options;
  if (signal?.aborted) return Promise.reject(new AbortedError());

  return new Promise<AnalysisDetail>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    let settled = false;
    let analysing = false;

    const cleanup = () => signal?.removeEventListener('abort', onCallerAbort);
    const fail = (error: Error) => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(error);
    };
    const onCallerAbort = () => {
      xhr.abort(); // fires onabort → AbortedError
    };
    const reportAnalysing = () => {
      if (analysing) return;
      analysing = true;
      onProgress?.({ phase: 'analysing' });
    };

    xhr.open('POST', `${API_BASE_URL}${API_PREFIX}${route}`);
    xhr.setRequestHeader('Accept', 'application/json');
    xhr.setRequestHeader('X-Client-Id', getClientId());
    // Do NOT set Content-Type: the browser adds multipart/form-data with its boundary.
    xhr.responseType = 'text';
    xhr.timeout = timeoutMs;

    xhr.upload.onprogress = (event) => {
      if (analysing || !event.lengthComputable || event.total === 0) return;
      const percent = Math.min(100, Math.round((event.loaded / event.total) * 100));
      onProgress?.({ phase: 'uploading', percent });
    };
    xhr.upload.onload = reportAnalysing; // every byte is out; the server is now running the model

    xhr.onload = () => {
      if (settled) return;
      const text = typeof xhr.responseText === 'string' ? xhr.responseText : '';
      const requestId = xhr.getResponseHeader('X-Request-ID');
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const detail = JSON.parse(text) as AnalysisDetail;
          settled = true;
          cleanup();
          resolve(detail);
        } catch {
          fail(
            new ApiError({
              code: 'INVALID_RESPONSE',
              status: xhr.status,
              title: 'Invalid response',
              detail: 'The server returned a body that is not valid JSON.',
              requestId,
            }),
          );
        }
        return;
      }
      fail(
        parseApiError({
          status: xhr.status,
          body: text,
          requestIdHeader: requestId,
          retryAfterHeader: xhr.getResponseHeader('Retry-After'),
        }),
      );
    };
    xhr.onerror = () => fail(new NetworkError('offline'));
    xhr.ontimeout = () => fail(new NetworkError('timeout'));
    xhr.onabort = () => fail(new AbortedError());

    signal?.addEventListener('abort', onCallerAbort, { once: true });

    onProgress?.({ phase: 'uploading', percent: 0 });
    xhr.send(buildPredictFormData(input));
  });
}
