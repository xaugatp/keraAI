/**
 * `GET /api/v1/models` — which models exist, their status and whether their weights are a
 * placeholder. All three models are built and `ready` today; `status: 'unavailable'` is reserved
 * for a model that fails to load at startup (missing weights, bad checkpoint), not a future one.
 * No `X-Client-Id` needed (and omitting it avoids a CORS preflight).
 */
import { API_PREFIX } from './config.ts';
import { apiFetch } from './client.ts';
import type { ModelInfo } from './types.ts';

export function getModels(signal?: AbortSignal): Promise<ModelInfo[]> {
  return apiFetch<ModelInfo[]>(`${API_PREFIX}/models`, { signal, withClientId: false });
}
