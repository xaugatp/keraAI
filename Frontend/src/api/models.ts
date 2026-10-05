/**
 * `GET /api/v1/models` — which models exist, their status and whether their weights are a
 * placeholder. Includes planned models (status `unavailable`) so the UI can show "coming soon".
 * No `X-Client-Id` needed (and omitting it avoids a CORS preflight).
 */
import { API_PREFIX } from './config.ts';
import { apiFetch } from './client.ts';
import type { ModelInfo } from './types.ts';

export function getModels(signal?: AbortSignal): Promise<ModelInfo[]> {
  return apiFetch<ModelInfo[]>(`${API_PREFIX}/models`, { signal, withClientId: false });
}
