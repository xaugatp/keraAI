/**
 * History / detail / delete endpoints (`/api/v1/analyses`).
 *
 * All three send `X-Client-Id`: "mine" is whatever this device created, and `scope: 'samples'`
 * adds the public pre-computed samples. A 404 `ANALYSIS_NOT_FOUND` is also returned for another
 * device's analysis — the backend never reveals that it exists.
 */
import { API_PREFIX } from './config.ts';
import { apiFetch } from './client.ts';
import type { AnalysisDetail, AnalysisSummaryPage, ModelKey, Scope } from './types.ts';

const BASE = `${API_PREFIX}/analyses`;

export interface ListAnalysesParams {
  /** Only this model (omit for all). */
  modelKey?: ModelKey;
  /** Default `'all_visible'` (mine + samples). */
  scope?: Scope;
  /** 1-based (default 1). */
  page?: number;
  /** 1-100 (default 20). */
  pageSize?: number;
}

/** One page of summaries, newest first. */
export function listAnalyses(
  params: ListAnalysesParams = {},
  signal?: AbortSignal,
): Promise<AnalysisSummaryPage> {
  return apiFetch<AnalysisSummaryPage>(BASE, {
    query: {
      model_key: params.modelKey,
      scope: params.scope,
      page: params.page,
      page_size: params.pageSize,
    },
    signal,
  });
}

/** Full detail with FRESH signed image URLs (re-call this when an `<img>` fails — see urls.ts). */
export function getAnalysis(id: string, signal?: AbortSignal): Promise<AnalysisDetail> {
  return apiFetch<AnalysisDetail>(`${BASE}/${encodeURIComponent(id)}`, { signal });
}

/** Soft-delete one of YOUR analyses (204). Samples are immutable → `ApiError` code `SAMPLE_IMMUTABLE`. */
export function deleteAnalysis(id: string, signal?: AbortSignal): Promise<void> {
  return apiFetch<void>(`${BASE}/${encodeURIComponent(id)}`, { method: 'DELETE', signal });
}
