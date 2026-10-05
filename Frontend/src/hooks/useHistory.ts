/**
 * One page of history (`GET /analyses`) with refresh and delete.
 *
 * `scope`: 'mine' (this device) · 'samples' (public) · 'all_visible'. The previous page stays on
 * screen while the next one loads, so paging does not flash an empty table.
 */
import { useCallback } from 'react';
import { deleteAnalysis, listAnalyses } from '../api/analyses.ts';
import type { AnalysisSummary, AnalysisSummaryPage, ModelKey, Scope } from '../api/types.ts';
import { useAsync } from './useAsync.ts';

export interface UseHistoryParams {
  modelKey?: ModelKey;
  scope?: Scope;
  /** 1-based. */
  page?: number;
  pageSize?: number;
}

export interface UseHistoryResult {
  /** The whole envelope (`total`, `total_pages`, ...); null until the first load. */
  page: AnalysisSummaryPage | null;
  items: AnalysisSummary[];
  loading: boolean;
  error: unknown;
  /** Re-fetch the current page. */
  refresh: () => void;
  /**
   * Delete one of YOUR analyses, then refresh. Rejects with the `ApiError` on failure
   * (`SAMPLE_IMMUTABLE`, `ANALYSIS_NOT_FOUND`, ...) so the caller can show `userMessageFor(error)`.
   */
  remove: (id: string) => Promise<void>;
}

export function useHistory({
  modelKey,
  scope = 'all_visible',
  page = 1,
  pageSize = 20,
}: UseHistoryParams = {}): UseHistoryResult {
  const { data, loading, error, reload } = useAsync(
    (signal) => listAnalyses({ modelKey, scope, page, pageSize }, signal),
    [modelKey, scope, page, pageSize],
    { keepPreviousData: true },
  );

  const remove = useCallback(
    async (id: string) => {
      await deleteAnalysis(id);
      reload();
    },
    [reload],
  );

  return { page: data, items: data?.items ?? [], loading, error, refresh: reload, remove };
}
