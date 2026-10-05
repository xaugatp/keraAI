/**
 * One analysis in full (`GET /analyses/{id}`), with fresh signed image URLs.
 *
 * Image URLs expire after ~1 hour. If an `<img>` fails (`onError`), call `reload()` to get new
 * links instead of retrying the stale URL. Pass `null`/`undefined` to fetch nothing.
 */
import { getAnalysis } from '../api/analyses.ts';
import type { AnalysisDetail } from '../api/types.ts';
import { useAsync } from './useAsync.ts';

export interface UseAnalysisResult {
  analysis: AnalysisDetail | null;
  loading: boolean;
  error: unknown;
  reload: () => void;
}

export function useAnalysis(id: string | null | undefined): UseAnalysisResult {
  const { data, loading, error, reload } = useAsync(
    (signal) => getAnalysis(id as string, signal),
    [id],
    { enabled: Boolean(id) },
  );
  return { analysis: data, loading, error, reload };
}
