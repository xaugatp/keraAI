/**
 * The public sample analyses for one model (`GET /analyses?scope=samples&model_key=...`).
 *
 * Samples are pre-computed by the real pipeline, so opening one needs NO inference: list them
 * here (thumbnail + title), then `getAnalysis(sample.id)` for the full result.
 */
import { listAnalyses } from '../api/analyses.ts';
import type { AnalysisSummary, ModelKey } from '../api/types.ts';
import { useAsync } from './useAsync.ts';

export interface UseSamplesResult {
  samples: AnalysisSummary[];
  loading: boolean;
  error: unknown;
  reload: () => void;
}

export function useSamples(modelKey: ModelKey): UseSamplesResult {
  const { data, loading, error, reload } = useAsync(
    (signal) => listAnalyses({ modelKey, scope: 'samples', pageSize: 100 }, signal),
    [modelKey],
  );
  return { samples: data?.items ?? [], loading, error, reload };
}
