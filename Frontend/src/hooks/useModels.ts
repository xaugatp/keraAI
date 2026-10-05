/**
 * `GET /models`, loaded once and shared by every component that calls this hook.
 *
 * WHY a module-level cache: the header, the About page and the workspace all need the model list;
 * it is tiny and static for the life of the server, so one request serves them all. A failed load
 * is NOT cached, so the next mount (or `reload()`) tries again — important because the backend may
 * simply not have been running when the page first loaded. Because the request is shared between
 * components it is not aborted on unmount; an unmounted consumer just ignores the result.
 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import { getModels } from '../api/models.ts';
import type { ModelInfo, ModelKey } from '../api/types.ts';

let cached: ModelInfo[] | null = null;
let inflight: Promise<ModelInfo[]> | null = null;

function loadModels(force: boolean): Promise<ModelInfo[]> {
  if (!force && cached) return Promise.resolve(cached);
  if (!inflight) {
    inflight = getModels()
      .then((models) => {
        cached = models;
        return models;
      })
      .finally(() => {
        inflight = null;
      });
  }
  return inflight;
}

export interface UseModelsResult {
  /** null until loaded (or if loading failed). */
  models: ModelInfo[] | null;
  byKey: Partial<Record<ModelKey, ModelInfo>>;
  loading: boolean;
  error: unknown;
  /** True when any READY model is running on placeholder weights → results are not real predictions. */
  anyPlaceholder: boolean;
  /** Force a fresh request (e.g. once the server comes back online). */
  reload: () => void;
}

export function useModels(): UseModelsResult {
  const [models, setModels] = useState<ModelInfo[] | null>(cached);
  const [loading, setLoading] = useState(cached === null);
  const [error, setError] = useState<unknown>(null);
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    let active = true;
    const force = nonce > 0;
    if (!force && cached) {
      setModels(cached);
      setLoading(false);
      return;
    }
    setLoading(true);
    loadModels(force).then(
      (result) => {
        if (!active) return;
        setModels(result);
        setError(null);
        setLoading(false);
      },
      (err: unknown) => {
        if (!active) return;
        setError(err);
        setLoading(false);
      },
    );
    return () => {
      active = false;
    };
  }, [nonce]);

  const reload = useCallback(() => setNonce((n) => n + 1), []);

  const byKey = useMemo(() => {
    const map: Partial<Record<ModelKey, ModelInfo>> = {};
    for (const model of models ?? []) map[model.key] = model;
    return map;
  }, [models]);

  const anyPlaceholder = useMemo(
    () => (models ?? []).some((m) => m.status === 'ready' && m.is_placeholder),
    [models],
  );

  return { models, byKey, loading, error, anyPlaceholder, reload };
}
