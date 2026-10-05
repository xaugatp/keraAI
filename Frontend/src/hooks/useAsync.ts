/**
 * Shared plumbing for the data hooks: run an async loader, expose `{data, error, loading}`,
 * cancel the in-flight request on unmount / dependency change, and allow a manual `reload()`.
 *
 * WHY hand-rolled: the spec forbids a data-fetching library; this is ~40 lines and covers every
 * read in the app. The loader receives an `AbortSignal`; pass it to the api functions.
 */
import { useCallback, useEffect, useState } from 'react';
import type { DependencyList } from 'react';
import { AbortedError } from '../api/errors.ts';

export interface AsyncState<T> {
  data: T | null;
  /** The thrown value (`ApiError` / `NetworkError`) — render it with `userMessageFor(error)`. */
  error: unknown;
  loading: boolean;
}

export interface UseAsyncOptions {
  /** When false nothing is fetched (e.g. no id yet). Default true. */
  enabled?: boolean;
  /** Keep showing the previous data while a new request is in flight (pagination). Default false. */
  keepPreviousData?: boolean;
}

export function useAsync<T>(
  loader: (signal: AbortSignal) => Promise<T>,
  deps: DependencyList,
  options: UseAsyncOptions = {},
): AsyncState<T> & { reload: () => void } {
  const { enabled = true, keepPreviousData = false } = options;
  const [state, setState] = useState<AsyncState<T>>({ data: null, error: null, loading: enabled });
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    if (!enabled) {
      setState({ data: null, error: null, loading: false });
      return;
    }
    const controller = new AbortController();
    setState((prev) => ({ data: keepPreviousData ? prev.data : null, error: null, loading: true }));
    loader(controller.signal).then(
      (data) => {
        if (!controller.signal.aborted) setState({ data, error: null, loading: false });
      },
      (error: unknown) => {
        if (controller.signal.aborted || error instanceof AbortedError) return;
        setState((prev) => ({ data: keepPreviousData ? prev.data : null, error, loading: false }));
      },
    );
    return () => controller.abort();
    // `loader` is intentionally not a dependency: callers list the inputs it closes over in `deps`.
  },[...deps, enabled, keepPreviousData, nonce]);

  const reload = useCallback(() => setNonce((n) => n + 1), []);
  return { ...state, reload };
}
