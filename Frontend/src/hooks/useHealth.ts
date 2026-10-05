/**
 * Polls `/health/ready` for the header status pill.
 *
 * States:  checking (first probe pending) · ok · degraded (server answered 503 — some component
 * is down; `failing` names them) · offline (no usable answer: server/tunnel down or too slow).
 *
 * Polling pauses while the tab is hidden (no point waking the owner's laptop for a background
 * tab) and probes immediately when the tab becomes visible again. Probes are chained with
 * setTimeout rather than setInterval so a slow response can never overlap the next probe.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { getReady } from '../api/health.ts';
import type { ReadyComponent } from '../api/health.ts';
import { AbortedError } from '../api/errors.ts';

export type HealthStatus = 'checking' | 'ok' | 'degraded' | 'offline';

export interface UseHealthResult {
  status: HealthStatus;
  components: Record<string, ReadyComponent>;
  /** Names of components that are not ok (e.g. `['database']`); empty unless `degraded`. */
  failing: string[];
  /** `Date.now()` of the last completed probe, or null before the first. */
  checkedAt: number | null;
  /** Probe right now (also restarts the polling timer). */
  refresh: () => void;
}

interface Snapshot {
  status: HealthStatus;
  components: Record<string, ReadyComponent>;
  checkedAt: number | null;
}

export function useHealth(intervalMs = 30_000): UseHealthResult {
  const [snapshot, setSnapshot] = useState<Snapshot>({ status: 'checking', components: {}, checkedAt: null });
  const probeRef = useRef<() => void>(() => {});

  useEffect(() => {
    let disposed = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let controller: AbortController | null = null;

    const stop = () => {
      if (timer !== undefined) clearTimeout(timer);
      timer = undefined;
      controller?.abort();
      controller = null;
    };

    const schedule = () => {
      if (disposed || document.hidden) return;
      timer = setTimeout(probe, intervalMs);
    };

    const probe = () => {
      stop();
      if (disposed || document.hidden) return;
      const mine = new AbortController();
      controller = mine;
      getReady(mine.signal).then(
        (result) => {
          if (disposed || mine.signal.aborted) return;
          setSnapshot({
            status: result.ok ? 'ok' : 'degraded',
            components: result.components,
            checkedAt: Date.now(),
          });
          schedule();
        },
        (error: unknown) => {
          if (disposed || mine.signal.aborted || error instanceof AbortedError) return;
          setSnapshot({ status: 'offline', components: {}, checkedAt: Date.now() });
          schedule();
        },
      );
    };

    const onVisibility = () => {
      if (document.hidden) stop();
      else probe();
    };

    probeRef.current = probe;
    document.addEventListener('visibilitychange', onVisibility);
    probe();

    return () => {
      disposed = true;
      document.removeEventListener('visibilitychange', onVisibility);
      stop();
    };
  }, [intervalMs]);

  const refresh = useCallback(() => probeRef.current(), []);
  const failing = Object.entries(snapshot.components)
    .filter(([, component]) => component.status !== 'ok')
    .map(([name]) => name);

  return { ...snapshot, failing, refresh };
}
