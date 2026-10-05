/**
 * App-level state for the detect → leaf-analysis flow, shared by the views.
 *
 *   file  the photo the user picked/captured (memory only: a `File` cannot be persisted, so a page
 *         refresh loses it — the user re-selects it if they want to run another model)
 *   tree  the Model 1 result for it            leaf  the Model 2 result for it
 *
 * WHY only `lastAnalysisId` is persisted (sessionStorage, per tab): results live in the backend
 * database, so after a refresh `restoreLast()` simply re-fetches that one analysis (which also
 * returns FRESH signed image URLs — stored URLs would expire). Never store `AnalysisDetail`
 * itself or image URLs here.
 *
 * `setFile()` deliberately does NOT clear `tree`/`leaf` — the views decide when a new photo means
 * a new session (call `reset()` first for that).
 */
import { createContext, useCallback, useContext, useMemo, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { getAnalysis } from '../api/analyses.ts';
import { ApiError } from '../api/errors.ts';
import type { AnalysisDetail, ModelKey } from '../api/types.ts';

const STORAGE_KEY = 'keraai.lastAnalysis.v1';

interface StoredRef {
  id: string;
  modelKey: ModelKey;
}

function readStored(): StoredRef | null {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed === 'object' && parsed !== null) {
      const { id, modelKey } = parsed as Record<string, unknown>;
      if (typeof id === 'string' && typeof modelKey === 'string') {
        return { id, modelKey: modelKey as ModelKey };
      }
    }
  } catch {
    // storage blocked or corrupted — behave as if nothing was stored
  }
  return null;
}

function writeStored(ref: StoredRef | null): void {
  try {
    if (ref) sessionStorage.setItem(STORAGE_KEY, JSON.stringify(ref));
    else sessionStorage.removeItem(STORAGE_KEY);
  } catch {
    // private mode / blocked storage: the in-memory state still works for this page load
  }
}

export interface AnalysisContextValue {
  file: File | null;
  tree: AnalysisDetail | null;
  leaf: AnalysisDetail | null;
  setFile: (file: File | null) => void;
  /** Set (or clear with null) the Model 1 result; a non-null value also becomes the "last analysis". */
  setTree: (analysis: AnalysisDetail | null) => void;
  /** Set (or clear with null) the Model 2 result; a non-null value also becomes the "last analysis". */
  setLeaf: (analysis: AnalysisDetail | null) => void;
  /** Clear file, both results and the persisted last-analysis reference. */
  reset: () => void;
  /**
   * A read-only record opened from History — deliberately separate from `tree`/`leaf`, which are
   * the LIVE detect -> analyse-leaf session. Opening a past result must never be mistaken for (or
   * clobber) a session in progress, and viewing it must never offer "Analyse the Leaf"-style
   * hand-offs that imply a File is available (a history row has none).
   */
  viewing: AnalysisDetail | null;
  setViewing: (analysis: AnalysisDetail | null) => void;
  /**
   * After a refresh: re-fetch the last analysis of this tab and put it into `tree` or `leaf`.
   * Resolves the analysis, or `null` when there is nothing to restore (nothing stored, or it was
   * deleted — the stale reference is dropped). Rejects with the `ApiError`/`NetworkError` for
   * other failures so the caller can show `userMessageFor(error)`.
   */
  restoreLast: () => Promise<AnalysisDetail | null>;
}

const AnalysisContext = createContext<AnalysisContextValue | null>(null);

export function AnalysisProvider({ children }: { children: ReactNode }) {
  const [file, setFile] = useState<File | null>(null);
  const [tree, setTreeState] = useState<AnalysisDetail | null>(null);
  const [leaf, setLeafState] = useState<AnalysisDetail | null>(null);
  const [viewing, setViewing] = useState<AnalysisDetail | null>(null);
  // Guards restoreLast() against clobbering results the user produced while it was in flight.
  const generation = useRef(0);

  const setTree = useCallback((analysis: AnalysisDetail | null) => {
    generation.current += 1;
    setTreeState(analysis);
    if (analysis) writeStored({ id: analysis.id, modelKey: analysis.model_key });
  }, []);

  const setLeaf = useCallback((analysis: AnalysisDetail | null) => {
    generation.current += 1;
    setLeafState(analysis);
    if (analysis) writeStored({ id: analysis.id, modelKey: analysis.model_key });
  }, []);

  const reset = useCallback(() => {
    generation.current += 1;
    setFile(null);
    setTreeState(null);
    setLeafState(null);
    writeStored(null);
  }, []);

  const restoreLast = useCallback(async (): Promise<AnalysisDetail | null> => {
    const stored = readStored();
    if (!stored) return null;
    const startedAt = generation.current;
    let analysis: AnalysisDetail;
    try {
      analysis = await getAnalysis(stored.id);
    } catch (error) {
      if (error instanceof ApiError && error.code === 'ANALYSIS_NOT_FOUND') {
        writeStored(null);
        return null;
      }
      throw error;
    }
    if (generation.current !== startedAt) return analysis; // user moved on; don't overwrite
    if (analysis.model_key === 'leaf_segmentation') setLeafState(analysis);
    else if (analysis.model_key === 'tree_classification') setTreeState(analysis);
    return analysis;
  }, []);

  const value = useMemo<AnalysisContextValue>(
    () => ({ file, tree, leaf, setFile, setTree, setLeaf, reset, viewing, setViewing, restoreLast }),
    [file, tree, leaf, setTree, setLeaf, reset, viewing, restoreLast],
  );

  return <AnalysisContext.Provider value={value}>{children}</AnalysisContext.Provider>;
}

/** Access the shared flow state. Must be used under `<AnalysisProvider>` (App wraps everything). */
export function useAnalysisState(): AnalysisContextValue {
  const ctx = useContext(AnalysisContext);
  if (!ctx) throw new Error('useAnalysisState must be used inside <AnalysisProvider>.');
  return ctx;
}
