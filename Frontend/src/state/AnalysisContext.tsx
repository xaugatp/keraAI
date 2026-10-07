/**
 * App-level state for the detect → leaf-analysis flow, shared by the views.
 *
 *   file     the photo the user picked/captured (memory only: a `File` cannot be persisted, so a
 *            page refresh loses it — the user re-selects it if they want to run another model)
 *   tree     the Model 1 result for it   leaf  the Model 2 result for it   disease  the Model 3 result for it
 *   viewing  a read-only record opened from History — deliberately separate from tree/leaf/disease
 *            (see below)
 *
 * WHY each slot persists only an id (sessionStorage), not the `AnalysisDetail` itself: results
 * live in the backend database, so after a refresh `restoreTree()`/`restoreLeaf()`/
 * `restoreDisease()`/`restoreViewing()` simply re-fetch that one analysis (which also returns
 * FRESH signed image URLs — stored URLs would expire). App.tsx calls the one the restored tab
 * actually needs on mount, so reloading the page lands back on the same tab WITH its data, not on
 * Home.
 *
 * WHY four independent slots (not one shared "last analysis" ref): tree, leaf and disease can all
 * be set in the same session (the Stage-1 -> Stage-2 -> Stage-3 hand-offs) and reloading on an
 * earlier stage's page must not restore a later stage's result into it just because it happened
 * more recently — each view's data must come back independently of what else the user did
 * afterward. `viewing` is kept separate again: opening a past record from History must never be
 * mistaken for, or clobber, a session in progress, and viewing it must never offer
 * "Analyse the Leaf"-style hand-offs that imply a File is available (a history row has none).
 *
 * `setFile()` deliberately does NOT clear `tree`/`leaf` — the views decide when a new photo means
 * a new session (call `reset()` first for that).
 */
import { createContext, useCallback, useContext, useMemo, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { getAnalysis } from '../api/analyses.ts';
import { ApiError } from '../api/errors.ts';
import type { AnalysisDetail } from '../api/types.ts';

type Slot = 'tree' | 'leaf' | 'disease' | 'viewing';

function storageKey(slot: Slot): string {
  return `keraai.last.${slot}.v1`;
}

function readStoredId(slot: Slot): string | null {
  try {
    return sessionStorage.getItem(storageKey(slot));
  } catch {
    return null; // private mode / blocked storage
  }
}

function writeStoredId(slot: Slot, id: string | null): void {
  try {
    if (id) sessionStorage.setItem(storageKey(slot), id);
    else sessionStorage.removeItem(storageKey(slot));
  } catch {
    // private mode / blocked storage: the in-memory state still works for this page load
  }
}

export interface AnalysisContextValue {
  file: File | null;
  tree: AnalysisDetail | null;
  leaf: AnalysisDetail | null;
  disease: AnalysisDetail | null;
  setFile: (file: File | null) => void;
  /** Set (or clear with null) the Model 1 result; a non-null value also becomes the restorable "last tree result". */
  setTree: (analysis: AnalysisDetail | null) => void;
  /** Set (or clear with null) the Model 2 result; a non-null value also becomes the restorable "last leaf result". */
  setLeaf: (analysis: AnalysisDetail | null) => void;
  /** Set (or clear with null) the Model 3 result; a non-null value also becomes the restorable "last disease result". */
  setDisease: (analysis: AnalysisDetail | null) => void;
  /** Clear file, all results, the read-only `viewing` record and every persisted reference. */
  reset: () => void;
  viewing: AnalysisDetail | null;
  setViewing: (analysis: AnalysisDetail | null) => void;
  /**
   * Re-fetch the last tree/leaf/disease/viewing analysis (by whichever one's persisted id exists)
   * and put it back into that same slot. Resolves the analysis, or `null` when there is nothing
   * stored, or it was deleted (the stale reference is then dropped). Rejects with the
   * ApiError/NetworkError for other failures so the caller can show `userMessageFor(error)`.
   * Called once, on mount, by App.tsx — only for whichever slot the restored tab actually needs.
   */
  restoreTree: () => Promise<AnalysisDetail | null>;
  restoreLeaf: () => Promise<AnalysisDetail | null>;
  restoreDisease: () => Promise<AnalysisDetail | null>;
  restoreViewing: () => Promise<AnalysisDetail | null>;
}

const AnalysisContext = createContext<AnalysisContextValue | null>(null);

export function AnalysisProvider({ children }: { children: ReactNode }) {
  const [file, setFile] = useState<File | null>(null);
  const [tree, setTreeState] = useState<AnalysisDetail | null>(null);
  const [leaf, setLeafState] = useState<AnalysisDetail | null>(null);
  const [disease, setDiseaseState] = useState<AnalysisDetail | null>(null);
  const [viewing, setViewingState] = useState<AnalysisDetail | null>(null);
  // Guards each restoreX() against clobbering a result the user produced while it was in flight —
  // one shared counter is enough since a restore only ever writes back to the slot it read from.
  const generation = useRef(0);

  const setTree = useCallback((analysis: AnalysisDetail | null) => {
    generation.current += 1;
    setTreeState(analysis);
    writeStoredId('tree', analysis?.id ?? null);
  }, []);

  const setLeaf = useCallback((analysis: AnalysisDetail | null) => {
    generation.current += 1;
    setLeafState(analysis);
    writeStoredId('leaf', analysis?.id ?? null);
  }, []);

  const setDisease = useCallback((analysis: AnalysisDetail | null) => {
    generation.current += 1;
    setDiseaseState(analysis);
    writeStoredId('disease', analysis?.id ?? null);
  }, []);

  const setViewing = useCallback((analysis: AnalysisDetail | null) => {
    generation.current += 1;
    setViewingState(analysis);
    writeStoredId('viewing', analysis?.id ?? null);
  }, []);

  const reset = useCallback(() => {
    generation.current += 1;
    setFile(null);
    setTreeState(null);
    setLeafState(null);
    setDiseaseState(null);
    setViewingState(null);
    writeStoredId('tree', null);
    writeStoredId('leaf', null);
    writeStoredId('disease', null);
    writeStoredId('viewing', null);
  }, []);

  const restoreSlot = useCallback(
    (slot: Slot, apply: (analysis: AnalysisDetail) => void): (() => Promise<AnalysisDetail | null>) =>
      async () => {
        const id = readStoredId(slot);
        if (!id) return null;
        const startedAt = generation.current;
        let analysis: AnalysisDetail;
        try {
          analysis = await getAnalysis(id);
        } catch (error) {
          if (error instanceof ApiError && error.code === 'ANALYSIS_NOT_FOUND') {
            writeStoredId(slot, null);
            return null;
          }
          throw error;
        }
        if (generation.current !== startedAt) return analysis; // user moved on; don't overwrite
        apply(analysis);
        return analysis;
      },
    [],
  );

  const restoreTree = useCallback(restoreSlot('tree', setTreeState), [restoreSlot]);
  const restoreLeaf = useCallback(restoreSlot('leaf', setLeafState), [restoreSlot]);
  const restoreDisease = useCallback(restoreSlot('disease', setDiseaseState), [restoreSlot]);
  const restoreViewing = useCallback(restoreSlot('viewing', setViewingState), [restoreSlot]);

  const value = useMemo<AnalysisContextValue>(
    () => ({
      file,
      tree,
      leaf,
      disease,
      setFile,
      setTree,
      setLeaf,
      setDisease,
      reset,
      viewing,
      setViewing,
      restoreTree,
      restoreLeaf,
      restoreDisease,
      restoreViewing,
    }),
    [
      file,
      tree,
      leaf,
      disease,
      setTree,
      setLeaf,
      setDisease,
      reset,
      viewing,
      setViewing,
      restoreTree,
      restoreLeaf,
      restoreDisease,
      restoreViewing,
    ],
  );

  return <AnalysisContext.Provider value={value}>{children}</AnalysisContext.Provider>;
}

/** Access the shared flow state. Must be used under `<AnalysisProvider>` (App wraps everything). */
export function useAnalysisState(): AnalysisContextValue {
  const ctx = useContext(AnalysisContext);
  if (!ctx) throw new Error('useAnalysisState must be used inside <AnalysisProvider>.');
  return ctx;
}
