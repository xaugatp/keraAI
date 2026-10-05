/**
 * Pure guard for the Stage-2 hand-off (`LeafAnalysisView`): decides whether the `File` the user
 * just analysed with Model 1 should be auto-submitted to Model 2, without the user re-picking it.
 *
 * WHY a ref-tracked "already submitted" file instead of just `file !== null && leaf === null`:
 * while the hand-off submission is in flight (or after it fails), `leaf` is still `null` — without
 * remembering which `File` was already sent, every re-render would fire `predict()` again. The
 * caller stores the `File` it last sent (in a `useRef`) and passes it back as
 * `alreadySubmittedFile`; a genuinely new `File` (the user picked a different photo, or `reset()`
 * cleared everything so `alreadySubmittedFile` is reset too) compares unequal and is allowed
 * through again. A failed hand-off is retried only via the explicit Retry button, not by this
 * effect firing again on its own.
 */
import type { AnalysisDetail } from './types.ts';

export interface ShouldAutoSubmitHandoffInput {
  /** `useAnalysisState().file` — the photo Model 1 just analysed, or `null` for a direct visit. */
  file: File | null;
  /** `useAnalysisState().leaf` — the Model 2 result for the current session, if any. */
  leaf: AnalysisDetail | null;
  /** The `File` this view already sent to Model 2 (a `useRef<File | null>`), or `null`. */
  alreadySubmittedFile: File | null;
}

/** True exactly when there is a hand-off file, no leaf result yet, and it has not been sent. */
export function shouldAutoSubmitHandoff({
  file,
  leaf,
  alreadySubmittedFile,
}: ShouldAutoSubmitHandoffInput): boolean {
  return file !== null && leaf === null && file !== alreadySubmittedFile;
}
