import React from 'react';
import { ApiError, NetworkError, requestIdFor, userMessageFor } from '../api';

/**
 * `idle`     nothing in flight (the modal is not shown even if `open` is true)
 * `preparing` validating/building the request, before any bytes go out
 * `uploading` bytes are going out — `progress` (0-100) is real, from `upload.onprogress`
 * `analysing` upload finished; the server is running the model (indeterminate — no fake number)
 * `done`     the result came back (the caller usually navigates away immediately)
 * `error`    `error` holds the thrown `ApiError`/`NetworkError`
 */
export type ProcessingPhase = 'idle' | 'preparing' | 'uploading' | 'analysing' | 'done' | 'error';

export interface ProcessingModalProps {
  open: boolean;
  phase: ProcessingPhase;
  /** 0-100; only meaningful while `phase === 'uploading'`. */
  progress?: number;
  /** Set when `phase === 'error'`. */
  error?: ApiError | NetworkError | null;
  onCancel: () => void;
  /** Re-run the same submission. Omit to hide the Retry button. */
  onRetry?: () => void;
}

type Step = 'preparing' | 'uploading' | 'analysing';
const STEPS: Step[] = ['preparing', 'uploading', 'analysing'];
const STEP_LABEL: Record<Step, string> = {
  preparing: 'Preparing photo',
  uploading: 'Uploading photo',
  analysing: 'Running the model',
};

function stepStatus(step: Step, phase: ProcessingPhase): 'done' | 'current' | 'pending' {
  if (phase === 'done') return 'done';
  const stepIdx = STEPS.indexOf(step);
  const phaseIdx = STEPS.indexOf(phase as Step);
  if (phaseIdx === -1) return 'pending';
  if (stepIdx < phaseIdx) return 'done';
  if (stepIdx === phaseIdx) return 'current';
  return 'pending';
}

const TITLE: Record<ProcessingPhase, string> = {
  idle: '',
  preparing: 'Preparing photo…',
  uploading: 'Uploading photo…',
  analysing: 'Analysing the photo…',
  done: 'Analysis complete',
  error: 'Something went wrong',
};

/**
 * Real-state processing dialog: every phase is driven by the caller (`predict()`'s progress
 * callback and its resolved/rejected promise), never by a timer in here. No fake endpoints,
 * hardware names or latency numbers — `analysing` has no percentage because the backend gives
 * none.
 */
export const ProcessingModal: React.FC<ProcessingModalProps> = ({
  open,
  phase,
  progress,
  error,
  onCancel,
  onRetry,
}) => {
  if (!open || phase === 'idle') return null;

  const isError = phase === 'error';
  const isDone = phase === 'done';
  const requestId = requestIdFor(error);

  return (
    <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4 animate-fade-in">
      <div className="bg-white w-full max-w-md rounded-2xl p-6 sm:p-8 shadow-2xl border border-[#dae2fd] flex flex-col gap-6">
        <div className="flex items-center gap-3">
          <div
            className={`w-10 h-10 rounded-xl flex items-center justify-center shrink-0 ${
              isError ? 'bg-[#ffdad6] text-[#ba1a1a]' : 'bg-[#e2e7ff] text-[#006948]'
            }`}
          >
            <span className={`material-symbols-outlined text-[24px] ${isError || isDone ? '' : 'animate-spin'}`}>
              {isError ? 'error' : isDone ? 'check_circle' : 'progress_activity'}
            </span>
          </div>
          <div>
            <h3 className="text-lg font-extrabold text-[#131b2e]">{TITLE[phase]}</h3>
            {!isError && (
              <p className="text-xs text-[#3d4a42]">KeraAI is processing your photo. This usually takes a few seconds.</p>
            )}
          </div>
        </div>

        {isError ? (
          <div className="flex flex-col gap-3">
            <div className="p-3.5 rounded-xl bg-[#ffdad6] border border-[#ba1a1a]/30 text-[#ba1a1a] text-sm">
              {userMessageFor(error)}
            </div>
            {requestId && (
              <p className="font-mono text-[11px] text-[#3d4a42]">
                Request ID: <span className="font-semibold">{requestId}</span>
              </p>
            )}
            <div className="flex items-center justify-end gap-3 pt-1">
              <button
                type="button"
                onClick={onCancel}
                className="px-4 py-2 rounded-xl bg-[#eaedff] text-[#131b2e] hover:bg-[#dae2fd] font-semibold text-sm transition-all cursor-pointer"
              >
                Close
              </button>
              {onRetry && (
                <button
                  type="button"
                  onClick={onRetry}
                  className="px-4 py-2 rounded-xl bg-[#006948] text-white hover:bg-[#00855d] font-semibold text-sm transition-all cursor-pointer"
                >
                  Retry
                </button>
              )}
            </div>
          </div>
        ) : (
          <>
            {/* Overall bar: a real percentage while uploading, an indeterminate pulse otherwise. */}
            <div className="w-full bg-[#eaedff] rounded-full h-2 overflow-hidden">
              {phase === 'uploading' ? (
                <div
                  className="bg-[#006948] h-2 rounded-full transition-all duration-300 ease-out"
                  style={{ width: `${Math.max(0, Math.min(100, progress ?? 0))}%` }}
                />
              ) : (
                <div
                  className={`h-2 rounded-full bg-[#006948] ${isDone ? 'w-full' : 'w-1/3 animate-pulse'}`}
                />
              )}
            </div>

            <div className="flex flex-col gap-3 font-mono text-xs">
              {STEPS.map((step) => {
                const status = stepStatus(step, phase);
                return (
                  <div key={step} className="flex items-center gap-3">
                    {status === 'done' ? (
                      <span className="w-5 h-5 rounded-full bg-[#006948] text-white flex items-center justify-center text-[12px] font-bold shrink-0">
                        ✓
                      </span>
                    ) : status === 'current' ? (
                      <span className="w-5 h-5 rounded-full border-2 border-[#006948] text-[#006948] flex items-center justify-center animate-pulse text-[10px] shrink-0">
                        ●
                      </span>
                    ) : (
                      <span className="w-5 h-5 rounded-full border border-[#bccac0] text-[#bccac0] flex items-center justify-center text-[10px] shrink-0">
                        ○
                      </span>
                    )}
                    <span className={status === 'pending' ? 'text-[#3d4a42]/60' : 'text-[#131b2e] font-semibold'}>
                      {STEP_LABEL[step]}
                      {step === 'uploading' && phase === 'uploading' && typeof progress === 'number'
                        ? ` — ${Math.round(progress)}%`
                        : ''}
                    </span>
                  </div>
                );
              })}
            </div>

            <div className="flex items-center justify-end pt-1">
              <button
                type="button"
                onClick={onCancel}
                className="px-4 py-2 rounded-xl bg-[#eaedff] text-[#131b2e] hover:bg-[#dae2fd] font-semibold text-sm transition-all cursor-pointer"
              >
                Cancel
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
};
