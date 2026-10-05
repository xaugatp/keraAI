/**
 * Read-only "view a past result" page — what History rows open into.
 *
 * WHY this exists as its own view instead of reusing Stage1ResultView/LeafAnalysisView directly:
 * those two are the LIVE workspaces (LeafAnalysisView in particular embeds its own camera/upload/
 * sample entry UI, since it must also work as a standalone starting point). Opening a saved history
 * row into them made the page look like an invitation to start a new scan, and the two looked
 * inconsistent with each other (Stage1ResultView has no camera/upload chrome, LeafAnalysisView
 * does). This view has none of that: it only ever displays an already-fetched `AnalysisDetail`,
 * the same way for both models, with one small "Start New Detection" exit and nothing else.
 */
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { ViewTab } from '../types';
import { useAnalysisState } from '../state/AnalysisContext';
import { useModels } from '../hooks/useModels';
import { absoluteImageUrl, getAnalysis, isLeafSegDetails, isTreeDetails } from '../api';
import type { AnalysisDetail, LeafSegLabel, ModelInfo, TreeVerdict } from '../api';

interface AnalysisDetailViewProps {
  onNavigate: (tab: ViewTab) => void;
}

function fmtMs(ms: number | null | undefined): string {
  return ms == null ? '—' : `${Math.round(ms)} ms`;
}

function fmtPct(value: number | null | undefined, decimals = 1): string {
  return value == null ? '—' : `${value.toFixed(decimals)}%`;
}

/** `value` is a 0-1 probability. */
function fmtProbability(value: number | null | undefined): string {
  return value == null ? '—' : `${(value * 100).toFixed(1)}%`;
}

function fmtCoord(value: number | null | undefined, positive: string, negative: string): string {
  if (value == null) return '—';
  return `${Math.abs(value).toFixed(6)}° ${value >= 0 ? positive : negative}`;
}

// --- Shared chrome: both model bodies render inside this, so the two never look different ------

function Breadcrumb({ onNavigate }: { onNavigate: (tab: ViewTab) => void }) {
  return (
    <div className="flex items-center gap-2 text-xs font-mono uppercase text-[#3d4a42] font-semibold">
      <button onClick={() => onNavigate('home')} className="hover:text-[#006948] transition-colors cursor-pointer">
        Home
      </button>
      <span className="material-symbols-outlined text-[14px]">chevron_right</span>
      <button onClick={() => onNavigate('history')} className="hover:text-[#006948] transition-colors cursor-pointer">
        Analysis History
      </button>
      <span className="material-symbols-outlined text-[14px]">chevron_right</span>
      <span className="text-[#006948] font-bold">Record</span>
    </div>
  );
}

function PlaceholderChip({ modelInfo }: { modelInfo: ModelInfo | undefined }) {
  if (!modelInfo?.is_placeholder) return null;
  return (
    <span
      title="Results come from placeholder weights and are not real predictions"
      className="px-1.5 py-0.5 rounded bg-[#ffeed2] border border-[#ffb95f] text-[#825100] text-[10px] font-bold"
    >
      Demo weights
    </span>
  );
}

function RecordMeta({ analysis }: { analysis: AnalysisDetail }) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] font-mono text-[#3d4a42]">
      <span>ID: {analysis.id}</span>
      <span>{new Date(analysis.created_at).toLocaleString()}</span>
      <span className="capitalize">
        {analysis.is_sample ? 'Public sample' : `From ${analysis.source}`}
      </span>
    </div>
  );
}

function Footer({ onNavigate }: { onNavigate: (tab: ViewTab) => void }) {
  return (
    <div className="flex flex-col sm:flex-row gap-3 pt-2">
      <button
        type="button"
        onClick={() => onNavigate('history')}
        className="flex-1 py-3 px-4 rounded-xl bg-[#eaedff] text-[#131b2e] hover:bg-[#dae2fd] font-semibold text-sm transition-all flex items-center justify-center gap-2 border border-[#dae2fd] cursor-pointer"
      >
        <span className="material-symbols-outlined text-[18px]">arrow_back</span>
        <span>Back to History</span>
      </button>
      <button
        type="button"
        onClick={() => onNavigate('detect')}
        className="flex-1 py-3 px-4 rounded-xl bg-[#006948] text-white hover:bg-[#00855d] font-semibold text-sm transition-all flex items-center justify-center gap-2 shadow-sm cursor-pointer"
      >
        <span className="material-symbols-outlined text-[18px]">add_a_photo</span>
        <span>Start New Detection</span>
      </button>
    </div>
  );
}

/** Signed URLs expire after ~1h: fetch a fresh copy of the record once, then give up and show a fallback. */
function useImageRetry(analysis: AnalysisDetail) {
  const [current, setCurrent] = useState(analysis);
  const [failed, setFailed] = useState(false);
  const retriedRef = useRef(false);

  useEffect(() => {
    setCurrent(analysis);
    setFailed(false);
    retriedRef.current = false;
  }, [analysis]);

  const onError = useCallback(() => {
    if (retriedRef.current) {
      setFailed(true);
      return;
    }
    retriedRef.current = true;
    getAnalysis(analysis.id)
      .then(setCurrent)
      .catch(() => setFailed(true));
  }, [analysis.id]);

  return { current, failed, onError };
}

// --- Model 1: tree classification ----------------------------------------------------------------

const VERDICT_STYLE: Record<TreeVerdict, { icon: string; bannerClass: string; cardBorder: string; title: string }> = {
  banana_tree: {
    icon: 'check_circle',
    bannerClass: 'bg-[#85f8c4]/30 border border-[#85f8c4] text-[#002114]',
    cardBorder: 'border-[#dae2fd]',
    title: 'Banana Tree Confirmed',
  },
  uncertain: {
    icon: 'help',
    bannerClass: 'bg-[#ffeed2] border border-[#ffb95f] text-[#5c3c00]',
    cardBorder: 'border-[#ffb95f]/60',
    title: 'Uncertain',
  },
  not_banana_tree: {
    icon: 'info',
    bannerClass: 'bg-[#ffdad6] border border-[#ba1a1a]/30 text-[#ba1a1a]',
    cardBorder: 'border-[#ffdad6]',
    title: 'Banana Tree Not Detected',
  },
};

const CIRCLE_R = 42;
const CIRCUMFERENCE = 2 * Math.PI * CIRCLE_R;

function TreeResultBody({ analysis, modelInfo }: { analysis: AnalysisDetail; modelInfo: ModelInfo | undefined }) {
  const { current, failed, onError } = useImageRetry(analysis);
  const details = isTreeDetails(current.details) ? current.details : null;
  const prediction = current.prediction;
  const verdict = details?.verdict ?? null;
  const style = verdict ? VERDICT_STYLE[verdict] : null;
  const confidencePct = prediction?.confidence != null ? prediction.confidence * 100 : null;
  const dashOffset = confidencePct != null ? CIRCUMFERENCE * (1 - confidencePct / 100) : CIRCUMFERENCE;
  const resolutionText =
    current.image.width != null && current.image.height != null
      ? `${current.image.width} × ${current.image.height} px`
      : '—';

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
      <div className="lg:col-span-7 flex flex-col gap-4">
        <div className="relative w-full aspect-[4/3] rounded-2xl overflow-hidden bg-[#283044] shadow-md border border-[#dae2fd]">
          {!failed ? (
            <img
              alt="Analyzed plant specimen"
              className="w-full h-full object-cover"
              src={absoluteImageUrl(current.image.original_url)}
              onError={onError}
            />
          ) : (
            <div className="w-full h-full flex flex-col items-center justify-center gap-2 text-white/70">
              <span className="material-symbols-outlined text-[36px]">broken_image</span>
              <span className="text-sm">Image unavailable</span>
            </div>
          )}
          <div className="absolute bottom-4 left-4 right-4 bg-white/90 backdrop-blur-md p-3 rounded-xl flex items-center justify-between text-[#131b2e] shadow-md border border-white/60">
            <div className="flex items-center gap-3">
              <span className="material-symbols-outlined text-[18px] text-[#006948]">photo_camera</span>
              <span className="font-mono text-xs font-semibold">{resolutionText}</span>
              <span className="w-1 h-3 bg-[#dae2fd] rounded-full" />
              <span className="font-mono text-xs text-[#006948] font-bold flex items-center gap-1">
                <span className="material-symbols-outlined text-[14px]">pin_drop</span>
                {fmtCoord(current.location?.latitude, 'N', 'S')}, {fmtCoord(current.location?.longitude, 'E', 'W')}
              </span>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <div className="bg-white p-3.5 rounded-xl border border-[#dae2fd] flex flex-col gap-0.5">
            <span className="text-[11px] text-[#3d4a42]">Resolution</span>
            <span className="text-xs font-bold text-[#131b2e]">{resolutionText}</span>
          </div>
          <div className="bg-white p-3.5 rounded-xl border border-[#dae2fd] flex flex-col gap-0.5">
            <span className="text-[11px] text-[#3d4a42]">GPS Accuracy</span>
            <span className="text-xs font-bold text-[#131b2e]">
              {current.location?.accuracy_m != null ? `±${current.location.accuracy_m.toFixed(1)} m` : '—'}
            </span>
          </div>
          <div className="bg-white p-3.5 rounded-xl border border-[#dae2fd] flex flex-col gap-0.5">
            <span className="text-[11px] text-[#3d4a42]">Captured</span>
            <span className="text-xs font-bold text-[#006948] truncate">
              {current.location?.captured_at ? new Date(current.location.captured_at).toLocaleString() : '—'}
            </span>
          </div>
        </div>
      </div>

      <div className="lg:col-span-5 flex flex-col gap-6">
        <div className={`bg-white p-7 rounded-2xl shadow-sm border ${style?.cardBorder ?? 'border-[#dae2fd]'} flex flex-col gap-6`}>
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono font-bold uppercase tracking-wider text-[#006948]">
              Tree Classification Result
            </span>
            <PlaceholderChip modelInfo={modelInfo} />
          </div>

          <div className={`flex items-center gap-3 p-3.5 rounded-xl ${style?.bannerClass ?? 'bg-[#eaedff] border border-[#dae2fd] text-[#131b2e]'}`}>
            <span className="material-symbols-outlined text-[26px]">{style?.icon ?? 'help'}</span>
            <span className="text-base font-extrabold tracking-tight">{style?.title ?? 'Result unavailable'}</span>
          </div>

          {verdict === 'uncertain' && details && (
            <p className="text-xs text-[#5c3c00] leading-relaxed -mt-3">
              Confidence was below the {Math.round(details.threshold * 100)}% threshold needed for a confident verdict.
            </p>
          )}

          <div className="flex items-center justify-between">
            <div className="flex flex-col gap-0.5">
              <span className="text-xs text-[#3d4a42] uppercase font-mono font-bold tracking-wider">Confidence</span>
              <span className="text-3xl font-extrabold text-[#131b2e]">
                {confidencePct != null ? `${confidencePct.toFixed(1)}%` : '—'}
              </span>
              <span className="text-xs text-[#3d4a42]">
                {prediction?.display_label ?? 'No prediction'}
                {prediction?.is_uncertain ? ' • uncertain' : ''}
              </span>
            </div>
            <div className="relative w-20 h-20 flex items-center justify-center">
              <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
                <circle cx="50" cy="50" r={CIRCLE_R} fill="transparent" stroke="#eaedff" strokeWidth="8" />
                <circle
                  cx="50"
                  cy="50"
                  r={CIRCLE_R}
                  fill="transparent"
                  stroke="#006948"
                  strokeWidth="8"
                  strokeDasharray={CIRCUMFERENCE}
                  strokeDashoffset={dashOffset}
                  strokeLinecap="round"
                />
              </svg>
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <span className="text-sm font-extrabold text-[#131b2e]">
                  {confidencePct != null ? `${confidencePct.toFixed(0)}%` : '—'}
                </span>
                <span className="font-mono text-[8px] uppercase text-[#006948] font-bold">MATCH</span>
              </div>
            </div>
          </div>

          {details && details.probabilities.length > 0 && (
            <div className="flex flex-col gap-2">
              {details.probabilities.map((p) => (
                <div key={p.class_key} className="flex flex-col gap-1">
                  <div className="flex items-center justify-between text-xs font-mono">
                    <span className="text-[#3d4a42]">{p.display_name}</span>
                    <span className="font-bold text-[#131b2e]">{(p.probability * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full h-2 rounded-full bg-[#eaedff] overflow-hidden">
                    <div
                      className="h-2 rounded-full bg-[#006948]"
                      style={{ width: `${Math.min(100, Math.max(0, p.probability * 100))}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          )}

          <div className="p-4 rounded-xl bg-[#f2f3ff] border border-[#dae2fd] flex flex-col gap-2 font-mono text-xs">
            <span className="text-[10px] uppercase font-bold text-[#3d4a42] tracking-wider">Model &amp; Timing</span>
            <div className="flex justify-between py-1 border-b border-[#dae2fd]">
              <span className="text-[#3d4a42]">Model version:</span>
              <span className="font-bold text-[#131b2e]">{current.model_version}</span>
            </div>
            <div className="flex justify-between py-1">
              <span className="text-[#3d4a42]">Total time:</span>
              <span className="font-bold text-[#006948]">{fmtMs(current.timings_ms.total)}</span>
            </div>
          </div>

          <RecordMeta analysis={current} />
        </div>
      </div>
    </div>
  );
}

// --- Model 2: leaf segmentation -------------------------------------------------------------------

const LABEL_STYLE: Record<LeafSegLabel, { icon: string; bannerClass: string; cardBorder: string; title: string; description: string }> = {
  affected: {
    icon: 'warning',
    bannerClass: 'bg-[#ffdad6] border border-[#ba1a1a]/30 text-[#ba1a1a]',
    cardBorder: 'border-[#ffdad6]',
    title: 'Affected Tissue Detected',
    description: 'Damaged or diseased tissue was found on this leaf.',
  },
  healthy: {
    icon: 'check_circle',
    bannerClass: 'bg-[#85f8c4]/30 border border-[#85f8c4] text-[#002114]',
    cardBorder: 'border-[#dae2fd]',
    title: 'Healthy Leaf',
    description: 'No significant tissue damage was found on this leaf.',
  },
  no_leaf: {
    icon: 'help',
    bannerClass: 'bg-[#ffeed2] border border-[#ffb95f] text-[#5c3c00]',
    cardBorder: 'border-[#ffb95f]/60',
    title: 'No Banana Leaf Detected',
    description: 'No banana leaf was detected in this photo.',
  },
};

const SPLIT_MIN = 5;
const SPLIT_MAX = 95;
const SPLIT_KEY_STEP = 5;

function LeafResultBody({ analysis, modelInfo }: { analysis: AnalysisDetail; modelInfo: ModelInfo | undefined }) {
  const { current, failed, onError } = useImageRetry(analysis);
  const details = isLeafSegDetails(current.details) ? current.details : null;
  const rawLabel = current.prediction?.label ?? null;
  const label: LeafSegLabel | null =
    rawLabel === 'affected' || rawLabel === 'healthy' || rawLabel === 'no_leaf' ? rawLabel : null;
  const style = label ? LABEL_STYLE[label] : null;
  const hasLeafTissue = label !== null && label !== 'no_leaf';
  const healthyPct = details && hasLeafTissue ? Math.max(0, 100 - details.affected_area_pct_of_leaf) : null;
  const affectedPct = details && hasLeafTissue ? details.affected_area_pct_of_leaf : null;
  const originalUrl = absoluteImageUrl(current.image.original_url);
  const resultUrl = absoluteImageUrl(current.image.result_url);

  const [viewMode, setViewMode] = useState<'overlay' | 'original'>('overlay');
  const [isSliderActive, setIsSliderActive] = useState(false);
  const [splitPosition, setSplitPosition] = useState(50);
  const sliderContainerRef = useRef<HTMLDivElement | null>(null);
  const isDraggingRef = useRef(false);

  const updateSplitFromClientX = useCallback((clientX: number) => {
    const el = sliderContainerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const pct = Math.max(SPLIT_MIN, Math.min(SPLIT_MAX, ((clientX - rect.left) / rect.width) * 100));
    setSplitPosition(pct);
  }, []);
  const handleSliderPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    isDraggingRef.current = true;
    e.currentTarget.setPointerCapture(e.pointerId);
    updateSplitFromClientX(e.clientX);
  };
  const handleSliderPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!isDraggingRef.current) return;
    updateSplitFromClientX(e.clientX);
  };
  const handleSliderPointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
    isDraggingRef.current = false;
    if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId);
  };
  const handleSliderKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.key === 'ArrowLeft') {
      setSplitPosition((p) => Math.max(SPLIT_MIN, p - SPLIT_KEY_STEP));
      e.preventDefault();
    } else if (e.key === 'ArrowRight') {
      setSplitPosition((p) => Math.min(SPLIT_MAX, p + SPLIT_KEY_STEP));
      e.preventDefault();
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
      <div className="lg:col-span-7 flex flex-col gap-4">
        <div className="flex flex-wrap items-center justify-between gap-3 bg-white p-3.5 rounded-xl border border-[#dae2fd]">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-xs font-mono font-bold text-[#131b2e] uppercase">View:</span>
            <div className="inline-flex p-1 rounded-lg bg-[#f2f3ff] border border-[#dae2fd] text-xs font-mono">
              <button
                type="button"
                onClick={() => {
                  setViewMode('overlay');
                  setIsSliderActive(false);
                }}
                disabled={!resultUrl}
                className={`px-2.5 py-1 rounded transition-colors cursor-pointer disabled:cursor-not-allowed disabled:opacity-40 ${
                  viewMode === 'overlay' && !isSliderActive ? 'bg-[#006948] text-white font-bold shadow-sm' : 'text-[#3d4a42] hover:text-[#131b2e]'
                }`}
              >
                Overlay
              </button>
              <button
                type="button"
                onClick={() => {
                  setViewMode('original');
                  setIsSliderActive(false);
                }}
                className={`px-2.5 py-1 rounded transition-colors cursor-pointer ${
                  viewMode === 'original' && !isSliderActive ? 'bg-[#131b2e] text-white font-bold shadow-sm' : 'text-[#3d4a42] hover:text-[#131b2e]'
                }`}
              >
                Original
              </button>
            </div>
          </div>
          <button
            type="button"
            onClick={() => setIsSliderActive((v) => !v)}
            disabled={!resultUrl}
            className={`px-3 py-1.5 rounded-lg text-xs font-mono font-semibold flex items-center gap-1.5 border transition-all cursor-pointer disabled:cursor-not-allowed disabled:opacity-40 ${
              isSliderActive ? 'bg-[#00855d] text-white border-[#00855d]' : 'bg-[#eaedff] text-[#131b2e] border-[#dae2fd] hover:bg-[#dae2fd]'
            }`}
            title="Drag (or use the arrow keys) to compare the overlay against the original photo"
          >
            <span className="material-symbols-outlined text-[16px]">compare</span>
            <span>Split Compare</span>
          </button>
        </div>

        <div
          ref={sliderContainerRef}
          className="relative w-full aspect-[4/3] rounded-2xl overflow-hidden bg-[#131b2e] shadow-xl border border-[#dae2fd] select-none touch-none"
        >
          {!failed ? (
            <>
              <img alt="Original leaf photo" className="absolute inset-0 w-full h-full object-cover" src={originalUrl ?? undefined} onError={onError} />
              {resultUrl && (
                <img
                  alt="Leaf segmentation overlay — green outlines the leaf, red marks affected tissue"
                  className="absolute inset-0 w-full h-full object-cover"
                  src={resultUrl}
                  onError={onError}
                  style={
                    isSliderActive
                      ? { clipPath: `inset(0 ${100 - splitPosition}% 0 0)` }
                      : viewMode === 'overlay'
                        ? undefined
                        : { display: 'none' }
                  }
                />
              )}
              {isSliderActive && (
                <div
                  role="slider"
                  tabIndex={0}
                  aria-label="Comparison slider: overlay versus original photo"
                  aria-orientation="horizontal"
                  aria-valuemin={SPLIT_MIN}
                  aria-valuemax={SPLIT_MAX}
                  aria-valuenow={Math.round(splitPosition)}
                  onPointerDown={handleSliderPointerDown}
                  onPointerMove={handleSliderPointerMove}
                  onPointerUp={handleSliderPointerUp}
                  onPointerCancel={handleSliderPointerUp}
                  onKeyDown={handleSliderKeyDown}
                  style={{ left: `${splitPosition}%` }}
                  className="absolute top-0 bottom-0 w-1 bg-white shadow-[0_0_14px_rgba(0,0,0,0.8)] cursor-ew-resize flex items-center justify-center z-30 touch-none focus:outline-none focus:ring-2 focus:ring-[#85f8c4]"
                >
                  <div className="w-8 h-8 rounded-full bg-white text-[#131b2e] flex items-center justify-center shadow-lg border border-[#bccac0]">
                    <span className="material-symbols-outlined text-[18px]">drag_indicator</span>
                  </div>
                </div>
              )}
              {!resultUrl && (
                <div className="absolute bottom-4 left-4 right-4 bg-white/90 backdrop-blur-md px-4 py-2 rounded-xl text-xs text-[#3d4a42] text-center">
                  No overlay image is available for this result.
                </div>
              )}
            </>
          ) : (
            <div className="w-full h-full flex flex-col items-center justify-center gap-2 text-white/70">
              <span className="material-symbols-outlined text-[36px]">broken_image</span>
              <span className="text-sm">Image unavailable</span>
            </div>
          )}
        </div>

        {hasLeafTissue ? (
          <div className="bg-white p-5 rounded-2xl border border-[#dae2fd] shadow-sm flex flex-col gap-3 font-mono">
            <span className="text-xs font-bold uppercase tracking-wider text-[#131b2e]">Leaf Tissue Breakdown</span>
            <div className="w-full h-4 rounded-full overflow-hidden flex bg-[#dae2fd]">
              <div
                style={{ width: `${healthyPct ?? 0}%` }}
                className="bg-[#006948] h-full transition-all duration-500 flex items-center justify-center text-[10px] font-bold text-white"
                title="Healthy"
              >
                {(healthyPct ?? 0) > 10 ? fmtPct(healthyPct, 0) : ''}
              </div>
              <div
                style={{ width: `${affectedPct ?? 0}%` }}
                className="bg-[#ba1a1a] h-full transition-all duration-500 flex items-center justify-center text-[10px] font-bold text-white"
                title="Affected"
              >
                {(affectedPct ?? 0) > 10 ? fmtPct(affectedPct, 0) : ''}
              </div>
            </div>
            <div className="grid grid-cols-2 gap-3 pt-1">
              <div className="p-3 rounded-xl bg-[#85f8c4]/20 border border-[#85f8c4] flex flex-col gap-1">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-[#006948] flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-[#006948]" />
                    Healthy
                  </span>
                  <span className="text-sm font-extrabold text-[#006948]">{fmtPct(healthyPct)}</span>
                </div>
              </div>
              <div className="p-3 rounded-xl bg-[#ffdad6]/40 border border-[#ba1a1a]/30 flex flex-col gap-1">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-[#ba1a1a] flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-[#ba1a1a]" />
                    Affected
                  </span>
                  <span className="text-sm font-extrabold text-[#ba1a1a]">{fmtPct(affectedPct)}</span>
                </div>
              </div>
            </div>
          </div>
        ) : (
          label === 'no_leaf' && (
            <div className="bg-white p-5 rounded-2xl border border-[#ffb95f]/60 shadow-sm text-sm text-[#5c3c00]">
              No tissue breakdown is shown because the model could not find enough leaf area in this photo.
            </div>
          )
        )}
      </div>

      <div className="lg:col-span-5 flex flex-col gap-6">
        <div className={`bg-white p-7 rounded-2xl shadow-sm border ${style?.cardBorder ?? 'border-[#dae2fd]'} flex flex-col gap-6`}>
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono font-bold uppercase tracking-wider text-[#006948]">
              Leaf Segmentation Result
            </span>
            <PlaceholderChip modelInfo={modelInfo} />
          </div>

          <div className={`flex items-center gap-3 p-3.5 rounded-xl ${style?.bannerClass ?? 'bg-[#eaedff] border border-[#dae2fd] text-[#131b2e]'}`}>
            <span className="material-symbols-outlined text-[26px]">{style?.icon ?? 'help'}</span>
            <span className="text-base font-extrabold tracking-tight">{style?.title ?? 'Result unavailable'}</span>
          </div>
          <p className="text-xs text-[#3d4a42] leading-relaxed -mt-3">
            {style?.description ?? 'This analysis did not return a recognised result.'}
          </p>

          <div className="p-4 rounded-xl bg-[#f2f3ff] border border-[#dae2fd] flex flex-col gap-2 font-mono text-xs">
            <span className="text-[10px] uppercase font-bold text-[#3d4a42] tracking-wider">Measurements</span>
            <div className="flex justify-between py-1 border-b border-[#dae2fd]">
              <span className="text-[#3d4a42]">Leaf coverage of photo:</span>
              <span className="font-bold text-[#131b2e]">{fmtPct(details?.leaf_area_pct_of_image)}</span>
            </div>
            <div className="flex justify-between py-1 border-b border-[#dae2fd]">
              <span className="text-[#3d4a42]">Lesion count:</span>
              <span className="font-bold text-[#131b2e]">{details?.lesion_count ?? '—'}</span>
            </div>
            <div className="flex justify-between py-1 border-b border-[#dae2fd]">
              <span className="text-[#3d4a42]">Largest lesion (% of leaf):</span>
              <span className="font-bold text-[#131b2e]">{fmtPct(details?.largest_lesion_pct_of_leaf)}</span>
            </div>
            <div className="flex justify-between py-1 border-b border-[#dae2fd]">
              <span className="text-[#3d4a42]">Mean leaf probability:</span>
              <span className="font-bold text-[#131b2e]">{fmtProbability(details?.mean_leaf_probability)}</span>
            </div>
            <div className="flex justify-between py-1">
              <span className="text-[#3d4a42]">Mean affected probability:</span>
              <span className="font-bold text-[#131b2e]">{fmtProbability(details?.mean_affected_probability)}</span>
            </div>
          </div>

          {details && (
            <details className="rounded-xl bg-[#faf8ff] border border-[#dae2fd] p-4 text-xs font-mono">
              <summary className="cursor-pointer font-bold text-[#131b2e] uppercase tracking-wider text-[10px]">
                Thresholds used
              </summary>
              <div className="grid grid-cols-2 gap-x-2 gap-y-1.5 mt-3 text-[#3d4a42]">
                <span>Leaf pixel threshold:</span>
                <span className="text-right font-bold text-[#131b2e]">{fmtProbability(details.thresholds.leaf)}</span>
                <span>Affected pixel threshold:</span>
                <span className="text-right font-bold text-[#131b2e]">{fmtProbability(details.thresholds.affected)}</span>
                <span>Min. leaf % (else no_leaf):</span>
                <span className="text-right font-bold text-[#131b2e]">{fmtPct(details.thresholds.min_leaf_pct)}</span>
                <span>Min. affected % (else healthy):</span>
                <span className="text-right font-bold text-[#131b2e]">{fmtPct(details.thresholds.min_affected_pct)}</span>
                <span>Min. lesion size %:</span>
                <span className="text-right font-bold text-[#131b2e]">{fmtPct(details.thresholds.min_lesion_pct)}</span>
                <span>Horizontal-flip TTA:</span>
                <span className="text-right font-bold text-[#131b2e]">{details.thresholds.tta_hflip ? 'Yes' : 'No'}</span>
              </div>
            </details>
          )}

          <div className="p-4 rounded-xl bg-[#f2f3ff] border border-[#dae2fd] flex flex-col gap-2 font-mono text-xs">
            <span className="text-[10px] uppercase font-bold text-[#3d4a42] tracking-wider">Model &amp; Timing</span>
            <div className="flex justify-between py-1 border-b border-[#dae2fd]">
              <span className="text-[#3d4a42]">Model version:</span>
              <span className="font-bold text-[#131b2e]">{current.model_version}</span>
            </div>
            <div className="flex justify-between py-1">
              <span className="text-[#3d4a42]">Total time:</span>
              <span className="font-bold text-[#006948]">{fmtMs(current.timings_ms.total)}</span>
            </div>
          </div>

          <RecordMeta analysis={current} />
        </div>
      </div>
    </div>
  );
}

// --- Entry point -----------------------------------------------------------------------------

export const AnalysisDetailView: React.FC<AnalysisDetailViewProps> = ({ onNavigate }) => {
  const { viewing } = useAnalysisState();
  const { byKey } = useModels();

  if (!viewing) {
    return (
      <div className="w-full max-w-3xl mx-auto px-6 py-16 flex flex-col items-center gap-4 text-center">
        <span className="material-symbols-outlined text-[40px] text-[#3d4a42]">history</span>
        <h1 className="text-2xl font-extrabold text-[#131b2e]">No record selected</h1>
        <p className="text-sm text-[#3d4a42]">Open Analysis History and click a row to view its full result here.</p>
        <button
          type="button"
          onClick={() => onNavigate('history')}
          className="mt-2 px-6 py-3 rounded-xl bg-[#006948] text-white hover:bg-[#00855d] font-semibold text-sm transition-all cursor-pointer"
        >
          Go to Analysis History
        </button>
      </div>
    );
  }

  return (
    <div className="w-full max-w-7xl mx-auto px-6 lg:px-12 py-10 flex flex-col gap-8">
      <Breadcrumb onNavigate={onNavigate} />

      {/* key={viewing.id} remounts the body (and its local state: image retries, the split
          slider) when a different record is opened without a full navigation away and back. */}
      {viewing.model_key === 'leaf_segmentation' ? (
        <LeafResultBody key={viewing.id} analysis={viewing} modelInfo={byKey.leaf_segmentation} />
      ) : viewing.model_key === 'tree_classification' ? (
        <TreeResultBody key={viewing.id} analysis={viewing} modelInfo={byKey.tree_classification} />
      ) : (
        <div className="p-6 rounded-2xl bg-[#ffdad6] border border-[#ba1a1a]/30 text-[#ba1a1a] text-sm">
          This record's model ({viewing.model_key}) has no detail view yet.
        </div>
      )}

      <Footer onNavigate={onNavigate} />
    </div>
  );
};
