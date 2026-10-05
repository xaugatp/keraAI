import React, { useCallback, useEffect, useState } from 'react';
import { ViewTab } from '../types';
import { useAnalysisState } from '../state/AnalysisContext';
import { useModels } from '../hooks/useModels';
import { absoluteImageUrl, getAnalysis, isTreeDetails } from '../api';
import type { TreeVerdict } from '../api';

interface Stage1ResultViewProps {
  onNavigate: (tab: ViewTab) => void;
}

const VERDICT_STYLE: Record<
  TreeVerdict,
  { icon: string; bannerClass: string; cardBorder: string; title: string }
> = {
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

function fmtMs(ms: number | null | undefined): string {
  return ms == null ? '—' : `${Math.round(ms)} ms`;
}

function fmtCoord(value: number | null | undefined, positive: string, negative: string): string {
  if (value == null) return '—';
  return `${Math.abs(value).toFixed(6)}° ${value >= 0 ? positive : negative}`;
}

export const Stage1ResultView: React.FC<Stage1ResultViewProps> = ({ onNavigate }) => {
  const { tree, file, setTree, reset } = useAnalysisState();
  const { byKey } = useModels();

  const [imageFailed, setImageFailed] = useState(false);
  const [imageRetried, setImageRetried] = useState(false);
  const [reloadingImage, setReloadingImage] = useState(false);

  // A new analysis arrived (or the view mounted with one already): give the image a clean slate.
  useEffect(() => {
    setImageFailed(false);
    setImageRetried(false);
  }, [tree?.id]);

  const handleImageError = useCallback(async () => {
    if (!tree || imageRetried || reloadingImage) {
      setImageFailed(true);
      return;
    }
    setReloadingImage(true);
    try {
      // Signed image URLs expire after ~1h; re-fetching the analysis gets fresh ones (urls.ts).
      const fresh = await getAnalysis(tree.id);
      setTree(fresh);
      setImageRetried(true);
    } catch {
      setImageFailed(true);
    } finally {
      setReloadingImage(false);
    }
  }, [tree, imageRetried, reloadingImage, setTree]);

  const startOver = useCallback(
    (tab: ViewTab) => {
      reset();
      onNavigate(tab);
    },
    [reset, onNavigate],
  );

  if (!tree) {
    return (
      <div className="w-full max-w-3xl mx-auto px-6 py-16 flex flex-col items-center gap-4 text-center">
        <span className="material-symbols-outlined text-[40px] text-[#3d4a42]">image_search</span>
        <h1 className="text-2xl font-extrabold text-[#131b2e]">No analysis selected</h1>
        <p className="text-sm text-[#3d4a42]">
          Upload a photo, take one with the camera, or open a sample from Plant Detection to see a result here.
        </p>
        <button
          type="button"
          onClick={() => onNavigate('detect')}
          className="mt-2 px-6 py-3 rounded-xl bg-[#006948] text-white hover:bg-[#00855d] font-semibold text-sm transition-all cursor-pointer"
        >
          Go to Plant Detection
        </button>
      </div>
    );
  }

  const details = isTreeDetails(tree.details) ? tree.details : null;
  const prediction = tree.prediction;
  const modelInfo = byKey.tree_classification;
  const verdict: TreeVerdict | null = details?.verdict ?? null;
  const style = verdict ? VERDICT_STYLE[verdict] : null;
  const confidencePct = prediction?.confidence != null ? prediction.confidence * 100 : null;

  const CIRCLE_R = 42;
  const CIRCUMFERENCE = 2 * Math.PI * CIRCLE_R;
  const dashOffset = confidencePct != null ? CIRCUMFERENCE * (1 - confidencePct / 100) : CIRCUMFERENCE;

  const canAnalyseLeaf = file !== null;
  const resolutionText =
    tree.image.width != null && tree.image.height != null ? `${tree.image.width} × ${tree.image.height} px` : '—';

  return (
    <div className="w-full max-w-7xl mx-auto px-6 lg:px-12 py-10 flex flex-col gap-8">
      {/* Breadcrumb */}
      <div className="flex items-center gap-2 text-xs font-mono uppercase text-[#3d4a42] font-semibold">
        <button onClick={() => onNavigate('home')} className="hover:text-[#006948] transition-colors cursor-pointer">
          Home
        </button>
        <span className="material-symbols-outlined text-[14px]">chevron_right</span>
        <button onClick={() => onNavigate('detect')} className="hover:text-[#006948] transition-colors cursor-pointer">
          Plant Detection
        </button>
        <span className="material-symbols-outlined text-[14px]">chevron_right</span>
        <span className="text-[#006948] font-bold">Detection Result</span>
      </div>

      {/* Main Two-Column Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Image (7 cols) */}
        <div className="lg:col-span-7 flex flex-col gap-4">
          <div className="relative w-full aspect-[4/3] rounded-2xl overflow-hidden bg-[#283044] shadow-md border border-[#dae2fd]">
            {!imageFailed ? (
              <img
                alt="Analyzed plant specimen"
                className="w-full h-full object-cover"
                src={absoluteImageUrl(tree.image.original_url)}
                onError={() => void handleImageError()}
              />
            ) : (
              <div className="w-full h-full flex flex-col items-center justify-center gap-2 text-white/70">
                <span className="material-symbols-outlined text-[36px]">broken_image</span>
                <span className="text-sm">Image unavailable</span>
              </div>
            )}

            {/* Bottom Floating Bar */}
            <div className="absolute bottom-4 left-4 right-4 bg-white/90 backdrop-blur-md p-3 rounded-xl flex items-center justify-between text-[#131b2e] shadow-md border border-white/60">
              <div className="flex items-center gap-3">
                <span className="material-symbols-outlined text-[18px] text-[#006948]">photo_camera</span>
                <span className="font-mono text-xs font-semibold">{resolutionText}</span>
                <span className="w-1 h-3 bg-[#dae2fd] rounded-full" />
                <span className="font-mono text-xs text-[#006948] font-bold flex items-center gap-1">
                  <span className="material-symbols-outlined text-[14px]">pin_drop</span>
                  {fmtCoord(tree.location?.latitude, 'N', 'S')}, {fmtCoord(tree.location?.longitude, 'E', 'W')}
                </span>
              </div>
            </div>
          </div>

          {/* Image / Capture Metrics */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div className="bg-white p-3.5 rounded-xl border border-[#dae2fd] flex flex-col gap-0.5">
              <span className="text-[11px] text-[#3d4a42]">Resolution</span>
              <span className="text-xs font-bold text-[#131b2e]">{resolutionText}</span>
            </div>
            <div className="bg-white p-3.5 rounded-xl border border-[#dae2fd] flex flex-col gap-0.5">
              <span className="text-[11px] text-[#3d4a42]">GPS Accuracy</span>
              <span className="text-xs font-bold text-[#131b2e]">
                {tree.location?.accuracy_m != null ? `±${tree.location.accuracy_m.toFixed(1)} m` : '—'}
              </span>
            </div>
            <div className="bg-white p-3.5 rounded-xl border border-[#dae2fd] flex flex-col gap-0.5">
              <span className="text-[11px] text-[#3d4a42]">Captured</span>
              <span className="text-xs font-bold text-[#006948] truncate">
                {tree.location?.captured_at ? new Date(tree.location.captured_at).toLocaleString() : '—'}
              </span>
            </div>
          </div>
        </div>

        {/* Right Column: Prediction Results (5 cols) */}
        <div className="lg:col-span-5 flex flex-col gap-6">
          <div className={`bg-white p-7 rounded-2xl shadow-sm border ${style?.cardBorder ?? 'border-[#dae2fd]'} flex flex-col gap-6`}>
            {/* Verdict Banner */}
            <div className={`flex items-center gap-3 p-3.5 rounded-xl ${style?.bannerClass ?? 'bg-[#eaedff] border border-[#dae2fd] text-[#131b2e]'}`}>
              <span className="material-symbols-outlined text-[26px]">{style?.icon ?? 'help'}</span>
              <span className="text-base font-extrabold tracking-tight">{style?.title ?? 'Result unavailable'}</span>
            </div>

            {verdict === 'uncertain' && details && (
              <p className="text-xs text-[#5c3c00] leading-relaxed -mt-3">
                Confidence was below the {Math.round(details.threshold * 100)}% threshold needed for a confident
                verdict. Try a clearer or closer photo of the plant.
              </p>
            )}
            {verdict === 'not_banana_tree' && (
              <p className="text-xs text-[#3d4a42] leading-relaxed">
                The model did not classify this photo as a banana tree.
              </p>
            )}

            {/* Confidence Meter */}
            <div className="flex items-center justify-between">
              <div className="flex flex-col gap-0.5">
                <span className="text-xs text-[#3d4a42] uppercase font-mono font-bold tracking-wider">
                  Model 1 Confidence
                </span>
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

            {/* Real 2-class probability bars */}
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

            {/* Technical Information Panel */}
            <div className="p-4 rounded-xl bg-[#f2f3ff] border border-[#dae2fd] flex flex-col gap-2 font-mono text-xs">
              <span className="text-[10px] uppercase font-bold text-[#3d4a42] tracking-wider">
                Model &amp; Timing
              </span>
              <div className="flex justify-between py-1 border-b border-[#dae2fd]">
                <span className="text-[#3d4a42]">Model version:</span>
                <span className="font-bold text-[#131b2e] flex items-center gap-1.5">
                  {tree.model_version}
                  {modelInfo?.is_placeholder && (
                    <span
                      title="Results come from placeholder weights and are not real predictions"
                      className="px-1.5 py-0.5 rounded bg-[#ffeed2] border border-[#ffb95f] text-[#825100] text-[10px] font-bold"
                    >
                      Demo weights
                    </span>
                  )}
                </span>
              </div>
              <div className="flex justify-between py-1 border-b border-[#dae2fd]">
                <span className="text-[#3d4a42]">Analyzed:</span>
                <span className="font-bold text-[#131b2e]">{new Date(tree.created_at).toLocaleString()}</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-[#3d4a42]">Total time:</span>
                <span className="font-bold text-[#006948]">{fmtMs(tree.timings_ms.total)}</span>
              </div>
            </div>

            {/* Next Stage Actions */}
            <div className="flex flex-col gap-3 pt-2">
              <div className="flex items-center justify-between">
                <span className="text-xs uppercase font-extrabold tracking-wider text-[#131b2e] font-mono">
                  Next Stage in Pipeline:
                </span>
              </div>

              {verdict !== 'not_banana_tree' ? (
                <>
                  <div className="p-4 rounded-xl bg-gradient-to-r from-[#eef3ff] to-[#f9faff] border-2 border-[#006948] hover:border-[#00855d] transition-all flex flex-col gap-2 shadow-sm">
                    <div className="flex items-center justify-between">
                      <span className="inline-flex items-center gap-1.5 text-xs font-extrabold text-[#006948] font-mono">
                        STAGE 02 — DETECT DISEASE &amp; AFFECTED AREA
                      </span>
                      <span className="px-2 py-0.5 rounded bg-[#ffdad6] text-[#ba1a1a] font-mono text-[10px] font-bold">
                        Model 3
                      </span>
                    </div>
                    <p className="text-xs text-[#3d4a42] leading-tight">
                      Classify disease and localize affected lesion boundaries with treatment recommendations.
                    </p>
                    <button
                      type="button"
                      onClick={() => onNavigate('detect-disease')}
                      className="w-full mt-1 py-3 px-4 rounded-xl bg-[#006948] hover:bg-[#00855d] text-white font-bold text-xs transition-all flex items-center justify-center gap-2 shadow-md shadow-[#006948]/20 cursor-pointer active:scale-95"
                    >
                      <span>Proceed to Detect Disease</span>
                      <span className="material-symbols-outlined text-[18px] font-bold">arrow_forward</span>
                    </button>
                  </div>

                  <div className="p-4 rounded-xl bg-gradient-to-r from-[#f0fbf7] to-[#f9fdfb] border-2 border-[#006a61] hover:border-[#00855d] transition-all flex flex-col gap-2 shadow-sm">
                    <div className="flex items-center justify-between">
                      <span className="inline-flex items-center gap-1.5 text-xs font-extrabold text-[#006a61] font-mono">
                        STAGE 03 — SEGMENT AFFECTED AREA
                      </span>
                      <span className="px-2 py-0.5 rounded bg-[#85f8c4]/40 text-[#006948] font-mono text-[10px] font-bold">
                        Model 2
                      </span>
                    </div>
                    <p className="text-xs text-[#3d4a42] leading-tight">
                      Pixel-level segmentation of healthy vs. affected leaf tissue.
                    </p>
                    <button
                      type="button"
                      onClick={() => canAnalyseLeaf && onNavigate('leaf-analysis')}
                      disabled={!canAnalyseLeaf}
                      title={
                        canAnalyseLeaf
                          ? undefined
                          : 'This result came from a sample photo — there is no original file to re-analyse. Upload or capture your own photo to use the leaf model.'
                      }
                      className="w-full mt-1 py-3 px-4 rounded-xl bg-[#006a61] hover:bg-[#00855d] disabled:bg-[#bccac0] disabled:cursor-not-allowed text-white font-bold text-xs transition-all flex items-center justify-center gap-2 shadow-md shadow-[#006a61]/20 cursor-pointer active:scale-95"
                    >
                      <span>Analyse the Leaf</span>
                      <span className="material-symbols-outlined text-[18px] font-bold">arrow_forward</span>
                    </button>
                  </div>

                  <button
                    type="button"
                    onClick={() => startOver('detect')}
                    className="py-2.5 px-4 rounded-xl bg-[#eaedff] text-[#131b2e] hover:bg-[#dae2fd] font-semibold text-xs transition-all flex items-center justify-center gap-1.5 border border-[#dae2fd] cursor-pointer mt-1"
                  >
                    <span className="material-symbols-outlined text-[16px]">refresh</span>
                    <span>Analyze Another Banana Plant</span>
                  </button>
                </>
              ) : (
                <button
                  type="button"
                  onClick={() => startOver('detect')}
                  className="w-full py-3.5 px-6 rounded-xl bg-[#006948] text-white hover:bg-[#00855d] font-semibold text-sm transition-all flex items-center justify-center gap-2 shadow-md cursor-pointer active:scale-95"
                >
                  <span className="material-symbols-outlined text-[18px]">photo_camera</span>
                  <span>Capture Another Plant Specimen</span>
                  <span className="material-symbols-outlined text-[18px]">arrow_forward</span>
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
