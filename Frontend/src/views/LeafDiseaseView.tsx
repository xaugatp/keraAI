import React, { useCallback, useEffect, useRef, useState } from 'react';
import { ViewTab } from '../types';
import { ASSETS } from '../data/mockData';
import { CameraModal } from '../components/CameraModal';
import { ProcessingModal } from '../components/ProcessingModal';
import type { ProcessingPhase } from '../components/ProcessingModal';
import { useAnalysisState } from '../state/AnalysisContext';
import { useModels } from '../hooks/useModels';
import { useSamples } from '../hooks/useSamples';
import {
  ACCEPTED_IMAGE_TYPES,
  AbortedError,
  ApiError,
  NetworkError,
  absoluteImageUrl,
  getAnalysis,
  isLeafDiseaseDetails,
  predict,
  resolveCapturedAt,
  shouldAutoSubmitHandoff,
  validateImageFile,
  withMinDuration,
} from '../api';
import type { AnalysisSummary, LeafDiseaseLabel, UploadSource } from '../api';

// Samples are a single, near-instant DB read (no inference happens) — without a floor, the
// ProcessingModal would flash and vanish, which reads as broken rather than fast.
const SAMPLE_OPEN_MIN_MS = 900;

interface LeafDiseaseViewProps {
  onNavigate: (tab: ViewTab) => void;
}

/** What the hand-off (and an independent capture) carries forward as the photo's location. */
interface LocationInput {
  latitude: number | null;
  longitude: number | null;
  accuracyM: number | null;
  capturedAt: Date | null;
}

const LABEL_STYLE: Record<
  LeafDiseaseLabel,
  { icon: string; bannerClass: string; cardBorder: string; title: string; description: string }
> = {
  black_sigatoka: {
    icon: 'warning',
    bannerClass: 'bg-[#ffdad6] border border-[#ba1a1a]/30 text-[#ba1a1a]',
    cardBorder: 'border-[#ffdad6]',
    title: 'Black Sigatoka Detected',
    description: 'This leaf shows signs of Black Sigatoka.',
  },
  yellow_sigatoka: {
    icon: 'warning',
    bannerClass: 'bg-[#ffeed2] border border-[#ffb95f]/60 text-[#825100]',
    cardBorder: 'border-[#ffeed2]',
    title: 'Yellow Sigatoka Detected',
    description: 'This leaf shows signs of Yellow Sigatoka.',
  },
  healthy: {
    icon: 'check_circle',
    bannerClass: 'bg-[#85f8c4]/30 border border-[#85f8c4] text-[#002114]',
    cardBorder: 'border-[#dae2fd]',
    title: 'Healthy Leaf',
    description: 'No disease above the configured threshold was found on this leaf.',
  },
  no_leaf: {
    icon: 'help',
    bannerClass: 'bg-[#ffeed2] border border-[#ffb95f] text-[#5c3c00]',
    cardBorder: 'border-[#ffb95f]/60',
    title: 'No Banana Leaf Detected',
    description: 'No banana leaf detected in this photo. Try a clearer, closer photo of a single leaf.',
  },
};

const DISEASE_CHANNEL_LABEL: Record<string, string> = {
  black_sigatoka: 'Black Sigatoka',
  yellow_sigatoka: 'Yellow Sigatoka',
};

function fmtPct(value: number | null | undefined, decimals = 1): string {
  return value == null ? '—' : `${value.toFixed(decimals)}%`;
}

/** `value` is a 0-1 probability. */
function fmtProbability(value: number | null | undefined): string {
  return value == null ? '—' : `${(value * 100).toFixed(1)}%`;
}

function fmtMs(ms: number | null | undefined): string {
  return ms == null ? '—' : `${Math.round(ms)} ms`;
}

const SPLIT_MIN = 5;
const SPLIT_MAX = 95;
const SPLIT_KEY_STEP = 5;

export const LeafDiseaseView: React.FC<LeafDiseaseViewProps> = ({ onNavigate }) => {
  const { disease, leaf, file, setDisease } = useAnalysisState();
  const { byKey } = useModels();
  const { samples, loading: samplesLoading } = useSamples('leaf_disease');

  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // The photo currently selected/captured (or handed off), kept around so "Retry" can re-submit
  // it unchanged.
  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const [pendingSource, setPendingSource] = useState<UploadSource | null>(null);
  const [pendingCapturedAt, setPendingCapturedAt] = useState<string | null>(null);
  const [pendingLocation, setPendingLocation] = useState<LocationInput | null>(null);
  const [pendingSample, setPendingSample] = useState<AnalysisSummary | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const [phase, setPhase] = useState<ProcessingPhase>('idle');
  const [progress, setProgress] = useState<number | undefined>(undefined);
  const [submitError, setSubmitError] = useState<ApiError | NetworkError | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const [pageError, setPageError] = useState<string | null>(null);

  // Guards the hand-off against re-submitting the same File on every re-render (see
  // `shouldAutoSubmitHandoff`'s doc comment for why a ref, not just `file && !disease`).
  const handoffFileRef = useRef<File | null>(null);

  // Overlay-vs-original comparison: a plain toggle, plus a drag/keyboard split slider that always
  // compares the two regardless of which toggle is selected.
  const [viewMode, setViewMode] = useState<'overlay' | 'original'>('overlay');
  const [isSliderActive, setIsSliderActive] = useState(false);
  const [splitPosition, setSplitPosition] = useState(50);
  const sliderContainerRef = useRef<HTMLDivElement | null>(null);
  const isDraggingRef = useRef(false);

  // Signed image URLs expire after ~1h: the same re-fetch-on-error-once pattern Stage1ResultView
  // uses for `image.original_url` / `image.result_url`.
  const [imageFailed, setImageFailed] = useState(false);
  const [imageRetried, setImageRetried] = useState(false);
  const [reloadingImage, setReloadingImage] = useState(false);

  useEffect(() => {
    setImageFailed(false);
    setImageRetried(false);
  }, [disease?.id]);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort();
    };
  }, []);

  const handleImageError = useCallback(async () => {
    if (!disease || imageRetried || reloadingImage) {
      setImageFailed(true);
      return;
    }
    setReloadingImage(true);
    try {
      const fresh = await getAnalysis(disease.id);
      setDisease(fresh);
      setImageRetried(true);
    } catch {
      setImageFailed(true);
    } finally {
      setReloadingImage(false);
    }
  }, [disease, imageRetried, reloadingImage, setDisease]);

  const submit = useCallback(
    async (
      photoFile: File,
      source: UploadSource,
      cameraCapturedAt: string | null,
      loc: LocationInput | null,
    ) => {
      setSubmitError(null);
      setPendingSample(null);
      setPhase('preparing');
      setProgress(undefined);

      const controller = new AbortController();
      abortControllerRef.current = controller;

      const capturedAt = resolveCapturedAt({ cameraCapturedAt, gpsCapturedAt: loc?.capturedAt ?? null });

      try {
        const detail = await predict(
          'leaf_disease',
          {
            file: photoFile,
            source,
            latitude: loc?.latitude ?? null,
            longitude: loc?.longitude ?? null,
            accuracyM: loc?.accuracyM ?? null,
            capturedAt,
          },
          {
            signal: controller.signal,
            onProgress: (p) => {
              if (p.phase === 'uploading') {
                setPhase('uploading');
                setProgress(p.percent);
              } else {
                setPhase('analysing');
                setProgress(undefined);
              }
            },
          },
        );
        setPhase('done');
        setDisease(detail);
      } catch (error) {
        if (error instanceof AbortedError) {
          setPhase('idle');
          return;
        }
        setSubmitError(error as ApiError | NetworkError);
        setPhase('error');
      } finally {
        abortControllerRef.current = null;
      }
    },
    [setDisease],
  );

  // Stage-3 hand-off: `file` is the SAME photo Model 2 (or Model 1) just analysed. Submit it
  // automatically, reusing the earlier location, without making the user re-pick anything.
  useEffect(() => {
    if (!file) return;
    if (!shouldAutoSubmitHandoff({ file, leaf: disease, alreadySubmittedFile: handoffFileRef.current })) return;
    handoffFileRef.current = file;

    // leaf.source can only be 'sample' when there is no original File to re-analyse, in which case
    // LeafAnalysisView never offers this hand-off — 'upload' is just a type-safe fallback.
    const source: UploadSource = leaf && leaf.source !== 'sample' ? leaf.source : 'upload';
    const loc: LocationInput | null = leaf?.location
      ? {
          latitude: leaf.location.latitude ?? null,
          longitude: leaf.location.longitude ?? null,
          accuracyM: leaf.location.accuracy_m ?? null,
          capturedAt: leaf.location.captured_at ? new Date(leaf.location.captured_at) : null,
        }
      : null;

    setPendingFile(file);
    setPendingSource(source);
    setPendingCapturedAt(null);
    setPendingLocation(loc);
    void submit(file, source, null, loc);
  }, [file, disease, leaf, submit]);

  const selectFile = useCallback(
    (pickedFile: File, source: UploadSource, capturedAt: string | null) => {
      const error = validateImageFile(pickedFile);
      if (error) {
        setPageError(error);
        return;
      }
      setPageError(null);
      setPreviewUrl(URL.createObjectURL(pickedFile));
      setPendingFile(pickedFile);
      setPendingSource(source);
      setPendingCapturedAt(capturedAt);
      // Independent entry on this view has no GPS UI (same as Model 2's standalone capture path).
      setPendingLocation(null);
      void submit(pickedFile, source, capturedAt, null);
    },
    [submit],
  );

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const pickedFile = e.target.files?.[0];
    e.target.value = ''; // allow re-selecting the same file
    if (pickedFile) selectFile(pickedFile, 'upload', null);
  };

  const handleDrop = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    setDragActive(false);
    const droppedFile = e.dataTransfer.files?.[0];
    if (droppedFile) selectFile(droppedFile, 'upload', null);
  };

  const handleCancel = useCallback(() => {
    abortControllerRef.current?.abort();
    setPhase('idle');
    setSubmitError(null);
  }, []);

  const openSample = useCallback(
    async (sample: AnalysisSummary) => {
      // Samples are pre-computed (D-06: no re-running the model), but they still go through the
      // same ProcessingModal as a real submission — a result appearing with no transition at all
      // reads as broken, not fast. withMinDuration only pads the UI wait; the result itself is the
      // real, already-computed row, fetched once.
      setPageError(null);
      setSubmitError(null);
      setPendingFile(null);
      setPendingSource(null);
      setPendingSample(sample);
      setPhase('preparing');

      const controller = new AbortController();
      abortControllerRef.current = controller;

      try {
        setPhase('analysing');
        setProgress(undefined);
        const detail = await withMinDuration(
          getAnalysis(sample.id, controller.signal),
          SAMPLE_OPEN_MIN_MS,
        );
        setPhase('done');
        setDisease(detail);
      } catch (error) {
        if (error instanceof AbortedError) {
          setPhase('idle');
          return;
        }
        setSubmitError(error as ApiError | NetworkError);
        setPhase('error');
      } finally {
        abortControllerRef.current = null;
      }
    },
    [setDisease],
  );

  const handleRetry = useCallback(() => {
    if (pendingFile && pendingSource) {
      void submit(pendingFile, pendingSource, pendingCapturedAt, pendingLocation);
    } else if (pendingSample) {
      void openSample(pendingSample);
    }
  }, [pendingFile, pendingSource, pendingCapturedAt, pendingLocation, pendingSample, submit, openSample]);

  // --- Comparison slider (pointer events: mouse, touch and pen; plus arrow-key support) --------

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

  // --- Derived view data -------------------------------------------------------------------------

  const details = disease && isLeafDiseaseDetails(disease.details) ? disease.details : null;
  const rawLabel = disease?.prediction?.label ?? null;
  const label: LeafDiseaseLabel | null =
    rawLabel === 'black_sigatoka' || rawLabel === 'yellow_sigatoka' || rawLabel === 'healthy' || rawLabel === 'no_leaf'
      ? rawLabel
      : null;
  const style = label ? LABEL_STYLE[label] : null;
  const modelInfo = byKey.leaf_disease;
  const hasLeafTissue = label !== null && label !== 'no_leaf';
  const originalUrl = disease ? absoluteImageUrl(disease.image.original_url) : null;
  const resultUrl = disease ? absoluteImageUrl(disease.image.result_url) : null;
  const hasPendingHandoff = file !== null && disease === null;

  return (
    <div className="w-full max-w-7xl mx-auto px-6 lg:px-12 py-10 flex flex-col gap-8">
      {/* Full-Screen Mobile-Optimized Camera Modal */}
      <CameraModal
        isOpen={isCameraOpen}
        mode="leaf"
        fallbackImage={ASSETS.leafAnalysisSpecimen}
        onClose={() => setIsCameraOpen(false)}
        onCapture={(capturedFile, capturedAt) => {
          setIsCameraOpen(false);
          selectFile(capturedFile, 'camera', capturedAt);
        }}
      />

      {/* Real-state Processing Modal: driven by the actual predict() call (hand-off or manual). */}
      <ProcessingModal
        open={phase !== 'idle'}
        phase={phase}
        progress={progress}
        error={submitError}
        onCancel={handleCancel}
        onRetry={phase === 'error' && (pendingFile || pendingSample) ? handleRetry : undefined}
      />

      {/* Hidden File Upload Input */}
      <input
        ref={fileInputRef}
        id="leaf-disease-upload"
        type="file"
        accept={ACCEPTED_IMAGE_TYPES.join(',')}
        className="sr-only"
        onChange={handleFileInputChange}
      />

      {/* Header */}
      <header className="flex flex-col gap-2 max-w-3xl">
        <div className="flex items-center gap-2 text-[#3d4a42] text-xs font-semibold tracking-wider uppercase font-mono">
          <button
            type="button"
            onClick={() => onNavigate('home')}
            className="hover:text-[#006948] transition-colors cursor-pointer"
          >
            Home
          </button>
          <span className="material-symbols-outlined text-[14px]">chevron_right</span>
          <span className="text-[#006948] font-bold">Detect Disease</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-[#131b2e] tracking-tight">
          Leaf Disease Identification
        </h1>
        <p className="text-base text-[#3d4a42] leading-relaxed">
          Model 3 tells a healthy banana leaf from one with Black Sigatoka or Yellow Sigatoka, and
          localises each disease on the leaf itself.
        </p>
      </header>

      {/* Selected photo preview (independent camera/upload entry only) */}
      {previewUrl && (
        <div className="bg-white p-4 rounded-2xl border border-[#dae2fd] shadow-sm flex items-center gap-4">
          <img
            src={previewUrl}
            alt="Selected leaf preview"
            className="w-16 h-16 rounded-xl object-cover border border-[#dae2fd] shrink-0"
          />
          <div className="flex flex-col flex-1 min-w-0">
            <span className="text-sm font-semibold text-[#131b2e] truncate">
              {pendingFile?.name ?? 'Captured photo'}
            </span>
            <span className="text-xs text-[#3d4a42]">
              {pendingSource === 'camera' ? 'From camera' : 'Uploaded file'}
            </span>
          </div>
        </div>
      )}

      {pageError && (
        <div className="p-3.5 rounded-xl bg-[#ffdad6] border border-[#ba1a1a]/30 text-[#ba1a1a] text-sm">
          {pageError}
        </div>
      )}

      {/* TWO LARGE WORKSPACE CARDS (independent entry: camera / upload / samples) */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-stretch">
        {/* Option 1: Use Camera */}
        <div className="order-1 md:order-2 bg-white p-7 rounded-2xl shadow-sm border border-[#dae2fd] flex flex-col justify-between gap-6 hover:shadow-md transition-shadow relative overflow-hidden">
          <div className="absolute top-0 right-0 px-3 py-1 bg-[#86f2e4]/30 text-[#006f66] font-mono text-[10px] font-bold rounded-bl-xl border-l border-b border-[#86f2e4]">
            IPHONE &bull; ANDROID READY
          </div>

          <div className="flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <span className="px-3 py-1 rounded-md bg-[#86f2e4]/30 text-[#006f66] font-mono text-xs font-bold">
                OPTION 01
              </span>
              <span className="material-symbols-outlined text-[#006a61] text-[24px]">photo_camera</span>
            </div>

            <div>
              <h2 className="text-xl font-bold text-[#131b2e]">Use Camera</h2>
              <p className="text-xs text-[#3d4a42] mt-1">
                Point your camera at a banana leaf to check it for disease.
              </p>
            </div>

            <div
              onClick={() => setIsCameraOpen(true)}
              className="relative w-full h-48 rounded-xl overflow-hidden bg-[#dae2fd] border border-[#bccac0] cursor-pointer group flex items-center justify-center shadow-inner"
            >
              <img
                alt="Camera targeting preview"
                className="absolute inset-0 w-full h-full object-cover opacity-85 group-hover:scale-105 transition-transform duration-500"
                src={ASSETS.leafAnalysisSpecimen}
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-black/30" />
              <div className="absolute inset-x-0 h-0.5 bg-gradient-to-r from-transparent via-[#85f8c4] to-transparent shadow-[0_0_12px_#85f8c4] animate-[pulse_2s_infinite]" />
              <div className="relative z-10 flex flex-col items-center gap-1 text-white text-center px-4">
                <span className="material-symbols-outlined text-[32px] text-[#85f8c4] drop-shadow">
                  center_focus_strong
                </span>
                <span className="text-sm font-bold drop-shadow">Camera Viewfinder</span>
                <span className="font-mono text-[11px] text-[#f5fff7] bg-[#00855d]/85 px-3 py-0.5 rounded-full backdrop-blur-sm shadow-sm">
                  Tap to launch
                </span>
              </div>
            </div>

            <div className="p-3.5 rounded-xl bg-[#f2f3ff] border border-[#dae2fd] flex items-center gap-2.5 text-xs text-[#3d4a42]">
              <span className="material-symbols-outlined text-[#006948] text-[20px] shrink-0">info</span>
              <span>Point camera directly toward the banana leaf blade with clear natural lighting.</span>
            </div>
          </div>

          <div className="pt-2">
            <button
              type="button"
              onClick={() => setIsCameraOpen(true)}
              className="w-full py-3.5 px-4 rounded-xl bg-[#006948] text-white hover:bg-[#00855d] font-semibold text-sm transition-all flex items-center justify-center gap-2 shadow-md shadow-[#006948]/25 cursor-pointer active:scale-[0.99]"
            >
              <span className="material-symbols-outlined text-[20px]">videocam</span>
              <span>Open Camera</span>
            </button>
          </div>
        </div>

        {/* Option 2: Upload Image */}
        <div className="order-2 md:order-1 bg-white p-7 rounded-2xl shadow-sm border border-[#dae2fd] flex flex-col justify-between gap-6 hover:shadow-md transition-shadow">
          <div className="flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <span className="px-3 py-1 rounded-md bg-[#eaedff] font-mono text-xs font-bold text-[#131b2e]">
                OPTION 02
              </span>
              <span className="material-symbols-outlined text-[#006948] text-[24px]">cloud_upload</span>
            </div>

            <div>
              <h2 className="text-xl font-bold text-[#131b2e]">Upload Image</h2>
              <p className="text-xs text-[#3d4a42] mt-1">
                Drag &amp; drop an image here or browse from your device.
              </p>
            </div>

            {/* Dropzone — real drag-and-drop */}
            <label
              htmlFor="leaf-disease-upload"
              onDragOver={(e) => {
                e.preventDefault();
                setDragActive(true);
              }}
              onDragLeave={() => setDragActive(false)}
              onDrop={handleDrop}
              className={`cursor-pointer flex flex-col items-center justify-center p-8 rounded-xl border-2 border-dashed transition-colors text-center relative group ${
                dragActive
                  ? 'bg-[#dae2fd] border-[#006948]'
                  : 'bg-[#f2f3ff] hover:bg-[#eaedff] border-[#bccac0] hover:border-[#006948]'
              }`}
            >
              <div className="w-12 h-12 rounded-full bg-white flex items-center justify-center shadow-sm text-[#006948] mb-3 border border-[#dae2fd] group-hover:scale-105 transition-transform">
                <span className="material-symbols-outlined text-[26px]">add_photo_alternate</span>
              </div>
              <span className="text-sm font-bold text-[#131b2e]">Drag &amp; drop an image here</span>
              <span className="text-xs text-[#3d4a42] mt-1">
                or browse from your device &bull; Supports JPG, PNG, WEBP
              </span>
            </label>

            {/* Real pre-computed samples */}
            <div className="flex flex-col gap-2 mt-1">
              <span className="font-mono text-xs text-[#3d4a42] font-semibold tracking-wide">
                SAMPLE SPECIMENS:
              </span>
              <div className="flex flex-wrap gap-2">
                {samplesLoading && <span className="text-xs text-[#3d4a42]">Loading samples…</span>}
                {!samplesLoading && samples.length === 0 && (
                  <span className="text-xs text-[#3d4a42]">No samples available yet.</span>
                )}
                {samples.map((sample) => (
                  <button
                    key={sample.id}
                    type="button"
                    onClick={() => void openSample(sample)}
                    className="px-3 py-1.5 rounded-lg bg-[#eaedff] hover:bg-[#dae2fd] text-[#131b2e] font-mono text-xs flex items-center gap-2 transition-colors border border-[#dae2fd] cursor-pointer"
                  >
                    <img
                      src={absoluteImageUrl(sample.thumbnail_url)}
                      alt=""
                      className="w-6 h-6 rounded object-cover shrink-0"
                    />
                    <span>{sample.title ?? sample.display_label}</span>
                  </button>
                ))}
              </div>
            </div>
          </div>

          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="w-full py-3.5 px-4 rounded-xl bg-[#eaedff] text-[#131b2e] hover:bg-[#dae2fd] font-semibold text-sm transition-all flex items-center justify-center gap-2 border border-[#dae2fd] shadow-sm cursor-pointer"
          >
            <span className="material-symbols-outlined text-[18px]">folder_open</span>
            <span>Choose Image</span>
          </button>
        </div>
      </div>

      {/* RESULT STUDIO — driven entirely by useAnalysisState().disease */}
      {disease && (
        <div className="flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono font-bold uppercase tracking-wider text-[#006948]">
              Leaf Disease Result
            </span>
            {modelInfo?.is_placeholder && (
              <span
                title="Results come from placeholder weights and are not real predictions"
                className="px-1.5 py-0.5 rounded bg-[#ffeed2] border border-[#ffb95f] text-[#825100] text-[10px] font-bold"
              >
                Demo weights
              </span>
            )}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
            {/* Left Column: Interactive Studio Canvas (7 Cols) */}
            <div className="lg:col-span-7 flex flex-col gap-4">
              {/* View toolbar */}
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
                        viewMode === 'overlay' && !isSliderActive
                          ? 'bg-[#006948] text-white font-bold shadow-sm'
                          : 'text-[#3d4a42] hover:text-[#131b2e]'
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
                        viewMode === 'original' && !isSliderActive
                          ? 'bg-[#131b2e] text-white font-bold shadow-sm'
                          : 'text-[#3d4a42] hover:text-[#131b2e]'
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
                    isSliderActive
                      ? 'bg-[#00855d] text-white border-[#00855d]'
                      : 'bg-[#eaedff] text-[#131b2e] border-[#dae2fd] hover:bg-[#dae2fd]'
                  }`}
                  title="Drag (or use the arrow keys) to compare the overlay against the original photo"
                >
                  <span className="material-symbols-outlined text-[16px]">compare</span>
                  <span>Split Compare</span>
                </button>
              </div>

              {/* Image area: original photo as the base layer, overlay PNG on top */}
              <div
                ref={sliderContainerRef}
                className="relative w-full aspect-[4/3] rounded-2xl overflow-hidden bg-[#131b2e] shadow-xl border border-[#dae2fd] select-none touch-none"
              >
                {!imageFailed ? (
                  <>
                    <img
                      alt="Original leaf photo"
                      className="absolute inset-0 w-full h-full object-cover"
                      src={originalUrl ?? undefined}
                      onError={() => void handleImageError()}
                    />
                    {resultUrl && (
                      <img
                        alt="Leaf disease overlay — green outlines the leaf, each disease shaded a different colour"
                        className="absolute inset-0 w-full h-full object-cover"
                        src={resultUrl}
                        onError={() => void handleImageError()}
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

              {/* Per-disease breakdown — real numbers from the server, only meaningful when a leaf was found */}
              {hasLeafTissue ? (
                <div className="bg-white p-5 rounded-2xl border border-[#dae2fd] shadow-sm flex flex-col gap-3 font-mono">
                  <span className="text-xs font-bold uppercase tracking-wider text-[#131b2e]">
                    Disease Breakdown (% of leaf)
                  </span>
                  {(details?.diseases ?? []).map((d) => (
                    <div
                      key={d.channel}
                      className="p-3 rounded-xl bg-[#f2f3ff] border border-[#dae2fd] flex flex-col gap-1"
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-bold text-[#131b2e]">
                          {DISEASE_CHANNEL_LABEL[d.channel] ?? d.channel}
                        </span>
                        <span className="text-sm font-extrabold text-[#131b2e]">
                          {fmtPct(d.area_pct_of_leaf)}
                        </span>
                      </div>
                      <div className="flex justify-between text-[11px] text-[#3d4a42]">
                        <span>Lesion count: {d.lesion_count}</span>
                        <span>Largest lesion: {fmtPct(d.largest_lesion_pct_of_leaf)}</span>
                        <span>Mean probability: {fmtProbability(d.mean_probability)}</span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                label === 'no_leaf' && (
                  <div className="bg-white p-5 rounded-2xl border border-[#ffb95f]/60 shadow-sm text-sm text-[#5c3c00]">
                    No disease breakdown is shown because the model could not find enough leaf area in
                    this photo.
                  </div>
                )
              )}
            </div>

            {/* Right Column: Status + Measurements (5 Cols) */}
            <div className="lg:col-span-5 flex flex-col gap-6">
              <div
                className={`bg-white p-7 rounded-2xl shadow-sm border ${style?.cardBorder ?? 'border-[#dae2fd]'} flex flex-col gap-6`}
              >
                <div className={`flex items-center gap-3 p-3.5 rounded-xl ${style?.bannerClass ?? 'bg-[#eaedff] border border-[#dae2fd] text-[#131b2e]'}`}>
                  <span className="material-symbols-outlined text-[26px]">{style?.icon ?? 'help'}</span>
                  <span className="text-base font-extrabold tracking-tight">
                    {style?.title ?? 'Result unavailable'}
                  </span>
                </div>
                <p className="text-xs text-[#3d4a42] leading-relaxed -mt-3">
                  {style?.description ?? 'This analysis did not return a recognised result.'}
                </p>

                {/* Measurements */}
                <div className="p-4 rounded-xl bg-[#f2f3ff] border border-[#dae2fd] flex flex-col gap-2 font-mono text-xs">
                  <span className="text-[10px] uppercase font-bold text-[#3d4a42] tracking-wider">
                    Measurements
                  </span>
                  <div className="flex justify-between py-1 border-b border-[#dae2fd]">
                    <span className="text-[#3d4a42]">Leaf coverage of photo:</span>
                    <span className="font-bold text-[#131b2e]">{fmtPct(details?.leaf_area_pct_of_image)}</span>
                  </div>
                  <div className="flex justify-between py-1">
                    <span className="text-[#3d4a42]">Mean leaf probability:</span>
                    <span className="font-bold text-[#131b2e]">
                      {fmtProbability(details?.mean_leaf_probability)}
                    </span>
                  </div>
                </div>

                {/* Thresholds — secondary/technical, collapsed by default */}
                {details && (
                  <details className="rounded-xl bg-[#faf8ff] border border-[#dae2fd] p-4 text-xs font-mono">
                    <summary className="cursor-pointer font-bold text-[#131b2e] uppercase tracking-wider text-[10px]">
                      Thresholds used
                    </summary>
                    <div className="grid grid-cols-2 gap-x-2 gap-y-1.5 mt-3 text-[#3d4a42]">
                      <span>Leaf pixel threshold:</span>
                      <span className="text-right font-bold text-[#131b2e]">
                        {fmtProbability(details.thresholds.leaf)}
                      </span>
                      <span>Black Sigatoka threshold:</span>
                      <span className="text-right font-bold text-[#131b2e]">
                        {fmtProbability(details.thresholds.black_sigatoka)}
                      </span>
                      <span>Yellow Sigatoka threshold:</span>
                      <span className="text-right font-bold text-[#131b2e]">
                        {fmtProbability(details.thresholds.yellow_sigatoka)}
                      </span>
                      <span>Min. leaf % (else no_leaf):</span>
                      <span className="text-right font-bold text-[#131b2e]">
                        {fmtPct(details.thresholds.min_leaf_pct)}
                      </span>
                      <span>Min. disease % (else healthy):</span>
                      <span className="text-right font-bold text-[#131b2e]">
                        {fmtPct(details.thresholds.min_disease_pct)}
                      </span>
                      <span>Min. lesion size %:</span>
                      <span className="text-right font-bold text-[#131b2e]">
                        {fmtPct(details.thresholds.min_lesion_pct)}
                      </span>
                      <span>Horizontal-flip TTA:</span>
                      <span className="text-right font-bold text-[#131b2e]">
                        {details.thresholds.tta_hflip ? 'Yes' : 'No'}
                      </span>
                    </div>
                  </details>
                )}

                {/* Model & timing */}
                <div className="p-4 rounded-xl bg-[#f2f3ff] border border-[#dae2fd] flex flex-col gap-2 font-mono text-xs">
                  <span className="text-[10px] uppercase font-bold text-[#3d4a42] tracking-wider">
                    Model &amp; Timing
                  </span>
                  <div className="flex justify-between py-1 border-b border-[#dae2fd]">
                    <span className="text-[#3d4a42]">Model version:</span>
                    <span className="font-bold text-[#131b2e]">{disease.model_version}</span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-[#dae2fd]">
                    <span className="text-[#3d4a42]">Analyzed:</span>
                    <span className="font-bold text-[#131b2e]">{new Date(disease.created_at).toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between py-1">
                    <span className="text-[#3d4a42]">Total time:</span>
                    <span className="font-bold text-[#006948]">{fmtMs(disease.timings_ms.total)}</span>
                  </div>
                </div>

                <button
                  type="button"
                  onClick={() => setIsCameraOpen(true)}
                  className="w-full py-3.5 px-6 rounded-xl bg-[#006948] text-white hover:bg-[#00855d] font-semibold text-sm transition-all flex items-center justify-center gap-2 shadow-sm cursor-pointer active:scale-95"
                >
                  <span className="material-symbols-outlined text-[20px]">videocam</span>
                  <span>Capture Another Leaf</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* No result yet: either a hand-off is about to run (the ProcessingModal covers it), or
          there is genuinely nothing to show. */}
      {!disease && hasPendingHandoff && (
        <div className="bg-white p-8 rounded-2xl border border-[#dae2fd] text-center text-sm text-[#3d4a42]">
          Preparing your disease check from the earlier photo…
        </div>
      )}

      {!disease && !hasPendingHandoff && (
        <div className="w-full max-w-3xl mx-auto px-6 py-12 flex flex-col items-center gap-4 text-center">
          <span className="material-symbols-outlined text-[40px] text-[#3d4a42]">coronavirus</span>
          <h2 className="text-2xl font-extrabold text-[#131b2e]">No disease check yet</h2>
          <p className="text-sm text-[#3d4a42]">
            Use the camera or upload above, try a sample, or analyse a leaf first and continue here
            with "Identify the Disease".
          </p>
          <button
            type="button"
            onClick={() => onNavigate('leaf-analysis')}
            className="mt-1 px-6 py-3 rounded-xl bg-[#006948] text-white hover:bg-[#00855d] font-semibold text-sm transition-all cursor-pointer"
          >
            Go to Leaf Analysis
          </button>
        </div>
      )}
    </div>
  );
};
