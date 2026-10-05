import React, { useCallback, useEffect, useRef, useState } from 'react';
import { ViewTab } from '../types';
import { ASSETS } from '../data/mockData';
import { CameraModal } from '../components/CameraModal';
import { ProcessingModal } from '../components/ProcessingModal';
import type { ProcessingPhase } from '../components/ProcessingModal';
import { useAnalysisState } from '../state/AnalysisContext';
import { useGeolocation } from '../hooks/useGeolocation';
import { useSamples } from '../hooks/useSamples';
import {
  ACCEPTED_IMAGE_TYPES,
  AbortedError,
  ApiError,
  NetworkError,
  absoluteImageUrl,
  getAnalysis,
  predict,
  resolveCapturedAt,
  validateImageFile,
  withMinDuration,
} from '../api';
import type { AnalysisSummary, UploadSource } from '../api';

// Samples are a single, near-instant DB read (no inference happens) — without a floor, the
// ProcessingModal would flash and vanish, which reads as broken rather than fast.
const SAMPLE_OPEN_MIN_MS = 900;

interface DetectionWorkspaceViewProps {
  onNavigate: (tab: ViewTab) => void;
}

export const DetectionWorkspaceView: React.FC<DetectionWorkspaceViewProps> = ({ onNavigate }) => {
  const { setFile, setTree } = useAnalysisState();
  const geo = useGeolocation();
  const { samples, loading: samplesLoading } = useSamples('tree_classification');

  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // The photo currently selected/captured, kept around so "Retry" can re-submit it unchanged.
  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const [pendingSource, setPendingSource] = useState<UploadSource | null>(null);
  const [pendingCapturedAt, setPendingCapturedAt] = useState<string | null>(null);
  const [pendingSample, setPendingSample] = useState<AnalysisSummary | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const [phase, setPhase] = useState<ProcessingPhase>('idle');
  const [progress, setProgress] = useState<number | undefined>(undefined);
  const [submitError, setSubmitError] = useState<ApiError | NetworkError | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const [pageError, setPageError] = useState<string | null>(null);

  // Manual GPS entry, if the owner prefers to type coordinates instead of using the device fix.
  const [manualOverride, setManualOverride] = useState<{ latitude: number; longitude: number } | null>(null);
  const [isEditingGps, setIsEditingGps] = useState(false);
  const [manualLat, setManualLat] = useState('');
  const [manualLng, setManualLng] = useState('');
  const [manualError, setManualError] = useState<string | null>(null);

  // Release the object URL of the previously selected photo whenever it changes, and on unmount.
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

  // Manual entry wins once set; requesting a fresh device fix drops it again.
  const location = manualOverride
    ? { latitude: manualOverride.latitude, longitude: manualOverride.longitude, accuracyM: null as number | null, capturedAt: null as Date | null }
    : geo.position
      ? { latitude: geo.position.latitude, longitude: geo.position.longitude, accuracyM: geo.position.accuracyM, capturedAt: geo.position.capturedAt }
      : null;

  const handleUseMyLocation = () => {
    setManualOverride(null);
    setIsEditingGps(false);
    geo.request();
  };

  const handleManualGpsSave = (e: React.FormEvent) => {
    e.preventDefault();
    const lat = Number.parseFloat(manualLat);
    const lng = Number.parseFloat(manualLng);
    if (!Number.isFinite(lat) || !Number.isFinite(lng) || lat < -90 || lat > 90 || lng < -180 || lng > 180) {
      setManualError('Enter a valid latitude (-90 to 90) and longitude (-180 to 180).');
      return;
    }
    setManualError(null);
    setManualOverride({ latitude: lat, longitude: lng });
    setIsEditingGps(false);
  };

  const submit = useCallback(
    async (file: File, source: UploadSource, cameraCapturedAt: string | null) => {
      setSubmitError(null);
      setPendingSample(null);
      setPhase('preparing');
      setProgress(undefined);

      const controller = new AbortController();
      abortControllerRef.current = controller;

      const capturedAt = resolveCapturedAt({
        cameraCapturedAt,
        gpsCapturedAt: location?.capturedAt ?? null,
      });

      try {
        const detail = await predict(
          'tree_classification',
          {
            file,
            source,
            latitude: location?.latitude ?? null,
            longitude: location?.longitude ?? null,
            accuracyM: location?.accuracyM ?? null,
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
        setFile(file);
        setTree(detail);
        onNavigate('stage1-result');
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
    [location, onNavigate, setFile, setTree],
  );

  const selectFile = useCallback(
    (file: File, source: UploadSource, capturedAt: string | null) => {
      const error = validateImageFile(file);
      if (error) {
        setPageError(error);
        return;
      }
      setPageError(null);
      setPreviewUrl(URL.createObjectURL(file));
      setPendingFile(file);
      setPendingSource(source);
      setPendingCapturedAt(capturedAt);
      void submit(file, source, capturedAt);
    },
    [submit],
  );

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = ''; // allow re-selecting the same file
    if (file) selectFile(file, 'upload', null);
  };

  const handleDrop = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    setDragActive(false);
    const file = e.dataTransfer.files?.[0];
    if (file) selectFile(file, 'upload', null);
  };

  const handleCancel = useCallback(() => {
    abortControllerRef.current?.abort();
    setPhase('idle');
    setSubmitError(null);
  }, []);

  const openSample = useCallback(
    async (sample: AnalysisSummary) => {
      // Samples are pre-computed (D-06: no re-running the model), but they still go through the
      // same ProcessingModal as a real submission — a result appearing with literally no
      // transition reads as broken, not fast. withMinDuration only pads the UI wait; the result
      // itself is the real, already-computed row, fetched once.
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
        setFile(null);
        setTree(detail);
        onNavigate('stage1-result');
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
    [onNavigate, setFile, setTree],
  );

  const handleRetry = useCallback(() => {
    if (pendingFile && pendingSource) void submit(pendingFile, pendingSource, pendingCapturedAt);
    else if (pendingSample) void openSample(pendingSample);
  }, [pendingFile, pendingSource, pendingCapturedAt, pendingSample, submit, openSample]);

  return (
    <div className="w-full max-w-7xl mx-auto px-6 lg:px-12 py-10 flex flex-col gap-8">
      {/* Full-Screen Mobile-Optimized Camera Modal */}
      <CameraModal
        isOpen={isCameraOpen}
        mode="plant"
        fallbackImage={ASSETS.bananaFieldCalib}
        onClose={() => setIsCameraOpen(false)}
        onCapture={(file, capturedAt) => {
          setIsCameraOpen(false);
          selectFile(file, 'camera', capturedAt);
        }}
      />

      {/* Real-state Processing Modal: driven by the actual predict() call below. */}
      <ProcessingModal
        open={phase !== 'idle'}
        phase={phase}
        progress={progress}
        error={submitError}
        onCancel={handleCancel}
        onRetry={phase === 'error' && (pendingFile || pendingSample) ? handleRetry : undefined}
      />

      {/* Header */}
      <header className="flex flex-col gap-2 max-w-3xl">
        <div className="flex items-center gap-2 text-[#3d4a42] text-xs font-semibold tracking-wider uppercase">
          <button
            onClick={() => onNavigate('home')}
            className="hover:text-[#006948] transition-colors cursor-pointer"
          >
            Home
          </button>
          <span className="material-symbols-outlined text-[14px]">chevron_right</span>
          <span className="text-[#006948] font-bold">Plant Detection</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-[#131b2e] tracking-tight">
          Plant Detection
        </h1>
        <p className="text-base text-[#3d4a42] leading-relaxed">
          Upload an image or use your camera to analyze a banana plant.
        </p>
      </header>

      {/* Field Location (Latitude & Longitude) Feature Bar */}
      <div className="bg-white p-5 rounded-2xl border border-[#dae2fd] shadow-sm flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-[#e2e7ff] text-[#006948] flex items-center justify-center shrink-0">
            <span className="material-symbols-outlined text-[22px]">location_on</span>
          </div>
          <div className="flex flex-col">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-xs uppercase font-bold text-[#3d4a42] tracking-wider">
                Field GPS Coordinates
              </span>
              {geo.status === 'requesting' ? (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-[#eaedff] text-[#006948] font-mono text-[10px] font-bold">
                  <span className="material-symbols-outlined text-[12px] animate-spin">refresh</span>
                  Acquiring...
                </span>
              ) : location ? (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-[#85f8c4]/40 text-[#006948] font-mono text-[10px] font-bold">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#006948]" />
                  {manualOverride ? 'Manual' : 'GPS Locked'}{' '}
                  {location.accuracyM != null ? `(±${location.accuracyM.toFixed(1)}m)` : ''}
                </span>
              ) : (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-[#eaedff] text-[#3d4a42] font-mono text-[10px] font-bold">
                  No Location Set
                </span>
              )}
            </div>

            {/* Coordinates Display */}
            {!isEditingGps ? (
              <div className="flex items-center gap-4 mt-1 font-mono text-xs sm:text-sm">
                <span className="text-[#131b2e]">
                  <strong className="text-[#3d4a42] font-semibold">Lat:</strong>{' '}
                  <span className="font-bold text-[#006948]">
                    {location ? `${location.latitude.toFixed(6)}°` : '—'}
                  </span>
                </span>
                <span className="text-[#bccac0]">&bull;</span>
                <span className="text-[#131b2e]">
                  <strong className="text-[#3d4a42] font-semibold">Long:</strong>{' '}
                  <span className="font-bold text-[#006948]">
                    {location ? `${location.longitude.toFixed(6)}°` : '—'}
                  </span>
                </span>
              </div>
            ) : (
              <form onSubmit={handleManualGpsSave} className="flex items-center gap-2 mt-1.5 flex-wrap">
                <input
                  type="text"
                  placeholder="Latitude"
                  value={manualLat}
                  onChange={(e) => setManualLat(e.target.value)}
                  className="px-2.5 py-1 rounded-lg border border-[#dae2fd] text-xs font-mono w-28 bg-[#f2f3ff]"
                />
                <input
                  type="text"
                  placeholder="Longitude"
                  value={manualLng}
                  onChange={(e) => setManualLng(e.target.value)}
                  className="px-2.5 py-1 rounded-lg border border-[#dae2fd] text-xs font-mono w-28 bg-[#f2f3ff]"
                />
                <button
                  type="submit"
                  className="px-3 py-1 rounded-lg bg-[#006948] text-white text-xs font-semibold cursor-pointer"
                >
                  Save
                </button>
                <button
                  type="button"
                  onClick={() => setIsEditingGps(false)}
                  className="px-2 py-1 text-xs text-[#3d4a42] hover:text-[#131b2e] cursor-pointer"
                >
                  Cancel
                </button>
              </form>
            )}

            {(manualError || geo.error) && (
              <span className="text-[11px] text-[#ba1a1a] mt-0.5">{manualError ?? geo.error}</span>
            )}
          </div>
        </div>

        {/* GPS Action Buttons */}
        <div className="flex items-center gap-2 self-end md:self-center">
          <button
            type="button"
            onClick={handleUseMyLocation}
            disabled={geo.status === 'requesting'}
            className="px-3 py-1.5 rounded-xl bg-[#eaedff] hover:bg-[#dae2fd] text-[#131b2e] text-xs font-semibold font-mono flex items-center gap-1.5 transition-colors cursor-pointer border border-[#dae2fd]"
            title="Use this device's current location"
          >
            <span
              className={`material-symbols-outlined text-[16px] text-[#006948] ${geo.status === 'requesting' ? 'animate-spin' : ''}`}
            >
              my_location
            </span>
            <span>Use My Location</span>
          </button>

          {!isEditingGps && (
            <button
              type="button"
              onClick={() => {
                setManualLat(location ? location.latitude.toFixed(6) : '');
                setManualLng(location ? location.longitude.toFixed(6) : '');
                setIsEditingGps(true);
              }}
              className="p-1.5 rounded-xl text-[#3d4a42] hover:bg-[#eaedff] transition-colors cursor-pointer"
              title="Edit Coordinates manually"
            >
              <span className="material-symbols-outlined text-[18px]">edit</span>
            </button>
          )}
        </div>
      </div>

      {/* Selected photo preview (shown for both camera captures and uploads, until the result arrives) */}
      {previewUrl && (
        <div className="bg-white p-4 rounded-2xl border border-[#dae2fd] shadow-sm flex items-center gap-4">
          <img
            src={previewUrl}
            alt="Selected specimen preview"
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

      {/* Two Large Options (Camera prominent on mobile) */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-stretch">
        {/* Mobile-first reordering: On mobile, camera is first! */}
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
              <span className="material-symbols-outlined text-[#006a61] text-[24px]">
                photo_camera
              </span>
            </div>

            <div>
              <h2 className="text-xl font-bold text-[#131b2e]">Use Camera</h2>
              <p className="text-xs text-[#3d4a42] mt-1">
                Point your camera at a banana plant for instant field identification.
              </p>
            </div>

            {/* Viewfinder Preview Box */}
            <div
              onClick={() => setIsCameraOpen(true)}
              className="relative w-full h-48 rounded-xl overflow-hidden bg-[#dae2fd] border border-[#bccac0] cursor-pointer group flex items-center justify-center shadow-inner"
            >
              <img
                alt="Camera targeting preview"
                className="absolute inset-0 w-full h-full object-cover opacity-85 group-hover:scale-105 transition-transform duration-500"
                src={ASSETS.bananaFieldCalib}
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-black/30" />

              {/* Pulsing Scanline */}
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
              <span className="material-symbols-outlined text-[#006948] text-[20px] shrink-0">
                info
              </span>
              <span>Point camera toward the banana tree pseudostem or full canopy foliage.</span>
            </div>
          </div>

          {/* Camera Button: Single robust Open Camera button */}
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
              <span className="material-symbols-outlined text-[#006948] text-[24px]">
                cloud_upload
              </span>
            </div>

            <div>
              <h2 className="text-xl font-bold text-[#131b2e]">Upload Image</h2>
              <p className="text-xs text-[#3d4a42] mt-1">
                Drag &amp; drop an image here or browse from your device.
              </p>
            </div>

            {/* Dropzone — real drag-and-drop */}
            <label
              htmlFor="plant-image-upload"
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
              <input
                ref={fileInputRef}
                id="plant-image-upload"
                type="file"
                accept={ACCEPTED_IMAGE_TYPES.join(',')}
                className="sr-only"
                onChange={handleFileInputChange}
              />
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
    </div>
  );
};
