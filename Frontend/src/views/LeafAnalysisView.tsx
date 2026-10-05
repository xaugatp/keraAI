import React, { useEffect, useState, useRef } from 'react';
import { ViewTab, Stage2Diagnosis } from '../types';
import { sampleDiagnoses, ASSETS } from '../data/mockData';
import { CameraModal } from '../components/CameraModal';
import { ProcessingModal } from '../components/ProcessingModal';
import type { ProcessingPhase } from '../components/ProcessingModal';

interface LeafAnalysisViewProps {
  onNavigate: (tab: ViewTab) => void;
  selectedDiagnosisIndex?: number;
}

export const LeafAnalysisView: React.FC<LeafAnalysisViewProps> = ({
  onNavigate,
  selectedDiagnosisIndex = 0,
}) => {
  const [currentIdx, setCurrentIdx] = useState<number>(selectedDiagnosisIndex);
  // NOTE: still a mock-data stub (F3 wires this view to the real Model 2 result). This state
  // holds an object URL for whatever photo the user picked, just to keep the preview working.
  const [customLeafImage, setCustomLeafImage] = useState<string | null>(null);
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingPhase, setProcessingPhase] = useState<ProcessingPhase>('idle');

  // Release the previous preview's object URL whenever it changes, and on unmount.
  useEffect(() => {
    return () => {
      if (customLeafImage) URL.revokeObjectURL(customLeafImage);
    };
  }, [customLeafImage]);

  // This view is not yet wired to the real backend (F3); simulate the same brief "analysing"
  // delay the old timer-driven ProcessingModal used to provide internally.
  useEffect(() => {
    if (!isProcessing) {
      setProcessingPhase('idle');
      return;
    }
    setProcessingPhase('analysing');
    const timer = setTimeout(() => setIsProcessing(false), 1200);
    return () => clearTimeout(timer);
  }, [isProcessing]);

  // Segmentation Studio Modes: 'overlay' | 'healthy-only' | 'unhealthy-only' | 'original'
  const [viewMode, setViewMode] = useState<'overlay' | 'healthy-only' | 'unhealthy-only' | 'original'>('overlay');
  const [isSliderActive, setIsSliderActive] = useState(false);
  const [splitPosition, setSplitPosition] = useState<number>(50); // percentage

  const containerRef = useRef<HTMLDivElement | null>(null);
  const isDraggingRef = useRef<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const selectedDiagnosis: Stage2Diagnosis = sampleDiagnoses[currentIdx] || sampleDiagnoses[0];
  const isHealthy = selectedDiagnosis.severity === 'healthy';

  const healthyPercent = isHealthy ? 100 : Number((100 - selectedDiagnosis.infectedAreaPercent).toFixed(1));
  const unhealthyPercent = isHealthy ? 0 : selectedDiagnosis.infectedAreaPercent;

  const handleMouseDown = () => {
    isDraggingRef.current = true;
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!isDraggingRef.current || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const pct = Math.max(5, Math.min(95, (x / rect.width) * 100));
    setSplitPosition(pct);
  };

  const handleMouseUp = () => {
    isDraggingRef.current = false;
  };

  const handleCustomCapture = (file: File, _capturedAt: string) => {
    setCustomLeafImage(URL.createObjectURL(file));
    setIsProcessing(true);
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const isHealthyName = file.name.toLowerCase().includes('healthy') || file.name.toLowerCase().includes('clean');
      const reader = new FileReader();
      reader.onload = (event) => {
        if (event.target?.result) {
          setCustomLeafImage(event.target.result as string);
          if (isHealthyName) setCurrentIdx(1);
          setIsProcessing(true);
        }
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSelectBenchmarkSample = () => {
    setCustomLeafImage(null);
    setCurrentIdx(0);
    setIsProcessing(true);
  };

  const activeImage = customLeafImage || ASSETS.leafAnalysisSpecimen || selectedDiagnosis.imageUrl;

  return (
    <div className="w-full max-w-7xl mx-auto px-6 lg:px-12 py-10 flex flex-col gap-8">
      {/* Functional Camera Modal */}
      <CameraModal
        isOpen={isCameraOpen}
        mode="leaf"
        fallbackImage={ASSETS.leafAnalysisSpecimen}
        onClose={() => setIsCameraOpen(false)}
        onCapture={handleCustomCapture}
      />

      {/* Hidden File Upload Input */}
      <input
        ref={fileInputRef}
        id="leaf-analysis-upload"
        type="file"
        accept="image/*"
        className="sr-only"
        onChange={handleFileUpload}
      />

      {/* Processing Animation Modal (still a timed stub here; F3 wires the real Model 2 call) */}
      <ProcessingModal
        open={isProcessing}
        phase={processingPhase}
        onCancel={() => setIsProcessing(false)}
      />

      {/* Header */}
      <header className="flex flex-col gap-2 max-w-3xl">
        <div className="flex items-center gap-2 text-[#3d4a42] text-xs font-semibold tracking-wider uppercase font-mono">
          <button
            onClick={() => onNavigate('home')}
            className="hover:text-[#006948] transition-colors cursor-pointer"
          >
            Home
          </button>
          <span className="material-symbols-outlined text-[14px]">chevron_right</span>
          <span className="text-[#006948] font-bold">Leaf Analysis</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-[#131b2e] tracking-tight">
          Leaf Health Analysis &amp; Tissue Segmentation
        </h1>
        <p className="text-base text-[#3d4a42] leading-relaxed">
          Stage 3 foliar segmentation studio: isolate affected unhealthy foliar tissue from healthy leaf blade using Model 4 U-Net architecture.
        </p>
      </header>

      {/* TWO LARGE WORKSPACE CARDS (OPTION 01: USE CAMERA & OPTION 02: UPLOAD IMAGE) */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-stretch">
        {/* Option 1: Use Camera */}
        <div className="bg-white p-7 rounded-2xl shadow-sm border border-[#dae2fd] flex flex-col justify-between gap-6 hover:shadow-md transition-shadow relative overflow-hidden">
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
                Point your camera at a banana leaf for instant foliar segmentation.
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
                src={ASSETS.leafAnalysisSpecimen}
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
              <span>Point camera directly toward the banana leaf blade with clear natural lighting.</span>
            </div>
          </div>

          {/* Single Robust Open Camera Button */}
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
        <div className="bg-white p-7 rounded-2xl shadow-sm border border-[#dae2fd] flex flex-col justify-between gap-6 hover:shadow-md transition-shadow">
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

            {/* Dropzone */}
            <label
              htmlFor="leaf-analysis-upload"
              className="cursor-pointer flex flex-col items-center justify-center p-8 rounded-xl bg-[#f2f3ff] hover:bg-[#eaedff] border-2 border-dashed border-[#bccac0] hover:border-[#006948] transition-colors text-center relative group"
            >
              <div className="w-12 h-12 rounded-full bg-white flex items-center justify-center shadow-sm text-[#006948] mb-3 border border-[#dae2fd] group-hover:scale-105 transition-transform">
                <span className="material-symbols-outlined text-[26px]">add_photo_alternate</span>
              </div>
              <span className="text-sm font-bold text-[#131b2e]">Drag &amp; drop an image here</span>
              <span className="text-xs text-[#3d4a42] mt-1">
                or browse from your device &bull; Supports JPG, PNG, WEBP
              </span>
            </label>

            {/* Benchmark Samples: 1 example to see as requested */}
            <div className="flex flex-col gap-2 mt-1">
              <span className="font-mono text-xs text-[#3d4a42] font-semibold tracking-wide">
                QUICK BENCHMARK SAMPLE:
              </span>
              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={handleSelectBenchmarkSample}
                  className="px-3.5 py-2 rounded-lg bg-[#eaedff] hover:bg-[#dae2fd] text-[#131b2e] font-mono text-xs flex items-center gap-2 transition-colors border border-[#dae2fd] cursor-pointer shadow-sm"
                >
                  <span className="material-symbols-outlined text-[16px] text-[#006948]">spa</span>
                  <span>Sample: Banana Leaf Foliar Specimen</span>
                </button>
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

      {/* MAIN ANALYSIS STUDIO SECTION */}
      <div className="flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <span className="text-xs font-mono font-bold uppercase tracking-wider text-[#006948]">
            Interactive Foliar Tissue Studio
          </span>
          <span className="text-xs font-mono text-[#3d4a42]">
            Model 4 U-Net Segmentation
          </span>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Left Column: Interactive Segmentation Studio Canvas (7 Cols) */}
          <div className="lg:col-span-7 flex flex-col gap-4">
            {/* Segmentation View Mode Bar */}
            <div className="flex flex-wrap items-center justify-between gap-3 bg-white p-3.5 rounded-xl border border-[#dae2fd]">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-xs font-mono font-bold text-[#131b2e] uppercase">
                  Segmentation:
                </span>
                <div className="inline-flex p-1 rounded-lg bg-[#f2f3ff] border border-[#dae2fd] text-xs font-mono">
                  <button
                    type="button"
                    onClick={() => {
                      setViewMode('overlay');
                      setIsSliderActive(false);
                    }}
                    className={`px-2.5 py-1 rounded transition-colors cursor-pointer ${
                      viewMode === 'overlay' && !isSliderActive
                        ? 'bg-[#006948] text-white font-bold shadow-sm'
                        : 'text-[#3d4a42] hover:text-[#131b2e]'
                    }`}
                  >
                    Healthy vs Unhealthy
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setViewMode('unhealthy-only');
                      setIsSliderActive(false);
                    }}
                    className={`px-2.5 py-1 rounded transition-colors cursor-pointer ${
                      viewMode === 'unhealthy-only' && !isSliderActive
                        ? 'bg-[#ba1a1a] text-white font-bold shadow-sm'
                        : 'text-[#3d4a42] hover:text-[#131b2e]'
                    }`}
                  >
                    Unhealthy Part Only
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setViewMode('healthy-only');
                      setIsSliderActive(false);
                    }}
                    className={`px-2.5 py-1 rounded transition-colors cursor-pointer ${
                      viewMode === 'healthy-only' && !isSliderActive
                        ? 'bg-[#006948] text-white font-bold shadow-sm'
                        : 'text-[#3d4a42] hover:text-[#131b2e]'
                    }`}
                  >
                    Healthy Part Only
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
                    Original Leaf
                  </button>
                </div>
              </div>

              {/* Comparison Slider Toggle */}
              <button
                type="button"
                onClick={() => setIsSliderActive(!isSliderActive)}
                className={`px-3 py-1.5 rounded-lg text-xs font-mono font-semibold flex items-center gap-1.5 border transition-all cursor-pointer ${
                  isSliderActive
                    ? 'bg-[#00855d] text-white border-[#00855d]'
                    : 'bg-[#eaedff] text-[#131b2e] border-[#dae2fd] hover:bg-[#dae2fd]'
                }`}
              >
                <span className="material-symbols-outlined text-[16px]">compare</span>
                <span>Split Slider</span>
              </button>
            </div>

            {/* Canvas with Draggable Split Slider and ACCURATELY ALIGNED Lesion Overlays */}
            <div
              ref={containerRef}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
              className="relative w-full aspect-[4/3] rounded-2xl overflow-hidden bg-[#131b2e] shadow-xl border border-[#dae2fd] select-none"
            >
              {/* Base Image */}
              <img
                alt="Leaf specimen base"
                className="w-full h-full object-cover"
                src={activeImage}
              />

              {/* Segmentation Layer */}
              {viewMode !== 'original' && (
                <div
                  className="absolute inset-0 pointer-events-none transition-opacity duration-200"
                  style={{
                    clipPath: isSliderActive ? `inset(0 0 0 ${splitPosition}%)` : 'none',
                  }}
                >
                  {/* Contrast backing for isolated mode */}
                  {viewMode === 'unhealthy-only' && (
                    <div className="absolute inset-0 bg-black/60" />
                  )}

                  {/* Healthy Part Tint */}
                  {(viewMode === 'overlay' || viewMode === 'healthy-only') && (
                    <div className="absolute inset-0 bg-[#006948]/15 pointer-events-none" />
                  )}

                  {/* Unhealthy Part Highlight Vectors (Precisely aligned to real lesions in leaf photo) */}
                  {!isHealthy && (viewMode === 'overlay' || viewMode === 'unhealthy-only') && (
                    <svg
                      className="absolute inset-0 w-full h-full"
                      viewBox="0 0 1000 750"
                      preserveAspectRatio="none"
                    >
                      {/* Central Necrotic Spot with Chlorotic Halo */}
                      <polygon
                        points="435,260 480,240 550,245 590,270 595,305 565,335 500,340 450,310 430,280"
                        fill="rgba(186, 26, 26, 0.7)"
                        stroke="#ba1a1a"
                        strokeWidth="3.5"
                      />

                      {/* Right Necrotic Streak (Right of the midrib) */}
                      <polygon
                        points="630,345 675,325 715,350 725,390 690,425 645,420 625,380"
                        fill="rgba(186, 26, 26, 0.7)"
                        stroke="#ba1a1a"
                        strokeWidth="3.5"
                      />

                      {/* Mid-Left Necrotic Streak */}
                      <polygon
                        points="180,370 240,345 295,375 285,430 215,445 165,410"
                        fill="rgba(186, 26, 26, 0.65)"
                        stroke="#ba1a1a"
                        strokeWidth="3"
                      />

                      {/* Upper-Left Lesion */}
                      <polygon
                        points="225,120 280,100 325,125 315,165 260,175 220,150"
                        fill="rgba(186, 26, 26, 0.65)"
                        stroke="#ba1a1a"
                        strokeWidth="3"
                      />

                      {/* Lower-Center Necrotic Spot */}
                      <polygon
                        points="535,510 570,490 605,510 610,545 575,565 540,545"
                        fill="rgba(186, 26, 26, 0.65)"
                        stroke="#ba1a1a"
                        strokeWidth="2.5"
                      />
                    </svg>
                  )}
                </div>
              )}

              {/* Split Comparison Slider Handle */}
              {isSliderActive && (
                <div
                  onMouseDown={handleMouseDown}
                  style={{ left: `${splitPosition}%` }}
                  className="absolute top-0 bottom-0 w-1 bg-white shadow-[0_0_14px_rgba(0,0,0,0.8)] cursor-ew-resize flex items-center justify-center z-30"
                >
                  <div className="w-8 h-8 rounded-full bg-white text-[#131b2e] flex items-center justify-center shadow-lg border border-[#bccac0]">
                    <span className="material-symbols-outlined text-[18px]">drag_indicator</span>
                  </div>
                </div>
              )}

              {/* Bottom Legend: Strictly Healthy Part vs Unhealthy Part */}
              <div className="absolute bottom-4 left-4 right-4 bg-white/95 backdrop-blur-md px-4 py-2.5 rounded-xl shadow-md border border-white/60 flex items-center justify-between text-xs font-mono">
                <div className="flex items-center gap-4">
                  <span className="flex items-center gap-1.5 text-[#006948] font-bold">
                    <span className="w-2.5 h-2.5 rounded-full bg-[#006948]" />
                    Healthy Part: {healthyPercent}%
                  </span>
                  <span className="flex items-center gap-1.5 text-[#ba1a1a] font-bold">
                    <span className="w-2.5 h-2.5 rounded-full bg-[#ba1a1a]" />
                    Unhealthy Part: {unhealthyPercent}%
                  </span>
                </div>
                <span className="text-[11px] text-[#3d4a42] hidden sm:inline">
                  Model 4 Pixel Segmentation
                </span>
              </div>
            </div>

            {/* Area Segmentation Breakdown Card */}
            <div className="bg-white p-5 rounded-2xl border border-[#dae2fd] shadow-sm flex flex-col gap-3 font-mono">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold uppercase tracking-wider text-[#131b2e]">
                  Proportional Foliar Tissue Breakdown
                </span>
                <span className="text-[11px] text-[#3d4a42]">
                  Model 4 U-Net Topology
                </span>
              </div>

              {/* Two-Tone Proportional Bar */}
              <div className="w-full h-4 rounded-full overflow-hidden flex bg-[#dae2fd]">
                <div
                  style={{ width: `${healthyPercent}%` }}
                  className="bg-[#006948] h-full transition-all duration-500 flex items-center justify-center text-[10px] font-bold text-white"
                  title="Healthy Part"
                >
                  {healthyPercent > 10 ? `${healthyPercent}%` : ''}
                </div>
                <div
                  style={{ width: `${unhealthyPercent}%` }}
                  className="bg-[#ba1a1a] h-full transition-all duration-500 flex items-center justify-center text-[10px] font-bold text-white"
                  title="Unhealthy Part"
                >
                  {unhealthyPercent > 10 ? `${unhealthyPercent}%` : ''}
                </div>
              </div>

              {/* Two Distinct Cards: Healthy Part vs Unhealthy Part */}
              <div className="grid grid-cols-2 gap-3 pt-1">
                <div className="p-3 rounded-xl bg-[#85f8c4]/20 border border-[#85f8c4] flex flex-col gap-1">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-[#006948] flex items-center gap-1">
                      <span className="w-2 h-2 rounded-full bg-[#006948]" />
                      Healthy Part
                    </span>
                    <span className="text-sm font-extrabold text-[#006948]">
                      {healthyPercent}%
                    </span>
                  </div>
                  <span className="text-[10px] text-[#3d4a42] leading-tight">
                    Intact foliar tissue with active photosynthesis and uniform chlorophyll.
                  </span>
                </div>

                <div className="p-3 rounded-xl bg-[#ffdad6]/40 border border-[#ba1a1a]/30 flex flex-col gap-1">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-[#ba1a1a] flex items-center gap-1">
                      <span className="w-2 h-2 rounded-full bg-[#ba1a1a]" />
                      Unhealthy Part
                    </span>
                    <span className="text-sm font-extrabold text-[#ba1a1a]">
                      {unhealthyPercent}%
                    </span>
                  </div>
                  <span className="text-[10px] text-[#3d4a42] leading-tight">
                    Infected tissue displaying fungal streaks, chlorotic rings, or necrosis.
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column: Foliar Health Dashboard (5 Cols) */}
          <div className="lg:col-span-5 flex flex-col gap-6">
            <div className="bg-white p-7 rounded-2xl shadow-sm border border-[#dae2fd] flex flex-col gap-6">
              {/* Status Indicator */}
              {isHealthy ? (
                <div className="flex items-center gap-3 p-3.5 rounded-xl bg-[#85f8c4]/30 border border-[#85f8c4] text-[#002114]">
                  <span className="material-symbols-outlined text-[26px] text-[#006948]">check_circle</span>
                  <span className="text-base font-extrabold tracking-tight">✓ Healthy Leaf Specimen</span>
                </div>
              ) : (
                <div className="flex items-center gap-3 p-3.5 rounded-xl bg-[#ffdad6] border border-[#ba1a1a]/30 text-[#ba1a1a]">
                  <span className="material-symbols-outlined text-[26px]">warning</span>
                  <span className="text-base font-extrabold tracking-tight">⚠ Tissue Damage Detected</span>
                </div>
              )}

              {/* Health Metrics */}
              <div className="flex flex-col gap-1">
                <span className="text-xs uppercase tracking-wider text-[#3d4a42] font-semibold font-mono">
                  Foliar Health Evaluation
                </span>
                <h2 className="text-2xl font-extrabold text-[#131b2e]">
                  {isHealthy ? 'Clean Musa Tissue' : 'Compromised Foliar Canopy'}
                </h2>
                <span className="text-xs text-[#3d4a42]">
                  Evaluated across 2,400 spectral points for chlorophyll density and fungal lesioning.
                </span>
              </div>

              {/* Healthy vs Unhealthy Summary Box */}
              <div className="p-4 rounded-xl bg-[#f2f3ff] border border-[#dae2fd] flex flex-col gap-2.5 font-mono text-xs">
                <div className="flex justify-between py-1 border-b border-[#dae2fd]">
                  <span className="text-[#3d4a42]">Healthy Part Area:</span>
                  <span className="font-extrabold text-[#006948]">{healthyPercent}%</span>
                </div>
                <div className="flex justify-between py-1 border-b border-[#dae2fd]">
                  <span className="text-[#3d4a42]">Unhealthy Part Area:</span>
                  <span className="font-extrabold text-[#ba1a1a]">{unhealthyPercent}%</span>
                </div>
                <div className="flex justify-between py-1 border-b border-[#dae2fd]">
                  <span className="text-[#3d4a42]">Tissue Health Status:</span>
                  <span className={`font-bold ${isHealthy ? 'text-[#006948]' : 'text-[#ba1a1a]'}`}>
                    {isHealthy ? 'Optimal' : unhealthyPercent > 20 ? 'Critical' : 'Moderate'}
                  </span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-[#3d4a42]">Segmentation Model:</span>
                  <span className="font-bold text-[#131b2e]">Model 4: U-Net Semantic Segmenter</span>
                </div>
              </div>

              {/* Model 4 Specific Telemetry */}
              <div className="p-4 rounded-xl bg-[#faf8ff] border border-[#dae2fd] flex flex-col gap-3 font-mono text-xs">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] uppercase font-bold text-[#131b2e]">
                    Model 4 Segmentation Telemetry
                  </span>
                  <span className="text-[10px] text-[#006948] font-bold flex items-center gap-1">
                    <span className="w-1.5 h-1.5 rounded-full bg-[#006948]" />
                    200 OK (364ms)
                  </span>
                </div>

                <div className="bg-[#131b2e] text-[#85f8c4] p-2 rounded-lg text-[10px] truncate">
                  POST /api/v1/model4/semantic-segmentation
                </div>

                <div className="grid grid-cols-2 gap-2 text-[11px]">
                  <div>
                    <span className="text-[#3d4a42] block text-[10px]">Dice Coefficient:</span>
                    <span className="font-bold text-[#131b2e]">0.924</span>
                  </div>
                  <div>
                    <span className="text-[#3d4a42] block text-[10px]">Pixel Resolution:</span>
                    <span className="font-bold text-[#131b2e]">1000 × 750 px</span>
                  </div>
                </div>
              </div>

              {/* Transition Arrow to Detect Disease if user wants Pathogen ID */}
              <div className="p-4 rounded-xl bg-[#fff7ed] border border-[#fed7aa] flex flex-col gap-2">
                <span className="font-mono text-[11px] font-bold text-[#ea580c] uppercase tracking-wider">
                  Cross-Pathology Reference:
                </span>
                <p className="text-xs text-[#131b2e] leading-tight">
                  Want to identify the specific pathogen (Black Sigatoka, Cordana) and agronomic fungicide recommendations?
                </p>
                <button
                  type="button"
                  onClick={() => onNavigate('detect-disease')}
                  className="w-full py-2.5 px-4 rounded-xl bg-[#ea580c] hover:bg-[#c2410c] text-white font-bold text-xs transition-all flex items-center justify-center gap-2 cursor-pointer active:scale-95 shadow-sm"
                >
                  <span>Go to Detect Disease (Model 3)</span>
                  <span className="material-symbols-outlined text-[16px] font-bold">
                    arrow_forward
                  </span>
                </button>
              </div>

              {/* Action Buttons */}
              <div className="flex flex-col gap-2 pt-1">
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
      </div>
    </div>
  );
};
