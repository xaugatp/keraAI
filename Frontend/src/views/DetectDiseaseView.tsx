import React, { useState, useRef } from 'react';
import { ViewTab, Stage2Diagnosis } from '../types';
import { sampleDiagnoses, ASSETS } from '../data/mockData';
import { CameraModal } from '../components/CameraModal';
import { ProcessingModal } from '../components/ProcessingModal';

interface DetectDiseaseViewProps {
  onNavigate: (tab: ViewTab) => void;
  selectedDiagnosisIndex?: number;
}

export const DetectDiseaseView: React.FC<DetectDiseaseViewProps> = ({
  onNavigate,
  selectedDiagnosisIndex = 0,
}) => {
  const [selectedIdx, setSelectedIdx] = useState<number>(selectedDiagnosisIndex);
  const [customImage, setCustomImage] = useState<string | null>(null);
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [segmentationOverlayMode, setSegmentationOverlayMode] = useState<'all' | 'lesions' | 'heatmap' | 'original'>('all');
  const [overlayOpacity, setOverlayOpacity] = useState<number>(75);
  const [activeApiModal, setActiveApiModal] = useState<'model3' | 'model4' | null>(null);
  const [copiedKey, setCopiedKey] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const diagnosis: Stage2Diagnosis = sampleDiagnoses[selectedIdx] || sampleDiagnoses[0];
  const isHealthy = diagnosis.severity === 'healthy';

  const handleCustomCapture = (imgDataUrl: string) => {
    setCustomImage(imgDataUrl);
    setIsProcessing(true);
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const isHealthyName = file.name.toLowerCase().includes('healthy') || file.name.toLowerCase().includes('clean');
      const isCordana = file.name.toLowerCase().includes('cordana');
      const isYellow = file.name.toLowerCase().includes('yellow');
      const reader = new FileReader();
      reader.onload = (event) => {
        if (event.target?.result) {
          setCustomImage(event.target.result as string);
          if (isHealthyName) {
            setSelectedIdx(1);
          } else if (isCordana) {
            setSelectedIdx(3);
          } else if (isYellow) {
            setSelectedIdx(2);
          } else {
            setSelectedIdx(0);
          }
          setIsProcessing(true);
        }
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSelectBenchmarkSample = () => {
    setCustomImage(null);
    setSelectedIdx(0);
    setIsProcessing(true);
  };

  const copyToClipboard = (text: string, key: string) => {
    navigator.clipboard?.writeText(text);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(null), 2000);
  };

  const activeSpecimenImage = customImage || diagnosis.imageUrl;

  return (
    <div className="w-full max-w-7xl mx-auto px-6 lg:px-12 py-10 flex flex-col gap-8">
      {/* Universal Camera Modal */}
      <CameraModal
        isOpen={isCameraOpen}
        mode="leaf"
        fallbackImage={ASSETS.sigatokaLeafSample}
        onClose={() => setIsCameraOpen(false)}
        onCapture={handleCustomCapture}
      />

      {/* Hidden File Upload Input */}
      <input
        ref={fileInputRef}
        id="detect-disease-upload"
        type="file"
        accept="image/*"
        className="sr-only"
        onChange={handleFileUpload}
      />

      {/* Asynchronous Processing Modal */}
      <ProcessingModal
        isOpen={isProcessing}
        mode="leaf"
        onComplete={() => setIsProcessing(false)}
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
          <span className="text-[#006948] font-bold">Detect Disease</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-[#131b2e] tracking-tight">
          Detect Disease &amp; Affected Area
        </h1>
        <p className="text-base text-[#3d4a42] leading-relaxed">
          Stage 2 foliar pathology diagnostic workspace: classify disease pathogens (Model 3) and detect affected necrotic lesion boundaries with agronomic treatments.
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
                Point your camera at a banana leaf for instant disease identification.
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
                src={ASSETS.sigatokaLeafSample}
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
              <span>Position leaf specimen with visible spots or chlorosis inside targeting frame.</span>
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
              htmlFor="detect-disease-upload"
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

            {/* Benchmark Samples: 1 example to see */}
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
                  <span className="material-symbols-outlined text-[16px] text-[#ba1a1a]">coronavirus</span>
                  <span>Sample: Black Sigatoka Disease Specimen</span>
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

      {/* MAIN DIAGNOSTIC WORKSPACE SECTION */}
      <div className="flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <span className="text-xs font-mono font-bold uppercase tracking-wider text-[#ba1a1a]">
            Diagnostic Evaluation &amp; Pathogen Isolation
          </span>
          <span className="text-xs font-mono text-[#3d4a42]">
            Model 3 Pathology CNN
          </span>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Left Column: Interactive Leaf Specimen & Disease Lesion Overlay Canvas (7 Cols) */}
          <div className="lg:col-span-7 flex flex-col gap-4">
            {/* Lesion Overlay Controls Bar */}
            <div className="flex flex-wrap items-center justify-between gap-3 bg-white p-3.5 rounded-xl border border-[#dae2fd]">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-xs font-mono font-bold text-[#131b2e] uppercase">
                  Affected Area View:
                </span>
                <div className="inline-flex p-1 rounded-lg bg-[#f2f3ff] border border-[#dae2fd] text-xs font-mono">
                  <button
                    type="button"
                    onClick={() => setSegmentationOverlayMode('all')}
                    className={`px-2.5 py-1 rounded transition-colors cursor-pointer ${
                      segmentationOverlayMode === 'all'
                        ? 'bg-[#ba1a1a] text-white font-bold shadow-sm'
                        : 'text-[#3d4a42] hover:text-[#131b2e]'
                    }`}
                  >
                    Detected Lesions
                  </button>
                  <button
                    type="button"
                    onClick={() => setSegmentationOverlayMode('lesions')}
                    className={`px-2.5 py-1 rounded transition-colors cursor-pointer ${
                      segmentationOverlayMode === 'lesions'
                        ? 'bg-[#a36700] text-white font-bold shadow-sm'
                        : 'text-[#3d4a42] hover:text-[#131b2e]'
                    }`}
                  >
                    Bounding Boxes
                  </button>
                  <button
                    type="button"
                    onClick={() => setSegmentationOverlayMode('heatmap')}
                    className={`px-2.5 py-1 rounded transition-colors cursor-pointer ${
                      segmentationOverlayMode === 'heatmap'
                        ? 'bg-[#006948] text-white font-bold shadow-sm'
                        : 'text-[#3d4a42] hover:text-[#131b2e]'
                    }`}
                  >
                    Thermal Heatmap
                  </button>
                  <button
                    type="button"
                    onClick={() => setSegmentationOverlayMode('original')}
                    className={`px-2.5 py-1 rounded transition-colors cursor-pointer ${
                      segmentationOverlayMode === 'original'
                        ? 'bg-[#131b2e] text-white font-bold shadow-sm'
                        : 'text-[#3d4a42] hover:text-[#131b2e]'
                    }`}
                  >
                    Original Leaf
                  </button>
                </div>
              </div>

              {/* Opacity Slider */}
              {segmentationOverlayMode !== 'original' && (
                <div className="flex items-center gap-2 font-mono text-xs text-[#3d4a42]">
                  <span>Opacity:</span>
                  <input
                    type="range"
                    min="20"
                    max="100"
                    value={overlayOpacity}
                    onChange={(e) => setOverlayOpacity(Number(e.target.value))}
                    className="w-20 accent-[#006948] cursor-pointer"
                  />
                  <span className="w-8 font-bold text-[#131b2e]">{overlayOpacity}%</span>
                </div>
              )}
            </div>

            {/* Interactive Canvas Viewport */}
            <div className="relative w-full aspect-[4/3] rounded-2xl overflow-hidden bg-[#131b2e] shadow-xl border border-[#dae2fd] select-none">
              {/* Base Leaf Specimen */}
              <img
                alt="Analyzed leaf specimen"
                className="w-full h-full object-cover"
                src={activeSpecimenImage}
              />

              {/* SVG Diagnostic Lesions & Affected Areas */}
              {segmentationOverlayMode !== 'original' && !isHealthy && (
                <svg
                  className="absolute inset-0 w-full h-full pointer-events-none transition-opacity duration-200"
                  style={{ opacity: overlayOpacity / 100 }}
                  viewBox="0 0 1000 750"
                  preserveAspectRatio="none"
                >
                  <defs>
                    <radialGradient id="lesionHeatGrad1" cx="50%" cy="50%" r="50%">
                      <stop offset="0%" stopColor="#ba1a1a" stopOpacity="0.95" />
                      <stop offset="70%" stopColor="#ffb4ab" stopOpacity="0.5" />
                      <stop offset="100%" stopColor="#ba1a1a" stopOpacity="0.1" />
                    </radialGradient>
                    <radialGradient id="lesionHeatGrad2" cx="50%" cy="50%" r="50%">
                      <stop offset="0%" stopColor="#ffb95f" stopOpacity="0.9" />
                      <stop offset="80%" stopColor="#ba1a1a" stopOpacity="0.4" />
                      <stop offset="100%" stopColor="#ffb95f" stopOpacity="0.05" />
                    </radialGradient>
                  </defs>

                  {/* Lesion 1 (Apical Necrosis) */}
                  {(segmentationOverlayMode === 'all' || segmentationOverlayMode === 'heatmap') && (
                    <ellipse
                      cx="410"
                      cy="260"
                      rx="110"
                      ry="90"
                      fill={segmentationOverlayMode === 'heatmap' ? 'url(#lesionHeatGrad1)' : 'rgba(186, 26, 26, 0.65)'}
                      stroke="#ba1a1a"
                      strokeWidth="3"
                    />
                  )}

                  {/* Lesion 2 (Mid-rib streak) */}
                  {(segmentationOverlayMode === 'all' || segmentationOverlayMode === 'heatmap') && (
                    <ellipse
                      cx="620"
                      cy="480"
                      rx="95"
                      ry="80"
                      fill={segmentationOverlayMode === 'heatmap' ? 'url(#lesionHeatGrad2)' : 'rgba(186, 26, 26, 0.6)'}
                      stroke="#ba1a1a"
                      strokeWidth="3"
                    />
                  )}

                  {/* Lesion 3 (Basal Margin chlorosis) */}
                  {(segmentationOverlayMode === 'all' || segmentationOverlayMode === 'heatmap') && (
                    <polygon
                      points="120,440 210,410 260,470 230,580 140,610 95,510"
                      fill="rgba(255, 185, 95, 0.65)"
                      stroke="#a36700"
                      strokeWidth="2.5"
                    />
                  )}

                  {/* Bounding Boxes with Scientific Reticles */}
                  {(segmentationOverlayMode === 'all' || segmentationOverlayMode === 'lesions') && (
                    <>
                      <rect
                        x="290"
                        y="160"
                        width="240"
                        height="200"
                        fill="none"
                        stroke="#ba1a1a"
                        strokeWidth="2"
                        strokeDasharray="6,4"
                      />
                      <rect
                        x="515"
                        y="390"
                        width="210"
                        height="180"
                        fill="none"
                        stroke="#ba1a1a"
                        strokeWidth="2"
                        strokeDasharray="6,4"
                      />
                    </>
                  )}
                </svg>
              )}

              {/* Clean Specimen Badge if Healthy */}
              {isHealthy && (
                <div className="absolute inset-0 bg-[#006948]/10 flex items-center justify-center pointer-events-none">
                  <div className="bg-white/95 backdrop-blur-md px-6 py-3 rounded-2xl shadow-xl border border-[#85f8c4] flex items-center gap-2.5">
                    <span className="material-symbols-outlined text-[#006948] text-[28px]">verified</span>
                    <span className="font-bold text-[#131b2e] text-sm">
                      No Pathogen Lesions Detected &bull; 100% Intact Leaf
                    </span>
                  </div>
                </div>
              )}

              {/* Viewport Overlay HUD Bar */}
              <div className="absolute bottom-4 left-4 right-4 bg-white/95 backdrop-blur-md px-4 py-2.5 rounded-xl shadow-md border border-white/60 flex items-center justify-between text-xs font-mono">
                <div className="flex items-center gap-3">
                  <span className="flex items-center gap-1 font-bold text-[#131b2e]">
                    <span className={`w-2.5 h-2.5 rounded-full ${isHealthy ? 'bg-[#006948]' : 'bg-[#ba1a1a]'}`} />
                    {diagnosis.pathogenCommon}
                  </span>
                  <span className="text-[#3d4a42]">Confidence: {diagnosis.certainty}%</span>
                </div>
                <span className="text-[#ba1a1a] font-bold">
                  Affected Area: {isHealthy ? '0.0%' : `${diagnosis.infectedAreaPercent}%`}
                </span>
              </div>
            </div>

            {/* Diagnostic Telemetry Strip */}
            <div className="bg-white p-4 rounded-xl border border-[#dae2fd] shadow-sm grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
              <div>
                <span className="text-[#3d4a42] block text-[10px] uppercase font-bold">Model 3 Status</span>
                <span className="font-bold text-[#006948]">200 OK (288ms)</span>
              </div>
              <div>
                <span className="text-[#3d4a42] block text-[10px] uppercase font-bold">Pathogen Class</span>
                <span className="font-bold text-[#131b2e] truncate">{diagnosis.pathogenCommon}</span>
              </div>
              <div>
                <span className="text-[#3d4a42] block text-[10px] uppercase font-bold">Affected Surface</span>
                <span className={`font-bold ${isHealthy ? 'text-[#006948]' : 'text-[#ba1a1a]'}`}>
                  {isHealthy ? '0.0% (Clean)' : `${diagnosis.infectedAreaPercent}%`}
                </span>
              </div>
              <div>
                <span className="text-[#3d4a42] block text-[10px] uppercase font-bold">Inference Type</span>
                <span className="font-bold text-[#131b2e]">Multi-Class CNN</span>
              </div>
            </div>
          </div>

          {/* Right Column: Pathogen Details & Recommended Action Plan (5 Cols) */}
          <div className="lg:col-span-5 flex flex-col gap-6">
            <div className="bg-white p-7 rounded-2xl shadow-sm border border-[#dae2fd] flex flex-col gap-6">
              {/* Disease Severity Banner */}
              <div
                className={`flex items-center gap-3 p-3.5 rounded-xl border ${
                  isHealthy
                    ? 'bg-[#85f8c4]/30 border-[#85f8c4] text-[#002114]'
                    : diagnosis.severity === 'severe'
                    ? 'bg-[#ffdad6] border-[#ba1a1a]/30 text-[#ba1a1a]'
                    : 'bg-[#ffeed2] border-[#825100]/30 text-[#825100]'
                }`}
              >
                <span className="material-symbols-outlined text-[26px]">
                  {isHealthy ? 'check_circle' : 'coronavirus'}
                </span>
                <div className="flex flex-col">
                  <span className="text-base font-extrabold tracking-tight">
                    {diagnosis.pathogenCommon}
                  </span>
                  <span className="text-xs font-medium">
                    {isHealthy ? 'Specimen is free of foliar pathogens' : `Severity: ${diagnosis.severity.toUpperCase()}`}
                  </span>
                </div>
              </div>

              {/* Scientific Classification Information */}
              <div className="flex flex-col gap-1">
                <span className="text-xs uppercase tracking-wider text-[#3d4a42] font-semibold font-mono">
                  Scientific Diagnostics
                </span>
                <h2 className="text-2xl font-extrabold text-[#131b2e]">
                  {diagnosis.pathogenCommon}
                </h2>
                <p className="text-xs text-[#3d4a42] italic">
                  Causal Agent: {diagnosis.pathogenScientific}
                </p>
                <p className="text-xs text-[#3d4a42] mt-1 leading-relaxed">
                  {diagnosis.classificationClass} — Spectral PSI Index: {diagnosis.psiSeverityIndex}/100.
                </p>
              </div>

              {/* Affected Area & Confidence Metrics */}
              <div className="p-4 rounded-xl bg-[#f2f3ff] border border-[#dae2fd] flex flex-col gap-2.5 font-mono text-xs">
                <div className="flex justify-between py-1 border-b border-[#dae2fd]">
                  <span className="text-[#3d4a42]">Model Confidence:</span>
                  <span className="font-extrabold text-[#006948]">{diagnosis.certainty}%</span>
                </div>
                <div className="flex justify-between py-1 border-b border-[#dae2fd]">
                  <span className="text-[#3d4a42]">Affected Area (Necrotic):</span>
                  <span className="font-extrabold text-[#ba1a1a]">
                    {isHealthy ? '0.0%' : `${diagnosis.infectedAreaPercent}%`}
                  </span>
                </div>
                <div className="flex justify-between py-1 border-b border-[#dae2fd]">
                  <span className="text-[#3d4a42]">Recommended Action:</span>
                  <span className="font-bold text-[#131b2e]">
                    {isHealthy ? 'Regular Monitoring' : 'Immediate Foliar Intervention'}
                  </span>
                </div>
              </div>

              {/* Agronomic Treatment Actions */}
              <div className="flex flex-col gap-2.5">
                <span className="text-xs uppercase tracking-wider text-[#131b2e] font-bold font-mono">
                  Agronomic Management Protocol
                </span>
                <div className="flex flex-col gap-2">
                  {[
                    diagnosis.recommendations.chemical,
                    diagnosis.recommendations.cultural,
                    `Epidemic Risk: ${diagnosis.recommendations.epidemicRisk}`,
                    `Re-Inspection Schedule: ${diagnosis.recommendations.reInspectionHours}`,
                  ].map((treatment, idx) => (
                    <div
                      key={idx}
                      className="p-3 rounded-xl bg-[#faf8ff] border border-[#dae2fd] flex items-start gap-2.5 text-xs text-[#131b2e]"
                    >
                      <span className="material-symbols-outlined text-[#006948] text-[18px] shrink-0 mt-0.5">
                        task_alt
                      </span>
                      <span className="leading-snug">{treatment}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* PROMINENT NEXT STEP ARROW TO STAGE 3 (LEAF ANALYSIS & TISSUE SEGMENTATION) */}
              <div className="p-4 rounded-xl bg-[#e2e7ff] border-2 border-[#006948] flex flex-col gap-2.5 shadow-sm">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-[11px] font-bold text-[#006948] uppercase tracking-wider">
                    Next Step in Pipeline:
                  </span>
                  <span className="px-2 py-0.5 rounded-full bg-[#006948] text-white font-mono text-[10px] font-bold">
                    STAGE 03
                  </span>
                </div>
                <p className="text-xs text-[#131b2e] font-medium leading-tight">
                  Want to segment healthy vs. unhealthy foliar tissue with draggable split slider comparisons and exact area percentages?
                </p>
                <button
                  type="button"
                  onClick={() => onNavigate('leaf-analysis')}
                  className="w-full py-3 px-4 rounded-xl bg-[#006948] hover:bg-[#00855d] text-white font-bold text-xs transition-all flex items-center justify-center gap-2 shadow cursor-pointer active:scale-95 mt-1"
                >
                  <span>Segment Affected Area &amp; Leaf Analysis</span>
                  <span className="material-symbols-outlined text-[18px] font-bold animate-pulse">
                    arrow_forward
                  </span>
                </button>
              </div>

              {/* Independent Model 3 & Model 4 Endpoints */}
              <div className="border border-[#dae2fd] rounded-xl overflow-hidden bg-white">
                <div className="p-3.5 bg-[#eaedff] flex items-center justify-between font-mono text-xs font-bold text-[#131b2e]">
                  <div className="flex items-center gap-2">
                    <span className="material-symbols-outlined text-[#006948] text-[18px]">api</span>
                    <span>FastAPI Endpoints</span>
                  </div>
                </div>

                <div className="p-4 flex flex-col gap-3 font-mono text-xs">
                  {/* Model 3 Card */}
                  <div className="p-3 rounded-lg bg-[#faf8ff] border border-[#dae2fd] flex flex-col gap-2">
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-[#131b2e]">Model 3: Disease Classifier</span>
                      <span className="text-[10px] text-[#006948] font-bold">
                        200 OK ({diagnosis.model3.latencyMs}ms)
                      </span>
                    </div>
                    <div className="bg-[#131b2e] text-[#85f8c4] p-1.5 rounded text-[10px] flex items-center justify-between">
                      <span className="truncate">{diagnosis.model3.endpoint}</span>
                      <button
                        type="button"
                        onClick={() => copyToClipboard(diagnosis.model3.endpoint, 'm3')}
                        className="text-[#bccac0] hover:text-white cursor-pointer ml-1"
                      >
                        <span className="material-symbols-outlined text-[14px]">
                          {copiedKey === 'm3' ? 'check' : 'content_copy'}
                        </span>
                      </button>
                    </div>
                    <button
                      type="button"
                      onClick={() => setActiveApiModal('model3')}
                      className="text-[11px] font-semibold text-[#006948] hover:underline flex items-center gap-1 cursor-pointer"
                    >
                      <span>View Model 3 JSON Schema</span>
                      <span className="material-symbols-outlined text-[13px]">open_in_new</span>
                    </button>
                  </div>

                  {/* Model 4 Card */}
                  <div className="p-3 rounded-lg bg-[#faf8ff] border border-[#dae2fd] flex flex-col gap-2">
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-[#131b2e]">Model 4: Semantic Segmenter</span>
                      <span className="text-[10px] text-[#006948] font-bold">
                        200 OK ({diagnosis.model4.latencyMs}ms)
                      </span>
                    </div>
                    <div className="bg-[#131b2e] text-[#85f8c4] p-1.5 rounded text-[10px] flex items-center justify-between">
                      <span className="truncate">{diagnosis.model4.endpoint}</span>
                      <button
                        type="button"
                        onClick={() => copyToClipboard(diagnosis.model4.endpoint, 'm4')}
                        className="text-[#bccac0] hover:text-white cursor-pointer ml-1"
                      >
                        <span className="material-symbols-outlined text-[14px]">
                          {copiedKey === 'm4' ? 'check' : 'content_copy'}
                        </span>
                      </button>
                    </div>
                    <button
                      type="button"
                      onClick={() => setActiveApiModal('model4')}
                      className="text-[11px] font-semibold text-[#006948] hover:underline flex items-center gap-1 cursor-pointer"
                    >
                      <span>View Model 4 JSON Schema</span>
                      <span className="material-symbols-outlined text-[13px]">open_in_new</span>
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* JSON Schema Modal */}
      {activeApiModal && (
        <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-[#131b2e] text-white w-full max-w-2xl rounded-2xl p-6 shadow-2xl border border-white/20 flex flex-col gap-4 max-h-[85vh] overflow-y-auto">
            <div className="flex items-center justify-between border-b border-white/10 pb-3">
              <span className="font-mono text-sm font-bold text-[#85f8c4]">
                {activeApiModal === 'model3' ? diagnosis.model3.endpoint : diagnosis.model4.endpoint}
              </span>
              <button
                type="button"
                onClick={() => setActiveApiModal(null)}
                className="text-white/70 hover:text-white cursor-pointer"
              >
                <span className="material-symbols-outlined text-[20px]">close</span>
              </button>
            </div>
            <pre className="p-4 bg-[#0a0f18] rounded-xl font-mono text-xs text-[#85f8c4] overflow-x-auto border border-white/10">
              {activeApiModal === 'model3'
                ? diagnosis.model3.responseJson || JSON.stringify(diagnosis.model3, null, 2)
                : diagnosis.model4.responseJson || JSON.stringify(diagnosis.model4, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
};
