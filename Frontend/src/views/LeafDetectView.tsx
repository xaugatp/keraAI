import React, { useState, useRef } from 'react';
import { ViewTab } from '../types';
import { ASSETS, sampleDiagnoses } from '../data/mockData';
import { CameraModal } from '../components/CameraModal';
import { ProcessingModal } from '../components/ProcessingModal';

interface LeafDetectViewProps {
  onNavigate: (tab: ViewTab) => void;
  onSelectLeafDiagnosis?: (diagnosisIndex: number, customImg?: string) => void;
}

export const LeafDetectView: React.FC<LeafDetectViewProps> = ({
  onNavigate,
  onSelectLeafDiagnosis,
}) => {
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [pendingDiagnosisIdx, setPendingDiagnosisIdx] = useState<number>(0);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const nativeCameraInputRef = useRef<HTMLInputElement | null>(null);

  const handleStartAnalysis = (diagnosisIdx: number, customImg?: string) => {
    setPendingDiagnosisIdx(diagnosisIdx);
    if (onSelectLeafDiagnosis) {
      onSelectLeafDiagnosis(diagnosisIdx, customImg);
    }
    setIsProcessing(true);
  };

  const handleProcessingComplete = () => {
    setIsProcessing(false);
    onNavigate('leaf-result');
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const isHealthy = file.name.toLowerCase().includes('healthy') || file.name.toLowerCase().includes('clean');
      const reader = new FileReader();
      reader.onload = (event) => {
        handleStartAnalysis(isHealthy ? 1 : 0, event.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleNativeCameraCapture = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const isHealthy = file.name.toLowerCase().includes('healthy') || file.name.toLowerCase().includes('clean');
      const reader = new FileReader();
      reader.onload = (event) => {
        handleStartAnalysis(isHealthy ? 1 : 0, event.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  return (
    <div className="w-full max-w-7xl mx-auto px-6 lg:px-12 py-10 flex flex-col gap-8">
      {/* Hidden Universal Native Camera Input for iPhone & Android */}
      <input
        ref={nativeCameraInputRef}
        type="file"
        accept="image/*"
        capture="environment"
        className="sr-only"
        onChange={handleNativeCameraCapture}
      />

      {/* Camera Modal */}
      <CameraModal
        isOpen={isCameraOpen}
        mode="leaf"
        fallbackImage={ASSETS.sigatokaLeafSample}
        onClose={() => setIsCameraOpen(false)}
        onCapture={(img) => {
          setIsCameraOpen(false);
          handleStartAnalysis(0, img);
        }}
      />

      {/* Asynchronous Processing Modal */}
      <ProcessingModal
        isOpen={isProcessing}
        mode="leaf"
        onComplete={handleProcessingComplete}
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
          <span className="text-[#006948] font-bold">Detect Disease</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-[#131b2e] tracking-tight">
          Detect Disease &amp; Affected Area Segmentation
        </h1>
        <p className="text-base text-[#3d4a42] leading-relaxed">
          Upload or capture a close-up image of a banana leaf. Evaluates disease presence and segments the affected unhealthy area from the healthy leaf tissue.
        </p>
      </header>

      {/* Two Large Input Modes Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-stretch">
        {/* Option 1: Upload Leaf Image */}
        <div className="bg-white p-7 rounded-2xl shadow-sm border border-[#dae2fd] flex flex-col justify-between gap-6 hover:shadow-md transition-shadow">
          <div className="flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <span className="px-3 py-1 rounded-md bg-[#eaedff] font-mono text-xs font-bold text-[#131b2e]">
                OPTION 01
              </span>
              <span className="material-symbols-outlined text-[#006948] text-[24px]">
                cloud_upload
              </span>
            </div>

            <div>
              <h2 className="text-xl font-bold text-[#131b2e]">Upload Leaf Image</h2>
              <p className="text-xs text-[#3d4a42] mt-1">
                High-throughput analysis of pre-captured macroscopic foliar images.
              </p>
            </div>

            {/* Dropzone */}
            <label
              htmlFor="leaf-image-upload"
              className="cursor-pointer flex flex-col items-center justify-center p-8 rounded-xl bg-[#f2f3ff] hover:bg-[#eaedff] border-2 border-dashed border-[#bccac0] hover:border-[#006948] transition-colors text-center relative group"
            >
              <div className="w-12 h-12 rounded-full bg-white flex items-center justify-center shadow-sm text-[#006948] mb-3 border border-[#dae2fd] group-hover:scale-105 transition-transform">
                <span className="material-symbols-outlined text-[26px]">add_photo_alternate</span>
              </div>
              <span className="text-sm font-bold text-[#131b2e]">Drag &amp; drop a leaf image here</span>
              <span className="text-xs text-[#3d4a42] mt-1">
                or browse from your device &bull; Supports JPG, PNG, WEBP up to 25MB
              </span>
              <input
                ref={fileInputRef}
                id="leaf-image-upload"
                type="file"
                accept="image/*"
                className="sr-only"
                onChange={handleFileUpload}
              />
            </label>

            {/* Quick Benchmark Presets */}
            <div className="flex flex-col gap-2 mt-1">
              <span className="font-mono text-xs text-[#3d4a42] font-semibold tracking-wide">
                BENCHMARK SAMPLES:
              </span>
              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={() => handleStartAnalysis(0)}
                  className="px-3 py-1.5 rounded-lg bg-[#eaedff] hover:bg-[#dae2fd] text-[#131b2e] font-mono text-xs flex items-center gap-1.5 transition-colors border border-[#dae2fd] cursor-pointer"
                >
                  <span className="material-symbols-outlined text-[14px] text-[#ba1a1a]">warning</span>
                  <span>Sample 1: Black Sigatoka</span>
                </button>
                <button
                  type="button"
                  onClick={() => handleStartAnalysis(1)}
                  className="px-3 py-1.5 rounded-lg bg-[#eaedff] hover:bg-[#dae2fd] text-[#131b2e] font-mono text-xs flex items-center gap-1.5 transition-colors border border-[#dae2fd] cursor-pointer"
                >
                  <span className="material-symbols-outlined text-[14px] text-[#006948]">verified</span>
                  <span>Sample 2: Healthy Leaf</span>
                </button>
                <button
                  type="button"
                  onClick={() => handleStartAnalysis(2)}
                  className="px-3 py-1.5 rounded-lg bg-[#eaedff] hover:bg-[#dae2fd] text-[#131b2e] font-mono text-xs flex items-center gap-1.5 transition-colors border border-[#dae2fd] cursor-pointer"
                >
                  <span className="material-symbols-outlined text-[14px] text-[#825100]">grain</span>
                  <span>Sample 3: Yellow Sigatoka</span>
                </button>
                <button
                  type="button"
                  onClick={() => handleStartAnalysis(3)}
                  className="px-3 py-1.5 rounded-lg bg-[#eaedff] hover:bg-[#dae2fd] text-[#131b2e] font-mono text-xs flex items-center gap-1.5 transition-colors border border-[#dae2fd] cursor-pointer"
                >
                  <span className="material-symbols-outlined text-[14px] text-[#006a61]">spa</span>
                  <span>Sample 4: Cordana Spot</span>
                </button>
              </div>
            </div>
          </div>

          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="w-full py-3 px-4 rounded-xl bg-[#eaedff] text-[#131b2e] hover:bg-[#dae2fd] font-semibold text-sm transition-all flex items-center justify-center gap-2 border border-[#dae2fd] shadow-sm cursor-pointer"
          >
            <span className="material-symbols-outlined text-[18px]">folder_open</span>
            <span>Choose Leaf Image</span>
          </button>
        </div>

        {/* Option 2: Use Camera with Framing Guidance */}
        <div className="bg-white p-7 rounded-2xl shadow-sm border border-[#dae2fd] flex flex-col justify-between gap-6 hover:shadow-md transition-shadow">
          <div className="flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <span className="px-3 py-1 rounded-md bg-[#86f2e4]/30 text-[#006f66] font-mono text-xs font-bold">
                OPTION 02 &bull; RECOMMENDED FOR FIELD
              </span>
              <span className="material-symbols-outlined text-[#006a61] text-[24px]">
                photo_camera
              </span>
            </div>

            <div>
              <h2 className="text-xl font-bold text-[#131b2e]">Use Camera</h2>
              <p className="text-xs text-[#3d4a42] mt-1">
                Optimized mobile camera viewfinder calibrated for leaf surface macro-inspection.
              </p>
            </div>

            {/* Framing Guide Visual Box */}
            <div
              onClick={() => setIsCameraOpen(true)}
              className="relative w-full h-48 rounded-xl overflow-hidden bg-[#dae2fd] border border-[#bccac0] cursor-pointer group flex items-center justify-center shadow-inner"
            >
              <img
                alt="Sample leaf framing preview"
                className="absolute inset-0 w-full h-full object-cover opacity-85 group-hover:scale-105 transition-transform duration-500"
                src={ASSETS.sigatokaLeafSample}
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-black/30" />

              {/* Reticle */}
              <div className="relative z-10 w-44 h-32 border-2 border-dashed border-[#85f8c4] rounded-lg flex flex-col items-center justify-center p-2 text-center text-white backdrop-blur-[1px]">
                <span className="material-symbols-outlined text-[28px] text-[#85f8c4]">crop_free</span>
                <span className="text-xs font-bold mt-1 drop-shadow">Leaf Framing Guide</span>
                <span className="text-[10px] text-[#85f8c4] font-mono">Fill frame with leaf</span>
              </div>
            </div>

            {/* Guidance bullets */}
            <div className="p-3.5 rounded-xl bg-[#f2f3ff] border border-[#dae2fd] flex flex-col gap-2 text-xs text-[#3d4a42]">
              <div className="flex items-center gap-2 font-bold text-[#131b2e]">
                <span className="material-symbols-outlined text-[#006948] text-[18px]">lightbulb</span>
                <span>Camera Framing Guidance:</span>
              </div>
              <ul className="list-disc pl-5 flex flex-col gap-1 text-[11px] leading-relaxed">
                <li>Position one banana leaf inside the frame.</li>
                <li>Make sure the leaf is clearly visible and well lit.</li>
                <li>Avoid casting strong shadows with your hands or mobile phone.</li>
              </ul>
            </div>
          </div>

          <div className="flex flex-col gap-2">
            <button
              type="button"
              onClick={() => setIsCameraOpen(true)}
              className="w-full py-3 px-4 rounded-xl bg-[#006948] text-white hover:bg-[#00855d] font-semibold text-sm transition-all flex items-center justify-center gap-2 shadow-md shadow-[#006948]/20 cursor-pointer active:scale-[0.99]"
            >
              <span className="material-symbols-outlined text-[18px]">videocam</span>
              <span>Open Leaf Camera</span>
            </button>

            <button
              type="button"
              onClick={() => nativeCameraInputRef.current?.click()}
              className="w-full py-2.5 px-4 rounded-xl bg-[#eaedff] text-[#131b2e] hover:bg-[#dae2fd] font-semibold text-xs transition-all flex items-center justify-center gap-2 border border-[#dae2fd] cursor-pointer"
            >
              <span className="material-symbols-outlined text-[16px] text-[#006948]">smartphone</span>
              <span>Phone Camera App (iPhone / Android)</span>
            </button>
          </div>
        </div>
      </div>

      {/* Dual Model Architecture Explanation Banner */}
      <div className="bg-white p-6 rounded-2xl border border-[#dae2fd] shadow-sm flex flex-col sm:flex-row items-start gap-4">
        <div className="w-10 h-10 rounded-xl bg-[#e2e7ff] text-[#006948] flex items-center justify-center shrink-0">
          <span className="material-symbols-outlined text-[24px]">account_tree</span>
        </div>
        <div className="flex flex-col gap-2 flex-1">
          <div className="flex items-center gap-2 flex-wrap">
            <h3 className="text-base font-extrabold text-[#131b2e]">
              Two Independent AI Models Power Leaf Analysis
            </h3>
            <span className="px-2.5 py-0.5 rounded-full bg-[#85f8c4]/30 text-[#006948] text-[11px] font-mono font-bold">
              FastAPI Multi-Model Microservices
            </span>
          </div>
          <p className="text-xs text-[#3d4a42] leading-relaxed">
            The system executes two separate deep learning models in parallel for comprehensive foliar diagnostics:
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
            <div className="p-3 rounded-xl bg-[#faf8ff] border border-[#dae2fd] flex flex-col gap-1">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-[#131b2e]">Model 3: Leaf Disease Classifier</span>
                <span className="text-[10px] font-mono text-[#006948] font-semibold">218ms</span>
              </div>
              <span className="text-[11px] font-mono text-[#006948] bg-[#e2e7ff]/70 px-1.5 py-0.5 rounded w-fit">
                POST /api/v1/model3/disease-classification
              </span>
              <p className="text-[11px] text-[#3d4a42] mt-0.5">
                Evaluates botanical health, diagnoses fungal streak strains, and returns disease classification confidence.
              </p>
            </div>
            <div className="p-3 rounded-xl bg-[#faf8ff] border border-[#dae2fd] flex flex-col gap-1">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-[#131b2e]">Model 4: Leaf Semantic Segmenter</span>
                <span className="text-[10px] font-mono text-[#006948] font-semibold">364ms</span>
              </div>
              <span className="text-[11px] font-mono text-[#006948] bg-[#e2e7ff]/70 px-1.5 py-0.5 rounded w-fit">
                POST /api/v1/model4/semantic-segmentation
              </span>
              <p className="text-[11px] text-[#3d4a42] mt-0.5">
                U-Net architecture segments lesion coordinates pixel-by-pixel and calculates exact % of infected leaf area.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
