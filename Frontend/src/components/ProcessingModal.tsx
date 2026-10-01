import React, { useEffect, useState } from 'react';

interface ProcessingModalProps {
  isOpen: boolean;
  mode: 'plant' | 'leaf';
  onComplete: () => void;
}

export const ProcessingModal: React.FC<ProcessingModalProps> = ({
  isOpen,
  mode,
  onComplete,
}) => {
  const [currentStep, setCurrentStep] = useState<number>(1);
  const [progress, setProgress] = useState<number>(15);

  useEffect(() => {
    if (!isOpen) {
      setCurrentStep(1);
      setProgress(15);
      return;
    }

    // Step 1: Image captured (immediate)
    setCurrentStep(1);
    setProgress(30);

    const timer1 = setTimeout(() => {
      // Step 2: Detecting banana plant
      setCurrentStep(2);
      setProgress(60);
    }, 600);

    const timer2 = setTimeout(() => {
      // Step 3: Analyzing leaf (if leaf mode) or finishing plant check
      setCurrentStep(mode === 'leaf' ? 3 : 2);
      setProgress(85);
    }, 1200);

    const timer3 = setTimeout(() => {
      // Step 4: Identifying disease or finalizing
      setCurrentStep(mode === 'leaf' ? 4 : 2);
      setProgress(100);
    }, 1800);

    const timer4 = setTimeout(() => {
      onComplete();
    }, 2200);

    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
      clearTimeout(timer3);
      clearTimeout(timer4);
    };
  }, [isOpen, mode]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4 animate-fade-in">
      <div className="bg-white w-full max-w-md rounded-2xl p-6 sm:p-8 shadow-2xl border border-[#dae2fd] flex flex-col gap-6">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-[#e2e7ff] text-[#006948] flex items-center justify-center shrink-0">
            <span className="material-symbols-outlined text-[24px] animate-spin">progress_activity</span>
          </div>
          <div>
            <h3 className="text-lg font-extrabold text-[#131b2e]">
              {mode === 'plant' ? 'Analyzing plant image...' : 'Analyzing leaf pathology...'}
            </h3>
            <p className="text-xs text-[#3d4a42]">
              FastAPI asynchronous dispatch to neural vision models
            </p>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="w-full bg-[#eaedff] rounded-full h-2 overflow-hidden">
          <div
            className="bg-[#006948] h-2 rounded-full transition-all duration-300 ease-out"
            style={{ width: `${progress}%` }}
          />
        </div>

        {/* Stages Checklist */}
        <div className="flex flex-col gap-3 font-mono text-xs">
          {/* Stage 1 */}
          <div className="flex items-center gap-3">
            <span className="w-5 h-5 rounded-full bg-[#006948] text-white flex items-center justify-center text-[12px] font-bold shrink-0">
              ✓
            </span>
            <div className="flex flex-col">
              <span className="text-[#131b2e] font-semibold">Image captured &amp; normalized</span>
              <span className="text-[10px] text-[#3d4a42]">Tensor format: 512×512×3 RGB float32</span>
            </div>
          </div>

          {/* Stage 2 / Model 1 or Model 3 */}
          {mode === 'plant' ? (
            <div className="flex items-center gap-3">
              {currentStep >= 2 ? (
                <span className="w-5 h-5 rounded-full bg-[#006948] text-white flex items-center justify-center text-[12px] font-bold shrink-0">
                  ✓
                </span>
              ) : (
                <span className="w-5 h-5 rounded-full border-2 border-[#006948] text-[#006948] flex items-center justify-center animate-pulse text-[10px] shrink-0">
                  ●
                </span>
              )}
              <div className="flex flex-col">
                <span className={currentStep >= 2 ? 'text-[#131b2e] font-semibold' : 'text-[#3d4a42]'}>
                  Banana Tree Classifier (Model 1)
                </span>
                <span className="text-[10px] text-[#006948] font-bold">
                  POST /api/v1/models/tree-classifier/predict
                </span>
              </div>
            </div>
          ) : (
            <div className="flex items-center gap-3">
              {currentStep >= 2 ? (
                <span className="w-5 h-5 rounded-full bg-[#006948] text-white flex items-center justify-center text-[12px] font-bold shrink-0">
                  ✓
                </span>
              ) : (
                <span className="w-5 h-5 rounded-full border-2 border-[#006948] text-[#006948] flex items-center justify-center animate-pulse text-[10px] shrink-0">
                  ●
                </span>
              )}
              <div className="flex flex-col">
                <span className={currentStep >= 2 ? 'text-[#131b2e] font-semibold' : 'text-[#3d4a42]'}>
                  Model 3: Leaf Disease Classifier
                </span>
                <span className="text-[10px] text-[#006948] font-bold">
                  POST /api/v1/model3/disease-classification
                </span>
              </div>
            </div>
          )}

          {/* Stage 3 (Model 4 for leaf) */}
          {mode === 'leaf' && (
            <div className="flex items-center gap-3">
              {currentStep >= 3 ? (
                <span className="w-5 h-5 rounded-full bg-[#006948] text-white flex items-center justify-center text-[12px] font-bold shrink-0">
                  ✓
                </span>
              ) : currentStep === 2 ? (
                <span className="w-5 h-5 rounded-full border-2 border-[#006948] text-[#006948] flex items-center justify-center animate-pulse text-[10px] shrink-0">
                  ●
                </span>
              ) : (
                <span className="w-5 h-5 rounded-full border border-[#bccac0] text-[#bccac0] flex items-center justify-center text-[10px] shrink-0">
                  ○
                </span>
              )}
              <div className="flex flex-col">
                <span className={currentStep >= 3 ? 'text-[#131b2e] font-semibold' : 'text-[#3d4a42]/70'}>
                  Model 4: Leaf Semantic Segmenter (U-Net)
                </span>
                <span className="text-[10px] text-[#006948] font-bold">
                  POST /api/v1/model4/semantic-segmentation
                </span>
              </div>
            </div>
          )}

          {/* Stage 4 (Final Synthesis) */}
          <div className="flex items-center gap-3">
            {currentStep >= 4 ? (
              <span className="w-5 h-5 rounded-full bg-[#006948] text-white flex items-center justify-center text-[12px] font-bold shrink-0">
                ✓
              </span>
            ) : currentStep === 3 ? (
              <span className="w-5 h-5 rounded-full border-2 border-[#006948] text-[#006948] flex items-center justify-center animate-pulse text-[10px] shrink-0">
                ●
              </span>
            ) : (
              <span className="w-5 h-5 rounded-full border border-[#bccac0] text-[#bccac0] flex items-center justify-center text-[10px] shrink-0">
                ○
              </span>
            )}
            <div className="flex flex-col">
              <span className={currentStep >= 4 ? 'text-[#131b2e] font-semibold' : 'text-[#3d4a42]/50'}>
                {mode === 'leaf' ? 'Synthesizing Lesion Masks & Report' : 'Synthesizing Taxonomic Output'}
              </span>
              <span className="text-[10px] text-[#3d4a42]">
                Aggregating bounding geometry &amp; confidence scores
              </span>
            </div>
          </div>
        </div>

        <div className="text-[11px] text-[#3d4a42] bg-[#f2f3ff] p-2.5 rounded-xl border border-[#dae2fd] text-center font-mono">
          CUDA Worker #04 &bull; ONNX Tensor dispatch in progress (~380ms)
        </div>
      </div>
    </div>
  );
};
