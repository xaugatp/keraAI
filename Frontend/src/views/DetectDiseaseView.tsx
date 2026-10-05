import React from 'react';
import { ViewTab } from '../types';

interface DetectDiseaseViewProps {
  onNavigate: (tab: ViewTab) => void;
  /** Kept so existing call sites (App, History) still compile; Model 3 has no results to select yet. */
  selectedDiagnosisIndex?: number;
}

/**
 * Model 3 (leaf disease) is planned but not built. This panel says so plainly and shows no
 * results, numbers or recommendations: nothing here comes from a model.
 */
export const DetectDiseaseView: React.FC<DetectDiseaseViewProps> = ({ onNavigate }) => {
  return (
    <div className="w-full max-w-7xl mx-auto px-6 lg:px-12 py-10 flex flex-col gap-8">
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
          Model 3 &mdash; Leaf disease detection
        </h1>
      </header>

      <section
        aria-labelledby="model3-coming-soon"
        className="bg-white p-8 sm:p-10 rounded-2xl shadow-sm border border-[#dae2fd] flex flex-col gap-6 max-w-3xl"
      >
        <div className="flex items-center gap-3">
          <span className="w-12 h-12 rounded-full bg-[#ffeed2] border border-[#ffb95f] flex items-center justify-center text-[#825100]">
            <span className="material-symbols-outlined text-[26px]">hourglass_top</span>
          </span>
          <span className="px-3 py-1 rounded-md bg-[#ffeed2] text-[#825100] font-mono text-xs font-bold border border-[#ffb95f] uppercase tracking-wider">
            Coming soon
          </span>
        </div>

        <div className="flex flex-col gap-2">
          <h2 id="model3-coming-soon" className="text-xl font-bold text-[#131b2e]">
            This model is still being built
          </h2>
          <p className="text-sm text-[#3d4a42] leading-relaxed">
            Model 3 will tell a healthy banana leaf from a diseased one and, when it is diseased, say
            which kind of disease it is. It is planned but not trained yet, so there is nothing to run
            here and no results are shown.
          </p>
        </div>

        <div className="p-4 rounded-xl bg-[#f2f3ff] border border-[#dae2fd] flex items-start gap-2.5 text-xs text-[#3d4a42]">
          <span className="material-symbols-outlined text-[#006948] text-[20px] shrink-0">info</span>
          <span className="leading-relaxed">
            Available now: Model 1 checks whether a photo shows a banana tree, and Model 2 measures how
            much of a leaf is damaged.
          </span>
        </div>

        <div className="flex flex-col sm:flex-row gap-3">
          <button
            type="button"
            onClick={() => onNavigate('leaf-analysis')}
            className="py-3 px-5 rounded-xl bg-[#006948] text-white hover:bg-[#00855d] font-semibold text-sm transition-all flex items-center justify-center gap-2 shadow-md shadow-[#006948]/25 cursor-pointer active:scale-[0.99]"
          >
            <span className="material-symbols-outlined text-[18px]">layers</span>
            <span>Measure leaf damage (Model 2)</span>
          </button>
          <button
            type="button"
            onClick={() => onNavigate('detect')}
            className="py-3 px-5 rounded-xl bg-[#eaedff] text-[#131b2e] hover:bg-[#dae2fd] font-semibold text-sm transition-all flex items-center justify-center gap-2 border border-[#dae2fd] shadow-sm cursor-pointer"
          >
            <span className="material-symbols-outlined text-[18px]">center_focus_strong</span>
            <span>Detect a banana tree (Model 1)</span>
          </button>
        </div>
      </section>
    </div>
  );
};
