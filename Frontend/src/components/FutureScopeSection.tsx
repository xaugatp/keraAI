import React from 'react';
import { ASSETS } from '../data/mockData';

/**
 * Shared "what's next" pitch for drone-based farm surveillance, used on both the
 * Detect workspace and the About Models page so the two never drift out of sync.
 */
export const FutureScopeSection: React.FC = () => {
  return (
    <div className="bg-white p-6 sm:p-8 rounded-2xl shadow-sm border border-[#dae2fd] flex flex-col lg:flex-row gap-10 lg:items-center">
      <div className="flex-1 flex flex-col gap-4">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="px-2 py-0.5 rounded bg-[#e2e7ff] text-[#006948] font-mono text-[10px] font-bold uppercase">
            Future scope
          </span>
          <span className="px-2.5 py-0.5 rounded-full bg-[#fff8e1] text-[#825100] font-mono text-[10px] font-bold border border-[#ffb95f]/40">
            Open for pilot partnerships
          </span>
        </div>
        <h3 className="text-xl font-bold text-[#131b2e] leading-snug">
          From a single photo to an entire farm
        </h3>
        <p className="text-sm text-[#3d4a42] leading-relaxed">
          Right now this model looks at one photo at a time. The natural next step is teaching it
          to look at a whole plantation. Fly a drone over the field and the same technology can
          scan every tree, flag the ones that need attention, and build a simple health map for
          the entire farm without anyone walking the rows. If you grow bananas at scale, work in
          agritech, or research crop monitoring, we would love to talk about piloting this
          together.
        </p>
      </div>

      {/* Diagonal photo cascade: stacked and straight on mobile, tilted and overlapping from sm up */}
      <div className="flex-1 w-full flex flex-col gap-4 sm:block sm:relative sm:h-[300px]">
        <div className="relative sm:absolute sm:left-0 sm:top-0 w-full sm:w-[62%] aspect-[4/3] rounded-2xl overflow-hidden border border-[#dae2fd] shadow-md sm:shadow-xl sm:-rotate-6 sm:z-10 transition-transform duration-300 sm:hover:rotate-0 sm:hover:scale-105 sm:hover:z-30">
          <img
            src={ASSETS.droneSurveyPlantDetection}
            alt="A survey drone scanning a banana plantation and telling banana trees apart from other vegetation"
            className="w-full h-full object-cover"
          />
        </div>
        <div className="relative sm:absolute sm:right-0 sm:bottom-0 w-full sm:w-[62%] aspect-[4/3] rounded-2xl overflow-hidden border border-[#dae2fd] shadow-md sm:shadow-xl sm:rotate-6 sm:z-20 transition-transform duration-300 sm:hover:rotate-0 sm:hover:scale-105 sm:hover:z-30">
          <img
            src={ASSETS.droneSurveyDiseaseDiagnosis}
            alt="A drone-mounted camera reading the health of a banana leaf and picking up early signs of disease"
            className="w-full h-full object-cover"
          />
        </div>
      </div>
    </div>
  );
};
