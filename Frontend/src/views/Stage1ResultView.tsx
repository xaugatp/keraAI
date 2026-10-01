import React, { useState } from 'react';
import { ViewTab } from '../types';
import { defaultStage1Data, ASSETS } from '../data/mockData';

interface Stage1ResultViewProps {
  onNavigate: (tab: ViewTab) => void;
  activeStateMode?: 'positive' | 'negative';
  gpsLocation?: { latitude: number; longitude: number; accuracy?: number | null };
}

export const Stage1ResultView: React.FC<Stage1ResultViewProps> = ({
  onNavigate,
  activeStateMode = 'positive',
  gpsLocation,
}) => {
  const [activeTab, setActiveTab] = useState<'positive' | 'negative'>(activeStateMode);
  const [isZoomed, setIsZoomed] = useState(false);

  const displayLat = gpsLocation?.latitude ?? 27.71724;
  const displayLng = gpsLocation?.longitude ?? 85.32402;

  return (
    <div className="w-full max-w-7xl mx-auto px-6 lg:px-12 py-10 flex flex-col gap-8">
      {/* Top Breadcrumb & State Switcher for Demonstration */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex items-center gap-2 text-xs font-mono uppercase text-[#3d4a42] font-semibold">
          <button
            onClick={() => onNavigate('home')}
            className="hover:text-[#006948] transition-colors cursor-pointer"
          >
            Home
          </button>
          <span className="material-symbols-outlined text-[14px]">chevron_right</span>
          <button
            onClick={() => onNavigate('detect')}
            className="hover:text-[#006948] transition-colors cursor-pointer"
          >
            Plant Detection
          </button>
          <span className="material-symbols-outlined text-[14px]">chevron_right</span>
          <span className="text-[#006948] font-bold">Detection Result</span>
        </div>

        {/* Demo State Switcher */}
        <div className="inline-flex p-1 rounded-xl bg-[#e2e7ff] border border-[#dae2fd] shadow-inner">
          <button
            type="button"
            onClick={() => setActiveTab('positive')}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-bold transition-all cursor-pointer flex items-center gap-1.5 ${
              activeTab === 'positive'
                ? 'bg-white text-[#006948] shadow-sm'
                : 'text-[#3d4a42] hover:text-[#131b2e]'
            }`}
          >
            <span className="w-2 h-2 rounded-full bg-[#006948]" />
            <span>Banana Tree Detected (94.7%)</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveTab('negative')}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-bold transition-all cursor-pointer flex items-center gap-1.5 ${
              activeTab === 'negative'
                ? 'bg-white text-[#ba1a1a] shadow-sm'
                : 'text-[#3d4a42] hover:text-[#131b2e]'
            }`}
          >
            <span className="w-2 h-2 rounded-full bg-[#ba1a1a]" />
            <span>Non-Banana Tree (91.3%)</span>
          </button>
        </div>
      </div>

      {/* Main Two-Column Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Prominent Uploaded/Captured Image Display (7 cols) */}
        <div className="lg:col-span-7 flex flex-col gap-4">
          <div className="relative w-full aspect-[4/3] rounded-2xl overflow-hidden bg-[#283044] shadow-md border border-[#dae2fd]">
            <img
              alt="Analyzed plant specimen"
              className={`w-full h-full object-cover transition-transform duration-300 ${
                isZoomed ? 'scale-125' : 'scale-100'
              }`}
              src={activeTab === 'positive' ? ASSETS.bananaTreeConfirmed : ASSETS.negativeHouseplant}
            />

            {/* AI Bounding Box overlay */}
            {activeTab === 'positive' ? (
              <div className="absolute inset-[10%] rounded-xl shadow-[0_0_0_2px_#006948,0_0_24px_rgba(0,105,72,0.4)] pointer-events-none flex flex-col justify-between p-3">
                <div className="flex items-center justify-between">
                  <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-lg bg-[#006948]/90 backdrop-blur-md text-white font-mono text-xs shadow-sm">
                    <span className="material-symbols-outlined text-[14px]">nest_eco_leaf</span>
                    <span>Banana Tree (Musa spp.): 94.7%</span>
                  </div>
                  <div className="px-2 py-0.5 rounded bg-white/80 backdrop-blur-sm text-[#131b2e] font-mono text-[10px] font-semibold">
                    [x: 124, y: 88, w: 740, h: 960]
                  </div>
                </div>

                <div className="flex items-end justify-between">
                  <div className="font-mono text-[10px] text-white bg-[#131b2e]/70 px-2 py-0.5 rounded backdrop-blur-sm flex items-center gap-1.5">
                    <span className="material-symbols-outlined text-[12px] text-[#85f8c4]">location_on</span>
                    <span>GPS: {displayLat.toFixed(5)}°, {displayLng.toFixed(5)}°</span>
                  </div>
                  <span className="material-symbols-outlined text-[#85f8c4] text-[24px]">
                    filter_center_focus
                  </span>
                </div>
              </div>
            ) : (
              <div className="absolute inset-[14%] rounded-xl shadow-[0_0_0_2px_#ba1a1a,0_0_20px_rgba(186,26,26,0.3)] pointer-events-none flex flex-col justify-between p-3">
                <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-lg bg-[#ba1a1a] text-white font-mono text-xs shadow-sm w-fit">
                  <span className="material-symbols-outlined text-[14px]">cancel</span>
                  <span>Non-Musa Foliage Detected</span>
                </div>
                <div className="font-mono text-[10px] text-white bg-[#ba1a1a]/90 px-2 py-0.5 rounded backdrop-blur-sm w-fit">
                  Species Mismatch: Non-Target Houseplant
                </div>
              </div>
            )}

            {/* Bottom Floating Bar */}
            <div className="absolute bottom-4 left-4 right-4 bg-white/90 backdrop-blur-md p-3 rounded-xl flex items-center justify-between text-[#131b2e] shadow-md border border-white/60">
              <div className="flex items-center gap-3">
                <span className="material-symbols-outlined text-[18px] text-[#006948]">
                  photo_camera
                </span>
                <span className="font-mono text-xs font-semibold">1920 &times; 1440 px</span>
                <span className="w-1 h-3 bg-[#dae2fd] rounded-full" />
                <span className="font-mono text-xs text-[#006948] font-bold flex items-center gap-1">
                  <span className="material-symbols-outlined text-[14px]">pin_drop</span>
                  {displayLat.toFixed(4)}°, {displayLng.toFixed(4)}°
                </span>
              </div>

              <button
                type="button"
                onClick={() => setIsZoomed(!isZoomed)}
                className="p-1 rounded bg-[#eaedff] hover:bg-[#dae2fd] text-[#131b2e] transition-colors cursor-pointer"
                title={isZoomed ? 'Reset Zoom' : 'Zoom In'}
              >
                <span className="material-symbols-outlined text-[16px]">
                  {isZoomed ? 'zoom_out' : 'zoom_in'}
                </span>
              </button>
            </div>
          </div>

          {/* Plant Geometry Metrics */}
          <div className="grid grid-cols-3 gap-3">
            <div className="bg-white p-3.5 rounded-xl border border-[#dae2fd] flex flex-col gap-0.5">
              <span className="text-[11px] text-[#3d4a42]">Foliage Density</span>
              <span className="text-xs font-bold text-[#131b2e]">
                {activeTab === 'positive' ? 'Dense / Clustered' : 'Low / Non-Canopy'}
              </span>
            </div>
            <div className="bg-white p-3.5 rounded-xl border border-[#dae2fd] flex flex-col gap-0.5">
              <span className="text-[11px] text-[#3d4a42]">Pseudostem Integrity</span>
              <span className="text-xs font-bold text-[#131b2e]">
                {activeTab === 'positive' ? 'Intact (88.4%)' : 'Absent (0.0%)'}
              </span>
            </div>
            <div className="bg-white p-3.5 rounded-xl border border-[#dae2fd] flex flex-col gap-0.5">
              <span className="text-[11px] text-[#3d4a42]">Field Location</span>
              <span className="text-xs font-bold text-[#006948] truncate">
                {displayLat.toFixed(4)}°, {displayLng.toFixed(4)}°
              </span>
            </div>
          </div>
        </div>

        {/* Right Column: Prediction Results & Clear Step Progression (5 cols) */}
        <div className="lg:col-span-5 flex flex-col gap-6">
          {activeTab === 'positive' ? (
            /* BANANA TREE CONFIRMED CARD */
            <div className="bg-white p-7 rounded-2xl shadow-sm border border-[#dae2fd] flex flex-col gap-6">
              {/* Species Confirmation Banner */}
              <div className="flex items-center gap-3 p-3.5 rounded-xl bg-[#85f8c4]/30 border border-[#85f8c4] text-[#002114]">
                <span className="material-symbols-outlined text-[26px] text-[#006948]">
                  check_circle
                </span>
                <span className="text-base font-extrabold tracking-tight">
                  ✓ Banana Tree Confirmed
                </span>
              </div>

              {/* Confidence Meter */}
              <div className="flex items-center justify-between">
                <div className="flex flex-col gap-0.5">
                  <span className="text-xs text-[#3d4a42] uppercase font-mono font-bold tracking-wider">
                    Model 1 Confidence
                  </span>
                  <span className="text-3xl font-extrabold text-[#131b2e]">94.7%</span>
                  <span className="text-xs text-[#3d4a42]">Verified Musa genus canopy</span>
                </div>

                <div className="relative w-20 h-20 flex items-center justify-center">
                  <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
                    <circle
                      cx="50"
                      cy="50"
                      r="42"
                      fill="transparent"
                      stroke="#eaedff"
                      strokeWidth="8"
                    />
                    <circle
                      cx="50"
                      cy="50"
                      r="42"
                      fill="transparent"
                      stroke="#006948"
                      strokeWidth="8"
                      strokeDasharray="263.89"
                      strokeDashoffset="14.0"
                      strokeLinecap="round"
                    />
                  </svg>
                  <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <span className="text-sm font-extrabold text-[#131b2e]">94.7%</span>
                    <span className="font-mono text-[8px] uppercase text-[#006948] font-bold">MATCH</span>
                  </div>
                </div>
              </div>

              {/* Technical Information Panel */}
              <div className="p-4 rounded-xl bg-[#f2f3ff] border border-[#dae2fd] flex flex-col gap-2 font-mono text-xs">
                <span className="text-[10px] uppercase font-bold text-[#3d4a42] tracking-wider">
                  Plant &amp; Location Specs
                </span>
                <div className="flex justify-between py-1 border-b border-[#dae2fd]">
                  <span className="text-[#3d4a42]">Taxon:</span>
                  <span className="font-bold text-[#131b2e]">Musa acuminata Colla</span>
                </div>
                <div className="flex justify-between py-1 border-b border-[#dae2fd]">
                  <span className="text-[#3d4a42]">Field Latitude:</span>
                  <span className="font-bold text-[#006948]">{displayLat.toFixed(6)}° N</span>
                </div>
                <div className="flex justify-between py-1 border-b border-[#dae2fd]">
                  <span className="text-[#3d4a42]">Field Longitude:</span>
                  <span className="font-bold text-[#006948]">{displayLng.toFixed(6)}° E</span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-[#3d4a42]">Status:</span>
                  <span className="font-bold text-[#006948] flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-[#006948]" /> Ready for Pathology
                  </span>
                </div>
              </div>

              {/* TWO CRYSTAL-CLEAR PROGRESSION OPTIONS WITH PROMINENT ARROWS */}
              <div className="flex flex-col gap-3 pt-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs uppercase font-extrabold tracking-wider text-[#131b2e] font-mono">
                    Next Stage in Pipeline:
                  </span>
                  <span className="text-[11px] text-[#006948] font-bold">Choose Analysis Path</span>
                </div>

                {/* Option 1: Detect Disease & Affected Area */}
                <div className="p-4 rounded-xl bg-gradient-to-r from-[#eef3ff] to-[#f9faff] border-2 border-[#006948] hover:border-[#00855d] transition-all flex flex-col gap-2 shadow-sm">
                  <div className="flex items-center justify-between">
                    <span className="inline-flex items-center gap-1.5 text-xs font-extrabold text-[#006948] font-mono">
                      <span className="w-5 h-5 rounded-full bg-[#006948] text-white flex items-center justify-center text-[11px]">
                        1
                      </span>
                      STAGE 02 — DETECT DISEASE &amp; AFFECTED AREA
                    </span>
                    <span className="px-2 py-0.5 rounded bg-[#ffdad6] text-[#ba1a1a] font-mono text-[10px] font-bold">
                      Model 3
                    </span>
                  </div>
                  <p className="text-xs text-[#3d4a42] leading-tight">
                    Classify pathogens (Black Sigatoka, Cordana) and localize affected necrotic lesion boundaries with treatment recommendations.
                  </p>
                  <button
                    type="button"
                    onClick={() => onNavigate('detect-disease')}
                    className="w-full mt-1 py-3 px-4 rounded-xl bg-[#006948] hover:bg-[#00855d] text-white font-bold text-xs transition-all flex items-center justify-center gap-2 shadow-md shadow-[#006948]/20 cursor-pointer active:scale-95"
                  >
                    <span>Proceed to Detect Disease &amp; Affected Area</span>
                    <span className="material-symbols-outlined text-[18px] font-bold animate-pulse">
                      arrow_forward
                    </span>
                  </button>
                </div>

                {/* Option 2: Segment Affected Area & Leaf Health Analysis */}
                <div className="p-4 rounded-xl bg-gradient-to-r from-[#f0fbf7] to-[#f9fdfb] border-2 border-[#006a61] hover:border-[#00855d] transition-all flex flex-col gap-2 shadow-sm">
                  <div className="flex items-center justify-between">
                    <span className="inline-flex items-center gap-1.5 text-xs font-extrabold text-[#006a61] font-mono">
                      <span className="w-5 h-5 rounded-full bg-[#006a61] text-white flex items-center justify-center text-[11px]">
                        2
                      </span>
                      STAGE 03 — SEGMENT AFFECTED AREA
                    </span>
                    <span className="px-2 py-0.5 rounded bg-[#85f8c4]/40 text-[#006948] font-mono text-[10px] font-bold">
                      Model 4
                    </span>
                  </div>
                  <p className="text-xs text-[#3d4a42] leading-tight">
                    Pixel-level U-Net foliar segmentation strictly measuring Healthy Part vs. Unhealthy Part with interactive split slider.
                  </p>
                  <button
                    type="button"
                    onClick={() => onNavigate('leaf-analysis')}
                    className="w-full mt-1 py-3 px-4 rounded-xl bg-[#006a61] hover:bg-[#00855d] text-white font-bold text-xs transition-all flex items-center justify-center gap-2 shadow-md shadow-[#006a61]/20 cursor-pointer active:scale-95"
                  >
                    <span>Proceed to Segment Affected Area &amp; Leaf Health</span>
                    <span className="material-symbols-outlined text-[18px] font-bold animate-pulse">
                      arrow_forward
                    </span>
                  </button>
                </div>

                {/* Reset / Another Plant */}
                <button
                  type="button"
                  onClick={() => onNavigate('detect')}
                  className="py-2.5 px-4 rounded-xl bg-[#eaedff] text-[#131b2e] hover:bg-[#dae2fd] font-semibold text-xs transition-all flex items-center justify-center gap-1.5 border border-[#dae2fd] cursor-pointer mt-1"
                >
                  <span className="material-symbols-outlined text-[16px]">refresh</span>
                  <span>Analyze Another Banana Plant</span>
                </button>
              </div>
            </div>
          ) : (
            /* NON-BANANA RESULT CARD */
            <div className="bg-white p-7 rounded-2xl shadow-sm border border-[#ffdad6] flex flex-col gap-6">
              <div className="flex items-center gap-3 p-3.5 rounded-xl bg-[#ffdad6] border border-[#ba1a1a]/30 text-[#ba1a1a]">
                <span className="material-symbols-outlined text-[24px]">info</span>
                <span className="text-base font-extrabold tracking-tight">
                  Banana Tree Not Detected
                </span>
              </div>

              <p className="text-xs text-[#3d4a42] leading-relaxed">
                The uploaded image does not appear to contain a banana tree. KERA AI verified this specimen against herbarium records; foliar paddle venation and pseudostem signatures are absent.
              </p>

              <div className="p-4 rounded-xl bg-[#f2f3ff] border border-[#dae2fd] flex flex-col gap-2 font-mono text-xs">
                <span className="text-[10px] uppercase font-bold text-[#3d4a42] tracking-wider">
                  Field Capture Coordinates
                </span>
                <div className="flex justify-between py-1 border-b border-[#dae2fd]">
                  <span className="text-[#3d4a42]">Lat:</span>
                  <span className="font-bold text-[#131b2e]">{displayLat.toFixed(5)}°</span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-[#3d4a42]">Long:</span>
                  <span className="font-bold text-[#131b2e]">{displayLng.toFixed(5)}°</span>
                </div>
              </div>

              <div className="flex flex-col gap-2.5 pt-1">
                <button
                  type="button"
                  onClick={() => onNavigate('detect')}
                  className="w-full py-3.5 px-6 rounded-xl bg-[#006948] text-white hover:bg-[#00855d] font-semibold text-sm transition-all flex items-center justify-center gap-2 shadow-md cursor-pointer active:scale-95"
                >
                  <span className="material-symbols-outlined text-[18px]">photo_camera</span>
                  <span>Capture Another Plant Specimen</span>
                  <span className="material-symbols-outlined text-[18px]">arrow_forward</span>
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
