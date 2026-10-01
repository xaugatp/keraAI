import React, { useState } from 'react';
import { ViewTab } from '../types';
import { ASSETS } from '../data/mockData';

interface HomeViewProps {
  onNavigate: (tab: ViewTab) => void;
}

export const HomeView: React.FC<HomeViewProps> = ({ onNavigate }) => {
  const [activeMask, setActiveMask] = useState(true);

  return (
    <div className="flex flex-col w-full">
      {/* Top Atmospheric Glow */}
      <div className="relative w-full overflow-hidden">
        <div className="absolute -top-32 left-1/2 -translate-x-1/2 w-[920px] h-[480px] bg-[#85f8c4]/20 rounded-full blur-[120px] pointer-events-none -z-10" />
        <div className="absolute top-48 right-10 w-[420px] h-[360px] bg-[#89f5e7]/15 rounded-full blur-[100px] pointer-events-none -z-10" />

        {/* 1. Hero Section */}
        <section className="max-w-7xl mx-auto px-6 lg:px-12 pt-10 pb-20 lg:pt-16 lg:pb-28">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-12 items-center">
            {/* Left Column: Copy, Metadata & CTAs */}
            <div className="lg:col-span-6 flex flex-col items-start gap-5">
              {/* System Status Capsule */}
              <div className="inline-flex items-center gap-2 px-3.5 py-1 rounded-full bg-[#e2e7ff] shadow-sm border border-[#dae2fd]">
                <span className="w-2 h-2 rounded-full bg-[#006948] animate-pulse" />
                <span className="font-mono text-[11px] text-[#131b2e] uppercase tracking-wider font-semibold">
                  Dual-Engine Vision Pipeline
                </span>
                <span className="text-[#bccac0] font-mono text-[11px]">/</span>
                <span className="font-mono text-[11px] text-[#3d4a42]">ResNet-50 + U-Net</span>
              </div>

              {/* High-Impact Title */}
              <h1 className="text-3xl sm:text-4xl lg:text-[44px] font-extrabold text-[#131b2e] tracking-tight leading-[1.15]">
                AI-Powered Banana Plant{' '}
                <span className="text-[#006948]">Disease Detection</span>
              </h1>

              {/* Analytical Subtitle */}
              <p className="text-base text-[#3d4a42] max-w-xl leading-relaxed">
                Detect banana plants, analyze leaf health, and identify disease-affected areas using
                two-stage computer vision models with real-time semantic segmentation.
              </p>

              {/* Tech Architecture Chips */}
              <div className="flex flex-wrap items-center gap-2 pt-1">
                <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-[#eaedff] font-mono text-xs text-[#3d4a42] font-medium shadow-sm">
                  <span className="material-symbols-outlined text-[15px] text-[#006948]">bolt</span>
                  FastAPI Microservice
                </span>
                <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-[#eaedff] font-mono text-xs text-[#3d4a42] font-medium shadow-sm">
                  <span className="material-symbols-outlined text-[15px] text-[#006a61]">memory</span>
                  ResNet Banana Classifier
                </span>
                <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-[#eaedff] font-mono text-xs text-[#3d4a42] font-medium shadow-sm">
                  <span className="material-symbols-outlined text-[15px] text-[#825100]">polyline</span>
                  U-Net Semantic Segmentation
                </span>
              </div>

              {/* Dual CTAs */}
              <div className="flex flex-wrap items-center gap-4 pt-2">
                <button
                  type="button"
                  onClick={() => onNavigate('detect')}
                  className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-[#006948] text-white font-semibold text-sm shadow-md hover:bg-[#00855d] transition-all hover:scale-[1.02] active:scale-[0.99] shadow-[#006948]/20 cursor-pointer"
                >
                  <span className="material-symbols-outlined text-[20px]">photo_camera</span>
                  <span>Start Detection</span>
                </button>
                <a
                  href="#how-it-works"
                  className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-white text-[#131b2e] border border-[#e2e8f0] font-semibold text-sm shadow-sm hover:bg-[#f2f3ff] transition-all cursor-pointer"
                >
                  <span className="material-symbols-outlined text-[20px] text-[#3d4a42]">schema</span>
                  <span>How It Works</span>
                </a>
              </div>

              {/* Micro Spec Bar */}
              <div className="flex items-center gap-6 pt-2">
                <div className="flex items-center gap-1.5">
                  <span className="material-symbols-outlined text-[18px] text-[#006948]">check_circle</span>
                  <span className="font-mono text-xs text-[#3d4a42]">PyTorch TensorRT Core</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="material-symbols-outlined text-[18px] text-[#006948]">verified</span>
                  <span className="font-mono text-xs text-[#3d4a42]">Validated on Musa Acuminata</span>
                </div>
              </div>
            </div>

            {/* Right Column: Interactive Vision Canvas Card */}
            <div className="lg:col-span-6 relative">
              {/* Peripheral Decorative Accent */}
              <div className="absolute -inset-2 bg-gradient-to-tr from-[#68dba9]/20 via-[#006c4a]/5 to-[#ffb95f]/25 rounded-3xl blur-xl -z-10" />
              <div className="relative bg-white rounded-2xl shadow-xl overflow-hidden border border-[#dae2fd] group">
                {/* Terminal Chrome Header */}
                <div className="px-5 py-3.5 bg-[#eaedff] flex items-center justify-between border-b border-[#dae2fd]">
                  <div className="flex items-center gap-2">
                    <span className="w-3 h-3 rounded-full bg-[#ba1a1a]/70" />
                    <span className="w-3 h-3 rounded-full bg-[#a36700]/70" />
                    <span className="w-3 h-3 rounded-full bg-[#006948]/70" />
                    <span className="ml-2 font-mono text-xs text-[#3d4a42]">
                      stage2_inference_stream_sigatoka_09.png
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="px-2 py-0.5 rounded-full bg-[#dae2fd] text-[#3d4a42] font-mono text-[11px]">
                      Batch: 01
                    </span>
                    <span className="px-2 py-0.5 rounded-full bg-[#006948]/10 text-[#006948] font-mono text-[11px] font-semibold">
                      Live Vision HUD
                    </span>
                  </div>
                </div>

                {/* Viewport Image Container */}
                <div className="relative w-full aspect-[4/3] bg-[#d2d9f4] overflow-hidden select-none">
                  {/* Provided Image Asset */}
                  <img
                    alt="Banana plant leaf exhibiting Black Sigatoka leaf spot disease with visible brownish-black necrotic lesions"
                    className="w-full h-full object-cover object-center filter saturate-105"
                    src={ASSETS.sigatokaLeafSample}
                  />

                  {/* Overlay Gradient Ambient Layer */}
                  <div className="absolute inset-0 bg-gradient-to-t from-[#131b2e]/40 via-transparent to-transparent pointer-events-none" />

                  {/* Animated Polygon Segmentation Mask Highlight (SVG Overlay) */}
                  {activeMask && (
                    <svg
                      className="absolute inset-0 w-full h-full pointer-events-none"
                      preserveAspectRatio="none"
                      viewBox="0 0 800 600"
                    >
                      <defs>
                        <linearGradient id="heroSigatokaMaskGrad" x1="0%" x2="100%" y1="0%" y2="100%">
                          <stop offset="0%" stopColor="#ba1a1a" stopOpacity="0.5" />
                          <stop offset="50%" stopColor="#a36700" stopOpacity="0.45" />
                          <stop offset="100%" stopColor="#ba1a1a" stopOpacity="0.6" />
                        </linearGradient>
                        <pattern id="heroGridOverlay" width="20" height="20" patternUnits="userSpaceOnUse">
                          <path
                            d="M 20 0 L 0 0 0 20"
                            fill="none"
                            stroke="rgba(255,255,255,0.15)"
                            strokeWidth="0.5"
                          />
                        </pattern>
                      </defs>

                      {/* Grid Texture for Scientific HUD feel */}
                      <rect width="800" height="600" fill="url(#heroGridOverlay)" />

                      {/* Polygons representing segmented necrotic disease lesions */}
                      <polygon
                        className="animate-pulse"
                        fill="url(#heroSigatokaMaskGrad)"
                        points="210,140 330,120 460,180 430,280 340,320 220,290 180,210"
                        style={{ animationDuration: '3s' }}
                      />
                      <polygon
                        className="animate-pulse"
                        fill="url(#heroSigatokaMaskGrad)"
                        points="450,220 590,200 670,270 650,380 540,430 460,370"
                        style={{ animationDuration: '4s' }}
                      />
                      <polygon
                        className="animate-pulse"
                        fill="url(#heroSigatokaMaskGrad)"
                        points="140,340 280,310 320,440 260,520 160,510 110,430"
                        style={{ animationDuration: '3.5s' }}
                      />

                      {/* AI Precision Bounding Box around Primary Necrosis Center */}
                      <g className="transition-transform duration-700">
                        {/* Outer Focus Frame */}
                        <rect
                          x="150"
                          y="100"
                          width="540"
                          height="360"
                          rx="12"
                          fill="none"
                          stroke="#ffb95f"
                          strokeWidth="2"
                          strokeDasharray="8 6"
                          opacity="0.9"
                        />
                        {/* Target Corners */}
                        <path d="M 150 130 L 150 100 L 180 100" fill="none" stroke="#85f8c4" strokeWidth="4" strokeLinecap="round" />
                        <path d="M 660 100 L 690 100 L 690 130" fill="none" stroke="#85f8c4" strokeWidth="4" strokeLinecap="round" />
                        <path d="M 150 430 L 150 460 L 180 460" fill="none" stroke="#85f8c4" strokeWidth="4" strokeLinecap="round" />
                        <path d="M 660 460 L 690 460 L 690 430" fill="none" stroke="#85f8c4" strokeWidth="4" strokeLinecap="round" />
                        {/* Scanner Crosshairs */}
                        <line x1="420" y1="90" x2="420" y2="470" stroke="rgba(255,255,255,0.3)" strokeDasharray="4 4" strokeWidth="1" />
                        <line x1="140" y1="280" x2="700" y2="280" stroke="rgba(255,255,255,0.3)" strokeDasharray="4 4" strokeWidth="1" />
                      </g>
                    </svg>
                  )}

                  {/* Floating Precision Diagnostic Tag Top-Left */}
                  <div className="absolute top-4 left-4 flex flex-col gap-1.5 z-20">
                    <div className="backdrop-blur-md bg-white/90 px-3 py-1.5 rounded-lg shadow-md flex items-center gap-2 border border-white/60">
                      <span className="w-2.5 h-2.5 rounded-full bg-[#ba1a1a]" />
                      <span className="text-xs text-[#131b2e] font-semibold tracking-tight">
                        Black Sigatoka Detected (93.8%)
                      </span>
                    </div>
                    <div className="backdrop-blur-md bg-white/90 px-3 py-1 rounded-lg shadow-sm flex items-center gap-2 border border-white/60">
                      <span className="material-symbols-outlined text-[14px] text-[#825100]">pie_chart</span>
                      <span className="font-mono text-[11px] text-[#3d4a42] font-medium">
                        Infected Area: <strong className="text-[#131b2e]">27.4%</strong>
                      </span>
                    </div>
                  </div>

                  {/* Floating Viewport Lens / Stage Control Pill */}
                  <div className="absolute bottom-4 right-4 z-20 backdrop-blur-md bg-white/85 px-3 py-2 rounded-xl shadow-lg flex items-center gap-2 border border-white/60">
                    <button
                      onClick={() => setActiveMask(!activeMask)}
                      className="font-mono text-[11px] text-[#006948] font-bold hover:underline cursor-pointer"
                    >
                      {activeMask ? 'U-NET MASK: ACTIVE' : 'U-NET MASK: HIDDEN'}
                    </button>
                    <span className="w-1 h-3 bg-[#bccac0]/40 rounded-full" />
                    <div className="flex items-center gap-1">
                      <span className="inline-block w-2.5 h-2.5 rounded-full bg-[#ffdad6]" />
                      <span className="font-mono text-[11px] text-[#3d4a42]">Lesions (5)</span>
                    </div>
                  </div>

                  {/* Scanline Sweep Animation Effect */}
                  <div className="absolute inset-x-0 top-0 h-1 bg-gradient-to-r from-transparent via-[#006948] to-transparent opacity-80 animate-[bounce_4s_infinite]" />
                </div>

                {/* Card Bottom Telemetry Panel */}
                <div className="p-5 bg-white grid grid-cols-3 gap-2 border-t border-[#eaedff]">
                  <div className="flex flex-col">
                    <span className="text-[11px] text-[#3d4a42]">Class Confirmation</span>
                    <span className="font-mono text-sm text-[#006948] font-bold">Musa Leaf (99.8%)</span>
                  </div>
                  <div className="flex flex-col">
                    <span className="text-[11px] text-[#3d4a42]">Pathogen Taxonomy</span>
                    <span className="font-mono text-sm text-[#ba1a1a] font-bold">Pseudocercospora</span>
                  </div>
                  <div className="flex flex-col">
                    <span className="text-[11px] text-[#3d4a42]">Segmentation Loss</span>
                    <span className="font-mono text-sm text-[#131b2e] font-bold">0.031 Dice</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>
      </div>

      {/* 2. Three Feature Bento Cards Section */}
      <section className="w-full bg-[#f2f3ff] py-20 lg:py-24 border-y border-[#dae2fd]/60">
        <div className="max-w-7xl mx-auto px-6 lg:px-12 flex flex-col gap-12">
          {/* Section Header */}
          <div className="flex flex-col items-center text-center max-w-2xl mx-auto gap-2">
            <span className="font-mono text-xs uppercase tracking-wider text-[#006948] font-bold">
              Modular Computer Vision Pipeline
            </span>
            <h2 className="text-2xl lg:text-3xl font-extrabold text-[#131b2e] tracking-tight">
              End-to-End Plant Pathology Architecture
            </h2>
            <p className="text-sm text-[#3d4a42] leading-relaxed">
              Engineered to isolate background agricultural clutter, validate biological
              authenticity, and quantify leaf tissue decay with surgical granularity.
            </p>
          </div>

          {/* Feature Bento Cards Grid with Prominent Step Flow Connectors */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 relative">
            {/* Card 1: Banana Tree Detection & Geo Coordinates */}
            <div
              onClick={() => onNavigate('detect')}
              className="bg-white rounded-2xl p-7 shadow-sm hover:shadow-md transition-all flex flex-col justify-between gap-6 border-2 border-[#dae2fd] hover:border-[#006948] group cursor-pointer relative"
            >
              <div className="flex flex-col gap-4">
                <div className="flex items-center justify-between">
                  <div className="w-12 h-12 rounded-xl bg-[#85f8c4] flex items-center justify-center text-[#002114] shadow-sm">
                    <span className="material-symbols-outlined text-[26px]">location_on</span>
                  </div>
                  <span className="px-3 py-1 rounded-full bg-[#eaedff] text-[#006948] font-mono text-[11px] font-bold border border-[#dae2fd]">
                    STAGE 01
                  </span>
                </div>
                <div className="flex flex-col gap-1">
                  <span className="font-mono text-xs text-[#006948] font-bold">SPECIES &amp; GPS COORD</span>
                  <h3 className="text-xl font-extrabold text-[#131b2e] group-hover:text-[#006948] transition-colors">
                    Banana Tree Detection
                  </h3>
                </div>
                <p className="text-sm text-[#3d4a42] leading-relaxed">
                  Confirms Musa genus biological presence and automatically tags precise field geo-coordinates (Latitude &amp; Longitude) of the plant.
                </p>
              </div>

              <div className="flex flex-col gap-3">
                <div className="p-3 rounded-xl bg-[#eaedff] flex items-center justify-between font-mono text-xs">
                  <span className="text-[#3d4a42]">Geo-Coordinates:</span>
                  <span className="text-[#006948] font-bold">27.7172°, 85.3240°</span>
                </div>
                <div className="w-full py-2.5 px-4 rounded-xl bg-[#eaedff] group-hover:bg-[#006948] text-[#131b2e] group-hover:text-white font-bold text-xs transition-colors flex items-center justify-center gap-2">
                  <span>Start Plant Detection</span>
                  <span className="material-symbols-outlined text-[18px]">arrow_forward</span>
                </div>
              </div>
            </div>

            {/* Card 2: Detect Disease & Affected Areas */}
            <div
              onClick={() => onNavigate('detect-disease')}
              className="bg-white rounded-2xl p-7 shadow-sm hover:shadow-md transition-all flex flex-col justify-between gap-6 border-2 border-[#dae2fd] hover:border-[#ba1a1a] group cursor-pointer relative"
            >
              <div className="flex flex-col gap-4">
                <div className="flex items-center justify-between">
                  <div className="w-12 h-12 rounded-xl bg-[#ffdad6] flex items-center justify-center text-[#ba1a1a] shadow-sm">
                    <span className="material-symbols-outlined text-[26px]">coronavirus</span>
                  </div>
                  <span className="px-3 py-1 rounded-full bg-[#ffeed2] text-[#825100] font-mono text-[11px] font-bold border border-[#ffeed2]">
                    STAGE 02
                  </span>
                </div>
                <div className="flex flex-col gap-1">
                  <span className="font-mono text-xs text-[#ba1a1a] font-bold">PATHOGEN DIAGNOSTICS</span>
                  <h3 className="text-xl font-extrabold text-[#131b2e] group-hover:text-[#ba1a1a] transition-colors">
                    Detect Disease &amp; Affected Area
                  </h3>
                </div>
                <p className="text-sm text-[#3d4a42] leading-relaxed">
                  Deep multi-class CNN (Model 3) identifies Black Sigatoka, Yellow Sigatoka, or Cordana Spot, and pinpoints infected foliar necrotic areas.
                </p>
              </div>

              <div className="flex flex-col gap-3">
                <div className="p-3 rounded-xl bg-[#eaedff] flex items-center justify-between font-mono text-xs">
                  <span className="text-[#3d4a42]">Pathogen Model:</span>
                  <span className="text-[#ba1a1a] font-bold">Model 3 Diagnostic</span>
                </div>
                <div className="w-full py-2.5 px-4 rounded-xl bg-[#eaedff] group-hover:bg-[#ba1a1a] text-[#131b2e] group-hover:text-white font-bold text-xs transition-colors flex items-center justify-center gap-2">
                  <span>Detect Disease</span>
                  <span className="material-symbols-outlined text-[18px]">arrow_forward</span>
                </div>
              </div>
            </div>

            {/* Card 3: Segment Affected Area & Leaf Health */}
            <div
              onClick={() => onNavigate('leaf-analysis')}
              className="bg-white rounded-2xl p-7 shadow-sm hover:shadow-md transition-all flex flex-col justify-between gap-6 border-2 border-[#dae2fd] hover:border-[#006a61] group cursor-pointer relative"
            >
              <div className="flex flex-col gap-4">
                <div className="flex items-center justify-between">
                  <div className="w-12 h-12 rounded-xl bg-[#85f8c4] flex items-center justify-center text-[#002114] shadow-sm">
                    <span className="material-symbols-outlined text-[26px]">spa</span>
                  </div>
                  <span className="px-3 py-1 rounded-full bg-[#eaedff] text-[#006a61] font-mono text-[11px] font-bold border border-[#dae2fd]">
                    STAGE 03
                  </span>
                </div>
                <div className="flex flex-col gap-1">
                  <span className="font-mono text-xs text-[#006a61] font-bold">TISSUE SEGMENTATION</span>
                  <h3 className="text-xl font-extrabold text-[#131b2e] group-hover:text-[#006a61] transition-colors">
                    Segment Affected Area &amp; Leaf Health
                  </h3>
                </div>
                <p className="text-sm text-[#3d4a42] leading-relaxed">
                  Model 4 U-Net segmentation strictly isolates affected unhealthy areas from healthy foliar tissue, with interactive split-screen slider comparisons.
                </p>
              </div>

              <div className="flex flex-col gap-3">
                <div className="p-3 rounded-xl bg-[#eaedff] flex items-center justify-between font-mono text-xs">
                  <span className="text-[#3d4a42]">Segmentation:</span>
                  <span className="text-[#006a61] font-bold">Healthy vs Unhealthy</span>
                </div>
                <div className="w-full py-2.5 px-4 rounded-xl bg-[#eaedff] group-hover:bg-[#006a61] text-[#131b2e] group-hover:text-white font-bold text-xs transition-colors flex items-center justify-center gap-2">
                  <span>Segment Leaf Area</span>
                  <span className="material-symbols-outlined text-[18px]">arrow_forward</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 3. How It Works: 4-Step Process Section with Clear, Unclipped Arrows */}
      <section className="max-w-7xl mx-auto px-6 lg:px-12 py-20 lg:py-28 w-full" id="how-it-works">
        <div className="flex flex-col gap-10">
          {/* Section Title & Narrative */}
          <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
            <div className="flex flex-col gap-1.5 max-w-xl">
              <span className="font-mono text-xs uppercase tracking-wider text-[#006948] font-bold">
                Execution Workflow
              </span>
              <h2 className="text-2xl lg:text-3xl font-extrabold text-[#131b2e]">
                From Raw Field Capture to Actionable Agronomic Data
              </h2>
            </div>
            <div className="text-sm text-[#3d4a42] max-w-sm">
              A sequential cascading pipeline that prevents hallucinated classifications on
              non-target vegetation.
            </div>
          </div>

          {/* 4 Connected Process Cards with clear, prominent connecting arrows */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 relative">
            {/* Step 1 */}
            <div className="flex flex-col gap-2 relative">
              <div
                onClick={() => onNavigate('detect')}
                className="bg-[#eaedff] rounded-2xl p-6 flex flex-col gap-4 shadow-sm border border-[#dae2fd] hover:shadow-md hover:border-[#006948] transition-all cursor-pointer group h-full relative"
              >
                <div className="flex items-center justify-between">
                  <span className="w-8 h-8 rounded-full bg-[#dae2fd] text-[#131b2e] font-mono text-xs flex items-center justify-center font-bold group-hover:bg-[#006948] group-hover:text-white transition-colors">
                    01
                  </span>
                  <span className="material-symbols-outlined text-[#006948] text-[24px]">camera_enhance</span>
                </div>
                <div className="flex flex-col gap-1">
                  <h4 className="text-base font-bold text-[#131b2e] group-hover:text-[#006948] transition-colors">
                    Capture or Upload
                  </h4>
                  <span className="text-xs text-[#006948] font-semibold">Input Pre-processing</span>
                </div>
                <p className="text-xs text-[#3d4a42] leading-relaxed">
                  Acquire field photos via live camera or image upload. Automatic orientation normalization and RGB calibration applied.
                </p>

                <div className="mt-auto pt-2 flex items-center gap-1.5 text-xs font-bold text-[#006948] group-hover:translate-x-1 transition-transform">
                  <span>Start with Capture</span>
                  <span className="material-symbols-outlined text-[16px]">arrow_forward</span>
                </div>

                {/* Desktop Connecting Arrow (Unclipped in Gap) */}
                <div className="hidden lg:flex absolute -right-5 top-1/2 -translate-y-1/2 z-30 w-8 h-8 rounded-full bg-[#006948] shadow-lg border-2 border-white items-center justify-center text-white">
                  <span className="material-symbols-outlined text-[18px] font-bold">arrow_forward</span>
                </div>
              </div>

              {/* Mobile / Tablet Down Arrow */}
              <div className="lg:hidden flex items-center justify-center py-2 text-[#006948]">
                <div className="w-8 h-8 rounded-full bg-[#006948] text-white flex items-center justify-center shadow-md">
                  <span className="material-symbols-outlined text-[20px] font-bold">arrow_downward</span>
                </div>
              </div>
            </div>

            {/* Step 2 */}
            <div className="flex flex-col gap-2 relative">
              <div
                onClick={() => onNavigate('detect')}
                className="bg-[#eaedff] rounded-2xl p-6 flex flex-col gap-4 shadow-sm border border-[#dae2fd] hover:shadow-md hover:border-[#006a61] transition-all cursor-pointer group h-full relative"
              >
                <div className="flex items-center justify-between">
                  <span className="w-8 h-8 rounded-full bg-[#dae2fd] text-[#131b2e] font-mono text-xs flex items-center justify-center font-bold group-hover:bg-[#006a61] group-hover:text-white transition-colors">
                    02
                  </span>
                  <span className="material-symbols-outlined text-[#006a61] text-[24px]">location_on</span>
                </div>
                <div className="flex flex-col gap-1">
                  <h4 className="text-base font-bold text-[#131b2e] group-hover:text-[#006a61] transition-colors">
                    Detect Banana Plant
                  </h4>
                  <span className="text-xs text-[#006a61] font-semibold">Species &amp; Geo-Coordinates</span>
                </div>
                <p className="text-xs text-[#3d4a42] leading-relaxed">
                  Model 1 verifies Musa genus integrity and acquires field GPS latitude and longitude coordinates for plant mapping.
                </p>

                <div className="mt-auto pt-2 flex items-center gap-1.5 text-xs font-bold text-[#006a61] group-hover:translate-x-1 transition-transform">
                  <span>Verify Tree &amp; GPS</span>
                  <span className="material-symbols-outlined text-[16px]">arrow_forward</span>
                </div>

                {/* Desktop Connecting Arrow (Unclipped in Gap) */}
                <div className="hidden lg:flex absolute -right-5 top-1/2 -translate-y-1/2 z-30 w-8 h-8 rounded-full bg-[#006a61] shadow-lg border-2 border-white items-center justify-center text-white">
                  <span className="material-symbols-outlined text-[18px] font-bold">arrow_forward</span>
                </div>
              </div>

              {/* Mobile / Tablet Down Arrow */}
              <div className="lg:hidden flex items-center justify-center py-2 text-[#006a61]">
                <div className="w-8 h-8 rounded-full bg-[#006a61] text-white flex items-center justify-center shadow-md">
                  <span className="material-symbols-outlined text-[20px] font-bold">arrow_downward</span>
                </div>
              </div>
            </div>

            {/* Step 3 */}
            <div className="flex flex-col gap-2 relative">
              <div
                onClick={() => onNavigate('detect-disease')}
                className="bg-[#eaedff] rounded-2xl p-6 flex flex-col gap-4 shadow-sm border border-[#dae2fd] hover:shadow-md hover:border-[#825100] transition-all cursor-pointer group h-full relative"
              >
                <div className="flex items-center justify-between">
                  <span className="w-8 h-8 rounded-full bg-[#dae2fd] text-[#131b2e] font-mono text-xs flex items-center justify-center font-bold group-hover:bg-[#825100] group-hover:text-white transition-colors">
                    03
                  </span>
                  <span className="material-symbols-outlined text-[#825100] text-[24px]">coronavirus</span>
                </div>
                <div className="flex flex-col gap-1">
                  <h4 className="text-base font-bold text-[#131b2e] group-hover:text-[#825100] transition-colors">
                    Detect Disease
                  </h4>
                  <span className="text-xs text-[#825100] font-semibold">Pathogen Classification</span>
                </div>
                <p className="text-xs text-[#3d4a42] leading-relaxed">
                  Model 3 multi-class deep CNN diagnoses foliar pathogens (Sigatoka, Cordana) and generates treatment recommendations.
                </p>

                <div className="mt-auto pt-2 flex items-center gap-1.5 text-xs font-bold text-[#825100] group-hover:translate-x-1 transition-transform">
                  <span>Diagnose Pathogen</span>
                  <span className="material-symbols-outlined text-[16px]">arrow_forward</span>
                </div>

                {/* Desktop Connecting Arrow (Unclipped in Gap) */}
                <div className="hidden lg:flex absolute -right-5 top-1/2 -translate-y-1/2 z-30 w-8 h-8 rounded-full bg-[#825100] shadow-lg border-2 border-white items-center justify-center text-white">
                  <span className="material-symbols-outlined text-[18px] font-bold">arrow_forward</span>
                </div>
              </div>

              {/* Mobile / Tablet Down Arrow */}
              <div className="lg:hidden flex items-center justify-center py-2 text-[#825100]">
                <div className="w-8 h-8 rounded-full bg-[#825100] text-white flex items-center justify-center shadow-md">
                  <span className="material-symbols-outlined text-[20px] font-bold">arrow_downward</span>
                </div>
              </div>
            </div>

            {/* Step 4 */}
            <div className="flex flex-col gap-2 relative">
              <div
                onClick={() => onNavigate('leaf-analysis')}
                className="bg-[#eaedff] rounded-2xl p-6 flex flex-col gap-4 shadow-sm border border-[#dae2fd] hover:shadow-md hover:border-[#ba1a1a] transition-all cursor-pointer group h-full relative"
              >
                <div className="flex items-center justify-between">
                  <span className="w-8 h-8 rounded-full bg-[#dae2fd] text-[#131b2e] font-mono text-xs flex items-center justify-center font-bold group-hover:bg-[#ba1a1a] group-hover:text-white transition-colors">
                    04
                  </span>
                  <span className="material-symbols-outlined text-[#ba1a1a] text-[24px]">layers</span>
                </div>
                <div className="flex flex-col gap-1">
                  <h4 className="text-base font-bold text-[#131b2e] group-hover:text-[#ba1a1a] transition-colors">
                    Segment Affected Area
                  </h4>
                  <span className="text-xs text-[#ba1a1a] font-semibold">Healthy vs. Unhealthy Ratio</span>
                </div>
                <p className="text-xs text-[#3d4a42] leading-relaxed">
                  Model 4 U-Net generates pixel segmentation masks isolating affected necrotic regions from healthy tissue with split slider analysis.
                </p>

                <div className="mt-auto pt-2 flex items-center gap-1.5 text-xs font-bold text-[#ba1a1a] group-hover:translate-x-1 transition-transform">
                  <span>Analyze Healthy %</span>
                  <span className="material-symbols-outlined text-[16px]">arrow_forward</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 4. Research & Benchmarks Callout */}
      <section className="max-w-7xl mx-auto px-6 lg:px-12 pb-24 w-full">
        <div className="bg-[#e2e7ff] rounded-3xl p-8 lg:p-12 relative overflow-hidden shadow-sm border border-[#dae2fd]">
          {/* Background subtle technical gradient */}
          <div className="absolute -right-24 -bottom-24 w-96 h-96 bg-[#85f8c4]/40 rounded-full blur-3xl pointer-events-none" />

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center relative z-10">
            {/* Left text summary */}
            <div className="lg:col-span-5 flex flex-col gap-4">
              <div className="flex items-center gap-2">
                <span className="px-3 py-1 rounded-full bg-white text-[#131b2e] font-mono text-xs font-semibold shadow-sm">
                  Agro-CV Benchmarks v2.4
                </span>
                <span className="w-2 h-2 rounded-full bg-[#006948]" />
              </div>
              <h3 className="text-xl font-bold text-[#131b2e]">
                Validated Machine Learning for Tropical Foliar Pathology
              </h3>
              <p className="text-sm text-[#3d4a42] leading-relaxed">
                Engineered specifically to solve real-world plantation diagnostics under varying
                illumination, moisture glares, and lens distortions.
              </p>
              <div className="pt-2">
                <button
                  onClick={() => onNavigate('about-models')}
                  className="inline-flex items-center gap-1.5 text-[#006948] font-bold text-sm hover:underline cursor-pointer"
                >
                  <span>Inspect Full Validation Loss Curves</span>
                  <span className="material-symbols-outlined text-[18px]">arrow_forward</span>
                </button>
              </div>
            </div>

            {/* Right metric cards cluster */}
            <div className="lg:col-span-7 grid grid-cols-1 sm:grid-cols-3 gap-4">
              {/* Metric 1 */}
              <div className="bg-white rounded-2xl p-6 shadow-sm flex flex-col justify-between gap-3 border border-[#dae2fd]">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-[#3d4a42] uppercase font-semibold">Accuracy</span>
                  <span className="material-symbols-outlined text-[20px] text-[#006948]">verified</span>
                </div>
                <div className="flex flex-col">
                  <span className="text-3xl font-extrabold text-[#131b2e] tracking-tight">97.4%</span>
                  <span className="font-mono text-xs text-[#006948] font-semibold">Cross-Validation Score</span>
                </div>
                <span className="text-xs text-[#3d4a42]">Across Sigatoka, Cordana, and Healthy test splits</span>
              </div>

              {/* Metric 2 */}
              <div className="bg-white rounded-2xl p-6 shadow-sm flex flex-col justify-between gap-3 border border-[#dae2fd]">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-[#3d4a42] uppercase font-semibold">Latency</span>
                  <span className="material-symbols-outlined text-[20px] text-[#006a61]">speed</span>
                </div>
                <div className="flex flex-col">
                  <span className="text-3xl font-extrabold text-[#131b2e] tracking-tight">&lt;0.6s</span>
                  <span className="font-mono text-xs text-[#006a61] font-semibold">Inference Latency</span>
                </div>
                <span className="text-xs text-[#3d4a42]">End-to-end FastAPI endpoint response time</span>
              </div>

              {/* Metric 3 */}
              <div className="bg-white rounded-2xl p-6 shadow-sm flex flex-col justify-between gap-3 border border-[#dae2fd]">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-[#3d4a42] uppercase font-semibold">Corpus</span>
                  <span className="material-symbols-outlined text-[20px] text-[#825100]">collections_bookmark</span>
                </div>
                <div className="flex flex-col">
                  <span className="text-3xl font-extrabold text-[#131b2e] tracking-tight">12,000+</span>
                  <span className="font-mono text-xs text-[#825100] font-semibold">Annotated Musa Leaves</span>
                </div>
                <span className="text-xs text-[#3d4a42]">Segmented bounding coordinates &amp; masks</span>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};
