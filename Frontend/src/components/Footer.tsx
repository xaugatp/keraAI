import React, { useState } from 'react';
import { ViewTab } from '../types';

interface FooterProps {
  onNavigate: (tab: ViewTab) => void;
}

export const Footer: React.FC<FooterProps> = ({ onNavigate }) => {
  const [showContactModal, setShowContactModal] = useState(false);

  return (
    <footer className="w-full bg-white mt-auto border-t border-[#dae2fd]/80 shadow-[0_-1px_6px_rgba(0,0,0,0.03)]">
      {/* Contact Modal */}
      {showContactModal && (
        <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4 animate-fade-in">
          <div className="bg-white max-w-md w-full rounded-2xl p-6 shadow-2xl border border-[#dae2fd] flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-extrabold text-[#131b2e]">Research &amp; Demonstration Contact</h3>
              <button
                type="button"
                onClick={() => setShowContactModal(false)}
                className="w-8 h-8 rounded-full bg-[#f2f3ff] text-[#3d4a42] hover:bg-[#eaedff] flex items-center justify-center cursor-pointer"
              >
                ✕
              </button>
            </div>
            <p className="text-xs text-[#3d4a42] leading-relaxed">
              This application was developed as a university demonstration of computer vision models
              for agricultural botanical pathology. For questions regarding the dual-stage ResNet-50
              and U-Net deployment or dataset citations:
            </p>
            <div className="p-3 bg-[#f2f3ff] rounded-xl border border-[#dae2fd] flex flex-col gap-1 text-xs font-mono">
              <span className="text-[#3d4a42]">Lead Researcher: Saugat Poudel</span>
              <span className="text-[#006948] font-bold">saugat.poudel@radicalsystems.com.au</span>
              <span className="text-[#3d4a42]">Faculty of Agricultural Sciences &amp; Computer Vision</span>
            </div>
            <button
              type="button"
              onClick={() => setShowContactModal(false)}
              className="w-full py-2.5 rounded-xl bg-[#006948] text-white font-semibold text-xs cursor-pointer hover:bg-[#00855d]"
            >
              Close
            </button>
          </div>
        </div>
      )}

      <div className="max-w-7xl mx-auto px-6 lg:px-12 py-8 flex flex-col md:flex-row items-center justify-between gap-6 text-[#3d4a42]">
        <div className="flex flex-col items-center md:items-start gap-1">
          <div className="flex items-center gap-2">
            <span className="font-extrabold text-sm text-[#131b2e]">BananaVision AI</span>
            <span className="text-xs text-[#bccac0]">&bull;</span>
            <span className="font-mono text-xs text-[#006948] font-semibold">KERA AI</span>
            <span className="px-2 py-0.5 rounded-full bg-[#eaedff] font-mono text-[10px] text-[#3d4a42] border border-[#dae2fd]">
              Computer Vision Project
            </span>
          </div>
          <p className="text-xs text-[#3d4a42] text-center md:text-left max-w-xl">
            Two-stage deep learning platform for real-time Musa classification and semantic foliar disease segmentation.
          </p>
        </div>

        <div className="flex flex-wrap items-center justify-center gap-6 text-xs font-medium">
          <button
            type="button"
            onClick={() => onNavigate('about')}
            className="hover:text-[#006948] transition-colors cursor-pointer"
          >
            About
          </button>
          <a
            className="hover:text-[#006948] transition-colors flex items-center gap-1.5"
            href="https://github.com/academic-vision/kera-ai"
            target="_blank"
            rel="noreferrer"
          >
            <span className="material-symbols-outlined text-[16px]">code</span>
            <span>GitHub</span>
          </a>
          <button
            type="button"
            onClick={() => setShowContactModal(true)}
            className="hover:text-[#006948] transition-colors flex items-center gap-1.5 cursor-pointer"
          >
            <span className="material-symbols-outlined text-[16px]">mail</span>
            <span>Contact</span>
          </button>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 lg:px-12 pb-6 text-center md:text-left border-t border-[#dae2fd]/40 pt-4">
        <p className="text-[11px] text-[#3d4a42]/80">
          &copy; 2024–2026 BananaVision AI / KERA AI Academic Project. Distributed under MIT License
          for non-commercial botanical disease diagnosis research.
        </p>
      </div>
    </footer>
  );
};
