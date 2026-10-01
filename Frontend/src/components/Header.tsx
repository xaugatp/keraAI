import React, { useState } from 'react';
import { ViewTab } from '../types';
import { ASSETS } from '../data/mockData';

interface HeaderProps {
  currentTab: ViewTab;
  onNavigate: (tab: ViewTab) => void;
}

export const Header: React.FC<HeaderProps> = ({ currentTab, onNavigate }) => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navLinks: { id: ViewTab; label: string }[] = [
    { id: 'home', label: 'Home' },
    { id: 'detect', label: 'Detect' },
    { id: 'leaf-analysis', label: 'Leaf Analysis' },
    { id: 'detect-disease', label: 'Detect Disease' },
    { id: 'history', label: 'History' },
    { id: 'about', label: 'About' },
  ];

  const isLinkActive = (item: typeof navLinks[0]) => {
    if (currentTab === item.id) return true;
    if (item.id === 'detect' && (currentTab === 'plant-result' || currentTab === 'stage1-result' || currentTab === 'detection-workspace')) return true;
    if (item.id === 'detect-disease' && currentTab === 'leaf-detect') return true;
    if (item.id === 'leaf-analysis' && currentTab === 'leaf-result') return true;
    if (item.id === 'history' && currentTab === 'analysis-history') return true;
    if (item.id === 'about' && currentTab === 'about-models') return true;
    return false;
  };

  return (
    <header className="fixed top-0 left-0 right-0 z-40 bg-[#faf8ff]/95 backdrop-blur-xl border-b border-[#e2e8f0]/80 shadow-[0_1px_8px_rgba(0,0,0,0.03)]">
      <div className="h-20 max-w-7xl mx-auto px-6 lg:px-12 flex items-center justify-between gap-4">
        {/* Brand Lockup */}
        <button
          type="button"
          onClick={() => onNavigate('home')}
          className="flex items-center gap-3 text-left focus:outline-none group cursor-pointer"
        >
          <img
            alt="BananaVision AI Logo"
            className="h-9 w-auto object-contain transition-transform group-hover:scale-105"
            src={ASSETS.logo}
            onError={(e) => {
              e.currentTarget.style.display = 'none';
            }}
          />
          <div className="flex flex-col">
            <div className="flex items-center gap-2">
              <span className="font-extrabold text-[18px] tracking-tight text-[#131b2e]">
                BananaVision AI
              </span>
              <span className="px-2 py-0.5 rounded-full bg-[#eaedff] text-[#006948] font-mono text-[10px] font-bold uppercase tracking-wider border border-[#dae2fd]">
                KERA
              </span>
            </div>
            <span className="text-[12px] text-[#3d4a42] font-medium hidden sm:inline-block leading-tight">
              Academic Computer Vision Project Demo
            </span>
          </div>
        </button>

        {/* Desktop Navigation */}
        <nav className="hidden md:flex items-center gap-1 lg:gap-1.5">
          {navLinks.map((link) => {
            const active = isLinkActive(link);
            return (
              <button
                key={link.id}
                type="button"
                onClick={() => onNavigate(link.id)}
                className={`px-3 py-2 rounded-lg text-xs lg:text-sm font-medium transition-all cursor-pointer whitespace-nowrap ${
                  active
                    ? 'bg-[#00855d] text-white font-semibold shadow-sm'
                    : 'text-[#3d4a42] hover:text-[#131b2e] hover:bg-[#eaedff]/60'
                }`}
              >
                {link.label}
              </button>
            );
          })}
        </nav>

        {/* Right Status & Actions */}
        <div className="flex items-center gap-2 sm:gap-3">
          <div className="hidden lg:flex items-center gap-2 px-3 py-1 rounded-full bg-[#f2f3ff] border border-[#dae2fd]">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#006948] opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-[#006948]"></span>
            </span>
            <span className="font-mono text-[11px] font-medium text-[#3d4a42] whitespace-nowrap">
              FastAPI: Ready (42ms)
            </span>
          </div>

          <button
            type="button"
            onClick={() => onNavigate('detect')}
            className="inline-flex items-center gap-1.5 px-3.5 py-2 sm:px-4 sm:py-2.5 rounded-xl bg-[#006948] text-white text-xs sm:text-sm font-semibold hover:bg-[#00855d] active:scale-[0.98] transition-all shadow-[0_2px_8px_rgba(0,105,72,0.25)] whitespace-nowrap cursor-pointer"
          >
            <span className="material-symbols-outlined text-[18px]">center_focus_strong</span>
            <span>Start Detection</span>
          </button>

          {/* Mobile hamburger button */}
          <button
            type="button"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="md:hidden p-2 rounded-lg text-[#3d4a42] hover:bg-[#eaedff] cursor-pointer"
            aria-label="Toggle navigation menu"
          >
            <span className="material-symbols-outlined text-[24px]">
              {mobileMenuOpen ? 'close' : 'menu'}
            </span>
          </button>
        </div>
      </div>

      {/* Mobile Nav Drawer */}
      {mobileMenuOpen && (
        <div className="md:hidden bg-[#faf8ff] border-b border-[#dae2fd] px-6 py-4 flex flex-col gap-2 shadow-lg">
          {navLinks.map((link) => (
            <button
              key={link.id}
              type="button"
              onClick={() => {
                onNavigate(link.id);
                setMobileMenuOpen(false);
              }}
              className={`text-left px-3.5 py-2.5 rounded-lg text-sm font-medium transition-all ${
                isLinkActive(link)
                  ? 'bg-[#00855d] text-white font-semibold'
                  : 'text-[#3d4a42] hover:bg-[#eaedff]'
              }`}
            >
              {link.label}
            </button>
          ))}
          <div className="pt-2 flex items-center gap-2 font-mono text-xs text-[#3d4a42]">
            <span className="w-2 h-2 rounded-full bg-[#006948]"></span>
            FastAPI Tensor Core: Online (42ms)
          </div>
        </div>
      )}
    </header>
  );
};
