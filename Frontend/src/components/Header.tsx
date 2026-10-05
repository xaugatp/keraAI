import React, { useEffect, useState } from 'react';
import { ViewTab } from '../types';
import { useHealth } from '../hooks/useHealth';
import type { HealthStatus } from '../hooks/useHealth';
import { useModels } from '../hooks/useModels';

// Real server state for the status pill (replaces the former hard-coded "Ready (42ms)").
const HEALTH_STYLE: Record<HealthStatus, { label: string; dot: string; ping: boolean; pill: string }> = {
  checking: { label: 'API: Checking…', dot: 'bg-[#6d7a72]', ping: false, pill: 'bg-[#f2f3ff] border-[#dae2fd]' },
  ok: { label: 'API: Ready', dot: 'bg-[#006948]', ping: true, pill: 'bg-[#f2f3ff] border-[#dae2fd]' },
  degraded: { label: 'API: Degraded', dot: 'bg-[#a36700]', ping: false, pill: 'bg-[#ffeed2] border-[#ffb95f]' },
  offline: { label: 'API: Offline', dot: 'bg-[#ba1a1a]', ping: false, pill: 'bg-[#ffdad6] border-[#ba1a1a]/30' },
};

const DEMO_WEIGHTS_TITLE = 'Results come from placeholder weights and are not real predictions';

interface HeaderProps {
  currentTab: ViewTab;
  onNavigate: (tab: ViewTab) => void;
}

export const Header: React.FC<HeaderProps> = ({ currentTab, onNavigate }) => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const health = useHealth();
  const { anyPlaceholder, error: modelsError, reload: reloadModels } = useModels();

  // If the model list failed to load because the server was down, retry once it is reachable.
  useEffect(() => {
    if (health.status === 'ok' && modelsError) reloadModels();
  }, [health.status]); // re-run only on status changes, not on every failed reload

  const healthStyle = HEALTH_STYLE[health.status];
  const healthTitle =
    health.status === 'ok'
      ? 'KeraAI server is ready'
      : health.status === 'degraded'
        ? `KeraAI server is degraded${health.failing.length ? ` — not ready: ${health.failing.join(', ')}` : ''}`
        : health.status === 'offline'
          ? 'Cannot reach the KeraAI server. It may be offline.'
          : 'Checking the KeraAI server…';

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
      <div className="h-16 sm:h-20 max-w-7xl mx-auto px-4 sm:px-6 lg:px-12 flex items-center justify-between gap-2 sm:gap-4">
        {/* Brand Lockup — a single self-contained glyph (no remote image/network dependency, which
            on a phone can be slow or blocked and would otherwise leave the header blank). */}
        <button
          type="button"
          onClick={() => onNavigate('home')}
          className="flex items-center gap-2 text-left focus:outline-none group cursor-pointer shrink-0"
        >
          <span
            aria-hidden="true"
            className="text-[22px] sm:text-[26px] leading-none transition-transform group-hover:scale-110"
          >
            🍌
          </span>
          <span className="font-extrabold text-[17px] sm:text-[19px] tracking-tight text-[#131b2e]">
            KERA AI
          </span>
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
          {anyPlaceholder && (
            <span
              role="note"
              title={DEMO_WEIGHTS_TITLE}
              aria-label={DEMO_WEIGHTS_TITLE}
              className="hidden lg:inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-[#ffeed2] border border-[#ffb95f] font-mono text-[11px] font-bold text-[#825100] whitespace-nowrap cursor-help"
            >
              <span className="material-symbols-outlined text-[14px]">science</span>
              Demo weights
            </span>
          )}

          <div
            role="status"
            title={healthTitle}
            aria-label={healthTitle}
            className={`hidden lg:flex items-center gap-2 px-3 py-1 rounded-full border ${healthStyle.pill}`}
          >
            <span className="relative flex h-2 w-2">
              {healthStyle.ping && (
                <span className={`animate-ping absolute inline-flex h-full w-full rounded-full ${healthStyle.dot} opacity-75`}></span>
              )}
              <span className={`relative inline-flex rounded-full h-2 w-2 ${healthStyle.dot}`}></span>
            </span>
            <span className="font-mono text-[11px] font-medium text-[#3d4a42] whitespace-nowrap">
              {healthStyle.label}
            </span>
          </div>

          {/* Hidden below sm: "Detect" is already the first item in the hamburger menu, and on a
              narrow phone this button plus the brand and the hamburger button would not fit on
              one row without wrapping or shrinking illegibly. */}
          <button
            type="button"
            onClick={() => onNavigate('detect')}
            className="hidden sm:inline-flex items-center gap-1.5 px-3.5 py-2 md:px-4 md:py-2.5 rounded-xl bg-[#006948] text-white text-xs md:text-sm font-semibold hover:bg-[#00855d] active:scale-[0.98] transition-all shadow-[0_2px_8px_rgba(0,105,72,0.25)] whitespace-nowrap cursor-pointer"
          >
            <span className="material-symbols-outlined text-[18px]">center_focus_strong</span>
            <span>Start Detection</span>
          </button>

          {/* Mobile hamburger button — a real 44x44px touch target (Apple/Material's minimum),
              not just the icon's own small bounding box. */}
          <button
            type="button"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="md:hidden flex items-center justify-center w-11 h-11 -mr-1 rounded-lg text-[#3d4a42] hover:bg-[#eaedff] active:bg-[#dae2fd] cursor-pointer"
            aria-label="Toggle navigation menu"
            aria-expanded={mobileMenuOpen}
          >
            <span className="material-symbols-outlined text-[26px]">
              {mobileMenuOpen ? 'close' : 'menu'}
            </span>
          </button>
        </div>
      </div>

      {/* Mobile Nav Drawer */}
      {mobileMenuOpen && (
        <div className="md:hidden bg-[#faf8ff] border-b border-[#dae2fd] px-4 sm:px-6 py-4 flex flex-col gap-1.5 shadow-lg max-h-[calc(100dvh-4rem)] overflow-y-auto">
          {navLinks.map((link) => (
            <button
              key={link.id}
              type="button"
              onClick={() => {
                onNavigate(link.id);
                setMobileMenuOpen(false);
              }}
              className={`text-left px-3.5 py-3 rounded-lg text-[15px] font-medium transition-all ${
                isLinkActive(link)
                  ? 'bg-[#00855d] text-white font-semibold'
                  : 'text-[#3d4a42] hover:bg-[#eaedff]'
              }`}
            >
              {link.label}
            </button>
          ))}
          <div
            role="status"
            title={healthTitle}
            aria-label={healthTitle}
            className="pt-2 flex items-center gap-2 font-mono text-xs text-[#3d4a42]"
          >
            <span className={`w-2 h-2 rounded-full ${healthStyle.dot}`}></span>
            {healthStyle.label}
            {health.status === 'degraded' && health.failing.length > 0 && (
              <span className="text-[#825100]">({health.failing.join(', ')})</span>
            )}
          </div>
          {anyPlaceholder && (
            <div
              role="note"
              title={DEMO_WEIGHTS_TITLE}
              aria-label={DEMO_WEIGHTS_TITLE}
              className="flex items-center gap-1.5 font-mono text-xs font-bold text-[#825100]"
            >
              <span className="material-symbols-outlined text-[14px]">science</span>
              Demo weights — results are not real predictions
            </div>
          )}
        </div>
      )}
    </header>
  );
};
