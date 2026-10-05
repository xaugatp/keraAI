/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';
import { ViewTab } from './types';
import { Header } from './components/Header';
import { Footer } from './components/Footer';
import { HomeView } from './views/HomeView';
import { DetectionWorkspaceView } from './views/DetectionWorkspaceView';
import { Stage1ResultView } from './views/Stage1ResultView';
import { DetectDiseaseView } from './views/DetectDiseaseView';
import { LeafAnalysisView } from './views/LeafAnalysisView';
import { AnalysisHistoryView } from './views/AnalysisHistoryView';
import { AboutModelsView } from './views/AboutModelsView';
import { AnalysisProvider } from './state/AnalysisContext';

// The provider wraps the whole app so every view can call useAnalysisState().
// AppShell (the former App body, unchanged) sits inside it so it can use the context too.
export default function App() {
  return (
    <AnalysisProvider>
      <AppShell />
    </AnalysisProvider>
  );
}

function AppShell() {
  const [currentTab, setCurrentTab] = useState<ViewTab>('home');

  const handleNavigate = (tab: ViewTab) => {
    setCurrentTab(tab);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div className="bg-[#faf8ff] text-[#131b2e] min-h-screen flex flex-col font-sans selection:bg-[#85f8c4] selection:text-[#002114]">
      {/* Top Header */}
      <Header currentTab={currentTab} onNavigate={handleNavigate} />

      {/* Main Content Area */}
      <main className="w-full pt-20 bg-[#faf8ff] flex-1 flex flex-col">
        {currentTab === 'home' && <HomeView onNavigate={handleNavigate} />}

        {(currentTab === 'detect' || currentTab === 'detection-workspace') && (
          <DetectionWorkspaceView onNavigate={handleNavigate} />
        )}

        {(currentTab === 'plant-result' || currentTab === 'stage1-result') && (
          <Stage1ResultView onNavigate={handleNavigate} />
        )}

        {(currentTab === 'detect-disease' || currentTab === 'leaf-detect') && (
          <DetectDiseaseView onNavigate={handleNavigate} />
        )}

        {(currentTab === 'leaf-analysis' || currentTab === 'leaf-result') && (
          <LeafAnalysisView onNavigate={handleNavigate} />
        )}

        {(currentTab === 'history' || currentTab === 'analysis-history') && (
          <AnalysisHistoryView onNavigate={handleNavigate} />
        )}

        {(currentTab === 'about' || currentTab === 'about-models') && (
          <AboutModelsView onNavigate={handleNavigate} />
        )}
      </main>

      {/* Global Footer */}
      <Footer onNavigate={handleNavigate} />
    </div>
  );
}
