/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
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

export default function App() {
  const [currentTab, setCurrentTab] = useState<ViewTab>('home');
  const [stage1SampleMode, setStage1SampleMode] = useState<'positive' | 'negative'>('positive');
  const [selectedLeafIdx, setSelectedLeafIdx] = useState<number>(0);
  const [fieldGps, setFieldGps] = useState<{ latitude: number; longitude: number; accuracy?: number | null }>({
    latitude: 27.71724,
    longitude: 85.32402,
    accuracy: 4.2,
  });

  const handleNavigate = (tab: ViewTab) => {
    setCurrentTab(tab);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleSelectPlantSample = (sampleType: 'positive' | 'negative' | 'custom', customImg?: string) => {
    if (sampleType === 'negative') {
      setStage1SampleMode('negative');
    } else {
      setStage1SampleMode('positive');
    }
  };

  return (
    <div className="bg-[#faf8ff] text-[#131b2e] min-h-screen flex flex-col font-sans selection:bg-[#85f8c4] selection:text-[#002114]">
      {/* Top Header */}
      <Header currentTab={currentTab} onNavigate={handleNavigate} />

      {/* Main Content Area */}
      <main className="w-full pt-20 bg-[#faf8ff] flex-1 flex flex-col">
        {currentTab === 'home' && <HomeView onNavigate={handleNavigate} />}

        {(currentTab === 'detect' || currentTab === 'detection-workspace') && (
          <DetectionWorkspaceView
            onNavigate={handleNavigate}
            onSelectSample={handleSelectPlantSample}
            onGpsChange={setFieldGps}
          />
        )}

        {(currentTab === 'plant-result' || currentTab === 'stage1-result') && (
          <Stage1ResultView
            onNavigate={handleNavigate}
            activeStateMode={stage1SampleMode}
            gpsLocation={fieldGps}
          />
        )}

        {(currentTab === 'detect-disease' || currentTab === 'leaf-detect') && (
          <DetectDiseaseView
            onNavigate={handleNavigate}
            selectedDiagnosisIndex={selectedLeafIdx}
          />
        )}

        {(currentTab === 'leaf-analysis' || currentTab === 'leaf-result') && (
          <LeafAnalysisView
            onNavigate={handleNavigate}
            selectedDiagnosisIndex={selectedLeafIdx}
          />
        )}

        {(currentTab === 'history' || currentTab === 'analysis-history') && (
          <AnalysisHistoryView
            onNavigate={handleNavigate}
            onSelectAnalysis={(idx) => {
              setSelectedLeafIdx(idx);
              handleNavigate('detect-disease');
            }}
          />
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
