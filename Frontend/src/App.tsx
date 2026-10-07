/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect, useState } from 'react';
import { ViewTab } from './types';
import { Header } from './components/Header';
import { Footer } from './components/Footer';
import { HomeView } from './views/HomeView';
import { DetectionWorkspaceView } from './views/DetectionWorkspaceView';
import { Stage1ResultView } from './views/Stage1ResultView';
import { LeafDiseaseView } from './views/LeafDiseaseView';
import { LeafAnalysisView } from './views/LeafAnalysisView';
import { AnalysisHistoryView } from './views/AnalysisHistoryView';
import { AnalysisDetailView } from './views/AnalysisDetailView';
import { AboutModelsView } from './views/AboutModelsView';
import { AnalysisProvider, useAnalysisState } from './state/AnalysisContext';

// The provider wraps the whole app so every view can call useAnalysisState().
// AppShell (the former App body, unchanged) sits inside it so it can use the context too.
export default function App() {
  return (
    <AnalysisProvider>
      <AppShell />
    </AnalysisProvider>
  );
}

const TAB_STORAGE_KEY = 'keraai.currentTab.v1';

// No router (CLAUDE.md: adding one needs the owner's sign-off), so there is no URL to reload
// against — the browser always serves the same index.html regardless of tab. Persisting the tab
// name here is what lets a reload land back where you were instead of resetting to Home.
const VALID_TABS: ReadonlySet<ViewTab> = new Set<ViewTab>([
  'home',
  'detect',
  'detection-workspace',
  'plant-result',
  'stage1-result',
  'detect-disease',
  'leaf-detect',
  'leaf-analysis',
  'leaf-result',
  'history',
  'analysis-history',
  'history-detail',
  'about',
  'about-models',
]);

function isViewTab(value: string): value is ViewTab {
  return VALID_TABS.has(value as ViewTab);
}

function readStoredTab(): ViewTab | null {
  try {
    const raw = sessionStorage.getItem(TAB_STORAGE_KEY);
    return raw !== null && isViewTab(raw) ? raw : null;
  } catch {
    return null; // private mode / blocked storage: just start on Home
  }
}

function writeStoredTab(tab: ViewTab): void {
  try {
    sessionStorage.setItem(TAB_STORAGE_KEY, tab);
  } catch {
    // private mode / blocked storage: in-memory navigation still works for this page load
  }
}

function AppShell() {
  const [currentTab, setCurrentTab] = useState<ViewTab>(() => readStoredTab() ?? 'home');
  const { tree, leaf, disease, viewing, restoreTree, restoreLeaf, restoreDisease, restoreViewing } =
    useAnalysisState();

  const handleNavigate = (tab: ViewTab) => {
    setCurrentTab(tab);
    writeStoredTab(tab);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  // Reload lands back on the tab above, but `tree`/`leaf`/`disease`/`viewing` start empty every page
  // load (an AnalysisDetail is never itself persisted — only its id, see AnalysisContext). Re-fetch
  // exactly the one piece of data the restored tab needs. This runs once, for the tab the page
  // loaded with, not on every navigation — ordinary in-app navigation already carries fresh data via
  // setTree/setLeaf/setDisease/setViewing at the point of navigation. If the fetch fails (deleted
  // record, backend unreachable), the slot just stays null and the view shows its normal "nothing
  // selected" state rather than a crash.
  useEffect(() => {
    if ((currentTab === 'plant-result' || currentTab === 'stage1-result') && !tree) {
      restoreTree().catch(() => {});
    } else if ((currentTab === 'leaf-analysis' || currentTab === 'leaf-result') && !leaf) {
      restoreLeaf().catch(() => {});
    } else if ((currentTab === 'detect-disease' || currentTab === 'leaf-detect') && !disease) {
      restoreDisease().catch(() => {});
    } else if (currentTab === 'history-detail' && !viewing) {
      restoreViewing().catch(() => {});
    }
    // Deliberately empty deps: see the comment above — this must run exactly once, on mount.
  }, []);

  return (
    <div className="bg-[#faf8ff] text-[#131b2e] min-h-screen flex flex-col font-sans selection:bg-[#85f8c4] selection:text-[#002114]">
      {/* Top Header */}
      <Header currentTab={currentTab} onNavigate={handleNavigate} />

      {/* Main Content Area */}
      {/* pt-16/pt-20 must track the header's own h-16 sm:h-20 (Header.tsx), or content either
          hides under the fixed header on phones or leaves a gap above it on desktop. */}
      <main className="w-full pt-16 sm:pt-20 bg-[#faf8ff] flex-1 flex flex-col">
        {currentTab === 'home' && <HomeView onNavigate={handleNavigate} />}

        {(currentTab === 'detect' || currentTab === 'detection-workspace') && (
          <DetectionWorkspaceView onNavigate={handleNavigate} />
        )}

        {(currentTab === 'plant-result' || currentTab === 'stage1-result') && (
          <Stage1ResultView onNavigate={handleNavigate} />
        )}

        {(currentTab === 'detect-disease' || currentTab === 'leaf-detect') && (
          <LeafDiseaseView onNavigate={handleNavigate} />
        )}

        {(currentTab === 'leaf-analysis' || currentTab === 'leaf-result') && (
          <LeafAnalysisView onNavigate={handleNavigate} />
        )}

        {(currentTab === 'history' || currentTab === 'analysis-history') && (
          <AnalysisHistoryView onNavigate={handleNavigate} />
        )}

        {currentTab === 'history-detail' && <AnalysisDetailView onNavigate={handleNavigate} />}

        {(currentTab === 'about' || currentTab === 'about-models') && (
          <AboutModelsView onNavigate={handleNavigate} />
        )}
      </main>

      {/* Global Footer */}
      <Footer onNavigate={handleNavigate} />
    </div>
  );
}
