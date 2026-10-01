import React, { useState } from 'react';
import { ViewTab } from '../types';
import { ASSETS } from '../data/mockData';

interface AnalysisHistoryViewProps {
  onNavigate: (tab: ViewTab) => void;
  onSelectAnalysis?: (sampleIdx: number) => void;
}

export const AnalysisHistoryView: React.FC<AnalysisHistoryViewProps> = ({
  onNavigate,
  onSelectAnalysis,
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterSeverity, setFilterSeverity] = useState<string>('all');
  const [viewLayout, setViewLayout] = useState<'cards' | 'table'>('cards');

  const historyItems = [
    {
      id: 'KER-9924-MUSA',
      dateTime: 'Today — 10:42 AM',
      plantResult: 'Banana Tree ✓',
      leafHealth: 'Unhealthy Leaf ⚠',
      disease: 'Black Sigatoka',
      confidence: '93.8%',
      infectedArea: '27.4%',
      status: 'severe',
      imageUrl: ASSETS.sigatokaLeafSample,
      sampleIdx: 0,
    },
    {
      id: 'KER-9923-MUSA',
      dateTime: 'Today — 09:15 AM',
      plantResult: 'Banana Tree ✓',
      leafHealth: 'Healthy Leaf ✓',
      disease: 'No disease detected',
      confidence: '96.2%',
      infectedArea: '0.0%',
      status: 'healthy',
      imageUrl: ASSETS.healthyLeafControl,
      sampleIdx: 1,
    },
    {
      id: 'KER-9921-MUSA',
      dateTime: 'Yesterday — 04:40 PM',
      plantResult: 'Banana Tree ✓',
      leafHealth: 'Unhealthy Leaf ⚠',
      disease: 'Yellow Sigatoka (Early)',
      confidence: '88.4%',
      infectedArea: '11.2%',
      status: 'moderate',
      imageUrl: ASSETS.sigatokaLeafSample,
      sampleIdx: 2,
    },
    {
      id: 'KER-9920-MUSA',
      dateTime: 'Oct 24 — 09:22 AM',
      plantResult: 'Banana Tree ✓',
      leafHealth: 'Unhealthy Leaf ⚠',
      disease: 'Cordana Leaf Spot',
      confidence: '91.0%',
      infectedArea: '4.8%',
      status: 'low',
      imageUrl: ASSETS.healthyLeafControl,
      sampleIdx: 3,
    },
    {
      id: 'KER-9919-REJ',
      dateTime: 'Oct 23 — 02:05 PM',
      plantResult: 'Banana Tree Not Detected ✕',
      leafHealth: 'N/A (Species Mismatch)',
      disease: 'Non-Musa Foliage',
      confidence: '91.3%',
      infectedArea: 'N/A',
      status: 'rejected',
      imageUrl: ASSETS.negativeHouseplant,
      sampleIdx: -1,
    },
  ];

  const filteredItems = historyItems.filter((item) => {
    const matchesSearch =
      item.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.disease.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.plantResult.toLowerCase().includes(searchTerm.toLowerCase());

    if (filterSeverity === 'all') return matchesSearch;
    if (filterSeverity === 'severe') return matchesSearch && item.status === 'severe';
    if (filterSeverity === 'healthy') return matchesSearch && item.status === 'healthy';
    if (filterSeverity === 'moderate') return matchesSearch && (item.status === 'moderate' || item.status === 'low');
    if (filterSeverity === 'rejected') return matchesSearch && item.status === 'rejected';
    return matchesSearch;
  });

  const handleCardClick = (item: typeof historyItems[0]) => {
    if (item.status === 'rejected') {
      onNavigate('plant-result');
    } else {
      if (onSelectAnalysis && item.sampleIdx >= 0) {
        onSelectAnalysis(item.sampleIdx);
      }
      onNavigate('detect-disease');
    }
  };

  const exportCSV = () => {
    const headers = 'ID,Date,PlantResult,LeafHealth,Disease,Confidence,InfectedArea,Status\n';
    const rows = historyItems
      .map(
        (i) =>
          `"${i.id}","${i.dateTime}","${i.plantResult}","${i.leafHealth}","${i.disease}","${i.confidence}","${i.infectedArea}","${i.status}"`
      )
      .join('\n');
    const blob = new Blob([headers + rows], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'bananavision-analysis-history.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="w-full max-w-7xl mx-auto px-6 lg:px-12 py-10 flex flex-col gap-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-xs font-mono uppercase text-[#3d4a42] font-semibold">
            <span>DATABASE</span>
            <span>/</span>
            <span className="text-[#006948] font-bold">ANALYSIS HISTORY</span>
          </div>
          <h1 className="text-3xl font-extrabold text-[#131b2e] mt-1 tracking-tight">
            Analysis History
          </h1>
          <p className="text-sm text-[#3d4a42] mt-1 max-w-2xl leading-relaxed">
            Review previous field scans and complete AI predictions. Click any record to inspect the
            full segmentation and confidence breakdown.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={exportCSV}
            className="px-4 py-2 rounded-xl bg-white text-[#131b2e] text-xs font-semibold border border-[#dae2fd] hover:bg-[#eaedff] transition-all flex items-center gap-2 shadow-sm cursor-pointer"
          >
            <span className="material-symbols-outlined text-[16px]">file_download</span>
            <span>Export CSV</span>
          </button>
          <button
            type="button"
            onClick={() => onNavigate('detect')}
            className="px-4 py-2 rounded-xl bg-[#006948] text-white text-xs font-semibold hover:bg-[#00855d] transition-all flex items-center gap-2 shadow-sm cursor-pointer"
          >
            <span className="material-symbols-outlined text-[16px]">add</span>
            <span>Start Detection</span>
          </button>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="bg-white p-4 rounded-2xl shadow-sm border border-[#dae2fd] flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="relative w-full sm:w-80">
          <span className="material-symbols-outlined absolute left-3 top-2.5 text-[#3d4a42] text-[18px]">
            search
          </span>
          <input
            type="text"
            placeholder="Search disease, result, or ID..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-9 pr-4 py-2 rounded-lg bg-[#f2f3ff] text-xs text-[#131b2e] border border-[#dae2fd] focus:outline-none focus:ring-2 focus:ring-[#006948]"
          />
        </div>

        <div className="flex items-center justify-between sm:justify-end gap-3 w-full sm:w-auto">
          {/* Severity tabs */}
          <div className="flex items-center gap-1.5 overflow-x-auto pb-1 sm:pb-0">
            {[
              { id: 'all', label: 'All' },
              { id: 'severe', label: 'Unhealthy' },
              { id: 'healthy', label: 'Healthy' },
              { id: 'rejected', label: 'Rejected' },
            ].map((tab) => (
              <button
                key={tab.id}
                type="button"
                onClick={() => setFilterSeverity(tab.id)}
                className={`px-3 py-1.5 rounded-lg text-xs font-mono font-medium transition-all cursor-pointer ${
                  filterSeverity === tab.id
                    ? 'bg-[#006948] text-white font-bold shadow-sm'
                    : 'bg-[#f2f3ff] text-[#3d4a42] hover:bg-[#eaedff]'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* Layout Toggle */}
          <div className="flex items-center gap-1 bg-[#f2f3ff] p-1 rounded-lg border border-[#dae2fd]">
            <button
              type="button"
              onClick={() => setViewLayout('cards')}
              className={`p-1.5 rounded transition-all cursor-pointer ${
                viewLayout === 'cards' ? 'bg-white text-[#006948] shadow-sm' : 'text-[#3d4a42]'
              }`}
              title="Card View"
            >
              <span className="material-symbols-outlined text-[16px]">grid_view</span>
            </button>
            <button
              type="button"
              onClick={() => setViewLayout('table')}
              className={`p-1.5 rounded transition-all cursor-pointer ${
                viewLayout === 'table' ? 'bg-white text-[#006948] shadow-sm' : 'text-[#3d4a42]'
              }`}
              title="Table View"
            >
              <span className="material-symbols-outlined text-[16px]">view_list</span>
            </button>
          </div>
        </div>
      </div>

      {/* CARD VIEW LAYOUT (Requested in prompt) */}
      {viewLayout === 'cards' ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredItems.map((item) => (
            <div
              key={item.id}
              onClick={() => handleCardClick(item)}
              className="bg-white rounded-2xl p-5 shadow-sm border border-[#dae2fd] hover:border-[#006948] hover:shadow-md transition-all flex flex-col justify-between gap-4 cursor-pointer group"
            >
              <div className="flex items-start gap-4">
                {/* Thumbnail */}
                <div className="w-20 h-20 rounded-xl overflow-hidden bg-[#283044] shrink-0 border border-[#dae2fd]">
                  <img
                    alt={item.disease}
                    src={item.imageUrl}
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                  />
                </div>

                {/* Details */}
                <div className="flex flex-col min-w-0 flex-1">
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-xs font-semibold text-[#3d4a42]">
                      {item.dateTime}
                    </span>
                    <span className="font-mono text-xs font-bold text-[#131b2e]">
                      {item.confidence}
                    </span>
                  </div>

                  {/* Plant Detection Result */}
                  <div className="mt-1 flex items-center gap-1 font-mono text-xs font-bold text-[#006948]">
                    <span>{item.plantResult}</span>
                  </div>

                  {/* Leaf Health & Disease */}
                  <div
                    className={`mt-1 font-mono text-xs font-bold flex items-center gap-1 ${
                      item.status === 'healthy'
                        ? 'text-[#006948]'
                        : item.status === 'rejected'
                        ? 'text-[#ba1a1a]'
                        : 'text-[#ba1a1a]'
                    }`}
                  >
                    <span>{item.leafHealth}</span>
                  </div>

                  <h3 className="text-sm font-extrabold text-[#131b2e] truncate mt-0.5">
                    {item.disease}
                  </h3>
                </div>
              </div>

              <div className="pt-3 border-t border-[#eaedff] flex items-center justify-between font-mono text-[11px] text-[#3d4a42]">
                <span>ID: {item.id}</span>
                <span className="text-[#006948] font-bold group-hover:underline flex items-center gap-0.5">
                  <span>View Complete Analysis</span>
                  <span className="material-symbols-outlined text-[14px]">arrow_forward</span>
                </span>
              </div>
            </div>
          ))}
        </div>
      ) : (
        /* TABLE VIEW */
        <div className="bg-white rounded-2xl shadow-sm border border-[#dae2fd] overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-[#f2f3ff] border-b border-[#dae2fd] text-[11px] font-mono uppercase text-[#3d4a42]">
                  <th className="py-3 px-6">Thumbnail &amp; ID</th>
                  <th className="py-3 px-4">Date/Time</th>
                  <th className="py-3 px-4">Plant Detection</th>
                  <th className="py-3 px-4">Leaf Health</th>
                  <th className="py-3 px-4">Disease</th>
                  <th className="py-3 px-4">Confidence</th>
                  <th className="py-3 px-6 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#eaedff] text-xs">
                {filteredItems.map((item) => (
                  <tr
                    key={item.id}
                    onClick={() => handleCardClick(item)}
                    className="hover:bg-[#faf8ff] transition-colors cursor-pointer"
                  >
                    <td className="py-3 px-6 flex items-center gap-3">
                      <img
                        alt={item.disease}
                        src={item.imageUrl}
                        className="w-10 h-8 rounded-lg object-cover border border-[#dae2fd]"
                      />
                      <span className="font-mono text-xs font-bold text-[#131b2e]">
                        {item.id}
                      </span>
                    </td>
                    <td className="py-3 px-4 font-mono text-xs text-[#3d4a42]">{item.dateTime}</td>
                    <td className="py-3 px-4 font-mono text-xs font-bold text-[#006948]">
                      {item.plantResult}
                    </td>
                    <td className="py-3 px-4 font-mono text-xs font-bold">
                      <span
                        className={
                          item.status === 'healthy' ? 'text-[#006948]' : 'text-[#ba1a1a]'
                        }
                      >
                        {item.leafHealth}
                      </span>
                    </td>
                    <td className="py-3 px-4 font-bold text-[#131b2e]">{item.disease}</td>
                    <td className="py-3 px-4 font-mono font-bold text-[#131b2e]">
                      {item.confidence}
                    </td>
                    <td className="py-3 px-6 text-right">
                      <span className="text-xs font-semibold text-[#006948] hover:underline">
                        View &rarr;
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
