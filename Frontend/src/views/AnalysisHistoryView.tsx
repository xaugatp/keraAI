import React, { useMemo, useState } from 'react';
import { ViewTab } from '../types';
import { useAnalysisState } from '../state/AnalysisContext';
import { useHistory } from '../hooks/useHistory';
import { absoluteImageUrl, getAnalysis, userMessageFor } from '../api';
import type { AnalysisSummary, ModelKey, Scope } from '../api';

interface AnalysisHistoryViewProps {
  onNavigate: (tab: ViewTab) => void;
}

const PAGE_SIZE = 12;

const SCOPE_TABS: { id: Scope; label: string }[] = [
  { id: 'all_visible', label: 'All' },
  { id: 'mine', label: 'My Scans' },
  { id: 'samples', label: 'Samples' },
];

const MODEL_TABS: { id: ModelKey | 'all'; label: string }[] = [
  { id: 'all', label: 'Both Models' },
  { id: 'tree_classification', label: 'Model 1 · Tree' },
  { id: 'leaf_segmentation', label: 'Model 2 · Leaf' },
];

// The result view each model's history row opens into (this is a saved row, not a fresh photo —
// there is no File to re-submit, so it opens read-only, same as a sample).
const RESULT_TAB: Partial<Record<ModelKey, ViewTab>> = {
  tree_classification: 'stage1-result',
  leaf_segmentation: 'leaf-analysis',
};

function fmtDate(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function fmtConfidence(value: number | null): string {
  return value == null ? '—' : `${(value * 100).toFixed(1)}%`;
}

function fmtCoords(lat: number | null | undefined, lon: number | null | undefined): string {
  if (lat == null || lon == null) return '—';
  return `${lat.toFixed(4)}°, ${lon.toFixed(4)}°`;
}

function csvCell(value: string): string {
  return `"${value.replace(/"/g, '""')}"`;
}

export const AnalysisHistoryView: React.FC<AnalysisHistoryViewProps> = ({ onNavigate }) => {
  const { setFile, setTree, setLeaf } = useAnalysisState();

  const [scope, setScope] = useState<Scope>('all_visible');
  const [modelFilter, setModelFilter] = useState<ModelKey | 'all'>('all');
  const [page, setPage] = useState(1);
  const [searchTerm, setSearchTerm] = useState('');
  const [viewLayout, setViewLayout] = useState<'cards' | 'table'>('cards');
  const [actionError, setActionError] = useState<string | null>(null);
  const [openingId, setOpeningId] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const { page: pageInfo, items, loading, error, remove } = useHistory({
    scope,
    modelKey: modelFilter === 'all' ? undefined : modelFilter,
    page,
    pageSize: PAGE_SIZE,
  });

  const filteredItems = useMemo(() => {
    const q = searchTerm.trim().toLowerCase();
    if (!q) return items;
    // Client-side only: it searches the CURRENT page, not the whole history (there is no
    // server-side search endpoint). Good enough for a page of 12; said nowhere that it's more.
    return items.filter(
      (item) =>
        item.id.toLowerCase().includes(q) ||
        (item.title ?? '').toLowerCase().includes(q) ||
        item.display_label.toLowerCase().includes(q),
    );
  }, [items, searchTerm]);

  const changeScope = (next: Scope) => {
    setScope(next);
    setPage(1);
  };
  const changeModelFilter = (next: ModelKey | 'all') => {
    setModelFilter(next);
    setPage(1);
  };

  const openItem = async (item: AnalysisSummary) => {
    const resultTab = RESULT_TAB[item.model_key];
    if (!resultTab) return; // leaf_disease has no result view yet
    setActionError(null);
    setOpeningId(item.id);
    try {
      // Fresh fetch (not the summary row): signed image URLs expire, and only the detail has them.
      const detail = await getAnalysis(item.id);
      setFile(null); // a saved row has no original File to carry into another model's stage
      if (detail.model_key === 'leaf_segmentation') setLeaf(detail);
      else setTree(detail);
      onNavigate(resultTab);
    } catch (err) {
      setActionError(userMessageFor(err));
    } finally {
      setOpeningId(null);
    }
  };

  const deleteItem = async (item: AnalysisSummary, e: React.MouseEvent) => {
    e.stopPropagation();
    if (item.is_sample) return; // samples are immutable; the button isn't shown for them anyway
    if (!window.confirm('Delete this analysis? This cannot be undone.')) return;
    setActionError(null);
    setDeletingId(item.id);
    try {
      await remove(item.id);
    } catch (err) {
      setActionError(userMessageFor(err));
    } finally {
      setDeletingId(null);
    }
  };

  const exportCSV = () => {
    const headers = ['id', 'model_key', 'source', 'is_sample', 'label', 'confidence', 'latitude', 'longitude', 'created_at'];
    const rows = filteredItems.map((item) =>
      [
        item.id,
        item.model_key,
        item.source,
        String(item.is_sample),
        item.display_label,
        item.confidence != null ? String(item.confidence) : '',
        item.latitude != null ? String(item.latitude) : '',
        item.longitude != null ? String(item.longitude) : '',
        item.created_at,
      ]
        .map(csvCell)
        .join(','),
    );
    const blob = new Blob([[headers.join(','), ...rows].join('\n')], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `keraai-history-${scope}-page${page}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const totalPages = pageInfo?.total_pages ?? 1;

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
          <h1 className="text-3xl font-extrabold text-[#131b2e] mt-1 tracking-tight">Analysis History</h1>
          <p className="text-sm text-[#3d4a42] mt-1 max-w-2xl leading-relaxed">
            Your device's past scans, plus the public samples. Click any record to open its full result.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={exportCSV}
            disabled={filteredItems.length === 0}
            className="px-4 py-2 rounded-xl bg-white text-[#131b2e] text-xs font-semibold border border-[#dae2fd] hover:bg-[#eaedff] disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center gap-2 shadow-sm cursor-pointer"
          >
            <span className="material-symbols-outlined text-[16px]">file_download</span>
            <span>Export CSV (this page)</span>
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
      <div className="bg-white p-4 rounded-2xl shadow-sm border border-[#dae2fd] flex flex-col gap-4">
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="relative w-full sm:w-80">
            <span className="material-symbols-outlined absolute left-3 top-2.5 text-[#3d4a42] text-[18px]">
              search
            </span>
            <input
              type="text"
              placeholder="Search this page by id/title/result…"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              aria-label="Search this page of history"
              className="w-full pl-9 pr-4 py-2 rounded-lg bg-[#f2f3ff] text-xs text-[#131b2e] border border-[#dae2fd] focus:outline-none focus:ring-2 focus:ring-[#006948]"
            />
          </div>

          {/* Layout Toggle */}
          <div className="flex items-center gap-1 bg-[#f2f3ff] p-1 rounded-lg border border-[#dae2fd]">
            <button
              type="button"
              onClick={() => setViewLayout('cards')}
              aria-pressed={viewLayout === 'cards'}
              aria-label="Card view"
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
              aria-pressed={viewLayout === 'table'}
              aria-label="Table view"
              className={`p-1.5 rounded transition-all cursor-pointer ${
                viewLayout === 'table' ? 'bg-white text-[#006948] shadow-sm' : 'text-[#3d4a42]'
              }`}
              title="Table View"
            >
              <span className="material-symbols-outlined text-[16px]">view_list</span>
            </button>
          </div>
        </div>

        <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3">
          <div className="flex items-center gap-1.5">
            {SCOPE_TABS.map((tab) => (
              <button
                key={tab.id}
                type="button"
                onClick={() => changeScope(tab.id)}
                aria-pressed={scope === tab.id}
                className={`px-3 py-1.5 rounded-lg text-xs font-mono font-medium transition-all cursor-pointer ${
                  scope === tab.id
                    ? 'bg-[#006948] text-white font-bold shadow-sm'
                    : 'bg-[#f2f3ff] text-[#3d4a42] hover:bg-[#eaedff]'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
          <span className="hidden sm:inline text-[#dae2fd]">|</span>
          <div className="flex items-center gap-1.5">
            {MODEL_TABS.map((tab) => (
              <button
                key={tab.id}
                type="button"
                onClick={() => changeModelFilter(tab.id)}
                aria-pressed={modelFilter === tab.id}
                className={`px-3 py-1.5 rounded-lg text-xs font-mono font-medium transition-all cursor-pointer ${
                  modelFilter === tab.id
                    ? 'bg-[#eaedff] text-[#131b2e] font-bold border border-[#006948]'
                    : 'bg-[#f2f3ff] text-[#3d4a42] hover:bg-[#eaedff] border border-transparent'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {actionError && (
        <div className="p-3.5 rounded-xl bg-[#ffdad6] border border-[#ba1a1a]/30 text-[#ba1a1a] text-sm">
          {actionError}
        </div>
      )}

      {Boolean(error) && (
        <div className="p-3.5 rounded-xl bg-[#ffdad6] border border-[#ba1a1a]/30 text-[#ba1a1a] text-sm">
          {userMessageFor(error)}
        </div>
      )}

      {loading && items.length === 0 && (
        <div className="flex items-center justify-center py-16 text-[#3d4a42] text-sm gap-2">
          <span className="material-symbols-outlined animate-spin text-[20px]">progress_activity</span>
          Loading…
        </div>
      )}

      {!loading && !error && filteredItems.length === 0 && (
        <div className="flex flex-col items-center gap-3 py-16 text-center">
          <span className="material-symbols-outlined text-[36px] text-[#3d4a42]">inbox</span>
          <p className="text-sm text-[#3d4a42] max-w-sm">
            {searchTerm
              ? 'No rows on this page match your search.'
              : scope === 'mine'
                ? "You haven't analysed anything on this device yet."
                : 'Nothing here yet.'}
          </p>
          <button
            type="button"
            onClick={() => onNavigate('detect')}
            className="px-5 py-2.5 rounded-xl bg-[#006948] text-white text-sm font-semibold hover:bg-[#00855d] transition-all cursor-pointer"
          >
            Start a Detection
          </button>
        </div>
      )}

      {filteredItems.length > 0 &&
        (viewLayout === 'cards' ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredItems.map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => void openItem(item)}
                disabled={openingId === item.id}
                className="text-left bg-white rounded-2xl p-5 shadow-sm border border-[#dae2fd] hover:border-[#006948] hover:shadow-md transition-all flex flex-col justify-between gap-4 cursor-pointer group disabled:opacity-60"
              >
                <div className="flex items-start gap-4">
                  <div className="w-20 h-20 rounded-xl overflow-hidden bg-[#283044] shrink-0 border border-[#dae2fd]">
                    <img
                      alt=""
                      src={absoluteImageUrl(item.thumbnail_url)}
                      className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                    />
                  </div>
                  <div className="flex flex-col min-w-0 flex-1 gap-0.5">
                    <div className="flex items-center justify-between">
                      <span className="font-mono text-[10px] font-semibold text-[#3d4a42]">{fmtDate(item.created_at)}</span>
                      <span className="font-mono text-xs font-bold text-[#131b2e]">{fmtConfidence(item.confidence)}</span>
                    </div>
                    <span className="font-mono text-[10px] uppercase tracking-wide text-[#006948] font-bold">
                      {item.model_key === 'leaf_segmentation' ? 'Model 2 · Leaf' : 'Model 1 · Tree'}
                      {item.is_sample && ' · Sample'}
                    </span>
                    <h3 className="text-sm font-extrabold text-[#131b2e] truncate mt-0.5">
                      {item.title ?? item.display_label}
                    </h3>
                    <span className="text-[11px] text-[#3d4a42] font-mono">{fmtCoords(item.latitude, item.longitude)}</span>
                  </div>
                </div>

                <div className="pt-3 border-t border-[#eaedff] flex items-center justify-between font-mono text-[11px] text-[#3d4a42]">
                  <span className="truncate">ID: {item.id.slice(0, 8)}…</span>
                  <div className="flex items-center gap-3 shrink-0">
                    {!item.is_sample && (
                      <span
                        role="button"
                        tabIndex={0}
                        onClick={(e) => void deleteItem(item, e)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' || e.key === ' ') {
                            e.preventDefault();
                            void deleteItem(item, e as unknown as React.MouseEvent);
                          }
                        }}
                        aria-label="Delete this analysis"
                        className="text-[#ba1a1a] hover:underline cursor-pointer"
                      >
                        {deletingId === item.id ? 'Deleting…' : 'Delete'}
                      </span>
                    )}
                    <span className="text-[#006948] font-bold group-hover:underline flex items-center gap-0.5">
                      <span>{openingId === item.id ? 'Opening…' : 'View'}</span>
                      <span className="material-symbols-outlined text-[14px]">arrow_forward</span>
                    </span>
                  </div>
                </div>
              </button>
            ))}
          </div>
        ) : (
          <div className="bg-white rounded-2xl shadow-sm border border-[#dae2fd] overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="bg-[#f2f3ff] border-b border-[#dae2fd] text-[11px] font-mono uppercase text-[#3d4a42]">
                    <th className="py-3 px-6">Photo &amp; ID</th>
                    <th className="py-3 px-4">Date</th>
                    <th className="py-3 px-4">Model</th>
                    <th className="py-3 px-4">Result</th>
                    <th className="py-3 px-4">Confidence</th>
                    <th className="py-3 px-4">GPS</th>
                    <th className="py-3 px-6 text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#eaedff] text-xs">
                  {filteredItems.map((item) => (
                    <tr
                      key={item.id}
                      onClick={() => void openItem(item)}
                      className="hover:bg-[#faf8ff] transition-colors cursor-pointer"
                    >
                      <td className="py-3 px-6 flex items-center gap-3">
                        <img
                          alt=""
                          src={absoluteImageUrl(item.thumbnail_url)}
                          className="w-10 h-8 rounded-lg object-cover border border-[#dae2fd]"
                        />
                        <span className="font-mono text-xs font-bold text-[#131b2e]">{item.id.slice(0, 8)}…</span>
                      </td>
                      <td className="py-3 px-4 font-mono text-xs text-[#3d4a42]">{fmtDate(item.created_at)}</td>
                      <td className="py-3 px-4 font-mono text-xs font-bold text-[#006948]">
                        {item.model_key === 'leaf_segmentation' ? 'Leaf' : 'Tree'}
                        {item.is_sample && ' (sample)'}
                      </td>
                      <td className="py-3 px-4 font-bold text-[#131b2e]">{item.title ?? item.display_label}</td>
                      <td className="py-3 px-4 font-mono font-bold text-[#131b2e]">{fmtConfidence(item.confidence)}</td>
                      <td className="py-3 px-4 font-mono text-[#3d4a42]">{fmtCoords(item.latitude, item.longitude)}</td>
                      <td className="py-3 px-6 text-right">
                        {!item.is_sample && (
                          <button
                            type="button"
                            onClick={(e) => void deleteItem(item, e)}
                            className="text-[#ba1a1a] hover:underline font-semibold mr-3 cursor-pointer"
                          >
                            {deletingId === item.id ? 'Deleting…' : 'Delete'}
                          </button>
                        )}
                        <span className="text-xs font-semibold text-[#006948] hover:underline">
                          {openingId === item.id ? 'Opening…' : 'View →'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ))}

      {/* Pagination */}
      {pageInfo && pageInfo.total > 0 && (
        <div className="flex items-center justify-between text-xs font-mono text-[#3d4a42]">
          <span>
            {pageInfo.total} result{pageInfo.total === 1 ? '' : 's'} · page {pageInfo.page} of {totalPages}
          </span>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page <= 1 || loading}
              className="px-3 py-1.5 rounded-lg bg-[#f2f3ff] hover:bg-[#eaedff] disabled:opacity-40 disabled:cursor-not-allowed border border-[#dae2fd] cursor-pointer"
            >
              ← Prev
            </button>
            <button
              type="button"
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              disabled={page >= totalPages || loading}
              className="px-3 py-1.5 rounded-lg bg-[#f2f3ff] hover:bg-[#eaedff] disabled:opacity-40 disabled:cursor-not-allowed border border-[#dae2fd] cursor-pointer"
            >
              Next →
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
