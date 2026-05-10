import { useEffect, useState } from 'react';
import Papa from 'papaparse';
import initCornerstone from './lib/initCornerstone';
import Viewport from './components/Viewport';
import AssistantPanel from './components/AssistantPanel';
import { Database, Loader2, AlertCircle, CheckCircle2 } from 'lucide-react';
import type { ViewerContextSnapshot } from './lib/viewerContext';

interface SeriesData {
  id: string;
  files: number;
}

interface SeriesCsvRow {
  id?: string;
  files?: string;
}

const DEFAULT_SERIES_ID = 'SE000003';

export default function App() {
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [seriesList, setSeriesList] = useState<SeriesData[]>([]);
  const [selectedSeries, setSelectedSeries] = useState<SeriesData | null>(null);
  const [viewerContext, setViewerContext] = useState<ViewerContextSnapshot | null>(null);
  const [loadingMetadata, setLoadingMetadata] = useState(true);
  const [diagnostics, setDiagnostics] = useState<{
    csvLoaded: boolean;
    dicomFetch: boolean;
    sharedArrayBuffer: boolean;
  }>({
    csvLoaded: false,
    dicomFetch: false,
    sharedArrayBuffer: !!window.SharedArrayBuffer
  });

  useEffect(() => {
    const start = async () => {
      try {
        await initCornerstone();
        setReady(true);
      } catch (err: unknown) {
        setError(`Cornerstone Init Failed: ${err instanceof Error ? err.message : String(err)}`);
      }
    };
    start();

    // Diagnostics: Check CSV
    Papa.parse('/dataset/brain_mri_4.csv', {
      download: true,
      header: true,
      complete: (results: Papa.ParseResult<SeriesCsvRow>) => {
        const parsed = results.data
          .reduce<SeriesData[]>((series, row) => {
            if (!row.id || !row.files) return series;
            series.push({
              id: row.id,
              files: parseInt(row.files, 10),
            });
            return series;
          }, []);
        setSeriesList(parsed);
        if (parsed.length > 0) {
          setSelectedSeries(parsed.find((series) => series.id === DEFAULT_SERIES_ID) ?? parsed[0]);
        }
        setLoadingMetadata(false);
        setDiagnostics(prev => ({ ...prev, csvLoaded: true }));
      },
      error: (err) => {
        console.error('Error parsing CSV:', err);
        setLoadingMetadata(false);
        setError(`CSV Load Failed: ${err}`);
      }
    });

    // Diagnostics: Check DICOM Fetch
    fetch('/dataset/ST000001/SE000001/IM000001.dcm')
      .then(res => {
        if (res.ok) setDiagnostics(prev => ({ ...prev, dicomFetch: true }));
        else console.error('DICOM Fetch failed with status:', res.status);
      })
      .catch(err => console.error('DICOM Fetch error:', err));
  }, []);

  if (error) {
    return (
      <div className="flex h-screen flex-col items-center justify-center bg-neutral-900 text-white gap-4 p-8 text-center">
        <AlertCircle className="h-12 w-12 text-red-500" />
        <div className="text-xl font-bold">Initialization Error</div>
        <div className="text-neutral-400 font-mono text-sm max-w-lg bg-black/50 p-4 rounded">{error}</div>
        <button 
            className="mt-4 px-6 py-2 bg-blue-600 hover:bg-blue-500 rounded font-medium transition"
            onClick={() => window.location.reload()}
        >
            Retry
        </button>
      </div>
    );
  }

  if (!ready) {
    return (
      <div className="flex h-screen flex-col items-center justify-center bg-neutral-900 text-white gap-4">
        <Loader2 className="animate-spin h-8 w-8 text-blue-500" />
        <div className="text-xl font-medium">Initializing Cornerstone3D...</div>
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-neutral-900 text-white overflow-hidden font-sans">
      {/* Sidebar */}
      <div className="w-72 border-r border-neutral-700 bg-neutral-800 flex flex-col shadow-xl z-20">
        <div className="p-6 border-b border-neutral-700 flex items-center gap-3">
          <Database className="text-blue-500" />
          <h1 className="font-bold text-xl tracking-tight">Brain MRI Viewer</h1>
        </div>
        
        <div className="p-4 bg-neutral-700/30 text-xs font-semibold uppercase tracking-wider text-neutral-500 flex justify-between items-center">
          <span>Available Series</span>
          <div className="flex gap-1">
            <div title="CSV Loaded" className={diagnostics.csvLoaded ? 'text-green-500' : 'text-neutral-600'}><CheckCircle2 size={12}/></div>
            <div title="DICOM Accessible" className={diagnostics.dicomFetch ? 'text-green-500' : 'text-neutral-600'}><CheckCircle2 size={12}/></div>
            <div title="SharedArrayBuffer" className={diagnostics.sharedArrayBuffer ? 'text-green-500' : 'text-neutral-600'}><CheckCircle2 size={12}/></div>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto">
          {loadingMetadata ? (
            <div className="p-8 text-center text-neutral-500 italic">Loading series list...</div>
          ) : (
            seriesList.map((series) => (
              <button
                key={series.id}
                className={`w-full text-left p-4 border-b border-neutral-700/50 hover:bg-neutral-700 transition-all active:bg-neutral-600 ${
                  selectedSeries?.id === series.id ? 'bg-neutral-700 border-l-4 border-l-blue-500 shadow-inner' : 'border-l-4 border-l-transparent'
                }`}
                onClick={() => setSelectedSeries(series)}
              >
                <div className={`font-semibold ${selectedSeries?.id === series.id ? 'text-blue-400' : 'text-neutral-200'}`}>
                  {series.id}
                </div>
                <div className="text-sm text-neutral-500 mt-1 flex justify-between">
                  <span>Volume Data</span>
                  <span className="bg-neutral-900 px-2 py-0.5 rounded text-[10px]">{series.files} Slices</span>
                </div>
              </button>
            ))
          )}
        </div>
        
        <div className="p-4 text-[10px] text-neutral-500 text-center border-t border-neutral-700 italic">
          Built with Cornerstone3D & React
        </div>
      </div>

      {/* Main Viewport Area */}
      <div className="flex-1 relative bg-black flex flex-col">
        {selectedSeries ? (
          <div className="flex-1">
            <div className="absolute top-4 right-4 z-10 bg-black/60 backdrop-blur-md border border-white/10 px-3 py-1.5 rounded-full text-xs font-mono text-white/80 shadow-2xl flex gap-4">
              <div>Series: <span className="text-blue-400">{selectedSeries.id}</span></div>
              <div className="opacity-30">|</div>
              <div>Slices: <span className="text-blue-400">{selectedSeries.files}</span></div>
            </div>
            <Viewport
              seriesId={selectedSeries.id}
              fileCount={selectedSeries.files}
              onContextChange={setViewerContext}
            />
          </div>
        ) : (
          <div className="flex h-full items-center justify-center text-neutral-500 bg-neutral-900 animate-pulse">
            Select a series from the left to start viewing
          </div>
        )}
      </div>
      <AssistantPanel viewerContext={viewerContext} />
    </div>
  );
}
