import { useCallback, useEffect, useRef, useState } from 'react';
import { 
  RenderingEngine, 
  Enums, 
  volumeLoader, 
  setVolumesForViewports, 
  getRenderingEngine,
  imageLoader
} from '@cornerstonejs/core';
import * as cornerstoneTools from '@cornerstonejs/tools';
import {
  Circle,
  Crosshair,
  Eraser,
  Loader2,
  Move,
  MousePointer2,
  PencilLine,
  RotateCcw,
  Ruler,
  Square,
  SunMedium,
  Trash2,
  ZoomIn,
} from 'lucide-react';
import { getAnnotationSummaries } from '../lib/viewerContext';
import type { ViewerContextSnapshot } from '../lib/viewerContext';

interface ViewportProps {
  seriesId: string;
  fileCount: number;
  onContextChange?: (context: ViewerContextSnapshot) => void;
}

type PrimaryTool =
  | 'WindowLevel'
  | 'Pan'
  | 'Zoom'
  | 'Length'
  | 'Bidirectional'
  | 'Angle'
  | 'Probe'
  | 'RectangleROI'
  | 'EllipticalROI'
  | 'PlanarFreehandROI'
  | 'ArrowAnnotate'
  | 'Eraser';

type StackViewportLike = {
  setStack: (imageIds: string[], currentImageIdIndex?: number) => Promise<string>;
  resetCamera?: () => void;
  resetProperties?: () => void;
};

const renderingEngineId = 'myRenderingEngine';
const viewportId = 'MRI_VIEWPORT';
const toolGroupId = 'mriReviewToolGroup';

const primaryToolButtons: Array<{
  toolName: PrimaryTool;
  label: string;
  title: string;
  Icon: typeof MousePointer2;
}> = [
  {
    toolName: 'WindowLevel',
    label: 'WL',
    title: 'Window/level',
    Icon: SunMedium,
  },
  {
    toolName: 'Pan',
    label: 'Pan',
    title: 'Pan image',
    Icon: Move,
  },
  {
    toolName: 'Zoom',
    label: 'Zoom',
    title: 'Zoom image',
    Icon: ZoomIn,
  },
  {
    toolName: 'Length',
    label: 'Len',
    title: 'Linear measurement',
    Icon: Ruler,
  },
  {
    toolName: 'Bidirectional',
    label: 'Bi',
    title: 'Bidirectional lesion measurement',
    Icon: Crosshair,
  },
  {
    toolName: 'Angle',
    label: 'Ang',
    title: 'Angle measurement',
    Icon: PencilLine,
  },
  {
    toolName: 'Probe',
    label: 'Probe',
    title: 'Pixel probe',
    Icon: MousePointer2,
  },
  {
    toolName: 'RectangleROI',
    label: 'Rect',
    title: 'Rectangle ROI',
    Icon: Square,
  },
  {
    toolName: 'EllipticalROI',
    label: 'Oval',
    title: 'Elliptical ROI',
    Icon: Circle,
  },
  {
    toolName: 'PlanarFreehandROI',
    label: 'Free',
    title: 'Freehand ROI',
    Icon: PencilLine,
  },
  {
    toolName: 'ArrowAnnotate',
    label: 'Note',
    title: 'Arrow annotation',
    Icon: PencilLine,
  },
  {
    toolName: 'Eraser',
    label: 'Erase',
    title: 'Erase annotation',
    Icon: Eraser,
  },
];

const toolsToRegister = [
  cornerstoneTools.StackScrollTool,
  cornerstoneTools.WindowLevelTool,
  cornerstoneTools.PanTool,
  cornerstoneTools.ZoomTool,
  cornerstoneTools.LengthTool,
  cornerstoneTools.BidirectionalTool,
  cornerstoneTools.AngleTool,
  cornerstoneTools.ProbeTool,
  cornerstoneTools.RectangleROITool,
  cornerstoneTools.EllipticalROITool,
  cornerstoneTools.PlanarFreehandROITool,
  cornerstoneTools.ArrowAnnotateTool,
  cornerstoneTools.EraserTool,
];

const reviewToolNames = toolsToRegister.map((Tool) => Tool.toolName);

function getOrCreateToolGroup() {
  toolsToRegister.forEach((Tool) => cornerstoneTools.addTool(Tool));

  let toolGroup = cornerstoneTools.ToolGroupManager.getToolGroup(toolGroupId);
  if (!toolGroup) {
    toolGroup = cornerstoneTools.ToolGroupManager.createToolGroup(toolGroupId);
  }

  reviewToolNames.forEach((toolName) => {
    if (!toolGroup?.hasTool(toolName)) {
      toolGroup?.addTool(toolName);
    }
  });

  return toolGroup;
}

function applyToolBindings(activeToolName: PrimaryTool) {
  const toolGroup = cornerstoneTools.ToolGroupManager.getToolGroup(toolGroupId);
  if (!toolGroup) return;

  const { MouseBindings } = cornerstoneTools.Enums;

  reviewToolNames.forEach((toolName) => {
    toolGroup.setToolPassive(toolName, { removeAllBindings: true });
  });

  toolGroup.setToolActive(cornerstoneTools.StackScrollTool.toolName, {
    bindings: [{ mouseButton: MouseBindings.Wheel }],
  });
  toolGroup.setToolActive(cornerstoneTools.ZoomTool.toolName, {
    bindings: [{ mouseButton: MouseBindings.Secondary }],
  });
  toolGroup.setToolActive(cornerstoneTools.PanTool.toolName, {
    bindings: [{ mouseButton: MouseBindings.Auxiliary }],
  });
  toolGroup.setToolActive(activeToolName, {
    bindings: [{ mouseButton: MouseBindings.Primary }],
  });
}

/**
 * Robust metadata pre-fetch.
 */
async function prefetchMetadata(imageIds: string[]) {
  console.log('Prefetching metadata for', imageIds.length, 'images...');
  // We only really need the first one to define the volume, but all for accurate reconstruction
  const chunkSize = 10;
  for (let i = 0; i < imageIds.length; i += chunkSize) {
    const chunk = imageIds.slice(i, i + chunkSize);
    await Promise.all(chunk.map(id => imageLoader.loadImage(id)));
    console.log(`Loaded ${Math.min(i + chunkSize, imageIds.length)}/${imageIds.length} metadata`);
  }
}

export default function Viewport({ seriesId, fileCount, onContextChange }: ViewportProps) {
  const elementRef = useRef<HTMLDivElement>(null);
  const activeToolRef = useRef<PrimaryTool>('WindowLevel');
  const useStackRef = useRef(true);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [useStack, setUseStack] = useState(true);
  const [activeTool, setActiveTool] = useState<PrimaryTool>('WindowLevel');

  const notifyContextChange = useCallback(() => {
    const renderingEngine = getRenderingEngine(renderingEngineId);
    const viewport = renderingEngine?.getViewport(viewportId) as
      | ({ currentImageIdIndex?: number; getCurrentImageIdIndex?: () => number } & StackViewportLike)
      | undefined;
    const currentImageIdIndex = viewport?.getCurrentImageIdIndex?.() ?? viewport?.currentImageIdIndex;

    onContextChange?.({
      seriesId,
      fileCount,
      mode: useStackRef.current ? 'Stack (2D)' : 'Volume (3D)',
      activeTool: activeToolRef.current,
      sliceIndex: typeof currentImageIdIndex === 'number' ? currentImageIdIndex + 1 : undefined,
      annotations: getAnnotationSummaries(),
      capturedAt: new Date().toLocaleString(),
    });
  }, [fileCount, onContextChange, seriesId]);

  useEffect(() => {
    if (!elementRef.current || !seriesId || fileCount <= 0) return;

    let isMounted = true;

    const setup = async () => {
      try {
        setLoading(true);
        setError(null);
        
        const baseUrl = window.location.origin;
        const imageIds = Array.from({ length: fileCount }, (_, i) => {
          const num = String(i + 1).padStart(6, '0');
          return `wadouri:${baseUrl}/dataset/ST000001/${seriesId}/IM${num}.dcm`;
        });

        console.log('Image IDs generated:', imageIds.slice(0, 1));

        // 1. Get/Create Rendering Engine
        let renderingEngine = getRenderingEngine(renderingEngineId);
        if (!renderingEngine) {
            renderingEngine = new RenderingEngine(renderingEngineId);
        }

        if (useStack) {
            // STACK VIEWPORT MODE (Fallback)
            console.log('Using Stack Viewport Mode');
            const initialSliceIndex = Math.floor(imageIds.length / 2);
            const viewportInput = {
                viewportId,
                type: Enums.ViewportType.STACK,
                element: elementRef.current!,
                defaultOptions: {
                    background: [0, 0, 0] as [number, number, number],
                },
            };
            renderingEngine.enableElement(viewportInput);
            const viewport = renderingEngine.getViewport(viewportId) as unknown as StackViewportLike;
            await viewport.setStack(imageIds, initialSliceIndex);
            if (!isMounted) return;
            renderingEngine.renderViewports([viewportId]);
        } else {
            // VOLUME VIEWPORT MODE
            console.log('Using Volume Viewport Mode');
            
            // Pre-fetch at least one to check if it works
            try {
                await imageLoader.loadImage(imageIds[0]);
                console.log('First image loaded successfully, metadata cached');
            } catch (e) {
                console.error('Failed to load first image:', e);
                throw new Error('DICOM files not accessible or invalid format.', { cause: e });
            }

            // Pre-fetch all metadata (needed for volume)
            await prefetchMetadata(imageIds);

            if (!isMounted) return;

            const volumeId = `cornerstone_volume_${seriesId}_${Date.now()}`;
            
            console.log('Creating volume:', volumeId);
            const volume = await volumeLoader.createAndCacheVolume(volumeId, { imageIds });

            const viewportInput = {
                viewportId,
                type: Enums.ViewportType.ORTHOGRAPHIC,
                element: elementRef.current!,
                defaultOptions: {
                    orientation: Enums.OrientationAxis.AXIAL,
                    background: [0, 0, 0] as [number, number, number],
                },
            };

            renderingEngine.enableElement(viewportInput);
            
            console.log('Loading volume pixels...');
            volume.load();

            await setVolumesForViewports(renderingEngine, [{ volumeId }], [viewportId]);
            renderingEngine.renderViewports([viewportId]);
        }

        // Setup Tools
        const toolGroup = getOrCreateToolGroup();
        toolGroup?.addViewport(viewportId, renderingEngineId);
        applyToolBindings(activeToolRef.current);

        console.log('Rendering complete');
        setLoading(false);
        window.setTimeout(notifyContextChange, 100);
      } catch (err: unknown) {
        console.error('Viewport setup error:', err);
        if (isMounted) {
            setError(err instanceof Error ? err.message : 'Error loading volume');
            setLoading(false);
        }
      }
    };

    setup();
    
    return () => {
        isMounted = false;
        const renderingEngine = getRenderingEngine(renderingEngineId);
        if (renderingEngine) {
            renderingEngine.disableElement(viewportId);
        }
    };
  }, [seriesId, fileCount, useStack, notifyContextChange]);

  useEffect(() => {
    useStackRef.current = useStack;
    activeToolRef.current = activeTool;
    applyToolBindings(activeTool);
    notifyContextChange();
  }, [activeTool, notifyContextChange, useStack]);

  const handleResetView = () => {
    const renderingEngine = getRenderingEngine(renderingEngineId);
    const viewport = renderingEngine?.getViewport(viewportId) as unknown as StackViewportLike | undefined;
    viewport?.resetCamera?.();
    viewport?.resetProperties?.();
    renderingEngine?.renderViewports([viewportId]);
    window.setTimeout(notifyContextChange, 100);
  };

  const handleClearAnnotations = () => {
    cornerstoneTools.annotation.state.removeAllAnnotations();
    const renderingEngine = getRenderingEngine(renderingEngineId);
    renderingEngine?.renderViewports([viewportId]);
    window.setTimeout(notifyContextChange, 100);
  };

  return (
    <div className="w-full h-full relative bg-neutral-950">
      <div className="absolute top-4 left-4 right-72 z-30 flex flex-wrap items-center gap-2">
        <button 
            onClick={() => setUseStack(!useStack)}
            className="h-8 px-3 bg-white/10 hover:bg-white/20 border border-white/20 rounded text-[10px] uppercase font-bold tracking-widest transition"
        >
            Mode: {useStack ? 'Stack (2D)' : 'Volume (3D)'}
        </button>

        <div className="flex flex-wrap items-center gap-1 rounded border border-white/15 bg-black/55 p-1 shadow-2xl backdrop-blur-md">
          {primaryToolButtons.map(({ toolName, label, title, Icon }) => (
            <button
              key={toolName}
              title={title}
              aria-label={title}
              onClick={() => setActiveTool(toolName)}
              className={`flex h-8 min-w-10 items-center justify-center gap-1 rounded px-2 text-[10px] font-bold uppercase tracking-wider transition ${
                activeTool === toolName
                  ? 'bg-blue-500 text-white shadow'
                  : 'bg-white/10 text-white/80 hover:bg-white/20'
              }`}
            >
              <Icon size={14} />
              <span>{label}</span>
            </button>
          ))}
        </div>

        <div className="flex items-center gap-1 rounded border border-white/15 bg-black/55 p-1 shadow-2xl backdrop-blur-md">
          <button
            title="Reset view"
            aria-label="Reset view"
            onClick={handleResetView}
            className="flex h-8 w-8 items-center justify-center rounded bg-white/10 text-white/80 transition hover:bg-white/20"
          >
            <RotateCcw size={14} />
          </button>
          <button
            title="Clear annotations"
            aria-label="Clear annotations"
            onClick={handleClearAnnotations}
            className="flex h-8 w-8 items-center justify-center rounded bg-white/10 text-white/80 transition hover:bg-white/20"
          >
            <Trash2 size={14} />
          </button>
        </div>
      </div>

      {loading && (
        <div className="absolute inset-0 z-20 flex items-center justify-center bg-black/80 backdrop-blur-sm">
            <div className="flex flex-col items-center gap-2">
                <Loader2 className="animate-spin text-blue-500" />
                <div className="text-blue-400 font-mono text-xs uppercase tracking-tighter">Initializing {useStack ? 'Stack' : 'Volume'}...</div>
            </div>
        </div>
      )}
      {error && (
        <div className="absolute top-16 left-4 z-30 text-red-500 bg-neutral-900 border border-red-500/50 p-4 rounded shadow-2xl max-w-md">
          <p className="font-bold mb-1">Rendering Error</p>
          <p className="text-xs font-mono opacity-80">{error}</p>
        </div>
      )}
      <div
        ref={elementRef}
        data-mri-viewport="true"
        className="w-full h-full"
        onContextMenu={(e) => e.preventDefault()}
        onWheel={() => window.setTimeout(notifyContextChange, 80)}
        onPointerUp={() => window.setTimeout(notifyContextChange, 80)}
      />
    </div>
  );
}
