import * as cornerstoneTools from '@cornerstonejs/tools';

export type ViewerMode = 'Stack (2D)' | 'Volume (3D)';

export interface AnnotationSummary {
  uid: string;
  toolName: string;
  label: string;
  pointCount: number;
  measurements: Record<string, string>;
}

export interface ViewerContextSnapshot {
  seriesId: string;
  fileCount: number;
  mode: ViewerMode;
  activeTool: string;
  sliceIndex?: number;
  annotations: AnnotationSummary[];
  capturedAt: string;
}

export interface ViewAttachment {
  dataUrl: string;
  image: HTMLImageElement;
}

type UnknownRecord = Record<string, unknown>;

function asRecord(value: unknown): UnknownRecord | undefined {
  return value && typeof value === 'object' ? (value as UnknownRecord) : undefined;
}

function formatValue(value: unknown): string | undefined {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return Math.abs(value) >= 10 ? value.toFixed(1) : value.toFixed(2);
  }

  if (typeof value === 'string' && value.trim()) {
    return value;
  }

  return undefined;
}

function collectMeasurements(value: unknown, prefix = '', output: Record<string, string> = {}) {
  const record = asRecord(value);
  if (!record) return output;

  Object.entries(record).forEach(([key, nestedValue]) => {
    if (Object.keys(output).length >= 10) return;

    const label = prefix ? `${prefix}.${key}` : key;
    const formatted = formatValue(nestedValue);
    if (formatted) {
      output[label] = formatted;
      return;
    }

    if (nestedValue && typeof nestedValue === 'object' && !Array.isArray(nestedValue)) {
      collectMeasurements(nestedValue, label, output);
    }
  });

  return output;
}

export function getAnnotationSummaries(): AnnotationSummary[] {
  try {
    const annotations = cornerstoneTools.annotation.state.getAllAnnotations() as unknown[];

    return annotations.map((annotation, index) => {
      const record = asRecord(annotation) ?? {};
      const metadata = asRecord(record.metadata) ?? {};
      const data = asRecord(record.data) ?? {};
      const handles = asRecord(data.handles) ?? {};
      const points = Array.isArray(handles.points) ? handles.points : [];
      const toolName = formatValue(metadata.toolName) ?? 'Annotation';
      const uid = formatValue(record.annotationUID) ?? `annotation-${index + 1}`;
      const label = formatValue(data.label) ?? `${toolName} ${index + 1}`;
      const measurements = collectMeasurements(data.cachedStats);

      return {
        uid,
        toolName,
        label,
        pointCount: points.length,
        measurements,
      };
    });
  } catch (error) {
    console.warn('Could not summarize Cornerstone annotations:', error);
    return [];
  }
}

export function captureCurrentViewportImage(): Promise<ViewAttachment> {
  const canvas = document.querySelector<HTMLCanvasElement>('[data-mri-viewport="true"] canvas');

  if (!canvas) {
    return Promise.reject(new Error('No active MRI viewport canvas was found.'));
  }

  const dataUrl = canvas.toDataURL('image/png');

  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve({ dataUrl, image });
    image.onerror = () => reject(new Error('Could not prepare the current viewport image.'));
    image.src = dataUrl;
  });
}

export function buildViewerContextText(context: ViewerContextSnapshot | null, includeAnnotations = true) {
  if (!context) {
    return 'Viewer context: not yet available.';
  }

  const lines = [
    `Viewer context captured ${context.capturedAt}`,
    `Series: ${context.seriesId}`,
    `Mode: ${context.mode}`,
    `Slice: ${context.sliceIndex ? `${context.sliceIndex} of ${context.fileCount}` : `not available of ${context.fileCount}`}`,
    `Active tool: ${context.activeTool}`,
  ];

  if (includeAnnotations) {
    if (context.annotations.length === 0) {
      lines.push('Annotations: none currently active.');
    } else {
      lines.push('Annotations:');
      context.annotations.forEach((annotation, index) => {
        const measurementText = Object.entries(annotation.measurements)
          .map(([key, value]) => `${key}=${value}`)
          .join(', ');
        lines.push(
          `${index + 1}. ${annotation.toolName} (${annotation.label}) with ${annotation.pointCount} points${
            measurementText ? `; measurements: ${measurementText}` : ''
          }`
        );
      });
    }
  }

  return lines.join('\n');
}
