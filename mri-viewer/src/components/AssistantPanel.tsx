import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertCircle,
  Bot,
  Clipboard,
  FileText,
  Image as ImageIcon,
  Loader2,
  RefreshCcw,
  Send,
  ShieldCheck,
  Sparkles,
  X,
} from 'lucide-react';
import { MedGemmaAssistantService } from '../lib/medGemmaAssistant';
import type { ModelStatus } from '../lib/medGemmaAssistant';
import {
  buildViewerContextText,
  captureCurrentViewportImage,
  getAnnotationSummaries,
} from '../lib/viewerContext';
import type { ViewAttachment, ViewerContextSnapshot } from '../lib/viewerContext';

interface AssistantPanelProps {
  viewerContext: ViewerContextSnapshot | null;
}

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  imageUrl?: string;
}

type QuickActionId =
  | 'describe-slice'
  | 'summarize-annotations'
  | 'draft-findings'
  | 'draft-impression'
  | 'explain-sequence'
  | 'review-questions';

interface ReportDraft {
  clinicalContext: string;
  technique: string;
  findings: string;
  impression: string;
  caveats: string;
}

type ContentBlock = { type: 'text' | 'think'; content: string };
type PanelView = 'chat' | 'report';

function parseMessageContent(text: string): ContentBlock[] {
  const blocks: ContentBlock[] = [];
  let currentPos = 0;

  while (currentPos < text.length) {
    const thinkStart = text.indexOf('<think>', currentPos);

    if (thinkStart === -1) {
      blocks.push({ type: 'text', content: text.substring(currentPos) });
      break;
    }

    if (thinkStart > currentPos) {
      blocks.push({ type: 'text', content: text.substring(currentPos, thinkStart) });
    }

    const thinkEnd = text.indexOf('</think>', thinkStart + 7);

    if (thinkEnd === -1) {
      blocks.push({ type: 'think', content: text.substring(thinkStart + 7) });
      break;
    }

    blocks.push({ type: 'think', content: text.substring(thinkStart + 7, thinkEnd) });
    currentPos = thinkEnd + 8;
  }

  return blocks;
}

const EMPTY_REPORT: ReportDraft = {
  clinicalContext: '',
  technique: 'Brain MRI review from local DICOM viewer. Sequence and protocol details require clinician verification.',
  findings: '',
  impression: '',
  caveats: 'AI draft for clinician review. Not a standalone diagnosis.',
};

const QUICK_ACTIONS: Array<{ id: QuickActionId; label: string; prompt: string }> = [
  {
    id: 'describe-slice',
    label: 'Describe current slice',
    prompt:
      'Describe the visible brain MRI slice for clinician review. Focus on anatomy, orientation, and visible image quality. Avoid diagnosis unless clearly supported by the provided context.',
  },
  {
    id: 'summarize-annotations',
    label: 'Summarize annotations',
    prompt:
      'Summarize the current annotations and measurements in reporting language. Preserve series and slice context when available.',
  },
  {
    id: 'draft-findings',
    label: 'Draft findings',
    prompt:
      'Draft a brain MRI Findings section from the supplied viewer context and user-confirmed annotations. Include uncertainty and missing-context caveats.',
  },
  {
    id: 'draft-impression',
    label: 'Draft impression',
    prompt:
      'Draft a concise Impression section based only on the supplied viewer context and annotations. Do not invent findings.',
  },
  {
    id: 'explain-sequence',
    label: 'Explain sequence/anatomy',
    prompt:
      'Explain likely sequence characteristics and visible neuroanatomy for a radiology trainee. Clearly separate education from case-specific interpretation.',
  },
  {
    id: 'review-questions',
    label: 'List review questions',
    prompt:
      'List focused questions a radiologist should review before finalizing documentation for this study.',
  },
];

function makeId() {
  return typeof crypto !== 'undefined' && 'randomUUID' in crypto ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`;
}

function sectionFromResponse(response: string, title: string) {
  const escapedTitle = title.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const pattern = new RegExp(`(?:^|\\n)#{0,3}\\s*${escapedTitle}\\s*:?\\s*\\n([\\s\\S]*?)(?=\\n#{0,3}\\s*(?:Clinical context|Indication|Technique|Findings|Impression|Caveats|Needs review)\\s*:?\\s*\\n|$)`, 'i');
  return response.match(pattern)?.[1]?.trim() ?? '';
}

function mergeReportDraft(previous: ReportDraft, response: string, action?: QuickActionId): ReportDraft {
  const clinicalContext = sectionFromResponse(response, 'Clinical context') || sectionFromResponse(response, 'Indication');
  const technique = sectionFromResponse(response, 'Technique');
  const findings = sectionFromResponse(response, 'Findings');
  const impression = sectionFromResponse(response, 'Impression');
  const caveats = sectionFromResponse(response, 'Caveats') || sectionFromResponse(response, 'Needs review');

  return {
    clinicalContext: clinicalContext || previous.clinicalContext,
    technique: technique || previous.technique,
    findings: findings || (action === 'draft-findings' ? response.trim() : previous.findings),
    impression: impression || (action === 'draft-impression' ? response.trim() : previous.impression),
    caveats: caveats || previous.caveats,
  };
}

function reportToText(report: ReportDraft) {
  return [
    `Clinical context\n${report.clinicalContext || '[Not provided]'}`,
    `Technique\n${report.technique || '[Needs review]'}`,
    `Findings\n${report.findings || '[Draft findings not generated]'}`,
    `Impression\n${report.impression || '[Draft impression not generated]'}`,
    `Caveats/needs review\n${report.caveats || EMPTY_REPORT.caveats}`,
  ].join('\n\n');
}

export default function AssistantPanel({ viewerContext }: AssistantPanelProps) {
  const serviceRef = useRef<MedGemmaAssistantService | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [panelView, setPanelView] = useState<PanelView>('chat');
  const [status, setStatus] = useState<ModelStatus>('Loading');
  const [statusDetail, setStatusDetail] = useState('Preparing local MedGemma runtime...');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [attachment, setAttachment] = useState<ViewAttachment | null>(null);
  const [includeAnnotations, setIncludeAnnotations] = useState(true);
  const [reportDraft, setReportDraft] = useState<ReportDraft>(EMPTY_REPORT);

  const annotationCount = viewerContext?.annotations.length ?? getAnnotationSummaries().length;
  const isBusy = status === 'Loading' || status === 'Generating';
  const canSubmit = Boolean((input.trim() || attachment) && status === 'Ready');

  const contextChips = useMemo(() => {
    const chips = [];
    if (viewerContext) {
      chips.push(viewerContext.seriesId);
      chips.push(viewerContext.mode);
      if (viewerContext.sliceIndex) chips.push(`Slice ${viewerContext.sliceIndex}/${viewerContext.fileCount}`);
    }
    if (includeAnnotations) chips.push(`${annotationCount} annotations`);
    if (attachment) chips.push('Current view attached');
    return chips;
  }, [annotationCount, attachment, includeAnnotations, viewerContext]);

  const loadModel = useCallback(async () => {
    if (!serviceRef.current) {
      serviceRef.current = new MedGemmaAssistantService();
    }

    setStatus('Loading');
    setStatusDetail('Loading local MedGemma model and WASM runtime...');

    try {
      await serviceRef.current.load();
      setStatus('Ready');
      setStatusDetail(serviceRef.current.modelDetail);
    } catch (error) {
      setStatus(navigator.gpu ? 'Error' : 'Unavailable');
      setStatusDetail(error instanceof Error ? error.message : String(error));
    }
  }, []);

  useEffect(() => {
    const loadTimer = window.setTimeout(() => {
      void loadModel();
    }, 0);

    return () => {
      window.clearTimeout(loadTimer);
      serviceRef.current?.close();
    };
  }, [loadModel]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, [messages]);

  const attachCurrentView = async () => {
    try {
      const viewAttachment = await captureCurrentViewportImage();
      setAttachment(viewAttachment);
    } catch (error) {
      setStatus('Error');
      setStatusDetail(error instanceof Error ? error.message : String(error));
    }
  };

  const resetAssistant = () => {
    serviceRef.current?.reset();
    setMessages([]);
    setInput('');
    setAttachment(null);
    setReportDraft(EMPTY_REPORT);
    setPanelView('chat');
    void loadModel();
  };

  const submitPrompt = async (promptText: string, action?: QuickActionId, overrideAttachment?: ViewAttachment | null) => {
    if (!serviceRef.current || status !== 'Ready') return;

    const currentAttachment = overrideAttachment === undefined ? attachment : overrideAttachment;
    const prompt = `${buildViewerContextText(viewerContext, includeAnnotations)}

Task:
${promptText}

For report drafting, use these headings when relevant:
Clinical context
Technique
Findings
Impression
Caveats/needs review`;

    const userMessage: ChatMessage = {
      id: makeId(),
      role: 'user',
      content: promptText,
      imageUrl: currentAttachment?.dataUrl,
    };
    const assistantId = makeId();

    setMessages((previous) => [
      ...previous,
      userMessage,
      { id: assistantId, role: 'assistant', content: 'Preparing draft...' },
    ]);
    setInput('');
    setStatus('Generating');
    setStatusDetail('Generating local AI draft...');

    try {
      const finalText = await serviceRef.current.generate({
        prompt,
        image: currentAttachment?.image,
        onPartial: (partialText) => {
          setMessages((previous) =>
            previous.map((message) =>
              message.id === assistantId ? { ...message, content: partialText || 'Generating...' } : message
            )
          );
        },
      });

      setMessages((previous) =>
        previous.map((message) => (message.id === assistantId ? { ...message, content: finalText } : message))
      );
      setReportDraft((previous) => mergeReportDraft(previous, finalText, action));
      setStatus('Ready');
      setStatusDetail('Draft complete. Clinician review is required before use.');
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setMessages((previous) =>
        previous.map((item) =>
          item.id === assistantId ? { ...item, content: `Model error: ${message}` } : item
        )
      );
      setStatus('Error');
      setStatusDetail(message);
    } finally {
      setAttachment(null);
    }
  };

  const handleSubmit = () => {
    if (!canSubmit) return;
    void submitPrompt(input.trim() || 'Describe the attached current MRI viewport.');
  };

  const handleQuickAction = (action: (typeof QUICK_ACTIONS)[number]) => {
    if (status !== 'Ready') return;
    const shouldAttachImage = action.id === 'describe-slice' && !attachment;
    if (shouldAttachImage) {
      void captureCurrentViewportImage()
        .then((viewAttachment) => {
          setAttachment(viewAttachment);
          return submitPrompt(action.prompt, action.id, viewAttachment);
        })
        .catch((error) => {
          setStatus('Error');
          setStatusDetail(error instanceof Error ? error.message : String(error));
        });
      return;
    }

    void submitPrompt(action.prompt, action.id);
  };

  const copyReport = () => {
    void navigator.clipboard.writeText(reportToText(reportDraft));
  };

  return (
    <aside className="flex h-full w-[480px] shrink-0 flex-col border-l border-neutral-700 bg-neutral-900 text-neutral-100 shadow-2xl xl:w-[520px]">
      <div className="border-b border-neutral-700 px-4 py-3">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <Bot className="h-5 w-5 text-cyan-400" />
              <h2 className="truncate text-sm font-bold uppercase tracking-wider">MedGemma Assistant</h2>
            </div>
            <p className="mt-1 text-[11px] leading-4 text-neutral-400">
              AI draft for clinician review. Not a standalone diagnosis.
            </p>
          </div>
          <button
            title="Reset assistant"
            aria-label="Reset assistant"
            onClick={resetAssistant}
            className="flex h-8 w-8 shrink-0 items-center justify-center rounded border border-white/10 bg-white/5 text-neutral-300 transition hover:bg-white/10"
          >
            <RefreshCcw size={15} />
          </button>
        </div>

        <div className="mt-3 rounded border border-white/10 bg-black/25 px-3 py-2">
          <div className="flex items-center gap-2 text-xs font-semibold">
            {status === 'Loading' || status === 'Generating' ? <Loader2 className="h-4 w-4 animate-spin text-blue-400" /> : null}
            {status === 'Ready' ? <ShieldCheck className="h-4 w-4 text-emerald-400" /> : null}
            {status === 'Unavailable' || status === 'Error' ? <AlertCircle className="h-4 w-4 text-amber-400" /> : null}
            <span>{status}</span>
          </div>
          <p className="mt-1.5 text-[11px] leading-5 text-neutral-400">{statusDetail}</p>
        </div>
      </div>

      <div className="border-b border-neutral-700 px-4 py-3">
        <div className="flex items-center gap-2">
          <button
            onClick={() => setPanelView('chat')}
            className={`h-8 rounded-full px-3 text-[11px] font-semibold uppercase tracking-wider transition ${
              panelView === 'chat'
                ? 'bg-cyan-500 text-white shadow'
                : 'border border-white/10 bg-white/5 text-neutral-300 hover:bg-white/10'
            }`}
          >
            Chat
          </button>
          <button
            onClick={() => setPanelView('report')}
            className={`h-8 rounded-full px-3 text-[11px] font-semibold uppercase tracking-wider transition ${
              panelView === 'report'
                ? 'bg-cyan-500 text-white shadow'
                : 'border border-white/10 bg-white/5 text-neutral-300 hover:bg-white/10'
            }`}
          >
            Report
          </button>
          <div className="ml-auto flex items-center gap-1.5 rounded-full border border-white/10 bg-white/5 px-2 py-1">
            <button
              onClick={attachCurrentView}
              disabled={isBusy}
              className="flex h-7 items-center gap-1.5 rounded-full px-2 text-[11px] font-semibold text-cyan-200 transition hover:bg-cyan-500/15 disabled:opacity-45"
            >
              <ImageIcon size={14} />
              View
            </button>
            <label className="flex h-7 cursor-pointer items-center gap-1.5 rounded-full px-2 text-[11px] font-semibold text-neutral-200 transition hover:bg-white/10">
              <input
                type="checkbox"
                checked={includeAnnotations}
                onChange={(event) => setIncludeAnnotations(event.target.checked)}
                className="h-3 w-3 accent-cyan-400"
              />
              Annotations
            </label>
          </div>
        </div>

        <div className="mt-3 flex flex-wrap gap-1.5">
          {contextChips.map((chip) => (
            <span key={chip} className="rounded-full bg-neutral-700 px-2 py-1 text-[10px] font-medium text-neutral-300">
              {chip}
            </span>
          ))}
          {attachment && (
            <button
              onClick={() => setAttachment(null)}
              className="flex items-center gap-1 rounded-full bg-neutral-700 px-2 py-1 text-[10px] font-medium text-neutral-300 hover:bg-neutral-600"
            >
              Remove image
              <X size={11} />
            </button>
          )}
        </div>
      </div>

      {panelView === 'chat' ? (
        <div className="flex min-h-0 flex-1 flex-col">
          <div className="border-b border-neutral-700 px-4 py-3">
            <div className="grid grid-cols-2 gap-2">
              {QUICK_ACTIONS.map((action) => (
                <button
                  key={action.id}
                  onClick={() => handleQuickAction(action)}
                  disabled={status !== 'Ready'}
                  className="min-h-10 rounded border border-white/10 bg-white/5 px-3 py-2 text-left text-[11px] font-semibold text-neutral-200 transition hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-45"
                >
                  {action.label}
                </button>
              ))}
            </div>
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
            {messages.length === 0 ? (
              <div className="flex h-full flex-col items-center justify-center text-center text-neutral-500">
                <Sparkles className="mb-3 h-8 w-8 text-neutral-600" />
                <p className="max-w-xs text-sm leading-6">
                  Ask about the current series, selected slice, visible viewport, or annotations. Outputs stay draft and local.
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                {messages.map((message) => (
                  <div key={message.id} className={message.role === 'user' ? 'text-right' : 'text-left'}>
                    <div
                      className={`inline-block w-full max-w-full rounded-2xl p-3 text-left text-[13px] leading-6 ${
                        message.role === 'user'
                          ? 'bg-cyan-600 text-white'
                          : 'border border-white/10 bg-neutral-800 text-neutral-100'
                      }`}
                    >
                      {message.imageUrl && (
                        <img
                          src={message.imageUrl}
                          alt="Attached current MRI viewport"
                          className="mb-3 max-h-40 rounded border border-white/10"
                        />
                      )}
                      <div className="space-y-2 whitespace-pre-wrap break-words">
                        {parseMessageContent(message.content).map((block, idx) =>
                          block.type === 'think' ? (
                            <div
                              key={idx}
                              className="rounded-xl border border-cyan-400/20 bg-cyan-500/8 p-3 text-[12px] leading-6 text-cyan-50/90 shadow-inner"
                            >
                              <div className="mb-2 flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-[0.22em] text-cyan-100/70">
                                <Bot size={10} />
                                Reasoning
                              </div>
                              <div className="whitespace-pre-wrap break-words text-cyan-50/90">
                                {block.content.trim()}
                              </div>
                            </div>
                          ) : (
                            <div key={idx} className={idx > 0 ? 'pt-1' : ''}>
                              {block.content}
                            </div>
                          )
                        )}
                      </div>
                    </div>
                  </div>
                ))}
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>

          <div className="border-t border-neutral-700 px-4 py-4">
            <div className="flex gap-2">
              <textarea
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
                    handleSubmit();
                  }
                }}
                disabled={status !== 'Ready'}
                placeholder={status === 'Ready' ? 'Ask about the current MRI review...' : 'Assistant is not ready yet...'}
                rows={4}
                className="min-h-24 max-h-48 flex-1 resize-none rounded-2xl border border-white/10 bg-neutral-950 px-3 py-3 text-sm leading-6 text-neutral-100 outline-none transition placeholder:text-neutral-600 focus:border-cyan-500 disabled:opacity-50"
              />
              <button
                onClick={handleSubmit}
                disabled={!canSubmit}
                title="Send prompt"
                aria-label="Send prompt"
                className="mt-auto flex h-12 w-12 items-center justify-center rounded-2xl bg-cyan-600 text-white transition hover:bg-cyan-500 disabled:cursor-not-allowed disabled:bg-neutral-700 disabled:text-neutral-500"
              >
                <Send size={18} />
              </button>
            </div>
          </div>
        </div>
      ) : (
        <div className="flex min-h-0 flex-1 flex-col px-4 py-4">
          <div className="mb-3 flex items-center justify-between">
            <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-neutral-300">
              <FileText size={14} />
              Report draft
            </div>
            <button
              onClick={copyReport}
              className="flex h-7 items-center gap-1 rounded border border-white/10 bg-white/5 px-2 text-[10px] font-semibold text-neutral-300 hover:bg-white/10"
            >
              <Clipboard size={12} />
              Copy
            </button>
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto rounded-2xl border border-white/10 bg-black/25 p-3">
            <div className="space-y-3">
              {Object.entries({
                clinicalContext: 'Clinical context',
                technique: 'Technique',
                findings: 'Findings',
                impression: 'Impression',
                caveats: 'Caveats/needs review',
              }).map(([key, label]) => (
                <label key={key} className="block text-[10px] font-semibold uppercase tracking-wider text-neutral-500">
                  {label}
                  <textarea
                    value={reportDraft[key as keyof ReportDraft]}
                    onChange={(event) =>
                      setReportDraft((previous) => ({ ...previous, [key]: event.target.value }))
                    }
                    rows={key === 'findings' ? 5 : 3}
                    className="mt-1 w-full resize-y rounded-xl border border-white/10 bg-neutral-950 p-3 text-sm leading-6 normal-case text-neutral-200 outline-none focus:border-cyan-500"
                  />
                </label>
              ))}
            </div>
          </div>
        </div>
      )}
    </aside>
  );
}
