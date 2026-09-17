'use client';

import { ChangeEvent, DragEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import {
  AlertTriangle,
  ArrowUp,
  Check,
  ChevronRight,
  Cpu,
  Database,
  Download,
  FileImage,
  History,
  Info,
  LoaderCircle,
  MessageSquare,
  MoreVertical,
  Paperclip,
  Plus,
  Radar,
  Settings,
  ShieldCheck,
  User,
  X,
} from 'lucide-react';

import { SatIcon } from '@/components/site/sat-icon';
import { Button } from '@/components/ui/button';
import { useAnalysis } from '@/hooks/use-analysis';
import {
  type AnalysisStage,
  type AnalysisSummary,
  type ClarificationPayload,
  type ClarificationResponse,
  SatQueryApiError,
  satqueryApi,
  type Warning,
} from '@/lib/satquery-api';

type UploadItem = {
  id: string;
  file: File;
  name: string;
  size: string;
  type: string;
  previewUrl?: string;
  uploadId?: string;
};

type Stage =
  | 'idle'
  | 'uploading'
  | 'validating'
  | 'routing'
  | 'inference'
  | 'evidence'
  | 'clarification'
  | 'complete'
  | 'failed';

type FileRole = 'before' | 'after' | 'optical' | 'sar' | 'single' | 'unknown';
type Modality = 'optical' | 'sar' | 'other';

type ChatHistoryItem = {
  id: string;
  title: string;
  timeAgo: string;
  section: 'Today' | 'Previous 7 days' | 'Older';
  isSample?: boolean;
};

const analysisStages = [
  { key: 'uploading',  label: 'Securing your imagery',       detail: 'Uploading original raster files to local evidence storage.' },
  { key: 'validating', label: 'Reading imagery and metadata', detail: 'Checking format, spatial bounds, acquisition timestamps, and resolution.' },
  { key: 'routing',    label: 'Interpreting your question',   detail: 'Constrained router selecting the specialist workflow (ChangeNet, SatVLM, SAR-FuseSeg).' },
  { key: 'inference',  label: 'Evaluating spatial patterns',  detail: 'Running GPU specialist inference with CUDA 12.8 acceleration.' },
  { key: 'evidence',   label: 'Building the evidence trace',  detail: 'Deterministic metric polygonization, UTM area projection, and confidence gating.' },
] as const;

const suggestions = [
  'Describe the visible features in this scene',
  'Where has the built-up area changed?',
  'Map the visible water regions',
];

function formatRelativeTime(dateString?: string | null): string {
  if (!dateString) return 'Just now';
  const date = new Date(dateString);
  const now = new Date();
  const diffSec = Math.floor((now.getTime() - date.getTime()) / 1000);
  if (diffSec < 60) return 'Just now';
  const diffMin = Math.floor(diffSec / 60);
  if (diffMin < 60) return `${diffMin} min ago`;
  const diffHours = Math.floor(diffMin / 60);
  if (diffHours < 24) return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
  const diffDays = Math.floor(diffHours / 24);
  if (diffDays === 1) return '1 day ago';
  if (diffDays < 7) return `${diffDays} days ago`;
  const diffWeeks = Math.floor(diffDays / 7);
  return `${diffWeeks} week${diffWeeks > 1 ? 's' : ''} ago`;
}

function classifySection(dateString?: string | null): 'Today' | 'Previous 7 days' | 'Older' {
  if (!dateString) return 'Today';
  const date = new Date(dateString);
  const now = new Date();
  const diffHours = (now.getTime() - date.getTime()) / (1000 * 60 * 60);
  if (diffHours < 24) return 'Today';
  if (diffHours < 24 * 7) return 'Previous 7 days';
  return 'Older';
}

const stageFromBackend = (stage?: AnalysisStage | null): Stage => {
  if (stage === 'validating' || stage === 'interpreting') return 'validating';
  if (stage === 'queued' || stage === 'routing') return 'routing';
  if (stage === 'preparing_scene' || stage === 'inference') return 'inference';
  if (stage === 'evidence' || stage === 'calibration' || stage === 'composition' || stage === 'reporting') return 'evidence';
  if (stage === 'done') return 'complete';
  if (stage === 'needs_clarification') return 'clarification';
  if (stage === 'failed') return 'failed';
  return 'routing';
};

const requestError = (error: unknown) =>
  error instanceof SatQueryApiError ? error.message : 'The local backend could not complete this request.';

export function WorkspacePage() {
  const router = useRouter();
  const fileInput = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const [files, setFiles] = useState<UploadItem[]>([]);
  const [query, setQuery] = useState('');
  const [submittedQuery, setSubmittedQuery] = useState('');
  const [stage, setStage] = useState<Stage>('idle');
  const [dragging, setDragging] = useState(false);
  const [analysisId, setAnalysisId] = useState('');
  const [backendStatus, setBackendStatus] = useState<'checking' | 'connected' | 'degraded' | 'offline'>('checking');
  const [notice, setNotice] = useState<string | null>(null);
  const [warnings, setWarnings] = useState<Warning[]>([]);
  const [clarification, setClarification] = useState<ClarificationPayload | null>(null);
  const [roles, setRoles] = useState<Record<string, FileRole>>({});
  const [modalities, setModalities] = useState<Record<string, Modality>>({});
  const [beforeDate, setBeforeDate] = useState('');
  const [afterDate, setAfterDate] = useState('');
  const [clarifiedQuestion, setClarifiedQuestion] = useState('');
  const [historyItems, setHistoryItems] = useState<ChatHistoryItem[]>([]);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [showSettings, setShowSettings] = useState(false);
  const [showProfile, setShowProfile] = useState(false);
  const [activeItemId, setActiveItemId] = useState<string>('');

  const { status: analysisStatus, result: analysisResult, refresh } = useAnalysis(analysisId);

  useEffect(() => {
    satqueryApi
      .health()
      .then((health) => setBackendStatus(health.status === 'ok' ? 'connected' : 'degraded'))
      .catch(() => setBackendStatus('offline'));

    setHistoryLoading(true);
    satqueryApi
      .listAnalyses(50, 0)
      .then((res) => {
        if (res.items && res.items.length > 0) {
          const mapped: ChatHistoryItem[] = res.items.map((item: AnalysisSummary) => ({
            id: item.analysis_id,
            title: item.question || item.task?.replaceAll('_', ' ') || 'Satellite analysis',
            timeAgo: formatRelativeTime(item.created_at),
            section: classifySection(item.created_at),
            isSample: false,
          }));
          setHistoryItems(mapped);
        } else {
          setHistoryItems([]);
        }
      })
      .catch(() => {
        setHistoryItems([]);
      })
      .finally(() => {
        setHistoryLoading(false);
      });
  }, []);

  useEffect(() => {
    if (!analysisStatus) return;
    if (analysisStatus.status === 'completed') {
      setStage('complete');
      setClarification(null);
    } else if (analysisStatus.status === 'failed') {
      setStage('failed');
      setNotice(analysisStatus.error?.message ?? analysisStatus.message ?? 'The analysis failed.');
    } else if (analysisStatus.status === 'needs_clarification') {
      const next = analysisStatus.clarification ?? null;
      setStage('clarification');
      setClarification(next);
      if (next) {
        setRoles(Object.fromEntries(next.upload_ids.map((id, index) => [id, next.allowed_roles[index] ?? next.allowed_roles[0] ?? 'unknown'])) as Record<string, FileRole>);
        setModalities(Object.fromEntries(next.upload_ids.map((id, index) => [id, index === 0 ? 'optical' : 'sar'])) as Record<string, Modality>);
        setClarifiedQuestion(submittedQuery);
      }
    } else {
      setStage(stageFromBackend(analysisStatus.stage));
    }
  }, [analysisStatus, submittedQuery]);

  const running = !['idle', 'complete', 'clarification', 'failed'].includes(stage);
  const conversationStarted = Boolean(submittedQuery);

  const inputMode = useMemo(() => {
    if (!files.length) return 'Waiting for imagery';
    if (files.length === 1) return 'Single-scene observation';
    return (submittedQuery || query).toLowerCase().includes('change') ? 'Bi-temporal comparison' : 'Optical + SAR fusion';
  }, [files.length, query, submittedQuery]);

  const currentStageIndex = analysisStages.findIndex((item) => item.key === stage);
  const currentStage = analysisStages[Math.max(currentStageIndex, 0)];

  const addFiles = (selected: FileList | File[]) => {
    const next = Array.from(selected).map((file, index) => ({
      id: `${file.name}-${file.lastModified}-${index}`,
      file,
      name: file.name,
      size: file.size < 1024 * 1024 ? `${Math.round(file.size / 1024)} KB` : `${(file.size / (1024 * 1024)).toFixed(1)} MB`,
      type: file.name.split('.').pop()?.toUpperCase() ?? 'FILE',
      previewUrl: file.type.startsWith('image/') ? URL.createObjectURL(file) : undefined,
    }));
    setFiles((prev) => [...prev, ...next].slice(0, 2));
  };

  const removeFile = (id: string) => {
    setFiles((prev) => {
      const file = prev.find((f) => f.id === id);
      if (file?.previewUrl) URL.revokeObjectURL(file.previewUrl);
      return prev.filter((f) => f.id !== id);
    });
  };

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) addFiles(event.target.files);
    event.target.value = '';
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    if (event.dataTransfer.files) addFiles(event.dataTransfer.files);
  };

  const handleHistoryClick = (item: ChatHistoryItem) => {
    setActiveItemId(item.id);
    if (!item.isSample) router.push(`/analysis/${item.id}`);
  };

  const resetConversation = () => {
    files.forEach((f) => { if (f.previewUrl) URL.revokeObjectURL(f.previewUrl); });
    setFiles([]); setQuery(''); setSubmittedQuery(''); setStage('idle');
    setAnalysisId(''); setClarification(null); setNotice(null); setWarnings([]); setActiveItemId('');
  };

  const startAnalysis = async () => {
    if (!files.length || !query.trim()) return;
    const q = query.trim();
    setSubmittedQuery(q); setStage('uploading'); setNotice(null); setWarnings([]);
    try {
      const uploadRes = await satqueryApi.upload(files.map((f) => f.file));
      const uploadIds = uploadRes.uploads.map((u) => u.upload_id);
      setStage('validating');
      const analysis = await satqueryApi.createAnalysis(uploadIds, q);
      setAnalysisId(analysis.analysis_id);
      setStage('routing');
    } catch (error) {
      setStage('failed');
      setNotice(requestError(error));
    }
  };

  const submitClarification = async () => {
    if (!analysisId || !clarification) return;
    setStage('routing');
    try {
      const payload: ClarificationResponse = {
        file_roles: roles,
        modalities: Object.values(modalities),
        before_date: beforeDate || undefined,
        after_date: afterDate || undefined,
        question: clarifiedQuestion || undefined,
      };
      await satqueryApi.submitClarification(analysisId, payload);
      refresh();
    } catch (error) {
      setStage('clarification');
      setNotice(requestError(error));
    }
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); void startAnalysis(); }
  };

  const todayItems    = historyItems.filter((it) => it.section === 'Today');
  const pastWeekItems = historyItems.filter((it) => it.section === 'Previous 7 days');
  const olderItems    = historyItems.filter((it) => it.section === 'Older');

  const HistoryGroup = ({ items, label }: { items: ChatHistoryItem[]; label: string }) =>
    items.length > 0 ? (
      <div>
        <div className="chat-history-group-title">{label}</div>
        {items.map((item) => (
          <button
            key={item.id}
            className={`chat-history-row ${activeItemId === item.id ? 'is-active' : ''}`}
            onClick={() => handleHistoryClick(item)}
          >
            <div className="chat-history-row-content">
              <MessageSquare />
              <div className="chat-history-row-text">
                <span className="chat-history-row-title">{item.title}</span>
                <span className="chat-history-row-time">{item.timeAgo}</span>
              </div>
            </div>
            <span className="chat-history-row-more"><MoreVertical /></span>
          </button>
        ))}
      </div>
    ) : null;

  return (
    <main className="chat-workspace-shell">
      <input ref={fileInput} type="file" hidden multiple accept=".tif,.tiff,.png,.jpg,.jpeg" onChange={handleFileChange} />

      <header className="chat-topbar" aria-label="SatQuery navigation">
        <Link href="/" className="chat-brand" aria-label="SatQuery AI home">
          <span className="chat-brand-logo"><SatIcon /></span>
          <span className="chat-brand-name">SATQUERY</span>
          <span className="chat-brand-ai">AI</span>
        </Link>
        <div className="relative">
          <button className="chat-profile-btn" aria-label="User profile" onClick={() => setShowProfile((p) => !p)}>
            <User />
          </button>
          {showProfile && (
            <div className="absolute right-0 top-12 z-50 w-64 rounded-xl border border-[var(--border)] bg-[#0d1013] p-4 shadow-2xl">
              <div className="flex items-center gap-3 border-b border-[var(--border)] pb-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-full bg-[var(--alloy)] text-[var(--solar-foil)]">
                  <User className="h-5 w-5" />
                </div>
                <div>
                  <strong className="block text-xs font-semibold text-[#dfe2e1]">Mission Analyst</strong>
                  <span className="text-[11px] text-[#657079]">SatQuery Intelligence</span>
                </div>
              </div>
              <div className="pt-3 space-y-2 text-xs text-[#8d969c]">
                <div className="flex justify-between"><span>Environment</span><span className="text-[#dfe2e1]">Local On-Premise</span></div>
                <div className="flex justify-between">
                  <span>Backend</span>
                  <span className={backendStatus === 'connected' ? 'text-emerald-400' : 'text-[var(--solar-foil)]'}>{backendStatus}</span>
                </div>
              </div>
              <div className="mt-4 border-t border-[var(--border)] pt-3">
                <Link href="/history" className="flex w-full items-center justify-center gap-2 rounded-lg bg-[var(--alloy)] py-2 text-xs font-medium text-[#dfe2e1]">
                  <History className="h-3.5 w-3.5" /> View Mission History
                </Link>
              </div>
            </div>
          )}
        </div>
      </header>

      <div className="chat-body-container">
        <aside className="chat-sidebar" aria-label="Conversation sidebar">
          <button className="chat-sidebar-new-btn" onClick={resetConversation}>
            <Plus /><span>New Chat</span>
          </button>

          <div className="chat-sidebar-history">
            {historyLoading ? (
              <div className="px-3 py-4 text-xs text-[#626b70] flex items-center gap-2">
                <LoaderCircle className="h-3.5 w-3.5 animate-spin text-[var(--solar-foil)]" />
                <span>Loading missions...</span>
              </div>
            ) : historyItems.length === 0 ? (
              <div className="px-3 py-6 text-center text-xs text-[#626b70]">
                <span>No previous missions found.</span>
              </div>
            ) : (
              <>
                <HistoryGroup items={todayItems} label="Today" />
                <HistoryGroup items={pastWeekItems} label="Previous 7 days" />
                <HistoryGroup items={olderItems} label="Older" />
              </>
            )}
          </div>

          <button className="chat-sidebar-settings-btn" onClick={() => setShowSettings(true)}>
            <Settings /><span>Settings</span>
          </button>
        </aside>

        <main className="chat-main-canvas">
          <div className="chat-canvas-header">
            <Link href="/history"><History aria-hidden="true" />History</Link>
            <button onClick={resetConversation}><Plus aria-hidden="true" />New chat</button>
          </div>

          <div className="chat-thread" aria-label="Satellite analysis conversation">
            {!conversationStarted ? (
              <div className="chat-empty-hero">
                <h1 className="chat-hero-title">Ask. <span className="accent">Analyze.</span> Discover.</h1>
                <p className="chat-hero-subtitle">Attach satellite imagery and ask a question about the scene.</p>
                <div className="chat-hero-cards">
                  {suggestions.map((s) => (
                    <button key={s} className="chat-hero-card" onClick={() => { setQuery(s); textareaRef.current?.focus(); }}>
                      <span>{s}</span><ChevronRight aria-hidden="true" />
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <div className="chat-messages">
                <article className="chat-message chat-message-user">
                  <div className="chat-message-label">You</div>
                  <div className="chat-user-bubble">
                    <p>{submittedQuery}</p>
                    {files.length > 0 && (
                      <div className="chat-inline-files">
                        {files.map((file) => (
                          <span key={file.id}><FileImage aria-hidden="true" /><span>{file.name}<small>{file.size}</small></span></span>
                        ))}
                      </div>
                    )}
                  </div>
                </article>

                <article className="chat-message chat-message-assistant" aria-live="polite">
                  <div className="chat-assistant-avatar" aria-hidden="true"><SatIcon /></div>
                  <div className="chat-assistant-content">
                    {stage === 'complete' ? (
                      <div className="chat-analysis-ready">
                        <span className="ready-label"><Check />Analysis complete</span>
                        {analysisResult?.answer ? (
                          <div className="chat-grounded-box mt-3 mb-2 rounded-xl border border-[var(--border)] bg-[#101316] p-4 text-[#dfe2e1]">
                            <div className="flex items-center gap-2 mb-2 text-xs font-semibold text-[var(--solar-foil)]">
                              <Radar className="h-4 w-4" />
                              <span>Grounded Specialist Answer</span>
                            </div>
                            <p className="text-sm leading-relaxed whitespace-pre-wrap">{analysisResult.answer}</p>
                          </div>
                        ) : (
                          <h2>Your evidence package is ready.</h2>
                        )}
                        <p className="text-xs text-[#858e92]">The response contains spatial artifacts, specialist confidence, warnings, and execution trace.</p>
                        <div className="chat-result-summary">
                          <div><span>Route</span><strong>{analysisStatus?.task?.replaceAll('_', ' ') ?? inputMode}</strong></div>
                          <div><span>Evidence ID</span><strong>{analysisId}</strong></div>
                          {analysisResult?.evidence?.area_value != null && (
                            <div><span>Measured Area</span><strong>{analysisResult.evidence.area_value.toLocaleString('en-IN', { maximumFractionDigits: 2 })} {analysisResult.evidence.area_unit ?? 'm²'}</strong></div>
                          )}
                          {analysisResult?.confidence?.decision && (
                            <div><span>Decision</span><strong>{analysisResult.confidence.decision}</strong></div>
                          )}
                        </div>
                        <div className="flex flex-wrap items-center gap-2 pt-1">
                          <Link href={`/analysis/${analysisId}`} className="inline-flex items-center gap-1.5 rounded-lg bg-[var(--starlight)] px-3.5 py-2 text-xs font-semibold text-[var(--void)] hover:opacity-90 transition-opacity">
                            Open interactive map <ChevronRight className="h-3.5 w-3.5" />
                          </Link>
                          <Button variant="outline" size="sm" onClick={() => window.open(satqueryApi.reportUrl(analysisId, 'html', false), '_blank', 'noopener,noreferrer')}>
                            View Report (HTML)
                          </Button>
                          <Button variant="outline" size="sm" onClick={() => { window.location.href = satqueryApi.reportUrl(analysisId, 'pdf', true); }}>
                            <Download className="mr-1.5 h-3.5 w-3.5" /> Download PDF
                          </Button>
                          <Button variant="outline" size="sm" onClick={() => { window.location.href = satqueryApi.reportUrl(analysisId, 'json', true); }}>
                            <Download className="mr-1.5 h-3.5 w-3.5" /> JSON
                          </Button>
                        </div>
                      </div>
                    ) : stage === 'clarification' && clarification ? (
                      <div className="chat-clarification">
                        <span className="ready-label"><AlertTriangle />Clarification required</span>
                        <h2>The backend needs one detail.</h2>
                        <p>{clarification.question}</p>
                        {clarification.missing_fields.includes('file_roles') && clarification.upload_ids.map((uploadId, index) => (
                          <label key={uploadId}>
                            <span>{files[index]?.name ?? uploadId}</span>
                            <select value={roles[uploadId] ?? 'unknown'} onChange={(e) => setRoles((c) => ({ ...c, [uploadId]: e.target.value as FileRole }))}>
                              {clarification.allowed_roles.map((role) => <option key={role} value={role}>{role}</option>)}
                            </select>
                          </label>
                        ))}
                        {clarification.missing_fields.includes('modality') && clarification.upload_ids.map((uploadId, index) => (
                          <label key={`modality-${uploadId}`}>
                            <span>{files[index]?.name ?? uploadId} modality</span>
                            <select value={modalities[uploadId] ?? 'other'} onChange={(e) => setModalities((c) => ({ ...c, [uploadId]: e.target.value as Modality }))}>
                              <option value="optical">Optical</option><option value="sar">SAR</option><option value="other">Other</option>
                            </select>
                          </label>
                        ))}
                        {clarification.missing_fields.includes('before_date') && (
                          <label><span>Before date</span><input type="date" value={beforeDate} onChange={(e) => setBeforeDate(e.target.value)} /></label>
                        )}
                        {clarification.missing_fields.includes('after_date') && (
                          <label><span>After date</span><input type="date" value={afterDate} onChange={(e) => setAfterDate(e.target.value)} /></label>
                        )}
                        {clarification.missing_fields.includes('question_intent') && (
                          <label><span>Clarified question</span><input value={clarifiedQuestion} onChange={(e) => setClarifiedQuestion(e.target.value)} /></label>
                        )}
                        <Button onClick={() => void submitClarification()}>Resume analysis <ChevronRight /></Button>
                      </div>
                    ) : stage === 'failed' ? (
                      <div className="chat-failure">
                        <span><AlertTriangle />Analysis stopped</span>
                        <h2>The request could not be completed.</h2>
                        <p>{notice}</p>
                        <Button variant="outline" onClick={resetConversation}>Start a new analysis</Button>
                      </div>
                    ) : (
                      <>
                        <div className="chat-thinking-title">
                          <strong>{currentStage?.label}</strong>
                          <span><i /><i /><i /></span>
                        </div>
                        <p>{analysisStatus?.message ?? currentStage?.detail}</p>
                        <ol className="analysis-trace">
                          {analysisStages.map((item, index) => {
                            const complete = index < currentStageIndex;
                            const current = index === currentStageIndex;
                            return (
                              <li key={item.key} className={complete ? 'is-complete' : current ? 'is-current' : ''}>
                                <span>{complete ? <Check /> : current ? <LoaderCircle /> : <i />}</span>
                                <div><strong>{item.label}</strong><small>{item.detail}</small></div>
                              </li>
                            );
                          })}
                        </ol>
                      </>
                    )}
                    {warnings.length > 0 && stage !== 'failed' && (
                      <div className="chat-backend-notice">
                        <AlertTriangle /><span><strong>{warnings.length} backend notice{warnings.length > 1 ? 's' : ''}</strong>{warnings[0].message}</span>
                      </div>
                    )}
                    {notice && stage === 'clarification' && (
                      <div className="chat-backend-notice is-error"><AlertTriangle /><span>{notice}</span></div>
                    )}
                  </div>
                </article>
              </div>
            )}
          </div>

          <div className="chat-dock">
            {files.length > 0 && !conversationStarted && (
              <div className="chat-dock-files">
                {files.map((file) => (
                  <div className="chat-dock-file" key={file.id}>
                    {file.previewUrl ? <img src={file.previewUrl} alt="" /> : <FileImage aria-hidden="true" />}
                    <div><strong>{file.name}</strong><span>{file.type} · {file.size}</span></div>
                    <button aria-label={`Remove ${file.name}`} onClick={() => removeFile(file.id)}><X /></button>
                  </div>
                ))}
              </div>
            )}
            <div
              className={`chat-dock-pill ${dragging ? 'is-dragging' : ''}`}
              onDragEnter={() => setDragging(true)}
              onDragLeave={() => setDragging(false)}
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
            >
              <button className="dock-attach-btn" aria-label="Attach satellite imagery" disabled={conversationStarted || files.length >= 2} onClick={() => fileInput.current?.click()}>
                <Paperclip aria-hidden="true" />
              </button>
              <textarea
                ref={textareaRef}
                className="dock-input"
                value={query}
                disabled={conversationStarted}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={conversationStarted ? 'Start a new chat to analyse another scene' : 'Ask a question about the attached imagery...'}
                maxLength={400}
                aria-label="Analysis question"
                rows={1}
              />
              <button
                className="dock-send-btn"
                aria-label="Send question"
                disabled={!files.length || !query.trim() || running || conversationStarted || backendStatus === 'offline'}
                onClick={() => void startAnalysis()}
              >
                <ArrowUp aria-hidden="true" />
              </button>
            </div>
            <div className="chat-dock-meta">
              <div className="flex items-center gap-1.5">
                <span>Attach up to two scenes</span>
                <Info className="h-3 w-3 text-[#727c82]" />
              </div>
              <div className="flex items-center gap-2">
                <span>Enter to send · Shift + Enter for a new line</span>
                <span className={`chat-online-dot is-${backendStatus}`} title={`Backend ${backendStatus}`} />
              </div>
            </div>
          </div>
        </main>
      </div>

      {showSettings && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={() => setShowSettings(false)} aria-modal="true" role="dialog">
          <div className="relative w-full max-w-md rounded-2xl border border-[var(--border)] bg-[#0d1013] p-6 shadow-2xl" onClick={(e) => e.stopPropagation()}>
            <button className="absolute right-4 top-4 rounded-lg p-1 text-[#657079] hover:text-[var(--starlight)]" onClick={() => setShowSettings(false)} aria-label="Close settings">
              <X className="h-4 w-4" />
            </button>
            <h2 className="mb-5 text-sm font-semibold text-[#dfe2e1]">Settings</h2>
            <div className="space-y-4 text-xs text-[#8d969c]">
              <div className="flex items-start gap-3 rounded-xl border border-[var(--border)] bg-[#111417] p-4">
                <Database className="mt-0.5 h-4 w-4 text-[var(--solar-foil)]" />
                <div><strong className="block font-medium text-[#dfe2e1]">API Endpoint</strong><span className="font-mono text-[11px]">http://localhost:8000/api/v1</span></div>
              </div>
              <div className="flex items-start gap-3 rounded-xl border border-[var(--border)] bg-[#111417] p-4">
                <Cpu className="mt-0.5 h-4 w-4 text-[var(--solar-foil)]" />
                <div><strong className="block font-medium text-[#dfe2e1]">GPU Compute</strong><span>RTX 4050 Laptop · CUDA 12.8 · PyTorch 2.11</span></div>
              </div>
              <div className="flex items-start gap-3 rounded-xl border border-[var(--border)] bg-[#111417] p-4">
                <ShieldCheck className="mt-0.5 h-4 w-4 text-[var(--solar-foil)]" />
                <div><strong className="block font-medium text-[#dfe2e1]">LangGraph Pipeline</strong><span>SatVLM · ChangeNet · SAR-FuseSeg · Evidence Engine</span></div>
              </div>
              <div className="flex items-start gap-3 rounded-xl border border-[var(--border)] bg-[#111417] p-4">
                <Info className="mt-0.5 h-4 w-4 text-[var(--solar-foil)]" />
                <div>
                  <strong className="block font-medium text-[#dfe2e1]">Backend Status</strong>
                  <span className={backendStatus === 'connected' ? 'text-emerald-400' : 'text-[var(--solar-foil)]'}>{backendStatus}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
