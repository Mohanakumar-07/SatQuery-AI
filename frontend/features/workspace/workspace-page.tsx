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
  Trash2,
  User,
  X,
} from 'lucide-react';

import { SatIcon } from '@/components/site/sat-icon';
import { Button } from '@/components/ui/button';
import { useAnalysis } from '@/hooks/use-analysis';
import {
  type AnalysisResult,
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

type ChatMessage = {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  files?: UploadItem[];
  analysisId?: string;
  result?: AnalysisResult | null;
  stage?: Stage;
  error?: string | null;
};

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
  { key: 'inference',  label: 'Evaluating spatial patterns',  detail: 'Running GPU specialist inference with CUDA acceleration.' },
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

  // Multi-turn conversation state: one persistent thread_id per session
  const [threadId, setThreadId] = useState<string>(() => `thread_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [activeUploadIds, setActiveUploadIds] = useState<string[]>([]);

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
  const [menuOpenId, setMenuOpenId] = useState<string | null>(null);
  const dragCounter = useRef(0);

  const { status: analysisStatus, result: analysisResult, refresh } = useAnalysis(analysisId);

  const loadHistory = () => {
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
  };

  useEffect(() => {
    const handleWindowClick = () => setMenuOpenId(null);
    const handleKeyDown = (e: globalThis.KeyboardEvent) => {
      if (e.key === 'Escape') setMenuOpenId(null);
    };
    window.addEventListener('click', handleWindowClick);
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('click', handleWindowClick);
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, []);

  useEffect(() => {
    satqueryApi
      .health()
      .then((health) => setBackendStatus(health.status === 'ok' ? 'connected' : 'degraded'))
      .catch(() => setBackendStatus('offline'));

    loadHistory();
  }, []);

  // Update assistant message state as the analysis pipeline progresses
  useEffect(() => {
    if (!analysisStatus || !analysisId) return;

    if (analysisStatus.status === 'completed') {
      setStage('complete');
      setClarification(null);
      if (analysisResult) {
        setMessages((prev) =>
          prev.map((m) =>
            m.analysisId === analysisId
              ? {
                  ...m,
                  stage: 'complete',
                  text: analysisResult.answer || 'Analysis complete.',
                  result: analysisResult,
                }
              : m
          )
        );
        loadHistory();
      }
    } else if (analysisStatus.status === 'failed') {
      const errStr = analysisStatus.error?.message ?? analysisStatus.message ?? 'The analysis failed.';
      setStage('failed');
      setNotice(errStr);
      setMessages((prev) =>
        prev.map((m) => (m.analysisId === analysisId ? { ...m, stage: 'failed', error: errStr } : m))
      );
    } else if (analysisStatus.status === 'needs_clarification') {
      const next = analysisStatus.clarification ?? null;
      setStage('clarification');
      setClarification(next);
      if (next) {
        setClarifiedQuestion(submittedQuery);
        setRoles(
          Object.fromEntries(
            next.upload_ids.map((id, index) => [
              id,
              next.allowed_roles[index] ?? next.allowed_roles[0] ?? 'unknown',
            ])
          ) as Record<string, FileRole>
        );
        setModalities(
          Object.fromEntries(
            next.upload_ids.map((id, index) => [id, index === 0 ? 'optical' : 'sar'])
          ) as Record<string, Modality>
        );
      }
    } else {
      const curStage = stageFromBackend(analysisStatus.stage);
      setStage(curStage);
      setMessages((prev) =>
        prev.map((m) => (m.analysisId === analysisId ? { ...m, stage: curStage } : m))
      );
    }
  }, [analysisStatus, analysisResult, analysisId, submittedQuery]);

  const running = !['idle', 'complete', 'clarification', 'failed'].includes(stage);
  const conversationStarted = messages.length > 0;

  const inputMode = useMemo(() => {
    if (!files.length && !activeUploadIds.length) return 'Waiting for imagery';
    if (files.length === 1 || activeUploadIds.length === 1) return 'Single-scene observation';
    return query.toLowerCase().includes('change') ? 'Bi-temporal comparison' : 'Optical + SAR fusion';
  }, [files.length, activeUploadIds.length, query]);

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

  const handleDragEnter = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    dragCounter.current += 1;
    if (event.dataTransfer?.items && event.dataTransfer.items.length > 0) {
      setDragging(true);
    }
  };

  const handleDragLeave = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    dragCounter.current -= 1;
    if (dragCounter.current <= 0) {
      dragCounter.current = 0;
      setDragging(false);
    }
  };

  const handleDragOver = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    dragCounter.current = 0;
    setDragging(false);
    if (event.dataTransfer.files && event.dataTransfer.files.length > 0) {
      addFiles(event.dataTransfer.files);
    }
  };

  const handleDeleteChat = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    setMenuOpenId(null);
    setHistoryItems((prev) => prev.filter((item) => item.id !== id));
    if (activeItemId === id || analysisId === id) {
      resetConversation();
    }
    try {
      await satqueryApi.deleteAnalysis(id);
    } catch (err) {
      console.warn('Could not delete analysis from database:', err);
    }
  };

  // Sidebar history click: load previous mission into conversation view (keeps user on Chat page!)
  const handleHistoryClick = async (item: ChatHistoryItem) => {
    setActiveItemId(item.id);
    if (item.isSample) return;

    try {
      const res = await satqueryApi.analysisResult(item.id);
      setThreadId(item.id);
      setAnalysisId(item.id);
      if (res.upload_ids && res.upload_ids.length > 0) {
        setActiveUploadIds(res.upload_ids);
      }
      const userMsg: ChatMessage = {
        id: `hist_user_${item.id}`,
        role: 'user',
        text: res.question || item.title,
      };
      const assistantMsg: ChatMessage = {
        id: `hist_asst_${item.id}`,
        role: 'assistant',
        text: res.answer || '',
        stage: 'complete',
        analysisId: res.analysis_id,
        result: res,
      };
      setMessages([userMsg, assistantMsg]);
      setStage('complete');
    } catch {
      // Fallback navigation if result cannot be loaded directly
      router.push(`/analysis/${item.id}`);
    }
  };

  const resetConversation = () => {
    files.forEach((f) => { if (f.previewUrl) URL.revokeObjectURL(f.previewUrl); });
    setFiles([]);
    setActiveUploadIds([]);
    setQuery('');
    setSubmittedQuery('');
    setStage('idle');
    setAnalysisId('');
    setClarification(null);
    setNotice(null);
    setWarnings([]);
    setActiveItemId('');
    setMessages([]);
    // Generate one fresh thread_id for the next chat
    setThreadId(`thread_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`);
  };

  // Sends a query (initial or follow-up) in the same conversation thread
  const startAnalysis = async () => {
    if (!query.trim()) return;
    if (!files.length && !activeUploadIds.length) return;

    const q = query.trim();
    setSubmittedQuery(q);
    const userMsgId = `user_${Date.now()}`;
    const assistantMsgId = `assistant_${Date.now()}`;

    const userMsg: ChatMessage = {
      id: userMsgId,
      role: 'user',
      text: q,
      files: files.length > 0 ? [...files] : undefined,
    };
    const assistantMsg: ChatMessage = {
      id: assistantMsgId,
      role: 'assistant',
      text: '',
      stage: 'uploading',
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setQuery('');
    setStage('uploading');
    setNotice(null);
    setWarnings([]);

    try {
      let uploadIds = activeUploadIds;
      // Upload any new files that don't have an uploadId yet
      const unuploaded = files.filter((f) => !f.uploadId);
      if (unuploaded.length > 0) {
        const uploadRes = await satqueryApi.upload(files.map((f) => f.file));
        uploadIds = uploadRes.uploads.map((u) => u.upload_id);
        setActiveUploadIds(uploadIds);
        setFiles((prev) =>
          prev.map((f, i) => ({ ...f, uploadId: uploadIds[i] ?? f.uploadId }))
        );
      }

      setStage('validating');
      // Pass the persistent thread_id so LangGraph reuses memory and prior messages
      const analysis = await satqueryApi.createAnalysis(uploadIds, q, threadId);
      setAnalysisId(analysis.analysis_id);

      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantMsgId
            ? { ...m, analysisId: analysis.analysis_id, stage: 'routing' }
            : m
        )
      );
      setStage('routing');
    } catch (error) {
      const errStr = requestError(error);
      setStage('failed');
      setNotice(errStr);
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantMsgId ? { ...m, stage: 'failed', error: errStr } : m
        )
      );
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
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      void startAnalysis();
    }
  };

  const todayItems    = historyItems.filter((it) => it.section === 'Today');
  const pastWeekItems = historyItems.filter((it) => it.section === 'Previous 7 days');
  const olderItems    = historyItems.filter((it) => it.section === 'Older');

  const HistoryGroup = ({ items, label }: { items: ChatHistoryItem[]; label: string }) =>
    items.length > 0 ? (
      <div>
        <div className="chat-history-group-title">{label}</div>
        {items.map((item) => (
          <div
            key={item.id}
            className={`chat-history-row ${activeItemId === item.id ? 'is-active' : ''}`}
            onClick={() => handleHistoryClick(item)}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                void handleHistoryClick(item);
              }
            }}
          >
            <div className="chat-history-row-content">
              <MessageSquare />
              <div className="chat-history-row-text">
                <span className="chat-history-row-title">{item.title}</span>
                <span className="chat-history-row-time">{item.timeAgo}</span>
              </div>
            </div>
            <button
              type="button"
              className={`chat-history-row-more ${menuOpenId === item.id ? 'is-open' : ''}`}
              aria-label="Chat options"
              onClick={(e) => {
                e.stopPropagation();
                setMenuOpenId((cur) => (cur === item.id ? null : item.id));
              }}
            >
              <MoreVertical />
            </button>
            {menuOpenId === item.id && (
              <div className="chat-history-dropdown" onClick={(e) => e.stopPropagation()}>
                <button
                  type="button"
                  className="chat-history-dropdown-item"
                  onClick={(e) => {
                    e.stopPropagation();
                    setMenuOpenId(null);
                    router.push(`/analysis/${item.id}`);
                  }}
                >
                  <Radar />
                  <span>View Analysis</span>
                </button>
                <button
                  type="button"
                  className="chat-history-dropdown-item is-danger"
                  onClick={(e) => void handleDeleteChat(e, item.id)}
                >
                  <Trash2 />
                  <span>Delete Chat</span>
                </button>
              </div>
            )}
          </div>
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

          <div className="chat-sidebar-history" data-lenis-prevent="true">
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

        <main
          className="chat-main-canvas"
          data-lenis-prevent="true"
          onDragEnter={handleDragEnter}
          onDragLeave={handleDragLeave}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
        >
          {dragging && (
            <div className="chat-canvas-dropzone">
              <FileImage />
              <strong>Drop satellite imagery here</strong>
              <p>Supports GeoTIFF, TIFF, PNG, and JPEG formats (up to 2 files)</p>
            </div>
          )}

          <div className="chat-canvas-header">
            <Link href="/history"><History aria-hidden="true" />History</Link>
            <button onClick={resetConversation}><Plus aria-hidden="true" />New chat</button>
          </div>

          <div className="chat-thread" data-lenis-prevent="true" aria-label="Satellite analysis conversation">
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
                {messages.map((msg) =>
                  msg.role === 'user' ? (
                    <article className="chat-message chat-message-user" key={msg.id}>
                      <div className="chat-message-label">You</div>
                      <div className="chat-user-bubble">
                        <p>{msg.text}</p>
                        {msg.files && msg.files.length > 0 && (
                          <div className="chat-inline-files">
                            {msg.files.map((file) => (
                              <span key={file.id}>
                                <FileImage aria-hidden="true" />
                                <span>{file.name}<small>{file.size}</small></span>
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    </article>
                  ) : (
                    <article className="chat-message chat-message-assistant" aria-live="polite" key={msg.id}>
                      <div className="chat-assistant-avatar" aria-hidden="true"><SatIcon /></div>
                      <div className="chat-assistant-content">
                        {msg.stage === 'complete' ? (
                          <div className="chat-analysis-ready">
                            <span className="ready-label"><Check />Analysis complete</span>
                            {msg.result?.answer ? (
                              <div className="chat-grounded-box mt-3 mb-2 rounded-xl border border-[var(--border)] bg-[#101316] p-4 text-[#dfe2e1]">
                                <div className="flex items-center gap-2 mb-2 text-xs font-semibold text-[var(--solar-foil)]">
                                  <Radar className="h-4 w-4" />
                                  <span>Grounded Specialist Answer</span>
                                </div>
                                <p className="text-sm leading-relaxed whitespace-pre-wrap">{msg.result.answer}</p>
                              </div>
                            ) : (
                              <h2>Your evidence package is ready.</h2>
                            )}
                            <p className="text-xs text-[#858e92]">The response contains spatial artifacts, specialist confidence, warnings, and execution trace.</p>
                            <div className="chat-result-summary">
                              <div><span>Route</span><strong>{msg.result?.task?.replaceAll('_', ' ') ?? inputMode}</strong></div>
                              <div><span>Evidence ID</span><strong>{msg.analysisId}</strong></div>
                              {msg.result?.evidence?.area_value != null && (
                                <div><span>Measured Area</span><strong>{msg.result.evidence.area_value.toLocaleString('en-IN', { maximumFractionDigits: 2 })} {msg.result.evidence.area_unit ?? 'm²'}</strong></div>
                              )}
                              {msg.result?.confidence?.decision && (
                                <div><span>Decision</span><strong>{msg.result.confidence.decision}</strong></div>
                              )}
                            </div>
                            <div className="flex flex-wrap items-center gap-2 pt-1">
                              {msg.analysisId && (
                                <Link
                                  href={`/analysis/${msg.analysisId}`}
                                  className="chat-view-analysis-btn"
                                >
                                  <Radar className="h-3.5 w-3.5" /> View Analysis <ChevronRight className="h-3.5 w-3.5" />
                                </Link>
                              )}
                              {msg.analysisId && (
                                <>
                                  <Button variant="outline" size="sm" onClick={() => window.open(satqueryApi.reportUrl(msg.analysisId!, 'html', false), '_blank', 'noopener,noreferrer')}>
                                    View Report (HTML)
                                  </Button>
                                  <Button variant="outline" size="sm" onClick={() => { window.location.href = satqueryApi.reportUrl(msg.analysisId!, 'pdf', true); }}>
                                    <Download className="mr-1.5 h-3.5 w-3.5" /> Download PDF
                                  </Button>
                                  <Button variant="outline" size="sm" onClick={() => { window.location.href = satqueryApi.reportUrl(msg.analysisId!, 'json', true); }}>
                                    <Download className="mr-1.5 h-3.5 w-3.5" /> JSON
                                  </Button>
                                </>
                              )}
                            </div>
                          </div>
                        ) : msg.stage === 'clarification' && clarification ? (
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
                        ) : msg.stage === 'failed' ? (
                          <div className="chat-failure">
                            <span><AlertTriangle />Analysis stopped</span>
                            <h2>The request could not be completed.</h2>
                            <p>{msg.error || notice}</p>
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
                  )
                )}
              </div>
            )}
          </div>

          <div className="chat-dock">
            {files.length > 0 && (
              <div className="chat-dock-files">
                {files.map((file) => (
                  <div className="chat-dock-file" key={file.id}>
                    {file.previewUrl ? <img src={file.previewUrl} alt="" /> : <FileImage aria-hidden="true" />}
                    <div><strong>{file.name}</strong><span>{file.type} · {file.size}</span></div>
                    {!running && <button aria-label={`Remove ${file.name}`} onClick={() => removeFile(file.id)}><X /></button>}
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
              <button
                className="dock-attach-btn"
                aria-label="Attach satellite imagery"
                disabled={running || files.length >= 2}
                onClick={() => fileInput.current?.click()}
              >
                <Paperclip aria-hidden="true" />
              </button>
              <textarea
                ref={textareaRef}
                className="dock-input"
                value={query}
                disabled={running}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={
                  running
                    ? 'Analyzing scene...'
                    : conversationStarted
                    ? 'Ask a follow-up question about this scene...'
                    : 'Ask a question about the attached imagery...'
                }
                maxLength={400}
                aria-label="Analysis question"
                rows={1}
              />
              <button
                className="dock-send-btn"
                aria-label="Send question"
                disabled={(!files.length && !activeUploadIds.length) || !query.trim() || running || backendStatus === 'offline'}
                onClick={() => void startAnalysis()}
              >
                <ArrowUp aria-hidden="true" />
              </button>
            </div>
            <div className="chat-dock-meta">
              <div className="flex items-center gap-1.5">
                <span>{conversationStarted ? 'Multi-turn conversation active' : 'Attach up to two scenes'}</span>
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
                <div><strong className="block font-medium text-[#dfe2e1]">GPU Compute</strong><span>CUDA Accelerated Inference</span></div>
              </div>
              <div className="flex items-start gap-3 rounded-xl border border-[var(--border)] bg-[#111417] p-4">
                <ShieldCheck className="mt-0.5 h-4 w-4 text-[var(--solar-foil)]" />
                <div><strong className="block font-medium text-[#dfe2e1]">LangGraph Pipeline</strong><span>Multi-turn memory · SatVLM · ChangeNet · SAR-FuseSeg</span></div>
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
