'use client';

import { ChangeEvent, DragEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from 'react';
import Link from 'next/link';
import { AlertTriangle, ArrowUp, Check, ChevronRight, FileImage, History, ImagePlus, LoaderCircle, Orbit, Paperclip, Plus, RotateCcw, Satellite, X } from 'lucide-react';

import { SiteHeader } from '@/components/site/site-header';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { useAnalysis } from '@/hooks/use-analysis';
import { type AnalysisStage, type ClarificationPayload, type ClarificationResponse, SatQueryApiError, satqueryApi, type Warning } from '@/lib/satquery-api';

type UploadItem = { id: string; file: File; name: string; size: string; type: string; previewUrl?: string; uploadId?: string };
type Stage = 'idle' | 'uploading' | 'validating' | 'routing' | 'inference' | 'evidence' | 'clarification' | 'complete' | 'failed';
type FileRole = 'before' | 'after' | 'optical' | 'sar' | 'single' | 'unknown';
type Modality = 'optical' | 'sar' | 'other';

const analysisStages = [
  { key: 'uploading', label: 'Securing your imagery', detail: 'Uploading the original raster files to local evidence storage.' },
  { key: 'validating', label: 'Reading imagery and metadata', detail: 'Checking format, coordinates, acquisition time, and resolution.' },
  { key: 'routing', label: 'Interpreting your question', detail: 'Selecting the permitted sensor-aware workflow from the input evidence.' },
  { key: 'inference', label: 'Evaluating spatial patterns', detail: 'Running the attached specialist pipeline for this mission.' },
  { key: 'evidence', label: 'Building the evidence trace', detail: 'Packaging regions, measurements, confidence, provenance, and reports.' },
] as const;

const suggestions = ['Describe the visible features in this scene', 'Where has the built-up area changed?', 'Map the visible water regions'];

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

const requestError = (error: unknown) => error instanceof SatQueryApiError ? error.message : 'The local backend could not complete this request.';

export function WorkspacePage() {
  const fileInput = useRef<HTMLInputElement>(null);
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
  const { status: analysisStatus, refresh: refreshAnalysis } = useAnalysis(analysisId);

  useEffect(() => {
    satqueryApi.health().then((health) => setBackendStatus(health.status === 'ok' ? 'connected' : 'degraded')).catch(() => setBackendStatus('offline'));
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
    } else setStage(stageFromBackend(analysisStatus.stage));
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
      size: file.size > 1_000_000 ? `${(file.size / 1_000_000).toFixed(1)} MB` : `${Math.ceil(file.size / 1000)} KB`,
      type: /\.tiff?$/i.test(file.name) ? 'GeoTIFF' : 'Raster image',
      previewUrl: file.type.startsWith('image/') && !/\.tiff?$/i.test(file.name) ? URL.createObjectURL(file) : undefined,
    }));
    setFiles((current) => {
      const combined = [...current, ...next].filter((item, index, all) => all.findIndex((candidate) => candidate.id === item.id) === index);
      combined.slice(2).forEach((item) => item.previewUrl && URL.revokeObjectURL(item.previewUrl));
      return combined.slice(0, 2);
    });
  };

  const removeFile = (id: string) => setFiles((current) => {
    const removed = current.find((item) => item.id === id);
    if (removed?.previewUrl) URL.revokeObjectURL(removed.previewUrl);
    return current.filter((item) => item.id !== id);
  });
  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) addFiles(event.target.files);
    event.target.value = '';
  };
  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    if (event.dataTransfer.files.length) addFiles(event.dataTransfer.files);
  };

  const resetConversation = () => {
    files.forEach((file) => file.previewUrl && URL.revokeObjectURL(file.previewUrl));
    setFiles([]); setQuery(''); setSubmittedQuery(''); setAnalysisId(''); setStage('idle'); setNotice(null); setWarnings([]); setClarification(null);
    setRoles({}); setModalities({}); setBeforeDate(''); setAfterDate('');
  };

  const startAnalysis = async () => {
    if (!files.length || !query.trim() || running) return;
    const question = query.trim();
    setSubmittedQuery(question); setQuery(''); setAnalysisId(''); setNotice(null); setWarnings([]); setClarification(null); setStage('uploading');
    try {
      const uploaded = await satqueryApi.upload(files.map((item) => item.file));
      const uploadIds = uploaded.uploads.map((item) => item.upload_id);
      setFiles((current) => current.map((item, index) => ({ ...item, uploadId: uploadIds[index] })));
      setWarnings([...uploaded.warnings, ...uploaded.uploads.flatMap((item) => item.warnings)]);
      setStage('validating');
      const validation = await satqueryApi.validate(uploadIds, question);
      setWarnings((current) => [...current, ...validation.warnings]);
      if (!validation.valid) {
        setStage('failed');
        setNotice(validation.errors.map((item) => item.message).join(' ') || 'The imagery did not pass validation.');
        return;
      }
      setStage('routing');
      const created = await satqueryApi.createAnalysis(uploadIds, question);
      setAnalysisId(created.analysis_id);
      setStage(stageFromBackend(created.stage));
    } catch (error) {
      setStage('failed'); setNotice(requestError(error));
      if (error instanceof TypeError) setBackendStatus('offline');
    }
  };

  const submitClarification = async () => {
    if (!analysisId || !clarification) return;
    const payload: ClarificationResponse = {};
    if (clarification.missing_fields.includes('file_roles')) payload.file_roles = roles;
    if (clarification.missing_fields.includes('modality')) payload.modalities = clarification.upload_ids.map((id) => modalities[id] ?? 'other');
    if (clarification.missing_fields.includes('before_date') && beforeDate) payload.before_date = beforeDate;
    if (clarification.missing_fields.includes('after_date') && afterDate) payload.after_date = afterDate;
    if (clarification.missing_fields.includes('question_intent') && clarifiedQuestion.trim()) payload.question = clarifiedQuestion.trim();
    setNotice(null); setStage('routing');
    try {
      await satqueryApi.submitClarification(analysisId, payload);
      setClarification(null); refreshAnalysis();
    } catch (error) {
      setStage('clarification'); setNotice(requestError(error));
    }
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); void startAnalysis(); }
  };

  return (
    <main className="app-shell chat-workspace-shell">
      <SiteHeader mode="app" />
      <input ref={fileInput} type="file" hidden multiple accept=".tif,.tiff,.png,.jpg,.jpeg" onChange={handleFileChange} />
      <section className="chat-workspace" aria-label="Satellite analysis conversation">
        <header className="chat-context-header">
          <div><span className={`chat-online-dot is-${backendStatus}`} /><span>Scene analysis</span><small>{inputMode} · {backendStatus === 'checking' ? 'connecting locally' : `backend ${backendStatus}`}</small></div>
          <div><Link href="/history"><History aria-hidden="true" />History</Link><button onClick={resetConversation}><Plus aria-hidden="true" />New chat</button></div>
        </header>

        <div className={`chat-thread ${conversationStarted ? 'has-conversation' : ''}`}>
          {!conversationStarted ? (
            <div className="chat-empty-state">
              <div className="chat-mark" aria-hidden="true"><Orbit /></div><span>SatQuery workspace</span><h1>What do you want to understand?</h1>
              <p>Attach satellite imagery, then ask a clear question about the scene.</p>
              <div className="chat-empty-actions"><button onClick={() => fileInput.current?.click()}><ImagePlus aria-hidden="true" />Upload imagery</button><button onClick={() => setQuery('Where has the built-up area changed between these dates?')}><RotateCcw aria-hidden="true" />Use sample question</button></div>
              <div className="chat-suggestions" aria-label="Suggested questions">{suggestions.map((suggestion) => <button key={suggestion} onClick={() => setQuery(suggestion)}>{suggestion}<ChevronRight aria-hidden="true" /></button>)}</div>
            </div>
          ) : (
            <div className="chat-messages">
              <article className="chat-message chat-message-user"><div className="chat-message-label">You</div><div className="chat-user-bubble"><p>{submittedQuery}</p><div className="chat-inline-files">{files.map((file) => <span key={file.id}><FileImage aria-hidden="true" /><span>{file.name}<small>{file.size}</small></span></span>)}</div></div></article>
              <article className="chat-message chat-message-assistant" aria-live="polite">
                <div className="chat-assistant-avatar" aria-hidden="true"><Satellite /></div>
                <div className="chat-assistant-content">
                  {stage === 'complete' ? (
                    <div className="chat-analysis-ready"><span className="ready-label"><Check />Analysis complete</span><h2>Your evidence package is ready.</h2><p>The response contains the backend result, spatial artifacts, separate specialist confidence, warnings, and execution trace.</p><div className="chat-result-summary"><div><span>Route</span><strong>{analysisStatus?.task?.replaceAll('_', ' ') ?? inputMode}</strong></div><div><span>Evidence ID</span><strong>{analysisId}</strong></div></div><Link href={`/analysis/${analysisId}`}>Open evidence <ChevronRight aria-hidden="true" /></Link></div>
                  ) : stage === 'clarification' && clarification ? (
                    <div className="chat-clarification">
                      <span className="ready-label"><AlertTriangle />Clarification required</span><h2>The backend needs one detail.</h2><p>{clarification.question}</p>
                      {clarification.missing_fields.includes('file_roles') && clarification.upload_ids.map((uploadId, index) => <label key={uploadId}><span>{files[index]?.name ?? uploadId}</span><select value={roles[uploadId] ?? 'unknown'} onChange={(event) => setRoles((current) => ({ ...current, [uploadId]: event.target.value as FileRole }))}>{clarification.allowed_roles.map((role) => <option key={role} value={role}>{role}</option>)}</select></label>)}
                      {clarification.missing_fields.includes('modality') && clarification.upload_ids.map((uploadId, index) => <label key={`modality-${uploadId}`}><span>{files[index]?.name ?? uploadId} modality</span><select value={modalities[uploadId] ?? 'other'} onChange={(event) => setModalities((current) => ({ ...current, [uploadId]: event.target.value as Modality }))}><option value="optical">Optical</option><option value="sar">SAR</option><option value="other">Other</option></select></label>)}
                      {clarification.missing_fields.includes('before_date') && <label><span>Before date</span><input type="date" value={beforeDate} onChange={(event) => setBeforeDate(event.target.value)} /></label>}
                      {clarification.missing_fields.includes('after_date') && <label><span>After date</span><input type="date" value={afterDate} onChange={(event) => setAfterDate(event.target.value)} /></label>}
                      {clarification.missing_fields.includes('question_intent') && <label><span>Clarified question</span><input value={clarifiedQuestion} onChange={(event) => setClarifiedQuestion(event.target.value)} /></label>}
                      <Button onClick={() => void submitClarification()}>Resume analysis <ChevronRight /></Button>
                    </div>
                  ) : stage === 'failed' ? (
                    <div className="chat-failure"><span><AlertTriangle />Analysis stopped</span><h2>The request could not be completed.</h2><p>{notice}</p><Button variant="outline" onClick={resetConversation}>Start a new analysis</Button></div>
                  ) : (
                    <><div className="chat-thinking-title"><strong>{currentStage.label}</strong><span><i /><i /><i /></span></div><p>{analysisStatus?.message ?? currentStage.detail}</p><ol className="analysis-trace">{analysisStages.map((item, index) => { const complete = index < currentStageIndex; const current = index === currentStageIndex; return <li key={item.key} className={complete ? 'is-complete' : current ? 'is-current' : ''}><span>{complete ? <Check /> : current ? <LoaderCircle /> : <i />}</span><div><strong>{item.label}</strong><small>{item.detail}</small></div></li>; })}</ol></>
                  )}
                  {warnings.length > 0 && stage !== 'failed' && <div className="chat-backend-notice"><AlertTriangle /><span><strong>{warnings.length} backend notice{warnings.length > 1 ? 's' : ''}</strong>{warnings[0].message}</span></div>}
                  {notice && stage === 'clarification' && <div className="chat-backend-notice is-error"><AlertTriangle /><span>{notice}</span></div>}
                </div>
              </article>
            </div>
          )}
        </div>

        <div className="chat-composer-dock">
          {files.length > 0 && !conversationStarted && <div className="composer-attachments" aria-label="Attached imagery">{files.map((file) => <div className="composer-file" key={file.id}>{file.previewUrl ? <img src={file.previewUrl} alt="" /> : <FileImage aria-hidden="true" />}<div><strong>{file.name}</strong><span>{file.type} · {file.size}</span></div><button aria-label={`Remove ${file.name}`} onClick={() => removeFile(file.id)}><X /></button></div>)}</div>}
          <div className={`chat-composer ${dragging ? 'is-dragging' : ''}`} onDragEnter={() => setDragging(true)} onDragLeave={() => setDragging(false)} onDragOver={(event) => event.preventDefault()} onDrop={handleDrop}>
            <Button variant="ghost" size="icon" className="composer-attach" aria-label="Attach satellite imagery" disabled={conversationStarted || files.length >= 2} onClick={() => fileInput.current?.click()}><Paperclip aria-hidden="true" /></Button>
            <Textarea value={query} disabled={conversationStarted} onChange={(event) => setQuery(event.target.value)} onKeyDown={handleKeyDown} placeholder={conversationStarted ? 'Start a new chat to analyse another scene' : 'Ask a question about the attached imagery'} maxLength={400} aria-label="Analysis question" />
            <Button size="icon" className="composer-send" aria-label="Send question" disabled={!files.length || !query.trim() || running || conversationStarted || backendStatus === 'offline'} onClick={() => void startAnalysis()}><ArrowUp aria-hidden="true" /></Button>
          </div>
          <div className="composer-meta"><span>{files.length ? `${files.length} scene${files.length > 1 ? 's' : ''} attached` : 'Attach up to two scenes'}</span><span>{backendStatus === 'offline' ? 'Start the backend on localhost:8000' : 'Enter to send · Shift + Enter for a new line'}</span></div>
        </div>
      </section>
    </main>
  );
}

