'use client';

import { useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import { AlertTriangle, Check, ChevronDown, CircleCheck, Download, FileJson, Layers3, LoaderCircle, MapPin, Radar, Route, ShieldCheck } from 'lucide-react';

import { SiteHeader } from '@/components/site/site-header';
import { SceneMap } from '@/components/visuals/scene-map';
import { Button } from '@/components/ui/button';
import { useAnalysis } from '@/hooks/use-analysis';
import { type AnalysisResult, type ClarificationPayload, type ClarificationResponse, resolveBackendUrl, satqueryApi } from '@/lib/satquery-api';

type FileRole = 'before' | 'after' | 'optical' | 'sar' | 'single' | 'unknown';
type Modality = 'optical' | 'sar' | 'other';

const readable = (value?: string | null) => value ? value.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase()) : 'Analysis';
const formatNumber = (value?: number | null, digits = 2) => value == null ? '—' : new Intl.NumberFormat('en-IN', { maximumFractionDigits: digits }).format(value);
const formatArea = (result: AnalysisResult) => {
  const evidence = result.evidence;
  if (evidence?.area_value == null) return '—';
  return `${formatNumber(evidence.area_value)} ${evidence.area_unit ?? ''}`.trim();
};

function ResultState({ analysisId, title, message, error = false }: { analysisId: string; title: string; message: string; error?: boolean }) {
  return (
    <main className="app-shell result-shell">
      <SiteHeader mode="app" backHref="/history" backLabel="Analysis history" />
      <div className="result-heading"><div><span className="section-kicker">Analysis / {analysisId}</span><h1>{title}</h1></div></div>
      <section className={`result-state-panel ${error ? 'is-error' : ''}`}>{error ? <AlertTriangle /> : <LoaderCircle className="spin-slow" />}<strong>{title}</strong><p>{message}</p><Link href="/history">Back to analysis history</Link></section>
    </main>
  );
}

function ClarificationPanel({ clarification, onResume }: { clarification: ClarificationPayload; onResume: (payload: ClarificationResponse) => Promise<void> }) {
  const [roles, setRoles] = useState<Record<string, FileRole>>({});
  const [modalities, setModalities] = useState<Record<string, Modality>>({});
  const [beforeDate, setBeforeDate] = useState('');
  const [afterDate, setAfterDate] = useState('');
  const [question, setQuestion] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setRoles(Object.fromEntries(clarification.upload_ids.map((id, index) => [id, clarification.allowed_roles[index] ?? clarification.allowed_roles[0] ?? 'unknown'])) as Record<string, FileRole>);
    setModalities(Object.fromEntries(clarification.upload_ids.map((id, index) => [id, index === 0 ? 'optical' : 'sar'])) as Record<string, Modality>);
  }, [clarification]);

  const submit = async () => {
    const payload: ClarificationResponse = {};
    if (clarification.missing_fields.includes('file_roles')) payload.file_roles = roles;
    if (clarification.missing_fields.includes('modality')) payload.modalities = clarification.upload_ids.map((id) => modalities[id] ?? 'other');
    if (clarification.missing_fields.includes('before_date') && beforeDate) payload.before_date = beforeDate;
    if (clarification.missing_fields.includes('after_date') && afterDate) payload.after_date = afterDate;
    if (clarification.missing_fields.includes('question_intent') && question.trim()) payload.question = question.trim();
    setSubmitting(true); setError(null);
    try { await onResume(payload); } catch (reason) { setError(reason instanceof Error ? reason.message : 'Could not resume the analysis.'); setSubmitting(false); }
  };

  return (
    <section className="result-clarification-panel">
      <AlertTriangle /><span>Clarification required</span><h2>{clarification.question}</h2>
      {clarification.missing_fields.includes('file_roles') && clarification.upload_ids.map((id) => <label key={id}><span>{id}</span><select value={roles[id] ?? 'unknown'} onChange={(event) => setRoles((current) => ({ ...current, [id]: event.target.value as FileRole }))}>{clarification.allowed_roles.map((role) => <option key={role} value={role}>{readable(role)}</option>)}</select></label>)}
      {clarification.missing_fields.includes('modality') && clarification.upload_ids.map((id) => <label key={`modality-${id}`}><span>{id} modality</span><select value={modalities[id] ?? 'other'} onChange={(event) => setModalities((current) => ({ ...current, [id]: event.target.value as Modality }))}><option value="optical">Optical</option><option value="sar">SAR</option><option value="other">Other</option></select></label>)}
      {clarification.missing_fields.includes('before_date') && <label><span>Before date</span><input type="date" value={beforeDate} onChange={(event) => setBeforeDate(event.target.value)} /></label>}
      {clarification.missing_fields.includes('after_date') && <label><span>After date</span><input type="date" value={afterDate} onChange={(event) => setAfterDate(event.target.value)} /></label>}
      {clarification.missing_fields.includes('question_intent') && <label><span>Clarified question</span><input value={question} onChange={(event) => setQuestion(event.target.value)} /></label>}
      {error && <p>{error}</p>}<Button disabled={submitting} onClick={() => void submit()}>{submitting ? <LoaderCircle className="spin-slow" /> : <Check />}Resume analysis</Button>
    </section>
  );
}

export function ResultPage({ analysisId }: { analysisId: string }) {
  const [showMask, setShowMask] = useState(true);
  const [showRegions, setShowRegions] = useState(true);
  const [traceOpen, setTraceOpen] = useState(false);
  const { status, result, error, refresh } = useAnalysis(analysisId);

  const allWarnings = useMemo(() => result ? [
    ...result.warnings,
    ...(result.evidence?.warnings ?? []),
    ...(result.confidence?.warnings ?? []),
    ...(result.disclaimer ? [{ code: 'DISCLAIMER', level: 'warning' as const, message: result.disclaimer }] : []),
  ] : [], [result]);

  if (error) return <ResultState analysisId={analysisId} title="Backend unavailable." message={error} error />;
  if (status?.status === 'failed') return <ResultState analysisId={analysisId} title="Analysis failed." message={status.error?.message ?? status.message ?? 'The backend stopped this analysis safely.'} error />;
  if (status?.status === 'needs_clarification' && status.clarification) {
    return <main className="app-shell result-shell"><SiteHeader mode="app" backHref="/history" backLabel="Analysis history" /><div className="result-heading"><div><span className="section-kicker">Analysis / {analysisId}</span><h1>Input clarification.</h1></div></div><ClarificationPanel clarification={status.clarification} onResume={async (payload) => { await satqueryApi.submitClarification(analysisId, payload); refresh(); }} /></main>;
  }
  if (!result) return <ResultState analysisId={analysisId} title={readable(status?.stage ?? 'Loading analysis')} message={status?.message ?? 'Reading analysis status from the local backend.'} />;

  const evidence = result.evidence;
  const overlayPath = evidence?.overlay?.url ?? result.artifacts.find((artifact) => artifact.kind === 'overlay' || artifact.kind === 'mask')?.url;
  const overlayUrl = overlayPath ? resolveBackendUrl(overlayPath) : null;
  const percent = evidence?.changed_percentage ?? evidence?.percentage;
  const regions = evidence?.regions ?? [];
  const coordinateLabel = evidence?.georeferenced ? evidence.measurement_crs ?? evidence.overlay?.crs ?? 'GEOREFERENCED EVIDENCE' : 'PIXEL-SPACE EVIDENCE';
  const authoritative = result.pipeline?.authoritative !== false;

  return (
    <main className="app-shell result-shell">
      <SiteHeader mode="app" backHref="/history" backLabel="Analysis history" />
      <div className="result-heading">
        <div><span className="section-kicker">Analysis complete / {analysisId}</span><h1>{readable(result.answer_type ?? result.task)}.</h1></div>
        <div className="result-actions"><span className="complete-badge"><CircleCheck />{authoritative ? 'Completed' : 'Non-authoritative'}</span><Button variant="outline" onClick={() => window.open(satqueryApi.reportUrl(analysisId, 'html', false), '_blank', 'noopener,noreferrer')}>View report</Button><Button onClick={() => { window.location.href = satqueryApi.reportUrl(analysisId, 'json', true); }}><Download />Download JSON</Button></div>
      </div>

      <section className="result-question"><span>Question</span><p>{result.question ?? 'Question unavailable'}</p></section>
      <div className="result-grid">
        <section className="result-map-panel">
          <div className="map-toolbar"><div><Layers3 /><span>Spatial evidence</span></div><div className="layer-toggles"><button className="active"><Check />Base</button><button className={showMask ? 'active' : ''} onClick={() => setShowMask(!showMask)}><span className="toggle-swatch mask" />Mask</button><button className={showRegions ? 'active' : ''} onClick={() => setShowRegions(!showRegions)}><span className="toggle-swatch region" />Regions</button></div></div>
          <SceneMap showMask={false} showRegions={false} overlayUrl={showMask ? overlayUrl : null} coordinateLabel={coordinateLabel} />
          <div className="map-legend result-legend"><span><i className="legend-mask" />{overlayUrl ? 'Backend artifact' : 'No raster artifact'}</span><span><i className="legend-boundary" />{evidence?.georeferenced ? 'Geographic evidence' : 'Pixel-space evidence'}</span></div>
          {result.artifacts.length > 0 && <div className="artifact-list"><span>Evidence artifacts</span>{result.artifacts.map((artifact) => <a key={artifact.artifact_id} href={resolveBackendUrl(artifact.url)} target="_blank" rel="noreferrer"><FileJson /><span><strong>{artifact.name}</strong><small>{readable(artifact.kind)} · {formatNumber(artifact.size_bytes / 1024, 0)} KB</small></span></a>)}</div>}
        </section>

        <aside className="result-evidence-panel">
          <div className="panel-heading"><span>Evidence package</span><strong>{readable(result.task)}</strong></div>
          <div className="result-answer"><Radar /><div><span>Grounded answer</span><p>{result.answer ?? 'The backend did not produce a grounded answer.'}</p></div></div>
          <div className="metric-grid">
            <div><span>Measured area</span><strong>{formatArea(result)}</strong><small>{evidence?.measurement_crs ?? (evidence?.georeferenced ? 'Georeferenced' : 'Pixel units')}</small></div>
            <div><span>Scene percentage</span><strong>{percent == null ? '—' : `${formatNumber(percent)}%`}</strong><small>Valid in-scope evidence</small></div>
            <div><span>Regions</span><strong>{(evidence?.region_count ?? regions.length) || '—'}</strong><small>Connected evidence areas</small></div>
            <div><span>Confidence decision</span><strong>{readable(result.confidence?.decision)}</strong><small>{result.confidence?.limiting_source ?? 'No combined score'}</small></div>
          </div>
          {result.confidence?.specialists.length ? <div className="specialist-confidence"><span>Separate specialist confidence</span>{result.confidence.specialists.map((specialist) => { const score = specialist.value ?? specialist.evidence_coverage; return <div key={specialist.source}><p><strong>{specialist.source}</strong><small>{readable(specialist.kind)} · {specialist.calibrated ? 'calibrated' : 'uncalibrated'}</small></p><b>{score == null ? '—' : `${formatNumber(score * 100)}%`}</b>{score != null && <i><span style={{ width: `${score * 100}%` }} /></i>}</div>; })}</div> : null}
          {showRegions && regions.length > 0 && <div className="region-list">{regions.map((region) => <div key={region.id}><MapPin /><span><strong>Region {region.id}</strong>{region.location ?? readable(region.class_name ?? region.space)}</span><b>{region.area_value == null ? '—' : `${formatNumber(region.area_value)} ${region.area_unit}`}</b></div>)}</div>}
          {allWarnings.map((warning) => <div className="result-warning" key={`${warning.code}-${warning.message}`}><AlertTriangle /><p><strong>{readable(warning.code)}</strong>{warning.message}</p></div>)}
        </aside>
      </div>

      <section className="trace-panel"><button onClick={() => setTraceOpen(!traceOpen)} aria-expanded={traceOpen}><span><Route /><strong>Execution trace</strong><small>{result.execution_trace.length} recorded stages · {result.models.map((model) => model.name).join(', ') || 'No model reported'}</small></span><ChevronDown className={traceOpen ? 'trace-chevron-open' : ''} /></button>{traceOpen && <div className="trace-steps">{result.execution_trace.map((step, index) => <div key={`${index}-${step}`}><span>{String(index + 1).padStart(2, '0')}</span><CircleCheck /><strong>{readable(step)}</strong><small>Recorded by backend</small></div>)}</div>}</section>
      <section className="result-provenance"><span><ShieldCheck />DECISION {result.confidence?.decision ?? 'UNAVAILABLE'}</span><span><FileJson />PIPELINE {result.pipeline?.mode ?? 'UNKNOWN'} / {authoritative ? 'AUTHORITATIVE' : 'NON-AUTHORITATIVE'}</span>{result.models.map((model) => <span key={`${model.name}-${model.version}`}><Radar />MODEL {model.name}{model.version ? ` ${model.version}` : ''}</span>)}</section>
    </main>
  );
}
