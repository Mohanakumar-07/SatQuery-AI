import type { ReactNode } from 'react';
import Link from 'next/link';
import {
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  CheckCircle2,
  Eye,
  FileScan,
  Layers3,
  MessageSquareText,
  Orbit,
  Radar,
  ScanSearch,
  ShieldCheck,
} from 'lucide-react';

import { ScrollMotion } from '@/components/motion/scroll-motion';
import { CinematicNav } from '@/components/site/cinematic-nav';
import { SiteFooter } from '@/components/site/site-footer';
import { SceneMap } from '@/components/visuals/scene-map';
import { StarField } from '@/components/visuals/star-field';
import { LiveCapabilityStatus } from '@/features/workflow/live-capability-status';

type WorkflowKey = 'capabilities' | 'process' | 'evidence';

const missionModes = [
  {
    code: 'OBS-01', label: 'Observe', detail: 'Single scene', icon: Eye,
    question: '“What features are visible in this scene?”',
    output: 'Scene description + grounded answer',
  },
  {
    code: 'CMP-02', label: 'Compare', detail: 'Two dates', icon: ScanSearch,
    question: '“Where has the built-up area changed?”',
    output: 'Change mask + measured regions',
  },
  {
    code: 'FUS-03', label: 'Fuse', detail: 'Optical + SAR', icon: Layers3,
    question: '“Map water and vegetation in this area.”',
    output: 'Class polygons + confidence',
  },
];

const processSteps = [
  { icon: FileScan, number: '01', title: 'Upload imagery', text: 'Add one scene, a temporal pair, or aligned optical and SAR inputs.' },
  { icon: MessageSquareText, number: '02', title: 'Ask a question', text: 'Describe what you need to understand in plain, operational language.' },
  { icon: Radar, number: '03', title: 'Interpret the mission', text: 'SatQuery validates metadata and identifies the permitted analysis route.' },
  { icon: Orbit, number: '04', title: 'Run the specialist', text: 'The relevant workflow processes the scene while preserving coordinates.' },
  { icon: ShieldCheck, number: '05', title: 'Inspect the evidence', text: 'Review the answer, masks, measurements, warnings, and execution trace.' },
];

const stages = [
  { key: 'capabilities', number: '01', href: '/capabilities', label: 'Choose mission' },
  { key: 'process', number: '02', href: '/process', label: 'Run analysis' },
  { key: 'evidence', number: '03', href: '/evidence', label: 'Verify evidence' },
] as const;

type WorkflowFrameProps = {
  active: WorkflowKey;
  index: string;
  eyebrow: string;
  title: ReactNode;
  summary: string;
  previous?: { href: string; label: string };
  next?: { href: string; label: string };
  children: ReactNode;
};

function WorkflowFrame({ active, index, eyebrow, title, summary, previous, next, children }: WorkflowFrameProps) {
  return (
    <main className="site-shell workflow-shell">
      <StarField dense />
      <ScrollMotion />
      <CinematicNav active={active} />

      <section className="workflow-page-hero" aria-labelledby={`${active}-title`}>
        <span className="workflow-page-index">{index}</span>
        <div data-reveal="text">
          <span className="section-kicker">{eyebrow}</span>
          <h1 id={`${active}-title`}>{title}</h1>
        </div>
        <p data-reveal="fade">{summary}</p>
      </section>

      <nav className="workflow-stage-rail" aria-label="Analysis workflow" data-reveal="fade">
        {stages.map((stage) => (
          <Link key={stage.key} href={stage.href} className={active === stage.key ? 'is-active' : undefined} aria-current={active === stage.key ? 'step' : undefined}>
            <span>{stage.number}</span><strong>{stage.label}</strong>
          </Link>
        ))}
      </nav>

      {children}

      <nav className="workflow-pager" aria-label="Workflow page navigation" data-reveal="fade">
        {previous ? <Link href={previous.href}><ArrowLeft aria-hidden="true" /><span><small>Previous stage</small>{previous.label}</span></Link> : <span />}
        {next ? <Link href={next.href}><span><small>Next stage</small>{next.label}</span><ArrowRight aria-hidden="true" /></Link> : <Link href="/workspace"><span><small>Workflow complete</small>Open workspace</span><ArrowRight aria-hidden="true" /></Link>}
      </nav>

      <SiteFooter />
    </main>
  );
}

export function CapabilitiesPage() {
  return (
    <WorkflowFrame
      active="capabilities"
      index="01 / 03"
      eyebrow="Mission selection"
      title={<>Choose how to<br />read the scene.</>}
      summary="The imagery and the question define the route. SatQuery selects the correct observation mode without exposing model plumbing."
      next={{ href: '/process', label: 'Run analysis' }}
    >
      <LiveCapabilityStatus />
      <section className="workflow-page-section workflow-capabilities" aria-label="Analysis capabilities">
        <div className="workflow-section-note" data-reveal="text">
          <span>INPUT → ROUTE</span>
          <p>One interface, three sensor-aware mission paths.</p>
        </div>
        <div className="capability-list">
          {missionModes.map((mode, index) => {
            const Icon = mode.icon;
            return (
              <article className="capability-row" key={mode.code} data-reveal="row" style={{ transitionDelay: `${index * 90}ms` }}>
                <span className="capability-code">{mode.code}</span>
                <div className="capability-icon"><Icon aria-hidden="true" /></div>
                <div className="capability-name"><span>{mode.detail}</span><h3>{mode.label}</h3></div>
                <p>{mode.question}</p>
                <div className="capability-output"><ArrowDown aria-hidden="true" /><span>{mode.output}</span></div>
              </article>
            );
          })}
        </div>
      </section>
    </WorkflowFrame>
  );
}

export function ProcessPage() {
  return (
    <WorkflowFrame
      active="process"
      index="02 / 03"
      eyebrow="Ground track"
      title={<>From raw scene<br />to verified answer.</>}
      summary="A five-stage execution path validates the input, selects a specialist, and preserves a trace of every decision."
      previous={{ href: '/capabilities', label: 'Choose mission' }}
      next={{ href: '/evidence', label: 'Verify evidence' }}
    >
      <section className="workflow-page-section workflow-process" aria-label="Analysis process">
        <aside className="workflow-section-note" data-reveal="text">
          <span>SEQ / 05 STAGES</span>
          <p>Each stage leaves a visible trace that can be inspected and challenged.</p>
        </aside>
        <div className="process-stack">
          {processSteps.map((step, index) => {
            const Icon = step.icon;
            return (
              <article className="process-step" key={step.number} data-reveal="row" style={{ transitionDelay: `${index * 70}ms` }}>
                <span className="step-number">{step.number}</span>
                <span className="step-node"><Icon aria-hidden="true" /></span>
                <div><h3>{step.title}</h3><p>{step.text}</p></div>
                <span className="step-status">TRACE ENABLED</span>
              </article>
            );
          })}
        </div>
      </section>
    </WorkflowFrame>
  );
}

export function EvidencePage() {
  return (
    <WorkflowFrame
      active="evidence"
      index="03 / 03"
      eyebrow="Evidence inspection"
      title={<>Verify before<br />you conclude.</>}
      summary="Every answer is paired with the spatial facts used to compose it. Weak evidence produces a warning or abstention."
      previous={{ href: '/process', label: 'Run analysis' }}
    >
      <section className="workflow-page-section workflow-evidence" aria-labelledby="evidence-detail-title">
        <div className="evidence-visual" data-reveal="fade">
          <div className="visual-topline"><span>ANALYSIS / SQ-260901</span><span>CHANGE DETECTION</span></div>
          <SceneMap />
          <div className="map-legend">
            <span><i className="legend-mask" />Detected change</span>
            <span><i className="legend-boundary" />Region boundary</span>
          </div>
        </div>
        <div className="evidence-copy" data-reveal="text">
          <span className="section-kicker">Evidence before language</span>
          <h2 id="evidence-detail-title">The answer is only the beginning.</h2>
          <p>Inspect the geometry, measurements, confidence, and execution trace behind the generated response.</p>
          <div className="evidence-readout">
            <div><span>Changed area</span><strong>12.48 km²</strong></div>
            <div><span>Regions detected</span><strong>02</strong></div>
            <div><span>Specialist confidence</span><strong>0.91</strong></div>
          </div>
          <ul className="trust-list">
            <li><CheckCircle2 aria-hidden="true" />Geographic masks and polygons</li>
            <li><CheckCircle2 aria-hidden="true" />Separate specialist confidence</li>
            <li><CheckCircle2 aria-hidden="true" />Warnings and abstention policy</li>
            <li><CheckCircle2 aria-hidden="true" />Reproducible execution trace</li>
          </ul>
        </div>
      </section>
    </WorkflowFrame>
  );
}
