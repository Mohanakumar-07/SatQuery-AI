import Link from 'next/link';
import {
  ArrowDown,
  ArrowUpRight,
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

import { CinematicNav } from '@/components/site/cinematic-nav';
import { SatIcon } from '@/components/site/sat-icon';
import { SiteFooter } from '@/components/site/site-footer';
import { RotatingEarth } from '@/components/visuals/rotating-earth';
import { SceneMap } from '@/components/visuals/scene-map';
import { StarField } from '@/components/visuals/star-field';
import { buttonVariants } from '@/components/ui/button';
import { LiveCapabilityStatus } from '@/features/workflow/live-capability-status';
import { cn } from '@/lib/utils';

const missionModes = [
  {
    code: 'OBS-01',
    label: 'Observe',
    detail: 'Single scene',
    icon: Eye,
    question: '“What features are visible in this scene?”',
    output: 'Scene description + grounded answer',
  },
  {
    code: 'CMP-02',
    label: 'Compare',
    detail: 'Two dates',
    icon: ScanSearch,
    question: '“Where has the built-up area changed?”',
    output: 'Change mask + measured regions',
  },
  {
    code: 'FUS-03',
    label: 'Fuse',
    detail: 'Optical + SAR',
    icon: Layers3,
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

export function LandingPage() {
  return (
    <main className="site-shell landing-shell">
      <StarField dense />
      <CinematicNav />

      {/* Hero Section */}
      <section id="home" className="cinematic-hero" aria-labelledby="hero-title">
        <div className="hero-crosshair hero-crosshair-left" aria-hidden="true"><i /><i /></div>
        <div className="hero-crosshair hero-crosshair-right" aria-hidden="true"><i /><i /></div>
        <div className="cinematic-copy">
          <h1 id="hero-title"><span>Ask the Earth.</span><span>Trace the evidence.</span></h1>
          <div className="cinematic-rule" aria-hidden="true"><i /><span /></div>
          <p>Evidence-backed satellite image analysis.</p>
          <div className="cinematic-actions">
            <Link href="/workspace" className={cn(buttonVariants({ size: 'lg' }), 'cinematic-primary')}>
              Start an analysis <ArrowUpRight aria-hidden="true" />
            </Link>
            <a href="#capabilities" className="cinematic-secondary">
              Explore capabilities <ArrowDown aria-hidden="true" />
            </a>
          </div>
        </div>

        <div className="space-stage" aria-hidden="true">
          <div className="hero-earth"><RotatingEarth /></div>
          <svg className="hero-orbit-trace" viewBox="0 0 1100 560" fill="none">
            <path d="M12 522C278 458 472 405 650 326C780 268 869 220 1086 172" />
            <circle cx="347" cy="445" r="4" /><circle cx="623" cy="339" r="5" /><circle cx="891" cy="214" r="4" />
          </svg>
          <div className="satellite-art-wrap">
            <img src="/sentinel-3-model.svg" alt="" className="hero-satellite-art" />
            <span>EO / SENTINEL-3</span>
          </div>
          <div className="hero-telemetry hero-telemetry-top"><span>ORB 814.5 KM</span><span>PASS 06:42:18</span></div>
          <div className="hero-telemetry hero-telemetry-side"><span>AZ 127.04°</span><span>VEL 7.46 KM/S</span></div>
        </div>
        <div className="hero-scale" aria-hidden="true">|&nbsp;&nbsp;|&nbsp;&nbsp;|&nbsp;&nbsp;|&nbsp;&nbsp;|&nbsp;&nbsp;|&nbsp;&nbsp;|</div>
      </section>

      {/* Live Capability Status Bar */}
      <div style={{ width: 'min(100% - 48px, 1440px)', margin: '0 auto 40px' }}>
        <LiveCapabilityStatus />
      </div>

      {/* Capabilities Section */}
      <section id="capabilities" className="landing-section capabilities-section" aria-labelledby="capabilities-title">
        <div className="chapter-intro">
          <span className="section-kicker">Mission selection</span>
          <h2 id="capabilities-title">Choose how to<br />read the scene.</h2>
          <p>The imagery and the question define the route. SatQuery selects the correct observation mode without exposing model plumbing.</p>
        </div>

        <div className="capability-list">
          {missionModes.map((mode) => {
            const Icon = mode.icon;
            return (
              <article className="capability-row" key={mode.code}>
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

      {/* Process Section */}
      <section id="process" className="landing-section process-section" aria-labelledby="process-title">
        <div className="process-layout">
          <div className="chapter-intro">
            <span className="section-kicker">Ground track / 05 Stages</span>
            <h2 id="process-title">From raw scene<br />to verified answer.</h2>
            <p>A five-stage execution path validates the input, selects a specialist, and preserves a trace of every decision.</p>
          </div>

          <div className="process-stack">
            {processSteps.map((step) => {
              const Icon = step.icon;
              return (
                <article className="process-step" key={step.number}>
                  <span className="step-number">{step.number}</span>
                  <div className="step-node"><Icon aria-hidden="true" /></div>
                  <div>
                    <h3>{step.title}</h3>
                    <p>{step.text}</p>
                  </div>
                  <span className="step-status">TRACE ENABLED</span>
                </article>
              );
            })}
          </div>
        </div>
      </section>

      {/* Evidence Section */}
      <section id="evidence" className="landing-section evidence-section" aria-labelledby="evidence-title">
        <div className="evidence-visual">
          <div className="visual-topline"><span>ANALYSIS / SQ-260901</span><span>CHANGE DETECTION</span></div>
          <SceneMap />
          <div className="map-legend">
            <span><i className="legend-mask" />Detected change</span>
            <span><i className="legend-boundary" />Region boundary</span>
          </div>
        </div>
        <div className="evidence-copy">
          <span className="section-kicker">Evidence before language</span>
          <h2 id="evidence-title">The answer is only the beginning.</h2>
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

      {/* Trust Band */}
      <div className="trust-band" aria-label="System guarantees">
        <span><ShieldCheck aria-hidden="true" />Evidence-backed claims only</span>
        <span><Orbit aria-hidden="true" />Sensor-aware model routing</span>
        <span><CheckCircle2 aria-hidden="true" />Reproducible execution audit</span>
      </div>

      {/* Clean Footer */}
      <SiteFooter />
    </main>
  );
}
