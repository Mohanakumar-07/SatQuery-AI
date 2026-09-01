import Link from 'next/link';
import { ArrowUpRight } from 'lucide-react';

import { CinematicNav } from '@/components/site/cinematic-nav';
import { SiteFooter } from '@/components/site/site-footer';
import { RotatingEarth } from '@/components/visuals/rotating-earth';
import { StarField } from '@/components/visuals/star-field';
import { buttonVariants } from '@/components/ui/button';
import { cn } from '@/lib/utils';

export function LandingPage() {
  return (
    <main className="site-shell landing-shell">
      <StarField dense />
      <CinematicNav />

      <section className="cinematic-hero" aria-labelledby="hero-title">
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
            <Link href="/capabilities" className="cinematic-secondary">
              Explore the workflow <ArrowUpRight aria-hidden="true" />
            </Link>
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
        <div className="hero-scale" aria-hidden="true">|&nbsp;&nbsp;|&nbsp;&nbsp;|&nbsp;&nbsp;|&nbsp;&nbsp;|&nbsp;&nbsp;|&nbsp;&nbsp;|&nbsp;&nbsp;|</div>
      </section>

      <SiteFooter />
    </main>
  );
}
