import Link from 'next/link';
import { ArrowLeft, ArrowUpRight, History } from 'lucide-react';

import { SatIcon } from '@/components/site/sat-icon';
import { buttonVariants } from '@/components/ui/button';
import { cn } from '@/lib/utils';

type SiteHeaderProps = {
  mode?: 'landing' | 'app';
  backHref?: string;
  backLabel?: string;
};

export function SiteHeader({ mode = 'landing', backHref, backLabel }: SiteHeaderProps) {
  return (
    <header className={cn('site-header', mode === 'app' && 'app-site-header')}>
      <Link href="/" className="wordmark" aria-label="SatQuery AI home">
        <span className="wordmark-icon"><SatIcon aria-hidden="true" /></span>
        <span className="wordmark-title">SATQUERY</span>
        <span className="wordmark-ai">AI</span>
      </Link>

      {mode === 'landing' ? (
        <nav className="desktop-nav" aria-label="Main navigation">
          <a href="#missions">Capabilities</a>
          <a href="#process">Process</a>
          <a href="#evidence">Evidence</a>
        </nav>
      ) : backHref ? (
        <Link href={backHref} className="header-backlink">
          <ArrowLeft aria-hidden="true" /> {backLabel ?? 'Back'}
        </Link>
      ) : (
        <nav className="desktop-nav" aria-label="Workspace navigation">
          <Link href="/workspace">New analysis</Link>
          <Link href="/history">History</Link>
        </nav>
      )}

      {mode === 'landing' ? (
        <Link href="/workspace" className={cn(buttonVariants(), 'header-cta')}>
          Open workspace <ArrowUpRight aria-hidden="true" />
        </Link>
      ) : null}
    </header>
  );
}
