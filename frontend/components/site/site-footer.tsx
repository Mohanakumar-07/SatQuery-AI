import Link from 'next/link';
import { ArrowUpRight, Satellite } from 'lucide-react';

export function SiteFooter() {
  return (
    <footer className="site-footer">
      <div className="footer-brand">
        <Satellite aria-hidden="true" />
        <span>SATQUERY AI</span>
      </div>
      <p>Evidence-backed Earth observation through natural language.</p>
      <div className="footer-links">
        <Link href="/workspace">Workspace <ArrowUpRight aria-hidden="true" /></Link>
        <Link href="/history">Analysis history</Link>
      </div>
      <div className="footer-meta">
        <span className="footer-coordinate">20.5937° N / 78.9629° E</span>
        <a href="https://commons.wikimedia.org/wiki/File:Sentinel-3_spacecraft_model.svg" target="_blank" rel="noreferrer">
          Sentinel-3 vector: SkywalkerPL / CC BY 4.0
        </a>
      </div>
    </footer>
  );
}
