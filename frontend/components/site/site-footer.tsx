import { SatIcon } from '@/components/site/sat-icon';

export function SiteFooter() {
  return (
    <footer className="site-footer">
      <div className="footer-brand">
        <SatIcon aria-hidden="true" />
        <span>SATQUERY AI</span>
      </div>
      <p className="footer-tagline">Evidence-backed Earth observation through natural language.</p>
      <span className="nav-link-status"><i /> System online</span>
    </footer>
  );
}
