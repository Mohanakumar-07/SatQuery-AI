import Link from 'next/link';
import { ArrowUpRight } from 'lucide-react';

type NavigationKey = 'capabilities' | 'process' | 'evidence' | 'history';

const navigation = [
  { key: 'capabilities', href: '/capabilities', label: 'Capabilities' },
  { key: 'process', href: '/process', label: 'Process' },
  { key: 'evidence', href: '/evidence', label: 'Evidence' },
  { key: 'history', href: '/history', label: 'History' },
] as const;

export function CinematicNav({ active }: { active?: NavigationKey }) {
  return (
    <header className="cinematic-header">
      <Link href="/" className="cinematic-wordmark" aria-label="SatQuery AI home">
        <strong>SATQUERY</strong><span>/ AI</span>
      </Link>
      <nav aria-label="Primary navigation">
        {navigation.map((item) => (
          <Link
            key={item.key}
            href={item.href}
            className={active === item.key ? 'is-active' : undefined}
            aria-current={active === item.key ? 'page' : undefined}
          >
            {item.label}
          </Link>
        ))}
      </nav>
      <div className="cinematic-nav-actions">
        <span className="nav-link-status"><i /> System online</span>
        <Link href="/workspace" className="cinematic-launch">
          <span>Open workspace</span><ArrowUpRight aria-hidden="true" />
        </Link>
      </div>
    </header>
  );
}
