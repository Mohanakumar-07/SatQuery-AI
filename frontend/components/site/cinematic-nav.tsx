import Link from 'next/link';
import { ArrowUpRight } from 'lucide-react';

import { SatIcon } from '@/components/site/sat-icon';

type NavigationKey = 'home' | 'capabilities' | 'process' | 'evidence';

const navigation = [
  { key: 'home', href: '/#home', label: 'Home' },
  { key: 'capabilities', href: '/#capabilities', label: 'Capabilities' },
  { key: 'process', href: '/#process', label: 'Process' },
  { key: 'evidence', href: '/#evidence', label: 'Evidence' },
] as const;

export function CinematicNav({ active }: { active?: NavigationKey }) {
  return (
    <header className="cinematic-header">
      <a href="/" className="cinematic-wordmark" aria-label="SatQuery AI home">
        <span className="cinematic-mark"><SatIcon /></span>
        <strong>SATQUERY</strong><span>/ AI</span>
      </a>
      <nav aria-label="Primary navigation">
        {navigation.map((item) => (
          <a
            key={item.key}
            href={item.href}
            className={active === item.key ? 'is-active' : undefined}
            aria-current={active === item.key ? 'page' : undefined}
          >
            {item.label}
          </a>
        ))}
      </nav>
      <div className="cinematic-nav-actions">
        <a href="/workspace" className="cinematic-launch">
          <span>Open workspace</span><ArrowUpRight aria-hidden="true" />
        </a>
      </div>
    </header>
  );
}
