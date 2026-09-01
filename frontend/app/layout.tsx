import type { Metadata } from 'next';
import { Barlow_Condensed, IBM_Plex_Mono, IBM_Plex_Sans } from 'next/font/google';

import { LenisProvider } from '@/components/providers/lenis-provider';

import './globals.css';

const display = Barlow_Condensed({
  variable: '--font-display',
  subsets: ['latin'],
  weight: ['500', '600', '700'],
});

const body = IBM_Plex_Sans({
  variable: '--font-body',
  subsets: ['latin'],
  weight: ['400', '500', '600'],
});

const mono = IBM_Plex_Mono({
  variable: '--font-mono',
  subsets: ['latin'],
  weight: ['400', '500'],
});

export const metadata: Metadata = {
  metadataBase: new URL(process.env.NEXT_PUBLIC_SITE_URL ?? 'http://localhost:3001'),
  title: {
    default: 'SatQuery AI — Ask the Earth',
    template: '%s — SatQuery AI',
  },
  description:
    'Evidence-backed satellite image analysis through natural-language questions.',
  openGraph: {
    title: 'SatQuery AI — Ask the Earth. Trace the evidence.',
    description: 'Evidence-backed satellite image analysis.',
    type: 'website',
    images: [{ url: '/og.png', width: 1731, height: 909, alt: 'SatQuery AI orbital mission interface' }],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'SatQuery AI — Ask the Earth. Trace the evidence.',
    description: 'Evidence-backed satellite image analysis.',
    images: ['/og.png'],
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="dark">
      <body className={`${display.variable} ${body.variable} ${mono.variable}`}>
        <LenisProvider>{children}</LenisProvider>
      </body>
    </html>
  );
}
