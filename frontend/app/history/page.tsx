import type { Metadata } from 'next';

import { HistoryPage } from '@/features/history/history-page';

export const metadata: Metadata = {
  title: 'Analysis history',
  description: 'Review previous SatQuery AI analysis missions.',
};

export default function History() {
  return <HistoryPage />;
}
