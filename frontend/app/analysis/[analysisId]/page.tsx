import type { Metadata } from 'next';

import { ResultPage } from '@/features/results/result-page';

type ResultRouteProps = { params: Promise<{ analysisId: string }> };

export async function generateMetadata({ params }: ResultRouteProps): Promise<Metadata> {
  const { analysisId } = await params;
  const title = `Analysis ${analysisId}`;
  const description = 'Evidence, confidence, measurements, and execution trace for this satellite analysis.';

  return {
    title,
    description,
    openGraph: { title, description, images: [] },
    twitter: { title, description, images: [] },
  };
}

export default async function AnalysisResult({ params }: ResultRouteProps) {
  const { analysisId } = await params;
  return <ResultPage analysisId={analysisId} />;
}
