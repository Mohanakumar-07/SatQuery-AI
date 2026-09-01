import type { Metadata } from 'next';

import { WorkspacePage } from '@/features/workspace/workspace-page';

export const metadata: Metadata = {
  title: 'Analysis workspace',
  description: 'Upload satellite imagery and begin an evidence-backed analysis.',
};

export default function Workspace() {
  return <WorkspacePage />;
}
