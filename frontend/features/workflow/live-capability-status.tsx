'use client';

import { useCallback, useEffect, useState } from 'react';
import { AlertTriangle, CheckCircle2, LoaderCircle, RefreshCw } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { type HealthResponse, type RegistryResponse, SatQueryApiError, satqueryApi } from '@/lib/satquery-api';

export function LiveCapabilityStatus() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [registry, setRegistry] = useState<RegistryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true); setError(null);
    try {
      const [nextHealth, nextRegistry] = await Promise.all([satqueryApi.health(), satqueryApi.models()]);
      setHealth(nextHealth); setRegistry(nextRegistry);
    } catch (reason) {
      setError(reason instanceof SatQueryApiError ? reason.message : 'The local backend is not reachable.');
    } finally { setLoading(false); }
  }, []);

  useEffect(() => { void load(); }, [load]);

  return (
    <section className="live-capability-status" aria-live="polite">
      <div><span>Local control plane</span><h2>Backend capability status</h2><p>This readout comes from the live health service and permitted model registry.</p></div>
      {loading ? <div className="capability-health-state"><LoaderCircle className="spin-slow" />Connecting to localhost</div> : error ? <div className="capability-health-state is-error"><AlertTriangle />{error}<Button variant="outline" onClick={() => void load()}><RefreshCw />Retry</Button></div> : <div className="capability-health-grid">
        <div><span>API</span><strong><CheckCircle2 />{health?.status ?? 'unknown'}</strong><small>v{health?.version ?? '—'} / {health?.environment ?? '—'}</small></div>
        <div><span>Pipeline</span><strong>{registry?.pipeline_mode ?? 'unknown'}</strong><small>{registry?.inference_ready ? 'Inference ready' : 'Inference is not attached'}</small></div>
        <div><span>Permitted specialists</span><strong>{registry?.permitted_models.length ?? 0}</strong><small>{registry?.permitted_models.join(', ') || 'None registered'}</small></div>
      </div>}
    </section>
  );
}

