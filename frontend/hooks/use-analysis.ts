'use client';

import { useCallback, useEffect, useState } from 'react';

import {
  type AnalysisResult,
  type AnalysisStatusResponse,
  SatQueryApiError,
  satqueryApi,
} from '@/lib/satquery-api';

type AnalysisPollingState = {
  status: AnalysisStatusResponse | null;
  result: AnalysisResult | null;
  error: string | null;
  loading: boolean;
};

export function useAnalysis(analysisId: string, intervalMs = 1400) {
  const [refreshVersion, setRefreshVersion] = useState(0);
  const [state, setState] = useState<AnalysisPollingState>({
    status: null,
    result: null,
    error: null,
    loading: Boolean(analysisId),
  });
  const refresh = useCallback(() => setRefreshVersion((version) => version + 1), []);

  useEffect(() => {
    if (!analysisId) {
      setState({ status: null, result: null, error: null, loading: false });
      return;
    }
    let active = true;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const poll = async () => {
      try {
        const status = await satqueryApi.analysisStatus(analysisId);
        if (!active) return;
        setState((current) => ({ ...current, status, error: null, loading: false }));
        if (status.status === 'completed') {
          const result = await satqueryApi.analysisResult(analysisId);
          if (active) setState({ status, result, error: null, loading: false });
          return;
        }
        if (status.status === 'failed' || status.status === 'needs_clarification') return;
        timer = setTimeout(poll, intervalMs);
      } catch (error) {
        if (!active) return;
        const message = error instanceof SatQueryApiError ? error.message : 'Could not reach the local SatQuery backend.';
        setState((current) => ({ ...current, error: message, loading: false }));
      }
    };

    setState((current) => ({ ...current, error: null, loading: true }));
    void poll();
    return () => {
      active = false;
      if (timer) clearTimeout(timer);
    };
  }, [analysisId, intervalMs, refreshVersion]);

  return { ...state, refresh };
}

