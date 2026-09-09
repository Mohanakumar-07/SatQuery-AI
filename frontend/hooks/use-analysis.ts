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
  const refresh = useCallback(() => setRefreshVersion((v) => v + 1), []);

  useEffect(() => {
    if (!analysisId) {
      setState({ status: null, result: null, error: null, loading: false });
      return;
    }
    let active = true;
    let timer: ReturnType<typeof setTimeout> | undefined;

    async function poll() {
      try {
        const polledStatus = await satqueryApi.analysisStatus(analysisId);
        if (!active) return;
        setState((cur) => ({ ...cur, status: polledStatus, error: null, loading: false }));
        if (polledStatus.status === 'completed') {
          const polledResult = await satqueryApi.analysisResult(analysisId);
          if (active) setState({ status: polledStatus, result: polledResult, error: null, loading: false });
          return;
        }
        if (polledStatus.status === 'failed' || polledStatus.status === 'needs_clarification') return;
        timer = setTimeout(poll, intervalMs);
      } catch (pollErr) {
        if (!active) return;
        const pollErrMsg = pollErr instanceof SatQueryApiError
          ? pollErr.message
          : 'Could not reach the local SatQuery backend.';
        setState((cur) => ({ ...cur, error: pollErrMsg, loading: false }));
      }
    }

    setState((cur) => ({ ...cur, error: null, loading: true }));
    void poll();
    return () => {
      active = false;
      if (timer) clearTimeout(timer);
    };
  }, [analysisId, intervalMs, refreshVersion]);

  return { ...state, refresh };
}

