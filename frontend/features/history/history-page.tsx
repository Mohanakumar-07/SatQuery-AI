'use client';

import Link from 'next/link';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { AlertTriangle, ArrowUpRight, Clock3, FileSearch, LoaderCircle, Plus, Search } from 'lucide-react';

import { SiteHeader } from '@/components/site/site-header';
import { Button, buttonVariants } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { type AnalysisSummary, SatQueryApiError, satqueryApi } from '@/lib/satquery-api';
import { cn } from '@/lib/utils';

const readable = (value?: string | null) => value ? value.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase()) : 'Pending';
const createdAt = (value?: string | null) => value ? new Intl.DateTimeFormat('en-IN', { dateStyle: 'medium', timeStyle: 'short' }).format(new Date(value)) : '—';

export function HistoryPage() {
  const [records, setRecords] = useState<AnalysisSummary[]>([]);
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadHistory = useCallback(async () => {
    setLoading(true); setError(null);
    try {
      const response = await satqueryApi.listAnalyses();
      setRecords(response.items);
    } catch (reason) {
      setError(reason instanceof SatQueryApiError ? reason.message : 'Could not reach the local SatQuery backend.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void loadHistory(); }, [loadHistory]);

  const filtered = useMemo(() => {
    const term = search.trim().toLowerCase();
    if (!term) return records;
    return records.filter((record) => `${record.analysis_id} ${record.question} ${record.input_type} ${record.task} ${record.status}`.toLowerCase().includes(term));
  }, [records, search]);

  return (
    <main className="app-shell history-shell">
      <SiteHeader mode="app" />
      <div className="app-page-heading history-heading">
        <div><span className="section-kicker">Mission archive</span><h1>Analysis history.</h1></div>
        <Link href="/workspace" className={cn(buttonVariants(), 'new-analysis-link')}><Plus />New analysis</Link>
      </div>
      <section className="history-panel">
        <div className="history-toolbar">
          <div><Clock3 /><span>{records.length.toString().padStart(2, '0')} backend mission{records.length === 1 ? '' : 's'}</span></div>
          <label className="history-search"><Search aria-hidden="true" /><Input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search missions" /></label>
        </div>
        {loading ? (
          <div className="history-empty"><LoaderCircle className="spin-slow" /><strong>Loading analyses</strong><span>Reading the local backend archive.</span></div>
        ) : error ? (
          <div className="history-empty"><AlertTriangle /><strong>Backend unavailable</strong><span>{error}</span><Button variant="outline" onClick={() => void loadHistory()}>Retry</Button></div>
        ) : filtered.length ? (
          <Table className="history-table">
            <TableHeader><TableRow><TableHead>Analysis ID</TableHead><TableHead>Question</TableHead><TableHead>Input</TableHead><TableHead>Workflow</TableHead><TableHead>Status</TableHead><TableHead>Created</TableHead><TableHead><span className="sr-only">Open</span></TableHead></TableRow></TableHeader>
            <TableBody>{filtered.map((record) => (
              <TableRow key={record.analysis_id}>
                <TableCell className="history-id">{record.analysis_id}</TableCell><TableCell className="history-question">{record.question}</TableCell>
                <TableCell>{readable(record.input_type)}</TableCell><TableCell>{readable(record.task)}</TableCell>
                <TableCell><span className={`status-badge status-${record.status}`}>{readable(record.status)}</span></TableCell><TableCell>{createdAt(record.created_at)}</TableCell>
                <TableCell><Link className="row-link" href={`/analysis/${record.analysis_id}`} aria-label={`Open ${record.analysis_id}`}><ArrowUpRight /></Link></TableCell>
              </TableRow>
            ))}</TableBody>
          </Table>
        ) : (
          <div className="history-empty"><FileSearch /><strong>{records.length ? 'No matching analyses' : 'No analyses yet'}</strong><span>{records.length ? 'Try a different analysis ID or question.' : 'Completed and in-progress backend analyses will appear here.'}</span></div>
        )}
      </section>
    </main>
  );
}

