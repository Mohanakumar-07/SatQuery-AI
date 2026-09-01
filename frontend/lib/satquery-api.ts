export const SATQUERY_API_URL = (
  process.env.NEXT_PUBLIC_SATQUERY_API_URL ?? 'http://localhost:8000/api/v1'
).replace(/\/$/, '');

export type AnalysisStatus = 'queued' | 'running' | 'needs_clarification' | 'completed' | 'failed';
export type AnalysisStage =
  | 'queued'
  | 'validating'
  | 'interpreting'
  | 'routing'
  | 'preparing_scene'
  | 'inference'
  | 'evidence'
  | 'calibration'
  | 'composition'
  | 'reporting'
  | 'done'
  | 'needs_clarification'
  | 'failed';

export type Warning = {
  code: string;
  level: 'info' | 'warning' | 'error';
  message: string;
  detail?: Record<string, unknown> | null;
};

export type ApiErrorPayload = {
  code: string;
  message: string;
  status?: number;
  detail?: Record<string, unknown> | null;
  request_id?: string;
};

export class SatQueryApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly detail?: Record<string, unknown> | null;
  readonly requestId?: string;

  constructor(status: number, payload: ApiErrorPayload) {
    super(payload.message);
    this.name = 'SatQueryApiError';
    this.status = status;
    this.code = payload.code;
    this.detail = payload.detail;
    this.requestId = payload.request_id;
  }
}

export type HealthResponse = {
  status: 'ok' | 'degraded' | 'error' | 'unknown';
  version: string;
  environment: string;
  time: string;
  components: Record<string, { name: string; status: string; message?: string | null }>;
  limits: Record<string, unknown>;
  warnings: Warning[];
};

export type ModelInfo = {
  internal_name: string;
  model_name: string;
  version?: string | null;
  role: string;
  status: 'available' | 'unavailable' | 'not_implemented' | 'loading' | 'error';
  tasks: string[];
  notes?: string | null;
  warnings: Warning[];
};

export type RegistryResponse = {
  registry_version: string;
  models: ModelInfo[];
  permitted_models: string[];
  unavailable_models: string[];
  inference_ready: boolean;
  warnings: Warning[];
  pipeline_mode?: string | null;
};

export type UploadRead = {
  upload_id: string;
  filename: string;
  stored_name: string;
  size_bytes: number;
  sha256: string;
  extension: string;
  declared_media_type?: string | null;
  detected_media_type: string;
  media_kind: string;
  url: string;
  probe_status: string;
  georeferenced?: boolean | null;
  crs?: string | null;
  width?: number | null;
  height?: number | null;
  band_count?: number | null;
  acquisition_date?: string | null;
  sensor?: string | null;
  modality?: string | null;
  errors: Warning[];
  warnings: Warning[];
  created_at?: string | null;
};

export type UploadResponse = {
  uploads: UploadRead[];
  duplicate_upload_ids: string[];
  warnings: Warning[];
};

export type ValidationResponse = {
  valid: boolean;
  detected_input_type?: string | null;
  detected_modalities: string[];
  crs?: string | null;
  aligned?: boolean | null;
  overlap_percentage?: number | null;
  routing_candidates: string[];
  warnings: Warning[];
  errors: Warning[];
  georeferenced: boolean;
  geographic_fields_allowed: boolean;
  missing_upload_ids: string[];
};

export type ClarificationPayload = {
  analysis_id: string;
  status: 'needs_clarification';
  missing_fields: Array<'file_roles' | 'before_date' | 'after_date' | 'modality' | 'question_intent'>;
  question: string;
  allowed_roles: Array<'before' | 'after' | 'optical' | 'sar' | 'single' | 'unknown'>;
  upload_ids: string[];
  resume_with: Record<string, unknown>;
};

export type AnalysisCreated = {
  analysis_id: string;
  status: AnalysisStatus;
  stage: AnalysisStage;
  progress: number;
  message: string;
  pipeline_mode?: string | null;
  links: Record<string, string>;
};

export type AnalysisStatusResponse = {
  analysis_id: string;
  status: AnalysisStatus;
  stage?: AnalysisStage | null;
  progress: number;
  message?: string | null;
  task?: string | null;
  pipeline_mode?: string | null;
  error?: { code: string; message: string; detail?: Record<string, unknown> | null } | null;
  clarification?: ClarificationPayload | null;
  recent_events: Array<Record<string, unknown>>;
  created_at?: string | null;
  updated_at?: string | null;
  finished_at?: string | null;
  duration_seconds?: number | null;
  links: Record<string, string>;
};

export type AnalysisSummary = {
  analysis_id: string;
  question: string;
  status: AnalysisStatus;
  stage?: AnalysisStage | null;
  input_type?: string | null;
  task?: string | null;
  models: string[];
  upload_ids: string[];
  created_at?: string | null;
  updated_at?: string | null;
  finished_at?: string | null;
  duration_seconds?: number | null;
  has_result: boolean;
  error_code?: string | null;
  links: Record<string, string>;
};

export type AnalysisListResponse = {
  items: AnalysisSummary[];
  total: number;
  limit: number;
  offset: number;
};

export type ArtifactLink = {
  artifact_id: string;
  name: string;
  kind: string;
  url: string;
  media_type: string;
  size_bytes: number;
  source?: string | null;
  bounds?: number[][] | null;
  crs?: string | null;
  synthetic: boolean;
  sha256?: string | null;
};

export type EvidenceRegion = {
  id: number | string;
  area_value?: number | null;
  area_unit: string;
  centroid?: number[] | null;
  space: 'geographic' | 'pixel';
  bounds?: number[][] | null;
  location?: string | null;
  mean_score?: number | null;
  class_name?: string | null;
};

export type Evidence = {
  kind: 'change' | 'land_cover' | 'scene' | 'none';
  georeferenced: boolean;
  area_value?: number | null;
  area_unit?: string | null;
  measurement_crs?: string | null;
  percentage?: number | null;
  changed_percentage?: number | null;
  region_count?: number | null;
  largest_region_location?: string | null;
  relative_location?: string | null;
  overlay?: { format: string; url: string; bounds?: number[][] | null; crs?: string | null; pixel_size?: number[] | null; space: 'geographic' | 'pixel' } | null;
  geojson_url?: string | null;
  regions: EvidenceRegion[];
  class_areas: Array<{ class_name: string; area_value?: number | null; area_unit: string; percentage?: number | null; region_count?: number | null; mean_confidence?: number | null }>;
  modality_contributions: Array<{ modality: string; model?: string | null; score?: number | null; area_value?: number | null; area_unit: string; notes?: string | null }>;
  synthetic: boolean;
  warnings: Warning[];
};

export type SpecialistConfidence = {
  source: string;
  kind: string;
  value?: number | null;
  answer_status?: string | null;
  evidence_coverage?: number | null;
  unsupported_claims?: number | null;
  calibrated: boolean;
  measured_on?: string | null;
};

export type AnalysisResult = {
  analysis_id: string;
  status: AnalysisStatus;
  input_interpretation?: {
    detected_input_type?: string | null;
    detected_modalities: string[];
    file_roles: Record<string, string>;
    intent?: string | null;
    rationale: string[];
    certainty?: number | null;
  } | null;
  task?: string | null;
  answer?: string | null;
  answer_type?: string | null;
  evidence?: Evidence | null;
  confidence?: {
    decision: 'accepted' | 'warning' | 'abstained';
    specialists: SpecialistConfidence[];
    limiting_score?: number | null;
    limiting_source?: string | null;
    answer_status?: string | null;
    evidence_coverage?: number | null;
    unsupported_claims?: number | null;
    abstain_reason?: string | null;
    rationale?: string | null;
    warnings: Warning[];
  } | null;
  models: Array<{ name: string; version?: string | null; role?: string | null; internal_name?: string | null }>;
  warnings: Warning[];
  execution_trace: string[];
  artifacts: ArtifactLink[];
  validation?: Record<string, unknown> | null;
  versions?: Record<string, string | null> | null;
  pipeline?: { mode: 'unattached' | 'stub' | 'python'; callable?: string | null; authoritative: boolean; worker?: string | null; note?: string | null } | null;
  disclaimer?: string | null;
  question?: string | null;
  upload_ids: string[];
  created_at?: string | null;
  finished_at?: string | null;
  duration_seconds?: number | null;
  links: Record<string, string>;
};

export type ClarificationResponse = {
  file_roles?: Record<string, 'before' | 'after' | 'optical' | 'sar' | 'single' | 'unknown'>;
  before_date?: string;
  after_date?: string;
  modalities?: Array<'optical' | 'sar' | 'other'>;
  question?: string;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${SATQUERY_API_URL}${path}`, {
    ...init,
    cache: 'no-store',
    headers: init?.body instanceof FormData
      ? init.headers
      : { 'Content-Type': 'application/json', ...init?.headers },
  });

  if (!response.ok) {
    let payload: { error?: ApiErrorPayload } | null = null;
    try {
      payload = (await response.json()) as { error?: ApiErrorPayload };
    } catch {
      // A proxy or server crash can return non-JSON. Keep the client error actionable.
    }
    throw new SatQueryApiError(response.status, payload?.error ?? {
      code: 'HTTP_ERROR',
      message: `The SatQuery API returned HTTP ${response.status}.`,
    });
  }

  return response.json() as Promise<T>;
}

export function resolveBackendUrl(path: string): string {
  if (/^https?:\/\//i.test(path)) return path;
  const base = new URL(SATQUERY_API_URL);
  return new URL(path, base.origin).toString();
}

export const satqueryApi = {
  health: () => request<HealthResponse>('/health'),
  models: () => request<RegistryResponse>('/models'),
  upload: (files: File[]) => {
    const body = new FormData();
    files.forEach((file) => body.append('files', file));
    return request<UploadResponse>('/uploads', { method: 'POST', body });
  },
  validate: (uploadIds: string[], question: string) => request<ValidationResponse>('/validation', {
    method: 'POST',
    body: JSON.stringify({ upload_ids: uploadIds, question }),
  }),
  createAnalysis: (uploadIds: string[], question: string) => request<AnalysisCreated>('/analyses', {
    method: 'POST',
    body: JSON.stringify({ upload_ids: uploadIds, question }),
  }),
  listAnalyses: (limit = 100, offset = 0) => request<AnalysisListResponse>(`/analyses?limit=${limit}&offset=${offset}`),
  analysisStatus: (analysisId: string) => request<AnalysisStatusResponse>(`/analyses/${encodeURIComponent(analysisId)}/status`),
  analysisResult: (analysisId: string) => request<AnalysisResult>(`/analyses/${encodeURIComponent(analysisId)}/result`),
  submitClarification: (analysisId: string, payload: ClarificationResponse) => request<AnalysisCreated>(
    `/analyses/${encodeURIComponent(analysisId)}/clarification`,
    { method: 'POST', body: JSON.stringify(payload) },
  ),
  reportUrl: (analysisId: string, format: 'html' | 'json' | 'pdf' = 'json', download = true) =>
    `${SATQUERY_API_URL}/analyses/${encodeURIComponent(analysisId)}/report?format=${format}&download=${download}`,
};

