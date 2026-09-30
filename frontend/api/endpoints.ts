import { api, qs } from './client'
import type {
  Alert, AlertStatus, Analytics, Annotation, Camera, ClassInfo, DatasetImage, DatasetStats, DatasetSummary,
  DetectionRow, Evaluation, EvaluationListItem, Experiment, ExperimentDetail, HardExample, Health,
  CctvTestResult, ImageDetectResult, Job, RoiConfig, RoiZone, RtspConfig, LiveFrame, ModelInfo, SearchResult, Settings, SystemEvent, SystemHealth,
  TestLabResponse, VideoResult,
} from './types'

export type AlertQuery = {
  limit?: number; offset?: number; camera_id?: string; class?: string; severity?: string; status?: string
  min_confidence?: number; start?: string; end?: string
}

export const Api = {
  health: () => api<Health>('/api/health'),
  systemHealth: () => api<SystemHealth>('/api/system/health'),
  classes: () => api<{ classes: ClassInfo[] }>('/api/classes'),
  events: (limit = 100, camera_id?: string) => api<{ events: SystemEvent[] }>(`/api/events${qs({ limit, camera_id })}`),
  search: (q: string) => api<SearchResult>(`/api/search${qs({ q })}`),
  authCheck: () => api<{ admin: boolean }>('/api/auth/check', { admin: true }),

  cameras: () => api<{ cameras: Camera[] }>('/api/cameras'),
  camera: (id: string) => api<Camera>(`/api/cameras/${encodeURIComponent(id)}`),
  createCamera: (body: Partial<Camera>) => api<Camera>('/api/cameras', { method: 'POST', body }),
  updateCamera: (id: string, body: Partial<Camera>) => api<Camera>(`/api/cameras/${encodeURIComponent(id)}`, { method: 'PUT', body }),
  deleteCamera: (id: string) => api<{ deleted: string }>(`/api/cameras/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  restartCamera: (id: string) => api<{ restarted: string; running: boolean }>(`/api/cameras/${encodeURIComponent(id)}/restart`, { method: 'POST' }),
  live: (id: string, since = 0, signal?: AbortSignal) => api<LiveFrame>(`/api/cameras/${encodeURIComponent(id)}/live${qs({ since })}`, { signal }),

  alerts: (q: AlertQuery = {}) => api<{ alerts: Alert[]; total: number }>(`/api/alerts${qs(q)}`),
  alert: (id: number) => api<Alert>(`/api/alerts/${id}`),
  updateAlert: (id: number, body: { status?: AlertStatus; notes?: string }) => api<Alert>(`/api/alerts/${id}`, { method: 'PATCH', body }),
  alertTimeline: (id: number) => api<{ observations: { ts: string; frame_idx: number; video_time_sec: number | null; class: string; confidence: number; status: string }[]; note: string | null }>(`/api/alerts/${id}/timeline`),

  detections: (q: { limit?: number; offset?: number; camera_id?: string; status?: string; job_id?: string; start?: string; end?: string; class?: string; model_version?: string; min_confidence?: number }) =>
    api<{ detections: DetectionRow[]; count: number; total: number }>(`/api/detections${qs(q)}`),
  analytics: (q: { start?: string; end?: string; camera_id?: string; class?: string; model_version?: string }) => api<Analytics>(`/api/analytics${qs(q)}`),

  cctvDefaults: () => api<{ defaults: Partial<RtspConfig> & { has_password?: boolean }; path_template: string; template_note: string }>('/api/cctv/defaults'),
  cctvTest: (body: { camera_id: string; rtsp: RtspConfig; password?: string }) => api<CctvTestResult>('/api/cctv/test', { method: 'POST', body }),
  cctvSave: (id: string, body: Record<string, unknown>) => api<Camera>(`/api/cctv/${encodeURIComponent(id)}`, { method: 'PUT', body }),
  cctvConnect: (id: string) => api<Camera>(`/api/cctv/${encodeURIComponent(id)}/connect`, { method: 'POST' }),
  cctvStop: (id: string) => api<Camera>(`/api/cctv/${encodeURIComponent(id)}/stop`, { method: 'POST' }),
  roi: (cameraId: string) => api<RoiConfig>(`/api/roi/${encodeURIComponent(cameraId)}`),
  saveRoi: (cameraId: string, zones: RoiZone[]) => api<RoiConfig>(`/api/roi/${encodeURIComponent(cameraId)}`, { method: 'PUT', body: { zones } }),

  detectImage: (form: FormData) => api<ImageDetectResult>('/api/detect/image', { form }),
  detectVideo: (form: FormData) => api<{ job_id: string; status: string }>('/api/detect/video', { form }),
  job: <R = VideoResult>(id: string) => api<Job<R>>(`/api/jobs/${id}`),
  jobs: (limit = 50) => api<{ jobs: Job[] }>(`/api/jobs${qs({ limit })}`),

  models: () => api<{ models: ModelInfo[]; active: string | null }>('/api/models'),
  registerModel: (body: { experiment_id: string; version: string; name?: string; notes?: string }) => api<{ version: string }>('/api/models/register', { method: 'POST', body }),
  activateModel: (version: string) => api<{ active: string }>('/api/models/activate', { method: 'POST', body: { version } }),

  settings: () => api<{ settings: Settings; defaults: Settings; overrides: Partial<Settings> }>('/api/settings'),
  saveSettings: (patch: Record<string, unknown>) => api<{ settings: Settings }>('/api/settings', { method: 'PUT', body: patch }),

  datasets: () => api<{ datasets: DatasetSummary[] }>('/api/datasets'),
  dataset: (name: string) => api<DatasetStats>(`/api/datasets/${encodeURIComponent(name)}`),
  datasetImages: (name: string, q: { split?: string; class?: string; size?: string; status?: string; offset?: number; limit?: number }) =>
    api<{ total: number; items: DatasetImage[] }>(`/api/datasets/${encodeURIComponent(name)}/images${qs(q)}`),
  annotation: (name: string, id: string) => api<Annotation>(`/api/datasets/${encodeURIComponent(name)}/annotation${qs({ id })}`),
  saveAnnotation: (name: string, id: string, boxes: { cls: number; cx: number; cy: number; w: number; h: number }[]) =>
    api<{ image: string; objects: number }>(`/api/datasets/${encodeURIComponent(name)}/annotation`, { method: 'PUT', body: { id, boxes } }),
  createDatasetVersion: (name: string, newName: string) => api<DatasetSummary>(`/api/datasets/${encodeURIComponent(name)}/versions`, { method: 'POST', body: { name: newName } }),

  experiments: () => api<{ experiments: Experiment[] }>('/api/experiments'),
  experiment: (id: string) => api<ExperimentDetail>(`/api/experiments/${encodeURIComponent(id)}`),

  startEvaluation: (body: { model_version?: string; dataset: string; split: string; compare_tiling: boolean; operating_threshold?: number }) =>
    api<{ job_id: string }>('/api/evaluations', { method: 'POST', body }),
  evaluations: () => api<{ evaluations: EvaluationListItem[] }>('/api/evaluations'),
  evaluation: (id: string) => api<Evaluation>(`/api/evaluations/${id}`),

  testlabRun: (body: { dataset: string; image: string; versions: string[]; confidence: number; iou?: number; image_size?: number; tiling: string }) =>
    api<TestLabResponse>('/api/testlab/run', { method: 'POST', body }),
  testlabUpload: (form: FormData) => api<TestLabResponse>('/api/testlab/upload', { form }),
  addReview: (body: { dataset: string; image: string; model_version: string; verdict: string; cls?: string; confidence?: number; bbox: number[] }) =>
    api<{ id: number }>('/api/testlab/reviews', { method: 'POST', body }),

  hardExamples: (q: { kind?: string; status?: string; offset?: number; limit?: number }) =>
    api<{ items: HardExample[]; total: number; counts: { kind: string; review_status: string; n: number }[] }>(`/api/hard-examples${qs(q)}`),
  updateHardExample: (id: number, review_status: string) => api<HardExample>(`/api/hard-examples/${id}`, { method: 'PATCH', body: { review_status } }),
}
