// Types mirror the FastAPI backend (backend/api/routes*.py). Keep in sync.

export type DetStatus = 'rejected' | 'weak' | 'candidate' | 'confirmed'
export type Severity = 'LOW' | 'MEDIUM' | 'HIGH'
export type AlertStatus = 'new' | 'acknowledged' | 'resolved' | 'false_positive'
export type Bbox = [number, number, number, number]

export interface Detection {
  class: string
  confidence: number
  bbox: Bbox
  status: DetStatus
  reason?: string
  track_id?: number
  priority?: string
}

export interface LiveSummary {
  raw_detections?: number
  rejected?: number
  weak?: number
  candidates?: number
  confirmed_in_frame?: number
  confirmed_litter?: number
  active_tracks?: number
  tracks_in_memory?: number
  new_alerts?: number
}

export interface Health {
  status: 'ok' | 'degraded'
  time: string
  uptime_sec: number
  model: { version: string; path: string; device: string; classes: string[]; train_imgsz: number } | null
  model_error: string | null
  cameras: { configured: number; running: number; online: number }
  admin_enabled: boolean
  notifications: { provider: string; configured: boolean }[]
  system: { cpu_percent: number; memory_percent: number; gpu: null | Record<string, unknown> }
  hardware: { cuda: boolean; cpu_count: number; gpu?: string; vram_gb?: number; recommended: Record<string, unknown> }
}

export interface RoiZone {
  name: string
  type: 'sidewalk' | 'road_edge' | 'ground' | 'parking' | 'entrance' | 'custom'
  mode: 'include' | 'exclude'
  polygon: [number, number][]
}

export interface CameraStatus {
  state: 'stopped' | 'connecting' | 'online' | 'reconnecting' | 'error' | 'disabled'
  fps?: number
  inference_ms?: number
  last_error?: string | null
  reconnects?: number
  frames_processed?: number
  frames_dropped?: number
  summary?: LiveSummary
  resolution?: [number, number] | null
  last_frame_age_sec?: number | null
  capture_fps?: number
  frames_captured?: number
}

export interface Camera {
  id: string
  name: string
  source: string
  location: string
  latitude?: number | null
  longitude?: number | null
  enabled: boolean
  roi: RoiZone[]
  settings: Record<string, unknown>
  notify: Record<string, unknown>
  created_at: string
  updated_at: string
  status: CameraStatus
  rtsp?: Partial<RtspConfig>
  rtsp_url?: string          // masked, never contains the password
  has_password?: boolean
}

export interface RtspConfig {
  host: string
  port: number
  channel: number
  stream_type: 'main' | 'sub'
  username: string
  path_template: string
}

export interface CctvTestResult {
  ok: boolean
  code: 'connected' | 'invalid_config' | 'unreachable' | 'timeout' | 'auth_failed' | 'channel_unavailable' | 'stream_unavailable'
  message: string
  stages: { stage: string; ok: boolean; detail: string }[]
  resolution: [number, number] | null
  fps: number | null
  url: string | null
}

export interface LiveFrame {
  camera_id: string
  state: CameraStatus['state']
  frame: null | {
    seq: number
    ts: number
    width: number
    height: number
    detections: Detection[]
    summary: LiveSummary
    inference_ms: number
    fps: number
    model_version: string
  }
  image: string | null
  roi: RoiZone[]
  fps: number
  inference_ms: number
  last_error: string | null
}

export interface Alert {
  id: number
  ts: string
  camera_id: string
  class: string
  confidence: number
  track_id: number | null
  session_id: string | null
  bbox: Bbox
  location: string | null
  video_time_sec: number | null
  severity: Severity
  status: AlertStatus
  occurrences: number
  last_seen_ts: string
  job_id: string | null
  model_version: string | null
  source_type: string | null
  notes: string | null
  snapshot_url: string | null
  crop_url: string | null
}

export interface DetectionRow {
  id: number
  ts: string
  camera_id: string
  source_type: string
  job_id: string | null
  frame_idx: number | null
  video_time_sec: number | null
  class: string
  confidence: number
  x1: number
  y1: number
  x2: number
  y2: number
  status: DetStatus
  reason: string | null
  track_id: number | null
  model_version: string | null
}

export interface ModelInfo {
  version: string
  name: string
  path: string
  trained_at: string | null
  classes: string[]
  metrics: Partial<{ precision: number; recall: number; f1: number; mAP50: number; mAP50_95: number; epoch: number; epochs_run: number; source: string }>
  active: boolean
  registered_at: string
  notes: string | null
  file_exists: boolean
  architecture: string
  train_imgsz: number | null
  experiment_id: string | null
  dataset: string | null
  size_mb: number | null
  latest_evaluation: null | { id: string; dataset: string; split: string; finished_at: string; overall: Prf & { mAP50: number | null; mAP50_95: number | null } }
}

export interface Settings {
  confidence: { weak_threshold: number; candidate_threshold: number; confirmed_threshold: number }
  inference: { image_size: number; iou_threshold: number; max_detections: number; device: string; half: boolean; agnostic_nms: boolean }
  tiling: { enabled: boolean; mode: 'auto' | 'always'; tile_size: number; overlap: number; auto_min_ratio: number; include_full_frame: boolean; merge_iou: number; cross_tile_ios: number }
  filters: { class_allowlist: string[] | null; class_blocklist: string[]; min_box_area_px: number; max_box_area_frac: number; max_aspect_ratio: number; roi_enabled: boolean; roi_anchor: 'footpoint' | 'center'; roi_margin_frac: number }
  tracking: { iou_match: number; low_iou_match: number; max_missed_frames: number; temporal_window: number; min_confirmations: number; stationarity_enabled: boolean; max_center_shift_frac: number }
  severity: { high: SeverityRule[]; medium: SeverityRule[] }
  alerts: { enabled: boolean; one_alert_per_track: boolean; dedup_window_sec: number; dedup_iou: number; dedup_center_frac: number; class_match: 'exact' | 'group' | 'any'; snapshot_jpeg_quality: number; crop_padding_frac: number; alert_on_image_uploads: boolean }
  video: { sample_fps: number; live_process_fps: number; live_log_interval_sec: number; annotate_all_frames: boolean; max_video_seconds: number }
}
export interface SeverityRule { priorities: string[]; classes: string[]; min_confidence: number; min_active_objects: number }

export interface Analytics {
  alerts_total: number
  confirmed_objects: number
  by_camera: { camera_id: string; n: number }[]
  by_class: { class: string; n: number }[]
  by_severity: { severity: Severity; n: number }[]
  by_status: { status: AlertStatus; n: number }[]
  by_hour: { hour: string; n: number }[]
  by_day: { day: string; n: number }[]
  confirmed_by_day: { day: string; n: number }[]
  detections_by_day: { day: string; status: DetStatus; n: number }[]
  detections_by_status: { status: DetStatus; n: number }[]
  rejection_reasons: { reason: string | null; n: number }[]
  confidence_histogram: { bin: number; n: number }[]
}

export interface ImageDetectResult {
  detections: Detection[]
  count: number
  counts: { confirmed: number; candidate: number; weak: number; rejected: number; raw: number }
  by_class: Record<string, number>
  alerts: Alert[]
  inference_ms: number
  tiled: boolean
  tiles: number
  image_size: [number, number]
  model_version: string
  roi_active?: boolean
  outside_roi_ignored?: number
  highest_confidence?: number | null
  average_confidence?: number | null
  annotated_image?: string
  filename: string
  camera_id: string
}

export interface Job<R = unknown> {
  id: string
  kind: 'video' | 'evaluation'
  status: 'queued' | 'running' | 'done' | 'failed'
  camera_id: string | null
  created_at: string
  finished_at: string | null
  input_name: string
  progress: number
  result: R | null
  error: string | null
  files?: Record<string, string>
}

export interface VideoProgress {
  frames_read: number
  frames_total: number
  frames_analysed: number
  processing_fps: number | null
  eta_sec: number | null
  detections?: number
  confirmed_objects?: number
  new_alerts?: number
}

export interface VideoResult {
  progress?: VideoProgress
  stats?: {
    video: { fps: number; frames: number; width: number; height: number; duration_sec: number; output_codec: string }
    frames_analysed: number
    sample_every_n_frames: number
    unique_confirmed_objects: number
    litter?: LitterSummary
    confirmed_by_class: Record<string, number>
    new_alerts: number
    detections_by_status: Record<string, number>
    rejection_reasons: Record<string, number>
    peak_per_frame: Record<string, number>
    mean_inference_ms: number | null
    model_runs?: number
    frames_reused_static?: number
    device?: string
    processing_fps: number | null
    wall_time_sec: number
    model_version: string
  }
  events?: { event: 'confirmed' | 'lost'; track_id: number; class: string; frame_idx: number; video_time_sec: number; confidence: number; alert_id?: number }[]
  alert_ids?: number[]
}

export interface LitterSummary {
  total: number
  roi_active: boolean
  inside_roi: number
  outside_roi_ignored: number
  highest_confidence: number | null
  average_confidence: number | null
  classes: { class: string; count: number; max_confidence: number; avg_confidence: number; timestamps_sec: number[] }[]
  objects: { track_id: number; class: string; first_seen_sec: number; max_confidence: number; avg_confidence: number }[]
}

export interface RoiConfig {
  camera_id: string
  zones: RoiZone[]
  active: boolean
}

export interface DatasetSummary {
  name: string
  classes: string[]
  splits: Record<string, { images?: number; objects?: number; negatives?: number; empty_labels?: number }>
  totals: { images: number; objects: number; negatives: number }
  created: number
  readme: string | null
  issue_counts: Record<string, number>
  error?: string
}
export interface DatasetStats extends DatasetSummary {
  path: string
  objects_per_class: Record<string, number>
  images_per_class: Record<string, number>
  size_distribution: { small: number; medium: number; large: number }
  box_area_px: null | { mean: number; median: number; min: number; max: number; histogram: number[]; histogram_edges_side_px: number[] }
  resolutions: Record<string, number>
  class_imbalance_ratio: number | null
  issues: { type: string; image: string; detail: string }[]
}
export interface DatasetImage { id: string; split: string; file: string; objects: number; classes: string[]; status: 'annotated' | 'empty' | 'negative' | 'issues' }
export interface GtBox { index: number; cls_id: number; class: string; bbox: Bbox; area_px: number; size: 'small' | 'medium' | 'large' }
export interface Annotation {
  id: string; split: string; negative: boolean; width: number; height: number
  ground_truth: GtBox[]; problems: string[]; editable: boolean
  reviews: { id: number; verdict: string; class: string | null; model_version: string; x1: number; y1: number; x2: number; y2: number }[]
}

export interface Experiment {
  id: string; name: string; path: string; date: string; model: string; dataset: string
  epochs_planned: number | null; epochs_run: number; imgsz: number | null; status: string
  best: null | { epoch: number; precision: number | null; recall: number | null; mAP50: number | null; mAP50_95: number | null }
  has_best_weights: boolean; error?: string
}
export interface ExperimentDetail extends Experiment {
  config: Record<string, unknown>
  augmentation: Record<string, unknown>
  epochs: Record<string, number | null>[]
  artifacts: string[]
}

export interface Prf { precision: number; recall: number; f1: number; tp: number; fp: number; fn: number }
export interface RecallBucket { instances: number; detected: number; recall: number | null }
export interface EvalMetrics {
  operating_threshold: number
  overall: Prf & { mAP50: number | null; mAP50_95: number | null }
  per_class: (Prf & { class: string; instances: number; ap50: number | null; ap50_95: number | null })[]
  by_size: Record<'small' | 'medium' | 'large', RecallBucket>
  tiny_recall: RecallBucket
  report_groups: Record<string, RecallBucket>
  threshold_sweep: (Prf & { threshold: number })[]
  best_f1_threshold: number
  negatives: { images: number; false_positives: number; fp_per_image: number | null }
  latency_ms: { mean: number | null; p95: number | null }
  images: number
  gt_objects: number
}
export interface Evaluation extends EvalMetrics {
  id: string; model_version: string; dataset: string; split: string; created_at: string; classes: string[]; mode: string
  modes: Record<string, EvalMetrics> | null
  tiling_comparison: null | { delta_mAP50: number; delta_small_recall: number; delta_precision: number; latency_factor: number; recommend_tiling: boolean; note: string }
  per_image: { id: string; split: string; negative: boolean; width: number; height: number; tp: number; fp: number; fn: number; gt: number }[]
  hard_examples_found: number
}
export interface EvaluationListItem { id: string; status: Job['status']; created_at: string; finished_at: string | null; input_name: string; progress: number; error: string | null; summary: null | { model_version: string; dataset: string; split: string; overall: EvalMetrics['overall']; tiling_comparison: Evaluation['tiling_comparison'] } }

export interface TestLabPred { cls_id: number; class: string; confidence: number; bbox: Bbox; match?: 'tp' | 'fp'; gt_index?: number | null }
export interface TestLabResult {
  model_version: string; error?: string; device?: string; latency_ms?: number; fps?: number; tiled?: boolean; tiles?: number
  detections?: TestLabPred[]; confidence_histogram?: number[]; missed_gt?: number[]
  counts?: { ground_truth: number; predicted: number; true_positive: number; false_positive: number; missed: number }
}
export interface TestLabResponse { image: { id?: string; width: number; height: number; split?: string; negative?: boolean }; ground_truth: GtBox[] | null; results: TestLabResult[] }

export interface HardExample {
  id: number; created_at: string; kind: string; source: string; dataset: string | null; split: string | null; image: string | null
  class: string | null; confidence: number | null; x1: number; y1: number; x2: number; y2: number
  model_version: string | null; review_status: 'pending' | 'approved' | 'rejected' | 'ignored'; notes: string | null
}

export interface SystemEvent { id: number; ts: string; level: 'info' | 'warning' | 'error'; source: string; camera_id: string | null; message: string; data_json: string | null }

export interface SystemHealth {
  overall: 'operational' | 'degraded' | 'offline'
  checked_at: string
  components: { component: string; status: 'operational' | 'degraded' | 'offline' | 'not_configured' | 'not_available'; detail: string; checked_at: string; [k: string]: unknown }[]
}

export interface SearchResult {
  query: string
  cameras: { id: string; name: string; location: string }[]
  alerts: Pick<Alert, 'id' | 'ts' | 'camera_id' | 'class' | 'confidence' | 'severity' | 'status' | 'track_id'>[]
  models: { version: string; name: string; active: boolean }[]
  experiments: { id: string; name: string }[]
  datasets: { name: string }[]
}

export interface ClassInfo { name: string; priority: string; bin: string; color: string | null; in_active_model: boolean }
