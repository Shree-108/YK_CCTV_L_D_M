// Four-screenshot video analysis (/api/sampled). Mirrors backend/sampling/reports.py public shapes.
import type { AnalysisInfo, Detection, Identity } from './types'
import type { Lifecycle } from '../lib/autoAnalysis'

/** A detection of a sampled screenshot: model output plus severity, estimated object and alert (when created). */
export type SampledDetection = Detection & { severity?: 'HIGH' | 'MEDIUM' | 'LOW'; object_id?: string; alert_id?: number }

export type SampledState = 'queued' | 'running' | 'done' | 'failed' | 'cancelled' | 'interrupted'
export type SampledStage = 'received' | 'validating' | 'sampling' | 'analysing' | 'generating_images' | 'generating_csv' | 'completed'
/** NO_LITTER = the model detected no litter on the screenshot (shown as "No litter detected by the model", never "clean"). */
export type FrameOutcome = 'LITTER_DETECTED' | 'NO_LITTER' | 'UNREADABLE' | 'FAILED'

export interface SampledMedia {
  id: string
  created_at: string
  filename: string
  size_bytes: number
  width: number
  height: number
  fps: number
  frames: number
  duration_sec: number
  suggested_recording_start: string | null
  preview_url?: string
  previously_received?: { media_id: string; received_at: string; filename: string; jobs: { id: string; state: SampledState; created_at: string }[] }[]
}

/** POST /api/sampled/media with auto_analyze=true: the analysis is created by the same request. */
export interface SampledAutoUpload extends Partial<SampledMedia> {
  job_id: string
  job_state: SampledState
  duplicate_of_active: boolean
}

export interface SampledFrame {
  idx: number
  frame_idx: number
  video_time_sec: number | null
  video_timestamp: string
  recording_timestamp: string | null
  status: 'planned' | 'saved' | 'unreadable' | 'failed'
  outcome: FrameOutcome | null
  detection_count: number
  error: string | null
  original_url: string | null
  annotated_url: string | null
  proposal_url: string | null
  detections: SampledDetection[]
  counts: { confirmed: number; candidate: number; weak: number; rejected: number; raw: number } | null
  by_class: Record<string, number>
  alert_ids: number[]
  inference_ms: number | null
  image_size: [number, number] | null
  planned_frame_idx: number | null
  sampling_note: string | null
  download_original_url: string | null
  download_annotated_url: string | null
  diagnostic_url?: string | null
}

/** analysis_summary.csv / .json of a job (backend/sampling/detection_report.py SUMMARY_FIELDS). */
export interface SampledReportSummary {
  job_id: string
  video_filename: string
  analysis_started_at: string
  analysis_completed_at: string
  processing_duration_seconds: number | ''
  screenshots_planned: number
  screenshots_analysed: number
  screenshots_failed: number
  sampling_adjustments: number
  total_objects_detected: number
  unique_objects_estimated: number
  total_high_severity: number
  total_medium_severity: number
  total_low_severity: number
  possible_litter_objects: number
  alerts_created: number
  overall_analysis_status: string
  review_required: 'TRUE' | 'FALSE'
  note: string
}

export interface SampledJob {
  id: string
  media_id: string
  video_name: string | null
  created_at: string
  started_at: string | null
  finished_at: string | null
  stage: SampledStage
  state: SampledState
  error: string | null
  store_id: string | null
  camera_id: string | null
  recording_start: string | null
  seed: number
  sample_count: number
  create_alerts: boolean
  model_version: string | null
  counts: null | {
    plan?: { idx: number; frame: number; seconds: number; segment: [number, number] }[]
    screenshots_planned?: number; screenshots_extracted?: number; screenshots_analysed?: number
    litter?: number; by_class?: Record<string, number>; alerts?: number; outcomes?: Record<string, number>
  }
  media: SampledMedia | null
  frames: SampledFrame[]
  csv_url: string | null
  mode: 'manual' | 'auto'
  sampling: 'random_quarters' | 'fixed_pct'
  lifecycle: Lifecycle
  reports: { detection_csv: string; summary_csv: string; summary_json: string } | null
  /** standard-format outputs (`<...>_detections.csv`, `<...>_summary.csv`, metadata) of the job */
  standard_outputs?: AnalysisInfo | null
  identity?: Identity | null
  summary: SampledReportSummary | null
}

export interface SampledSummary {
  videos_received: number
  analyses: number
  analyses_completed: number
  analyses_failed: number
  analyses_cancelled: number
  analyses_in_progress: number
  screenshots_extracted: number
  screenshots_analysed: number
  total_litter: number
  categories: Record<string, number>
}

export interface SampledAnalysisRow {
  id: string
  video_name: string | null
  store_id: string | null
  camera_id: string | null
  recording_start: string | null
  created_at: string
  finished_at: string | null
  state: SampledState
  stage: SampledStage
  error: string | null
  duration_sec: number | null
  screenshots_extracted: number
  screenshots_analysed: number
  litter: number
  timestamps: string[]
  thumbs: string[]
  csv_url: string | null
}

export interface SampledOverview {
  summary: SampledSummary
  latest: SampledJob | null
  newest_other: SampledJob | null
  months: { completion: string[]; recording: string[] }
  grouping_rules: { completion: string; recording: string }
}

export interface SampledMonth extends Omit<SampledSummary, 'videos_received'> {
  month: string
  by: 'completion' | 'recording'
  analyses_list: SampledAnalysisRow[]
  csv_url: string
}

export interface SampledDetectionRow {
  id: number
  ts: string
  class: string
  confidence: number
  x1: number; y1: number; x2: number; y2: number
  frame_idx: number
  video_time_sec: number | null
  video_timestamp: string
  recording_timestamp: string | null
  job_id: string
  model_version: string | null
  store_id: string | null
  camera_id: string | null
  video_name: string
  screenshot_idx: number | null
  evidence_url: string | null
}

export interface SampledHourlyRow {
  hour_start: string
  hour_end: string
  store_id: string
  camera_id: string
  analyses: number
  screenshots: number
  analysed: number
  with_litter: number
  litter: number
  by_class: Record<string, number>
}

export interface SampledCreateBody {
  media_id: string
  store_id?: string
  camera_id?: string
  recording_start?: string
  seed?: number
  create_alerts: boolean
  model_version?: string
  new_run?: boolean
  mode?: 'manual' | 'auto'
}
