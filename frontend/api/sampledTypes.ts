// Four-screenshot video analysis (/api/sampled). Mirrors backend/sampling/reports.py public shapes.
import type { Detection, RoiZone } from './types'

export type SampledState = 'queued' | 'running' | 'awaiting_roi' | 'done' | 'failed' | 'cancelled' | 'interrupted'
export type SampledStage = 'received' | 'validating' | 'sampling' | 'identifying_roi' | 'roi_confirmation' | 'analysing'
  | 'generating_images' | 'generating_csv' | 'completed'
export type FrameOutcome = 'LITTER_DETECTED' | 'NO_LITTER' | 'ROI_UNCERTAIN' | 'UNREADABLE' | 'FAILED'

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

export interface SampledFrame {
  idx: number
  frame_idx: number
  video_time_sec: number | null
  video_timestamp: string
  recording_timestamp: string | null
  status: 'planned' | 'saved' | 'unreadable' | 'failed'
  outcome: FrameOutcome | null
  roi_status: string | null
  roi: RoiZone[] | null
  detection_count: number
  error: string | null
  original_url: string | null
  annotated_url: string | null
  proposal_url: string | null
  detections: Detection[]
  counts: { confirmed: number; candidate: number; weak: number; rejected: number; raw: number } | null
  by_class: Record<string, number>
  alert_ids: number[]
  inference_ms: number | null
  image_size: [number, number] | null
  outside_roi_ignored: number | null
}

export interface RoiProposal {
  available: boolean
  method: string
  zones: RoiZone[]
  confidence: number | null
  status: 'proposed' | 'uncertain' | 'unavailable'
  static_view: boolean | null
  view_similarity: number[][]
  frames: { idx: number; zones: RoiZone[]; confidence: number; ground_share: number; indoor_share: number; labels: Record<string, number> }[]
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
  roi_mode: 'auto' | 'camera' | 'manual'
  auto_accept: boolean
  create_alerts: boolean
  roi: RoiZone[] | null
  roi_source: 'auto' | 'camera' | 'manual' | 'whole_frame' | null
  roi_status: 'pending' | 'proposed' | 'uncertain' | 'confirmed'
  roi_confidence: number | null
  proposal: { proposal: RoiProposal; prefill: RoiZone[]; prefill_source: 'auto' | 'camera' | null } | null
  model_version: string | null
  counts: null | {
    plan?: { idx: number; frame: number; seconds: number; segment: [number, number] }[]
    screenshots_planned?: number; screenshots_extracted?: number; screenshots_analysed?: number
    litter?: number; by_class?: Record<string, number>; alerts?: number; outcomes?: Record<string, number>
  }
  media: SampledMedia | null
  frames: SampledFrame[]
  csv_url: string | null
}

export interface SampledSummary {
  videos_received: number
  analyses: number
  analyses_completed: number
  analyses_failed: number
  analyses_cancelled: number
  analyses_in_progress: number
  awaiting_roi: number
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
  roi_status: string | null
  roi_source: string | null
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
  roi_mode: 'auto' | 'camera' | 'manual'
  auto_accept: boolean
  create_alerts: boolean
  model_version?: string
  new_run?: boolean
}

export interface RoiConfirmBody {
  zones: RoiZone[]
  frame_zones?: Record<number, RoiZone[]>
  skip_frames?: number[]
  whole_frame?: boolean
  save_for_camera?: boolean
}
