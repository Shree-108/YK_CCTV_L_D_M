// Results view API types (backend/services/results_view.py, /api/results/checks).
import type { BundleInfo, CheckStatus } from '../lib/results'

export interface ResultCheckRow {
  // the nine importer CSV fields, exactly as the CSV writes them (strings; empty = empty in the CSV)
  timestamp: string | null            // recording time, ISO 8601 +05:30; null = no trustworthy recording time
  camera_id: string
  image_name: string
  litter_class: string
  confidence_score: string            // 0-100, 2 decimals
  detection_status: CheckStatus
  remark: string
  cart_id: string
  store_name: string
  // context
  job_id: string
  idx: number
  job_created_at: string | null
  litter_objects: number
  under_review: number                // candidate detections only (same rule as the remark and the image header)
  time_known: boolean
  time_warning: string | null
  camera_raw: string | null
  camera_flag: string | null
  image_url: string
  store_key: string
  job_store_id: string
  store_flag: string | null
  import_ready: boolean
  import_issue: string | null
  bundle: BundleInfo | null
}

export interface ResultChecksResponse {
  rows: ResultCheckRow[]
  total: number
  summary: { total: number; clean: number; litter_detected: number; view_blocked: number; time_unknown: number; no_camera: number; import_ready: number }
  columns: string[]
  problems: string[]
  options: { stores: string[]; cameras: string[]; classes: string[] }
}

export type ResultChecksQuery = {
  store?: string; camera?: string; status?: string; litter_class?: string; min_confidence?: number
  date_from?: string; date_to?: string; job_id?: string; limit?: number; offset?: number
}
