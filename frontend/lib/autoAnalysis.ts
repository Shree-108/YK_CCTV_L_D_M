/**
 * Zero-click video analysis (Video Analysis page): drop a video -> it is checked, uploaded with
 * auto_analyze=true (the server creates the analysis in the same request) -> the page follows the job until the
 * four screenshots, the detections and the reports are there. No button press is needed after the drop.
 *
 * Pure functions (no React, no fetch) so the behaviour is unit-tested in frontend/tests/autoAnalysis.test.mjs.
 */

export type Lifecycle =
  | 'UPLOADING' | 'QUEUED' | 'PROCESSING' | 'EXTRACTING_SCREENSHOTS' | 'RUNNING_DETECTION' | 'GENERATING_REPORT'
  | 'COMPLETED' | 'FAILED' | 'CANCELLED'

export interface UploadLimits { max_video_mb: number; video_extensions: string[] }

/** Same list as the server (backend/api/deps.py VIDEO_EXTS); the server's /api/health value wins when known. */
export const DEFAULT_VIDEO_EXTENSIONS = ['.avi', '.m4v', '.mkv', '.mov', '.mp4', '.webm']

export interface FileLike { name: string; size: number; lastModified?: number }

/** Why this file cannot be analysed, or null. The server checks again (type, size, decodability). */
export function validateVideo(file: FileLike, limits?: UploadLimits | null): string | null {
  const exts = limits?.video_extensions?.length ? limits.video_extensions : DEFAULT_VIDEO_EXTENSIONS
  const dot = file.name.lastIndexOf('.')
  const ext = dot >= 0 ? file.name.slice(dot).toLowerCase() : ''
  if (!exts.includes(ext)) return `“${file.name}” is not a supported video. Supported: ${exts.map((e) => e.slice(1).toUpperCase()).join(', ')}.`
  if (file.size <= 0) return `“${file.name}” is empty.`
  if (limits?.max_video_mb && file.size > limits.max_video_mb * 1024 * 1024) {
    return `“${file.name}” is ${(file.size / 1024 / 1024).toFixed(0)} MB; the limit is ${limits.max_video_mb} MB.`
  }
  return null
}

export const fileKey = (f: FileLike) => `${f.name}|${f.size}|${f.lastModified ?? 0}`

export interface AutoUploadResult { job_id: string; duplicate_of_active?: boolean }

export interface AutoUploaderDeps<F extends FileLike> {
  validate: (file: F) => string | null
  upload: (file: F) => Promise<AutoUploadResult>
  onStart?: (file: F) => void
  onJob: (jobId: string, result: AutoUploadResult, file: F) => void
  onError: (error: Error, file: F) => void
}

/**
 * One drop = at most one upload = at most one analysis. While an upload is in flight, the same file dropped again
 * (a drop/change event delivered twice, a double drop) is ignored silently; a different file is refused with a
 * message instead of starting a second upload. The server adds a second guard (same file content -> same job).
 */
export function createAutoUploader<F extends FileLike>(deps: AutoUploaderDeps<F>) {
  let inFlight: string | null = null
  return {
    get busy() { return inFlight !== null },
    async drop(file: F | null | undefined): Promise<string | null> {
      if (!file) return null
      const key = fileKey(file)
      if (inFlight !== null) {
        if (inFlight !== key) deps.onError(new Error('A video is already being uploaded. Wait for it to finish.'), file)
        return null
      }
      const invalid = deps.validate(file)
      if (invalid) { deps.onError(new Error(invalid), file); return null }
      inFlight = key
      deps.onStart?.(file)
      try {
        const r = await deps.upload(file)
        if (!r?.job_id) throw new Error('The server accepted the video but did not start an analysis.')
        deps.onJob(r.job_id, r, file)
        return r.job_id
      } catch (e) {
        deps.onError(e instanceof Error ? e : new Error(String(e)), file)
        return null
      } finally {
        inFlight = null
      }
    },
  }
}

// ---------------------------------------------------------------- progress
export const LIFECYCLE_STEPS: { key: Lifecycle; label: string }[] = [
  { key: 'UPLOADING', label: 'Uploading' }, { key: 'QUEUED', label: 'Queued' }, { key: 'PROCESSING', label: 'Processing' },
  { key: 'EXTRACTING_SCREENSHOTS', label: 'Extracting screenshots' }, { key: 'RUNNING_DETECTION', label: 'Running detection' },
  { key: 'GENERATING_REPORT', label: 'Generating report' }, { key: 'COMPLETED', label: 'Completed' },
]

export const isTerminal = (l: Lifecycle | undefined) => l === 'COMPLETED' || l === 'FAILED' || l === 'CANCELLED'

/** Keep polling until the job has finished (or while it is not known yet). */
export const shouldPoll = (job: { lifecycle?: Lifecycle } | null | undefined) => !job || !isTerminal(job.lifecycle)

/** Index into LIFECYCLE_STEPS of the step in progress (FAILED/CANCELLED: the step reached before stopping). */
export function stepIndex(l: Lifecycle | undefined, lastActive?: Lifecycle): number {
  const i = LIFECYCLE_STEPS.findIndex((s) => s.key === l)
  if (i >= 0) return l === 'COMPLETED' ? LIFECYCLE_STEPS.length : i
  const j = LIFECYCLE_STEPS.findIndex((s) => s.key === lastActive)
  return j >= 0 ? j : 1
}

// ---------------------------------------------------------------- results
const OVERALL: Record<string, { text: string; tone: 'ok' | 'warn' | 'crit' | 'neutral' }> = {
  LITTER_OBSERVED: { text: 'Litter detected', tone: 'crit' },
  // never "clean": the model can miss litter
  NO_LITTER_OBSERVED: { text: 'No litter detected by the model', tone: 'neutral' },
  INCOMPLETE_LITTER_OBSERVED: { text: 'Litter detected (incomplete analysis)', tone: 'crit' },
  INCOMPLETE_NO_LITTER_IN_ANALYSED_SCREENSHOTS: { text: 'Incomplete: no litter detected in the readable screenshots', tone: 'warn' },
  FAILED: { text: 'FAILED', tone: 'crit' },
}

export const overallLabel = (s: string | undefined | null) => (s && OVERALL[s]) || { text: s ? s.replace(/_/g, ' ') : '—', tone: 'warn' as const }

export interface DetLike { class: string; confidence: number; status: string; severity?: string; object_id?: string }
export interface FrameLike {
  idx: number; frame_idx: number; video_time_sec: number | null; status: string; outcome: string | null
  detections: DetLike[]; sampling_note?: string | null; error?: string | null
}

/** What one screenshot card shows: litter detected with high confidence (class, confidence, severity) and the count. */
export function screenshotFacts(f: FrameLike) {
  const confirmed = f.detections.filter((d) => d.status === 'confirmed').sort((a, b) => b.confidence - a.confidence)
  return {
    count: confirmed.length,
    items: confirmed.map((d) => ({ cls: d.class, confidence: d.confidence, severity: d.severity ?? null, object: d.object_id ?? null })),
    possible: f.detections.filter((d) => d.status === 'candidate').length,
    adjusted: !!f.sampling_note,
    analysed: f.outcome === 'LITTER_DETECTED' || f.outcome === 'NO_LITTER',
  }
}

/** Results are shown only for a completed job whose reports exist (they are written before COMPLETED). */
export function resultsReady(job: { lifecycle?: Lifecycle; reports?: unknown | null } | null | undefined): boolean {
  return !!job && job.lifecycle === 'COMPLETED' && !!job.reports
}

/** Defaults applied to every dropped video (optional, chosen once - never asked per video). */
export interface UploadDefaults { camera_id: string; store_id: string }
export const DEFAULTS_KEY = 'ldm.autoAnalysis.defaults'
export const ID_RE = /^[A-Za-z0-9_-]{1,40}$/
export const STORE_RE = /^[A-Za-z0-9._ -]{1,64}$/

export function cleanDefaults(raw: unknown): UploadDefaults {
  const r = (raw && typeof raw === 'object' ? raw : {}) as Record<string, unknown>
  const cam = typeof r.camera_id === 'string' ? r.camera_id.trim() : ''
  const store = typeof r.store_id === 'string' ? r.store_id.trim() : ''
  return { camera_id: ID_RE.test(cam) ? cam : '', store_id: STORE_RE.test(store) ? store : '' }
}
