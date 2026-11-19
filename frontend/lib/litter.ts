/**
 * Litter-focus display rules (docs/ldm_litter_focus/API_CONTRACT.md): detection / processing status badges,
 * confidence text, the video sampling interval and store <-> camera matching.
 * Pure functions (no React, no fetch), unit-tested in frontend/tests/litter.test.mjs.
 */

type Tone = 'ok' | 'warn' | 'crit' | 'info' | 'exp' | 'neutral'

export interface StatusInfo { label: string; tone: Tone; hint: string }

const DETECTION: Record<string, StatusInfo> = {
  DETECTED: { label: 'DETECTED', tone: 'crit', hint: 'High confidence: counted as a litter event.' },
  TRACKED: { label: 'TRACKED', tone: 'info', hint: 'A later observation of an object that is already a litter event (not a new event).' },
  UNDER_REVIEW: { label: 'UNDER REVIEW', tone: 'warn', hint: 'Model output below the DETECTED rule: shown and reported, never counted as an event.' },
  REJECTED: { label: 'REJECTED', tone: 'neutral', hint: 'Removed by a filter or marked as a false positive.' },
  VERIFIED: { label: 'VERIFIED', tone: 'ok', hint: 'Confirmed by an operator.' },
  RESOLVED: { label: 'RESOLVED', tone: 'ok', hint: 'Cleaned up / closed by an operator.' },
}

/** Older responses carry only the tier (`status`); map it onto the contract vocabulary. */
const LEGACY: Record<string, string> = { confirmed: 'DETECTED', candidate: 'UNDER_REVIEW', weak: 'UNDER_REVIEW', rejected: 'REJECTED' }

export function detectionStatusOf(d: { detection_status?: string | null; status?: string | null }): string {
  return d.detection_status || LEGACY[d.status ?? ''] || 'UNDER_REVIEW'
}

const REASONS: Record<string, string> = {
  too_small: 'too small', too_large: 'too large', bad_aspect_ratio: 'bad aspect ratio', class_blocked: 'class blocked',
  class_not_allowed: 'class not allowed', moving: 'moving object',
}

export const reasonText = (r: string | null | undefined) => (r ? REASONS[r] ?? r.replace(/_/g, ' ') : '')

/** Badge for a detection / event status; a REJECTED badge carries its reason ("REJECTED · too small"). */
export function detectionStatusInfo(status: string | null | undefined, reason?: string | null): StatusInfo {
  const base = DETECTION[status ?? ''] ?? { label: (status || 'UNKNOWN').replace(/_/g, ' '), tone: 'neutral' as Tone, hint: '' }
  return status === 'REJECTED' && reason ? { ...base, label: `${base.label} · ${reasonText(reason)}` } : base
}

const PROCESSING: Record<string, StatusInfo> = {
  RUNNING: { label: 'Running', tone: 'info', hint: 'The analysis is still in progress.' },
  COMPLETED_LITTER_DETECTED: { label: 'Litter detected', tone: 'crit', hint: 'The model detected at least one litter object.' },
  // never "clean": the model can miss litter
  COMPLETED_NO_LITTER_DETECTED: { label: 'No litter detected by the model', tone: 'neutral', hint: 'The model found no DETECTED object. This is not a guarantee that the area is litter-free.' },
  FAILED: { label: 'FAILED', tone: 'crit', hint: 'The analysis did not complete; no detection or summary CSV was produced.' },
}

export function processingStatusInfo(s: string | null | undefined): StatusInfo {
  return PROCESSING[s ?? ''] ?? { label: (s || 'UNKNOWN').replace(/_/g, ' '), tone: 'neutral', hint: '' }
}

/** 0.8731 -> "0.87 · 87%" (both forms, as the operator asked). */
export function confidenceText(c: number | null | undefined): string {
  if (c === null || c === undefined || Number.isNaN(c)) return '—'
  return `${c.toFixed(2)} · ${Math.round(c * 100)}%`
}

// ---------------------------------------------------------------- video sampling interval
export type IntervalChoice = '' | '1' | '2' | '5' | 'custom'
export const INTERVAL_OPTIONS: { value: IntervalChoice; label: string }[] = [
  { value: '', label: 'Default (server setting)' }, { value: '1', label: 'Every 1 s' }, { value: '2', label: 'Every 2 s' },
  { value: '5', label: 'Every 5 s' }, { value: 'custom', label: 'Custom…' },
]
export const INTERVAL_MIN = 0.1
export const INTERVAL_MAX = 60

/** The `sample_interval_sec` form value: null = not sent (server default); an error when a custom value is invalid. */
export function intervalSeconds(choice: IntervalChoice, custom = ''): { value: number | null; error: string | null } {
  if (choice === '') return { value: null, error: null }
  const raw = choice === 'custom' ? custom.trim().replace(',', '.') : choice
  const v = Number(raw)
  if (!raw || !Number.isFinite(v)) return { value: null, error: 'Enter the interval in seconds.' }
  if (v < INTERVAL_MIN || v > INTERVAL_MAX) return { value: null, error: `Between ${INTERVAL_MIN} and ${INTERVAL_MAX} seconds.` }
  return { value: Math.round(v * 1000) / 1000, error: null }
}

/** `<input type="datetime-local">` gives "2026-11-19T14:05"; the API wants "2026-11-19T14:05:00" (local time, no zone). */
export function recordingStartValue(v: string): string | null {
  const t = v.trim()
  if (!t) return null
  if (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$/.test(t)) return `${t}:00`
  return /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$/.test(t) ? t : null
}

// ---------------------------------------------------------------- stores and cameras
export interface StoreLike { id: string; cameras: { id: string }[] }
export interface CameraLike { id: string; store_id?: string | null }

/** The store a registered camera belongs to ('' = none / camera unknown). The camera record wins over store lists. */
export function storeForCamera(cameraId: string, cameras: CameraLike[], stores: StoreLike[] = []): string {
  if (!cameraId) return ''
  const cam = cameras.find((c) => c.id === cameraId)
  if (cam) return cam.store_id ?? ''
  return stores.find((s) => s.cameras.some((c) => c.id === cameraId))?.id ?? ''
}

/** Cameras offered for a store ('' = all). */
export function camerasForStore<C extends CameraLike>(storeId: string, cameras: C[]): C[] {
  return storeId ? cameras.filter((c) => (c.store_id ?? '') === storeId) : cameras
}

export const STORE_ID_RE = /^[A-Za-z0-9._ -]{1,64}$/

// ---------------------------------------------------------------- analytics series
export interface SeriesPoint { bucket: string; events: number; observations: number }

/** Values of a granularity=day series for the given local days ("YYYY-MM-DD"); days without a bucket are 0. */
export function seriesForDays(series: SeriesPoint[], days: string[]): { events: number[]; observations: number[] } {
  const by = new Map<string, SeriesPoint>()
  for (const p of series) {
    const k = p.bucket.slice(0, 10)
    const prev = by.get(k)
    by.set(k, prev ? { bucket: k, events: prev.events + p.events, observations: prev.observations + p.observations } : p)
  }
  return { events: days.map((d) => by.get(d)?.events ?? 0), observations: days.map((d) => by.get(d)?.observations ?? 0) }
}

/** Short axis label of a bucket: hour "2026-11-19T14" -> "19 14h", day "2026-11-19" -> "11-19", anything else unchanged. */
export function bucketLabel(bucket: string): string {
  const h = /^\d{4}-(\d{2})-(\d{2})T(\d{2})/.exec(bucket)
  if (h) return `${h[2]} ${h[3]}h`
  const d = /^\d{4}-(\d{2})-(\d{2})$/.exec(bucket)
  return d ? `${d[1]}-${d[2]}` : bucket
}

/** 24 hourly values (0-23) from hourly_today, missing hours 0. */
export function hours24(rows: { hour: number; events: number }[]): number[] {
  const out = Array.from({ length: 24 }, () => 0)
  for (const r of rows) if (r.hour >= 0 && r.hour < 24) out[r.hour] += r.events
  return out
}

/** Last path segment of a download URL, for the saved file name. */
export function fileNameFromPath(path: string, fallback: string): string {
  const last = path.split('?')[0].split('/').filter(Boolean).pop()
  return last && last.includes('.') ? decodeURIComponent(last) : fallback
}
