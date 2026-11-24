/**
 * Results view display rules (one row = one check of one camera, the importer CSV fields; API: /api/results/checks).
 * Pure functions (no React, no fetch), unit-tested in frontend/tests/results.test.mjs.
 */

type Tone = 'ok' | 'warn' | 'crit' | 'info' | 'exp' | 'neutral'

export type CheckStatus = 'clean' | 'litter_detected' | 'view_blocked'

export const CHECK_STATUS: Record<CheckStatus, { label: string; tone: Tone; hint: string }> = {
  clean: { label: 'clean', tone: 'ok', hint: 'Analysed; no confirmed litter (possible objects under review are only mentioned in the remark).' },
  litter_detected: { label: 'litter_detected', tone: 'crit', hint: 'At least one confirmed litter object; class and score are the highest-confidence one.' },
  view_blocked: { label: 'view_blocked', tone: 'warn', hint: 'Not a usable picture (covered lens, no signal, darkness or corrupted frame); not analysed for litter.' },
}

export function checkStatusInfo(status: string) {
  return CHECK_STATUS[status as CheckStatus] ?? { label: status || '—', tone: 'neutral' as Tone, hint: 'Unknown status' }
}

/**
 * The CSV timestamp exactly as recorded (India time, recording start + screenshot offset), never converted to the
 * browser's time zone and never replaced by a processing time: null -> "time unknown".
 */
export function recordingTimeText(ts: string | null | undefined): { text: string; unknown: boolean } {
  if (!ts) return { text: 'time unknown', unknown: true }
  const m = /^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})([+-]\d{2}:\d{2})$/.exec(ts)
  return m ? { text: `${m[1]} ${m[2]} ${m[3]}`, unknown: false } : { text: ts, unknown: false }
}

/** "upload" or empty camera ids are not cameras: the analysis was started without one. */
export function cameraText(cameraId: string | null | undefined): { text: string; missing: boolean } {
  const c = (cameraId ?? '').trim()
  return !c || c.toLowerCase() === 'upload' ? { text: 'no camera assigned', missing: true } : { text: c, missing: false }
}

/** confidence_score is already 0-100 with 2 decimals (CSV text); empty for clean / view_blocked. */
export function confidenceCell(score: string | null | undefined): string {
  return score ? `${score}` : ''
}

export interface BundleInfo { ok?: boolean; folder?: string | null; import_ready?: boolean | null; csv?: string | null; reasons?: string[] | null; error?: string | null }

export function bundleBadge(b: BundleInfo | null | undefined): { label: string; tone: Tone; title: string } {
  if (!b) return { label: 'no bundle', tone: 'neutral', title: 'No result bundle recorded for this analysis (finished before bundles existed, or still being written).' }
  if (b.ok === false) return { label: 'bundle failed', tone: 'crit', title: b.error ?? 'The bundle could not be made.' }
  if (b.import_ready) return { label: 'import-ready', tone: 'ok', title: `${b.folder ?? ''}\n${b.csv ?? ''}`.trim() }
  return { label: 'no CSV', tone: 'warn', title: (b.reasons ?? []).join('\n') || 'The bundle has images and a manifest but no import CSV.' }
}

/** Query string for /api/results/checks(.csv) from the filter form (empty values dropped). */
export function resultsQuery(f: Record<string, string>): string {
  const p = new URLSearchParams()
  for (const [k, v] of Object.entries(f)) if (v !== undefined && v !== null && String(v).trim() !== '') p.set(k, String(v).trim())
  return p.toString()
}
