'use client'

import { useState } from 'react'
import { AlertCircle, Check, CircleSlash, Download, Eye, Loader2, ScanSearch, X } from 'lucide-react'
import { ApiError, api, downloadAuthed, url } from '../../api/client'
import type { FrameOutcome, SampledFrame, SampledJob, SampledStage, SampledState } from '../../api/sampledTypes'
import { useQuery } from '../../hooks/useQuery'
import { parseCsv } from '../../lib/csv'
import { cx, pct } from '../../lib/format'
import { useApp } from '../../stores/app'
import { Badge, Button, Segmented, type Tone } from '../../components/ui/primitives'
import { EmptyState, ErrorState, LoadingState } from '../../components/ui/states'
import { litterName } from '../detection/common'

export const OUTCOME: Record<FrameOutcome, { label: string; tone: Tone; hint: string }> = {
  LITTER_DETECTED: { label: 'Litter detected', tone: 'crit', hint: 'Litter detected with high confidence on the screenshot.' },
  // never "clean": the model can miss litter
  NO_LITTER: { label: 'No litter detected by the model', tone: 'neutral', hint: 'Analysed: the model detected no litter with high confidence.' },
  UNREADABLE: { label: 'Unreadable', tone: 'neutral', hint: 'Not analysed: the decoded picture was corrupted or blank.' },
  FAILED: { label: 'FAILED', tone: 'crit', hint: 'Processing failed for this screenshot.' },
}

export const STATE: Record<SampledState, { label: string; tone: Tone }> = {
  queued: { label: 'Queued', tone: 'neutral' }, running: { label: 'Processing', tone: 'warn' },
  done: { label: 'Completed', tone: 'ok' },
  failed: { label: 'Failed', tone: 'crit' }, cancelled: { label: 'Cancelled', tone: 'neutral' },
  interrupted: { label: 'Interrupted', tone: 'crit' },
}

export const isActive = (s: SampledState) => s === 'queued' || s === 'running'

/** Pipeline stages shown to the operator. "Preview" happens at upload time, before the job exists. */
const PIPELINE: { key: SampledStage | 'preview'; label: string }[] = [
  { key: 'received', label: 'Received' }, { key: 'validating', label: 'Validating' }, { key: 'preview', label: 'Preview' },
  { key: 'sampling', label: 'Sampling frames' }, { key: 'analysing', label: 'Analysing' },
  { key: 'generating_images', label: 'Generating images' }, { key: 'generating_csv', label: 'Generating CSV' },
  { key: 'completed', label: 'Completed' },
]

export function StageList({ job }: { job: SampledJob }) {
  const order = PIPELINE.map((p) => p.key)
  const cur = Math.max(order.indexOf(job.stage), 0)
  const failed = job.state === 'failed' || job.state === 'interrupted' || job.state === 'cancelled'
  return (
    <ol className="flex flex-col gap-1.5" aria-label="Processing status">
      {PIPELINE.map((p, i) => {
        const done = job.state === 'done' || i < cur || (p.key === 'preview' && cur >= 1)
        const now = !done && i === cur
        const stopped = now && failed
        return (
          <li key={p.key} className="flex items-center gap-2.5 text-[13px]" aria-current={now ? 'step' : undefined}>
            <span className={cx('flex size-5 shrink-0 items-center justify-center rounded-full border',
              done ? 'border-ok/40 bg-ok-bg text-ok' : stopped ? 'border-crit/40 bg-crit-bg text-crit'
                : now ? 'border-warn/40 bg-warn-bg text-warn' : 'border-line-strong text-fg-3')}>
              {done ? <Check className="size-3" aria-hidden /> : stopped ? <X className="size-3" aria-hidden />
                : now ? <Loader2 className="size-3 animate-spin" aria-hidden /> : <span className="text-[10px]">{i + 1}</span>}
            </span>
            <span className={cx(done ? 'text-fg-2' : now ? 'font-medium text-fg' : 'text-fg-3')}>{p.label}</span>
            {stopped && <Badge tone="crit" className="ml-auto">{STATE[job.state].label.toLowerCase()}</Badge>}
          </li>
        )
      })}
    </ol>
  )
}

/** One of the four screenshots: image (annotated / original), outcome, timestamps, detections. */
export function FrameCard({ frame, compact }: { frame: SampledFrame; compact?: boolean }) {
  const [view, setView] = useState<'annotated' | 'original'>('annotated')
  const src = view === 'annotated' ? frame.annotated_url ?? frame.original_url : frame.original_url
  const o = frame.outcome ? OUTCOME[frame.outcome] : null
  const confirmed = frame.detections.filter((d) => d.status === 'confirmed')
  return (
    <article className="flex min-w-0 flex-col overflow-hidden rounded-xl border border-line bg-panel">
      <div className="relative aspect-video bg-video">
        {src ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={url(src)} alt={`Screenshot ${frame.idx} at ${frame.video_timestamp}`} loading="lazy" className="size-full object-contain" />
        ) : (
          <div className="flex size-full flex-col items-center justify-center gap-1.5 p-4 text-center text-[12px] text-fg-3">
            <CircleSlash className="size-5" aria-hidden />{frame.status === 'planned' ? 'Not extracted yet' : 'No picture'}
          </div>
        )}
        <span className="absolute left-2 top-2 rounded-md bg-black/70 px-1.5 py-0.5 font-mono text-[11px] text-white">#{frame.idx} · {frame.video_timestamp.slice(0, 8)}</span>
        {o && <Badge tone={o.tone} dot className="absolute right-2 top-2 bg-black/70 backdrop-blur">{o.label}</Badge>}
      </div>
      <div className="flex flex-col gap-2 p-3">
        <div className="flex items-center justify-between gap-2">
          <div className="min-w-0 text-[12px] text-fg-3">
            <div>Video time <span className="font-mono text-fg-2">{frame.video_timestamp}</span> · frame {frame.frame_idx}</div>
            <div>Recording time <span className="text-fg-2">{frame.recording_timestamp ? frame.recording_timestamp.replace('T', ' ') : 'unknown'}</span></div>
          </div>
          {frame.annotated_url && !compact && (
            <Segmented size="sm" label={`Screenshot ${frame.idx} view`} value={view} onChange={setView}
              options={[{ value: 'annotated', label: 'Annotated', icon: ScanSearch }, { value: 'original', label: 'Original', icon: Eye }]} />
          )}
        </div>
        <div className="flex flex-wrap items-center gap-1.5 text-[12px]">
          <Badge tone={frame.detection_count ? 'crit' : 'neutral'}>{frame.detection_count} litter</Badge>
        </div>
        {confirmed.length > 0 && (
          <ul className="flex flex-wrap gap-1.5">
            {confirmed.map((d, i) => (
              <li key={i} className="inline-flex items-center gap-1.5 rounded-full border border-line bg-panel-2 px-2 py-0.5 text-[12px]">
                <span className="size-1.5 rounded-full bg-crit" aria-hidden /><span className="text-fg">{litterName(d.class)}</span>
                <span className="tabular text-fg-3">{pct(d.confidence, 0)}</span>
              </li>
            ))}
          </ul>
        )}
        {(frame.error || (o && frame.outcome !== 'LITTER_DETECTED' && frame.outcome !== 'NO_LITTER')) && (
          <p className="flex gap-1.5 text-[12px] text-fg-3"><AlertCircle className="mt-0.5 size-3.5 shrink-0" aria-hidden />{frame.error ?? o?.hint}</p>
        )}
      </div>
    </article>
  )
}

/** Preview (first rows) of a server-generated CSV report plus its download button. */
const LEGACY_COLS = ['screenshot', 'video_timestamp', 'recording_timestamp', 'class', 'confidence', 'detection_count', 'outcome', 'remark']

export function CsvPreview({ path, filename, max = 12, cols = LEGACY_COLS }: { path: string; filename: string; max?: number; cols?: string[] }) {
  const { toast } = useApp()
  const [busy, setBusy] = useState(false)
  const q = useQuery(`csv:${path}`, () => api<string>(path))
  const forbidden = q.error instanceof ApiError && q.error.status === 403
  const download = async () => {
    setBusy(true)
    try { await downloadAuthed(path, filename) } catch (e) { toast({ tone: 'crit', title: 'Download failed', body: (e as Error).message }) } finally { setBusy(false) }
  }
  const data = typeof q.data === 'string' ? parseCsv(q.data) : null
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-[12px] text-fg-3">{data ? `${data.rows.length} row${data.rows.length === 1 ? '' : 's'} · ${data.header.length} columns` : forbidden ? 'Only administrators can open CSV reports.' : ''}</p>
        <Button icon={Download} loading={busy} disabled={forbidden} onClick={download}>Download CSV</Button>
      </div>
      {q.error && !q.data ? (forbidden ? null : <ErrorState error={q.error} onRetry={q.refetch} compact />) : !data ? <LoadingState rows={3} /> :
        data.rows.length === 0 ? <EmptyState compact title="The CSV has no rows" /> : (
          <div className="overflow-x-auto rounded-lg border border-line">
            <table className="w-full border-collapse text-[12px]">
              <thead><tr className="border-b border-line bg-panel-2 text-left">
                {cols.map((c) => <th key={c} className="whitespace-nowrap px-3 py-2 text-[11px] font-semibold uppercase tracking-[0.05em] text-fg-3">{c.replace(/_/g, ' ')}</th>)}
              </tr></thead>
              <tbody>
                {data.rows.slice(0, max).map((r, i) => (
                  <tr key={i} className="border-b border-line/70 last:border-0">
                    {cols.map((c) => (
                      <td key={c} className={cx('px-3 py-2 align-top text-fg-2', c === 'remark' ? 'min-w-[220px]' : 'whitespace-nowrap')}>
                        {(c === 'class' || c === 'litter_class') && r[c] ? <span className="text-fg">{litterName(r[c])}</span>
                          : (c === 'confidence' || c === 'confidence_score') && r[c] ? pct(Number(r[c]), 1)
                          : c === 'outcome' && r[c] ? <Badge tone={OUTCOME[r[c] as FrameOutcome]?.tone ?? 'neutral'}>{OUTCOME[r[c] as FrameOutcome]?.label ?? r[c]}</Badge>
                          : r[c] || <span className="text-fg-3">—</span>}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
            {data.rows.length > max && <p className="border-t border-line px-3 py-2 text-[12px] text-fg-3">Showing {max} of {data.rows.length} rows — download for the full report.</p>}
          </div>
        )}
    </div>
  )
}
