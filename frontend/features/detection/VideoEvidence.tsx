'use client'

// Analysis information, detection summary, the four random evidence screenshots and the six-column CSV report of a
// finished full-video analysis. Everything shown comes from the server's result: screenshots are the pipeline's own
// frames (a frame without litter says so), and the CSV download is rebuilt from the database on every request.
import { useState } from 'react'
import { Camera, Download, FileSpreadsheet, Info } from 'lucide-react'
import { downloadAuthed, url } from '../../api/client'
import type { Job, LitterReportRow, VideoResult, VideoScreenshot } from '../../api/types'
import { cx, dateTime, duration, num } from '../../lib/format'
import { useApp } from '../../stores/app'
import { Badge, Button, Panel } from '../../components/ui/primitives'
import { DataTable } from '../../components/ui/table'
import { EmptyState } from '../../components/ui/states'
import { litterName } from './common'
import { RegionOverlay, RegionToggle } from '../../components/vision/Overlay'

const clock = (t: number) => `${Math.floor(t / 60)}:${(t % 60).toFixed(1).padStart(4, '0')}`
const unknown = (v: string | null | undefined) => !v || v === 'UNKNOWN'

function useDownload() {
  const { toast } = useApp()
  const [busy, setBusy] = useState<string | null>(null)
  const get = async (key: string, path: string, filename: string) => {
    setBusy(key)
    try { await downloadAuthed(path, filename) } catch (e) {
      toast({ tone: 'crit', title: 'Download failed', body: (e as Error).message })
    } finally { setBusy(null) }
  }
  return { busy, get }
}

/** Store, camera, file, duration, completion time and model of the analysis. */
export function AnalysisInfo({ job }: { job: Job<VideoResult> }) {
  const r = job.result ?? {}
  const id = r.identity
  const items: [string, string, boolean?][] = [
    ['Store ID', id?.store_id || 'UNKNOWN', unknown(id?.store_id)],
    ['Camera ID', id?.camera_id || 'UNKNOWN', unknown(id?.camera_id)],
    ['Video file', job.input_name],
    ['Video duration', r.stats ? duration(r.stats.video.duration_sec) : '—'],
    ['Analysis completed', job.finished_at ? dateTime(job.finished_at) : '—'],
    ['Model version', r.stats?.model_version ?? '—'],
  ]
  return (
    <Panel title="Analysis information" icon={Info}>
      <dl className="grid grid-cols-2 gap-2 md:grid-cols-3 xl:grid-cols-6">
        {items.map(([k, v, warn]) => (
          <div key={k} className="min-w-0 rounded-lg border border-line bg-panel-2 px-3 py-2">
            <dt className="text-[11px] font-medium text-fg-3">{k}</dt>
            <dd className={cx('truncate text-[13px] font-medium', warn ? 'text-warn' : 'text-fg')} title={v}>{v}</dd>
          </div>
        ))}
      </dl>
      {(unknown(id?.store_id) || unknown(id?.camera_id)) && (
        <p role="alert" className="mt-2 text-[12px] text-warn">Store or camera was not provided: the report records UNKNOWN for it.</p>
      )}
    </Panel>
  )
}

/** Exactly four annotated screenshots at random timestamps, each with its time and a download. */
export function ScreenshotGallery({ job }: { job: Job<VideoResult> }) {
  const shots = job.result?.screenshots ?? []
  const sel = job.result?.screenshot_selection
  const { busy, get } = useDownload()
  const id = job.result?.analysis?.analysis_id ?? job.id
  if (!shots.length) {
    return (
      <Panel title="Visual evidence" icon={Camera}>
        <EmptyState compact title="No screenshots for this analysis" body="Screenshots are made for analyses run after this feature was added. Analyse the video again to get them." />
      </Panel>
    )
  }
  const v = job.result?.stats?.video
  return (
    <Panel title="Visual evidence" icon={Camera}
      subtitle={`${shots.length} screenshots at random times across the video${sel ? ` · seed ${sel.seed}` : ''} · the whole video was still analysed at the sampling interval`}
      actions={shots.some((s) => s.roi?.applied) ? <RegionToggle /> : undefined}>
      <div className="grid gap-3 sm:grid-cols-2">
        {shots.map((s) => <Shot key={s.kind} s={s} size={v ? [v.width, v.height] : null} busy={busy === s.kind} onDownload={() => get(s.kind, s.url, `${id}_${s.file}`)} />)}
      </div>
    </Panel>
  )
}

function Shot({ s, size, busy, onDownload }: { s: VideoScreenshot; size: [number, number] | null; busy: boolean; onDownload: () => void }) {
  return (
    <figure className="flex flex-col overflow-hidden rounded-lg border border-line bg-panel-2">
      <a href={url(s.url)} target="_blank" rel="noreferrer" className="relative block bg-video" title="Open full size">
        {s.roi?.applied && <RegionOverlay regions={s.roi.polygons} width={size?.[0]} height={size?.[1]} needsReview={s.roi.needs_review} />}
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={url(s.url)} alt={`Screenshot ${s.index} at video time ${clock(s.video_time_sec)}: ${s.litter_detected ? `${s.litter_detected} litter object(s)` : 'no litter detected'}`}
          loading="lazy" className="aspect-video w-full object-contain" />
      </a>
      <figcaption className="flex flex-wrap items-center gap-x-3 gap-y-1.5 px-3 py-2 text-[12px]">
        <span className="font-medium text-fg">Screenshot {String(s.index).padStart(2, '0')}</span>
        <span className="font-mono text-info" title={`frame ${s.frame_idx}`}>{clock(s.video_time_sec)}</span>
        <span className="text-fg-3" title={s.timestamp_source === 'RECORDING_TIME' ? 'Recording time' : 'Processing time (recording start not given)'}>{s.timestamp}</span>
        {s.litter_detected
          ? <Badge tone="crit" dot>{s.litter_detected} litter</Badge>
          : <Badge tone="ok" dot>No litter detected</Badge>}
        {s.under_review > 0 && <Badge tone="warn" dot>{s.under_review} under review</Badge>}
        {!s.analysed_frame && <Badge tone="warn" title="This frame was not one of the sampled analysis frames; its boxes are unconfirmed">unconfirmed frame</Badge>}
        <Button size="sm" variant="ghost" icon={Download} loading={busy} onClick={onDownload} className="ml-auto">PNG</Button>
      </figcaption>
    </figure>
  )
}

/** The six-column CSV: summary counts, header list, rows and the download. */
export function LitterReportPanel({ job }: { job: Job<VideoResult> }) {
  const rep = job.result?.report
  const { busy, get } = useDownload()
  const id = job.result?.analysis?.analysis_id ?? job.id
  if (!rep) return null
  const c = rep.counts
  const confs = rep.rows.filter((r) => r.detection_status !== 'UNDER_REVIEW').map((r) => Number(r.confidence_score))
  return (
    <Panel title="Litter report (CSV)" icon={FileSpreadsheet} noPad
      subtitle={`One row per unique litter event · review changes are included when downloaded${rep.daily ? ` · added to the ${rep.daily.date} daily CSV (${num(rep.daily.total_rows)} rows at that point)` : ''}`}
      actions={<div className="flex gap-2">
        {rep.daily && <Button size="sm" variant="ghost" icon={Download} loading={busy === 'daily'} onClick={() => get('daily', rep.daily!.url, `litter_report_${rep.daily!.date}.csv`)}>Day CSV ({rep.daily.date})</Button>}
        <Button size="sm" variant="primary" icon={Download} loading={busy === 'csv'} onClick={() => get('csv', rep.url, `${id}_litter_report.csv`)}>This video CSV</Button>
      </div>}>
      <div className="flex flex-col gap-3 px-4 pb-3 pt-1">
        <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
          <Stat label="Unique litter events" value={num(c.DETECTED + c.VERIFIED)} />
          <Stat label="Requiring review" value={num(c.UNDER_REVIEW)} tone={c.UNDER_REVIEW ? 'warn' : undefined} />
          <Stat label="Categories" value={c.classes.length ? c.classes.map(litterName).join(', ') : 'None'} />
          <Stat label="Confidence (events)" value={confs.length ? `${Math.min(...confs).toFixed(2)} – ${Math.max(...confs).toFixed(2)}` : '—'} />
        </div>
        <div className="flex flex-wrap items-center gap-1.5 text-[11.5px] text-fg-3">
          Columns:
          {rep.columns.map((h) => <code key={h} className="rounded border border-line bg-panel-2 px-1.5 py-0.5 font-mono text-fg-2">{h}</code>)}
        </div>
      </div>
      <DataTable rows={rep.rows} rowKey={(r: LitterReportRow) => `${r.timestamp}-${r.litter_class}-${r.confidence_score}-${r.detection_status}`} dense
        empty={<EmptyState compact title="No litter detected by the model" body="The CSV contains the six headers and no rows. This is not a guarantee that the area is litter-free." />}
        columns={rep.columns.map((col) => ({
          key: col, header: col,
          cell: (r: LitterReportRow) => col === 'detection_status'
            ? <Badge tone={r.detection_status === 'UNDER_REVIEW' ? 'warn' : r.detection_status === 'VERIFIED' ? 'ok' : 'crit'} dot>{r.detection_status}</Badge>
            : <span className={cx('font-mono text-[12px]', col === 'litter_class' ? 'text-fg' : 'text-fg-2')}>{r[col as keyof LitterReportRow]}</span>,
          sortValue: (r: LitterReportRow) => r[col as keyof LitterReportRow],
        }))} />
    </Panel>
  )
}

function Stat({ label, value, tone }: { label: string; value: string; tone?: 'warn' }) {
  return (
    <div className="min-w-0 rounded-lg border border-line bg-panel-2 px-3 py-2">
      <div className="text-[11px] font-medium text-fg-3">{label}</div>
      <div className={cx('truncate text-[14px] font-semibold', tone === 'warn' ? 'text-warn' : 'text-fg')} title={value}>{value}</div>
    </div>
  )
}
