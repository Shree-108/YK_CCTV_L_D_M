'use client'

// Panels over GET /api/litter/analytics, shared by the Overview and Analytics pages. "Events" are unique litter
// events; "observations" are frame sightings. They are always labelled separately and never summed.
import Link from 'next/link'
import { url } from '../../api/client'
import type { LitterAnalysis, LitterAnalytics, ProcessingStatus } from '../../api/types'
import { className as cls, dateTime, num } from '../../lib/format'
import { hours24, processingStatusInfo } from '../../lib/litter'
import { Panel } from '../../components/ui/primitives'
import { BarList, Columns, SERIES, StackBar } from '../../components/ui/charts'
import { DataTable } from '../../components/ui/table'
import { EmptyState } from '../../components/ui/states'
import { ProcessingStatusBadge } from '../detection/AnalysisOutputs'

const PROCESSING_ORDER: ProcessingStatus[] = ['COMPLETED_LITTER_DETECTED', 'COMPLETED_NO_LITTER_DETECTED', 'FAILED', 'RUNNING']
const PROCESSING_COLOR: Record<ProcessingStatus, string> = {
  COMPLETED_LITTER_DETECTED: 'var(--crit)', COMPLETED_NO_LITTER_DETECTED: 'var(--fg-3)', FAILED: 'var(--warn)', RUNNING: 'var(--info)',
}

export function ByClassPanel({ a }: { a: LitterAnalytics }) {
  return (
    <Panel title="Litter events by class" subtitle="Unique events · frame observations shown beside each bar">
      {a.by_class.length === 0 ? <EmptyState compact title="No litter events" /> : (
        <BarList data={a.by_class.map((r, i) => ({
          label: cls(r.class), value: r.events, color: SERIES[i % SERIES.length],
          hint: <span className="ml-[8.5rem] block text-[11px] text-fg-3">{num(r.observations)} frame observations</span>,
        }))} />
      )}
    </Panel>
  )
}

export function HourlyTodayPanel({ a }: { a: LitterAnalytics }) {
  const h = hours24(a.hourly_today)
  return (
    <Panel title="Hourly frequency today" subtitle="Unique litter events per hour of day">
      {h.every((v) => v === 0) ? <EmptyState compact title="No litter events today" /> : (
        <Columns ariaLabel="Litter events per hour today" color="var(--crit)" data={h.map((v, i) => ({ label: String(i).padStart(2, '0'), value: v }))} />
      )}
    </Panel>
  )
}

export function ProcessingPanel({ a }: { a: LitterAnalytics }) {
  const parts = PROCESSING_ORDER.map((s) => ({ label: processingStatusInfo(s).label, value: a.processing[s] ?? 0, color: PROCESSING_COLOR[s], s }))
  const total = parts.reduce((x, p) => x + p.value, 0)
  return (
    <Panel title="Processing status" subtitle={`${num(total)} analyses`}>
      {total === 0 ? <EmptyState compact title="No analyses yet" /> : (
        <div className="flex flex-col gap-3">
          <StackBar parts={parts} />
          <ul className="flex flex-col gap-1.5 text-[12.5px]">
            {parts.map((p) => (
              <li key={p.s} className="flex items-center justify-between gap-2">
                <span className="flex items-center gap-2 text-fg-2"><span className="size-2 rounded-sm" style={{ background: p.color }} aria-hidden />{p.label}</span>
                <span className="tabular font-mono text-fg">{num(p.value)}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </Panel>
  )
}

export function StoreTable({ a }: { a: LitterAnalytics }) {
  return (
    <Panel title="Store-wise" noPad>
      <DataTable rows={a.by_store} rowKey={(r) => r.store_id} dense pageSize={10}
        empty={<EmptyState compact title="No litter events" />}
        columns={[
          { key: 'id', header: 'Store', cell: (r) => <span><span className="font-mono text-fg">{r.store_id}</span>{r.store_name && r.store_name !== 'UNKNOWN' && <span className="text-fg-3"> · {r.store_name}</span>}</span>, sortValue: (r) => r.store_id },
          { key: 'ev', header: 'Unique events', align: 'right', cell: (r) => num(r.events), sortValue: (r) => r.events },
          { key: 'obs', header: 'Frame observations', align: 'right', cell: (r) => num(r.observations), sortValue: (r) => r.observations },
        ]} />
    </Panel>
  )
}

export function CameraTable({ a }: { a: LitterAnalytics }) {
  return (
    <Panel title="Camera-wise" noPad>
      <DataTable rows={a.by_camera} rowKey={(r) => `${r.store_id}/${r.camera_id}`} dense pageSize={10}
        empty={<EmptyState compact title="No litter events" />}
        columns={[
          { key: 'id', header: 'Camera', cell: (r) => <span><span className="font-mono text-fg">{r.camera_id}</span>{r.camera_name && r.camera_name !== 'UNKNOWN' && <span className="text-fg-3"> · {r.camera_name}</span>}</span>, sortValue: (r) => r.camera_id },
          { key: 'store', header: 'Store', cell: (r) => <span className="font-mono">{r.store_id}</span>, sortValue: (r) => r.store_id },
          { key: 'ev', header: 'Unique events', align: 'right', cell: (r) => num(r.events), sortValue: (r) => r.events },
          { key: 'obs', header: 'Frame observations', align: 'right', cell: (r) => num(r.observations), sortValue: (r) => r.observations },
        ]} />
    </Panel>
  )
}

/** Where to open an analysis again: full-video jobs and quick checks open on Video Analysis. */
function analysisHref(r: LitterAnalysis): string | null {
  if (r.source_type === 'video' && r.job_id) return `/detect/video?video_job=${encodeURIComponent(r.job_id)}`
  if (r.source_type === 'sampled' && r.job_id) return `/detect/video?mode=quick&job=${encodeURIComponent(r.job_id)}`
  return null
}

export function RecentAnalysesTable({ rows }: { rows: LitterAnalysis[] }) {
  return (
    <DataTable rows={rows} rowKey={(r) => r.analysis_id} dense pageSize={10}
      empty={<EmptyState compact title="No analyses yet" body="Analyse a CCTV image or video to start." />}
      columns={[
        { key: 't', header: 'When', cell: (r) => dateTime(r.created_at ?? null), sortValue: (r) => r.created_at ?? '' },
        { key: 'src', header: 'Source', cell: (r) => <span className="text-fg">{r.input_name ?? r.analysis_id}</span>
          , sortValue: (r) => r.input_name ?? '' },
        { key: 'type', header: 'Type', cell: (r) => (r.source_type ?? '—').replace(/_/g, ' ') },
        { key: 'store', header: 'Store', cell: (r) => <span className="font-mono">{r.identity?.store_id ?? 'UNKNOWN'}</span> },
        { key: 'cam', header: 'Camera', cell: (r) => <span className="font-mono">{r.identity?.camera_id ?? 'UNKNOWN'}</span> },
        { key: 'st', header: 'Status', cell: (r) => <ProcessingStatusBadge status={r.processing_status} /> },
        { key: 'ev', header: 'Unique events', align: 'right', cell: (r) => num(r.unique_events), sortValue: (r) => r.unique_events },
        { key: 'obs', header: 'Observations', align: 'right', cell: (r) => num(r.frame_observations), sortValue: (r) => r.frame_observations },
        { key: 'open', header: '', cell: (r) => {
          const href = analysisHref(r)
          const img = r.urls?.annotated
          return href ? <Link href={href} className="text-[12px] text-info hover:underline">Open</Link>
            : img ? <a href={url(img)} target="_blank" rel="noreferrer" className="text-[12px] text-info hover:underline">Image</a> : null
        } },
      ]} />
  )
}
