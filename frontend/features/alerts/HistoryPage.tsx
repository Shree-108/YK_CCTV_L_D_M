'use client'

import { useSearchParams } from 'next/navigation'
import { useEffect, useState } from 'react'
import { Download, History } from 'lucide-react'
import { url } from '../../api/client'
import { Api } from '../../api/endpoints'
import type { DetectionRow } from '../../api/types'
import { useQuery } from '../../hooks/useQuery'
import { className as cls, dateTime, dayEndIso, dayStartIso, num, pct } from '../../lib/format'
import { Badge, Button, Field, Input, PageHeader, Panel, Select, StatusBadge } from '../../components/ui/primitives'
import { DataTable } from '../../components/ui/table'
import { EmptyState, ErrorState } from '../../components/ui/states'

const PAGE = 100

export default function HistoryPage() {
  const params = useSearchParams()
  const [f, setF] = useState({ camera_id: params.get('camera_id') ?? '', status: '', start: '', end: '', class: '', min_conf: '', model: '' })
  const [offset, setOffset] = useState(0)
  const cams = useQuery('cameras:list', Api.cameras)
  const models = useQuery('models', Api.models)
  const classes = useQuery('classes', Api.classes)
  const q = {
    limit: PAGE, offset, camera_id: f.camera_id || undefined, status: f.status || undefined,
    start: f.start ? dayStartIso(f.start) : undefined, end: f.end ? dayEndIso(f.end) : undefined,
    class: f.class || undefined, model_version: f.model || undefined, min_confidence: f.min_conf ? Number(f.min_conf) : undefined,
  }
  const rows = useQuery(`history:${JSON.stringify(q)}`, () => Api.detections(q), { poll: 15000 })
  useEffect(() => setOffset(0), [f.camera_id, f.status, f.start, f.end, f.class, f.min_conf, f.model])
  const exp = new URLSearchParams(Object.entries({ camera_id: f.camera_id, start: q.start ?? '', end: q.end ?? '' }).filter(([, v]) => v)).toString()

  return (
    <div className="flex flex-col gap-3">
      <PageHeader title="Detection history" description="Every detection the pipeline logged — including rejected ones with their reason — from images, videos and live cameras. Live cameras are logged at most once per interval (Settings → Video)."
        actions={<a href={url(`/api/export/detections.csv${exp ? `?${exp}` : ''}`)}><Button icon={Download}>Export CSV</Button></a>} />
      <Panel noPad>
        <div className="grid grid-cols-2 gap-2 border-b border-line p-3 md:grid-cols-4 xl:grid-cols-7">
          <Field label="From"><Input type="date" value={f.start} onChange={(e) => setF({ ...f, start: e.target.value })} /></Field>
          <Field label="To"><Input type="date" value={f.end} onChange={(e) => setF({ ...f, end: e.target.value })} /></Field>
          <Field label="Camera"><Select value={f.camera_id} onChange={(e) => setF({ ...f, camera_id: e.target.value })}>
            <option value="">All</option>{cams.data?.cameras.map((c) => <option key={c.id} value={c.id}>{c.id}</option>)}<option value="upload">upload</option></Select></Field>
          <Field label="Class"><Select value={f.class} onChange={(e) => setF({ ...f, class: e.target.value })}>
            <option value="">All</option>{classes.data?.classes.filter((c) => c.in_active_model).map((c) => <option key={c.name} value={c.name}>{cls(c.name)}</option>)}</Select></Field>
          <Field label="Min confidence"><Select value={f.min_conf} onChange={(e) => setF({ ...f, min_conf: e.target.value })}>
            <option value="">Any</option>{[0.3, 0.45, 0.6, 0.75, 0.9].map((v) => <option key={v} value={v}>≥ {v}</option>)}</Select></Field>
          <Field label="Model"><Select value={f.model} onChange={(e) => setF({ ...f, model: e.target.value })}>
            <option value="">All</option>{models.data?.models.map((m) => <option key={m.version}>{m.version}</option>)}</Select></Field>
          <Field label="Status"><Select value={f.status} onChange={(e) => setF({ ...f, status: e.target.value })}>
            <option value="">All</option>{['confirmed', 'candidate', 'weak', 'rejected'].map((s) => <option key={s}>{s}</option>)}</Select></Field>
        </div>
        {rows.error && !rows.data ? <ErrorState error={rows.error} onRetry={rows.refetch} compact /> : (
          <DataTable<DetectionRow> rows={rows.data?.detections ?? []} rowKey={(r) => r.id} dense
            serverPage={{ offset, limit: PAGE, total: rows.data?.total ?? 0, onChange: setOffset }}
            toolbar={<span className="text-[12px] text-fg-3">{rows.data ? `${num(rows.data.total)} matching detections` : 'Loading…'}</span>}
            empty={rows.data ? <EmptyState icon={History} compact title="No detections match" body="Run image/video detection or connect a camera to populate history." /> : <div className="p-3"><div className="skeleton h-24 rounded-md" /></div>}
            columns={[
              { key: 'ts', header: 'Time', cell: (r) => dateTime(r.ts), sortValue: (r) => r.ts },
              { key: 'cam', header: 'Camera', cell: (r) => <span className="font-mono">{r.camera_id}</span>, sortValue: (r) => r.camera_id },
              { key: 'src', header: 'Source', cell: (r) => r.source_type, hideable: true },
              { key: 'class', header: 'Class', cell: (r) => <span className="text-fg">{cls(r.class)}</span>, sortValue: (r) => r.class },
              { key: 'conf', header: 'Confidence', align: 'right', cell: (r) => <span className="font-mono">{pct(r.confidence, 1)}</span>, sortValue: (r) => r.confidence },
              { key: 'track', header: 'Track ID', cell: (r) => <span className="font-mono">{r.track_id ?? '—'}</span> },
              { key: 'model', header: 'Model', cell: (r) => <span className="font-mono text-[11px]">{r.model_version ?? '—'}</span>, hideable: true },
              { key: 'status', header: 'Status', cell: (r) => <StatusBadge status={r.status} />, sortValue: (r) => r.status },
              { key: 'reason', header: 'Reason', cell: (r) => r.reason ? <Badge>{r.reason.replace(/_/g, ' ')}</Badge> : '—', hideable: true },
              { key: 'frame', header: 'Frame / t', cell: (r) => r.video_time_sec != null ? <span className="font-mono">{r.video_time_sec.toFixed(1)}s</span> : (r.frame_idx ?? '—'), hideable: true, defaultHidden: true },
              { key: 'job', header: 'Job', cell: (r) => <span className="font-mono text-[11px]">{r.job_id ?? '—'}</span>, hideable: true, defaultHidden: true },
            ]} />
        )}
      </Panel>
    </div>
  )
}
