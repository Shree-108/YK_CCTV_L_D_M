'use client'

import Link from 'next/link'
import { useState } from 'react'
import { Download, FileBarChart, FileText } from 'lucide-react'
import { url } from '../../api/client'
import { Api } from '../../api/endpoints'
import { useQuery } from '../../hooks/useQuery'
import { dateTime, dayEndIso, dayStartIso, pct, todayLocal } from '../../lib/format'
import { Button, Field, Input, PageHeader, Panel, Select, StatusBadge } from '../../components/ui/primitives'
import { DataTable } from '../../components/ui/table'
import { EmptyState, ErrorState } from '../../components/ui/states'

export default function ReportsPage() {
  const [f, setF] = useState({ start: todayLocal(-6), end: todayLocal(), camera_id: '' })
  const cams = useQuery('cameras:list', Api.cameras)
  const evals = useQuery('evaluations', Api.evaluations)
  const jobs = useQuery('jobs', () => Api.jobs(100))
  const qs = new URLSearchParams(Object.entries({ start: dayStartIso(f.start), end: dayEndIso(f.end), camera_id: f.camera_id }).filter(([, v]) => v)).toString()

  return (
    <div className="flex flex-col gap-3">
      <PageHeader title="Reports" description="Exports of recorded data and the evaluation/video-processing reports already produced by the platform." />
      <Panel title="Data exports" icon={Download}>
        <div className="grid items-end gap-3 md:grid-cols-[1fr_1fr_1fr_auto_auto]">
          <Field label="From"><Input type="date" value={f.start} onChange={(e) => setF({ ...f, start: e.target.value })} /></Field>
          <Field label="To"><Input type="date" value={f.end} onChange={(e) => setF({ ...f, end: e.target.value })} /></Field>
          <Field label="Camera"><Select value={f.camera_id} onChange={(e) => setF({ ...f, camera_id: e.target.value })}>
            <option value="">All</option>{cams.data?.cameras.map((c) => <option key={c.id}>{c.id}</option>)}<option value="upload">upload</option></Select></Field>
          <a href={url(`/api/export/alerts.csv?${qs}`)}><Button icon={FileText}>Alerts CSV</Button></a>
          <a href={url(`/api/export/detections.csv?${qs}`)}><Button icon={FileText}>Detections CSV</Button></a>
        </div>
      </Panel>
      <Panel title="Evaluation reports" icon={FileBarChart} noPad>
        {evals.error && !evals.data ? <ErrorState error={evals.error} compact /> : (
          <DataTable rows={(evals.data?.evaluations ?? []).filter((e) => e.status === 'done')} rowKey={(e) => e.id} dense
            empty={<EmptyState compact title="No evaluation reports" body="Run an evaluation to generate model performance metrics." />}
            columns={[
              { key: 'm', header: 'Model', cell: (e) => <span className="font-mono text-fg">{e.summary?.model_version}</span> },
              { key: 'd', header: 'Dataset', cell: (e) => e.input_name },
              { key: 'p', header: 'Precision', align: 'right', cell: (e) => pct(e.summary?.overall.precision) },
              { key: 'r', header: 'Recall', align: 'right', cell: (e) => pct(e.summary?.overall.recall) },
              { key: 'map', header: 'mAP50', align: 'right', cell: (e) => pct(e.summary?.overall.mAP50) },
              { key: 't', header: 'Finished', cell: (e) => dateTime(e.finished_at) },
              { key: 'o', header: '', cell: (e) => <Link className="text-info hover:underline" href={`/evaluation?id=${e.id}`}>Open report</Link> },
            ]} />
        )}
      </Panel>
      <Panel title="Video processing reports" noPad>
        <DataTable rows={(jobs.data?.jobs ?? []).filter((j) => j.kind === 'video')} rowKey={(j) => j.id} dense
          empty={<EmptyState compact title="No processed videos" />}
          columns={[
            { key: 'n', header: 'Video', cell: (j) => <span className="text-fg">{j.input_name}</span> },
            { key: 'c', header: 'Camera', cell: (j) => <span className="font-mono">{j.camera_id}</span> },
            { key: 's', header: 'Status', cell: (j) => <StatusBadge status={j.status} /> },
            { key: 't', header: 'Submitted', cell: (j) => dateTime(j.created_at) },
            { key: 'f', header: 'Files', cell: (j) => j.status === 'done' ? (
              <span className="flex gap-2">{['stats', 'events', 'detections', 'video'].map((k) => <a key={k} className="text-info hover:underline" href={url(`/api/jobs/${j.id}/files/${k}`)}>{k}</a>)}</span>) : '—' },
          ]} />
      </Panel>
    </div>
  )
}
