'use client'

import { useMemo, useState } from 'react'
import { BarChart3 } from 'lucide-react'
import { Api } from '../../api/endpoints'
import type { Analytics } from '../../api/types'
import { useQuery } from '../../hooks/useQuery'
import { className as cls, dayEndIso, dayStartIso, num, pct, todayLocal } from '../../lib/format'
import { Field, Input, PageHeader, Panel, Select } from '../../components/ui/primitives'
import { BarList, Columns, Donut, Legend, LineChart, SERIES, StackBar } from '../../components/ui/charts'
import { MetricCard } from '../../components/ui/kpi'
import { EmptyState, ErrorState, LoadingState } from '../../components/ui/states'

function daysBetween(a: string, b: string): string[] {
  const out: string[] = []
  const d = new Date(`${a}T00:00:00`), end = new Date(`${b}T00:00:00`)
  while (d <= end && out.length < 400) {
    out.push(`${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`)
    d.setDate(d.getDate() + 1)
  }
  return out
}

const STATUS_COLORS: Record<string, string> = { confirmed: 'var(--ok)', candidate: 'var(--warn)', weak: 'var(--fg-3)', rejected: 'var(--crit)' }

export default function AnalyticsPage() {
  const [f, setF] = useState({ start: todayLocal(-29), end: todayLocal(), camera_id: '', class: '', model_version: '' })
  const cams = useQuery('cameras:list', Api.cameras)
  const models = useQuery('models', Api.models)
  const classes = useQuery('classes', Api.classes)
  const q = { start: dayStartIso(f.start), end: dayEndIso(f.end), camera_id: f.camera_id || undefined, class: f.class || undefined, model_version: f.model_version || undefined }
  const a = useQuery(`analytics:page:${JSON.stringify(q)}`, () => Api.analytics(q), { poll: 60000 })

  return (
    <div className="flex flex-col gap-3">
      <PageHeader title="Analytics" description="Trends and distributions computed from the alert, detection and confirmed-object tables. Hour-of-day uses UTC timestamps as stored." />
      <Panel>
        <div className="grid grid-cols-2 gap-2 md:grid-cols-5">
          <Field label="From"><Input type="date" value={f.start} max={f.end} onChange={(e) => setF({ ...f, start: e.target.value })} /></Field>
          <Field label="To"><Input type="date" value={f.end} min={f.start} onChange={(e) => setF({ ...f, end: e.target.value })} /></Field>
          <Field label="Camera"><Select value={f.camera_id} onChange={(e) => setF({ ...f, camera_id: e.target.value })}>
            <option value="">All cameras</option>{cams.data?.cameras.map((c) => <option key={c.id} value={c.id}>{c.id} · {c.name}</option>)}<option value="upload">upload</option></Select></Field>
          <Field label="Class"><Select value={f.class} onChange={(e) => setF({ ...f, class: e.target.value })}>
            <option value="">All classes</option>{classes.data?.classes.filter((c) => c.in_active_model).map((c) => <option key={c.name} value={c.name}>{cls(c.name)}</option>)}</Select></Field>
          <Field label="Model"><Select value={f.model_version} onChange={(e) => setF({ ...f, model_version: e.target.value })}>
            <option value="">All models</option>{models.data?.models.map((m) => <option key={m.version}>{m.version}</option>)}</Select></Field>
        </div>
      </Panel>
      {a.error && !a.data ? <Panel><ErrorState error={a.error} onRetry={a.refetch} /></Panel> : !a.data ? <Panel><LoadingState rows={6} /></Panel> :
        a.data.alerts_total === 0 && a.data.detections_by_status.length === 0 ? (
          <Panel><EmptyState icon={BarChart3} title="No data in this period"
            body="Analytics are computed only from real events. Connect a camera, process a video or run image detection with alerts enabled — nothing is simulated here." /></Panel>
        ) : <Charts a={a.data} days={daysBetween(f.start, f.end)} />}
    </div>
  )
}

function Charts({ a, days }: { a: Analytics; days: string[] }) {
  const byDay = useMemo(() => new Map(a.by_day.map((d) => [d.day, d.n])), [a])
  const confByDay = useMemo(() => new Map(a.confirmed_by_day.map((d) => [d.day, d.n])), [a])
  const detTotal = a.detections_by_status.reduce((s, r) => s + r.n, 0)
  const confirmed = a.detections_by_status.find((r) => r.status === 'confirmed')?.n ?? 0
  const rejected = a.detections_by_status.find((r) => r.status === 'rejected')?.n ?? 0
  const detDaily = useMemo(() => {
    const m = new Map<string, Record<string, number>>()
    a.detections_by_day.forEach((r) => m.set(r.day, { ...(m.get(r.day) ?? {}), [r.status]: r.n }))
    return m
  }, [a])
  const dow = useMemo(() => {
    const names = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    const c = [0, 0, 0, 0, 0, 0, 0]
    a.by_day.forEach((d) => { c[new Date(`${d.day}T12:00:00`).getDay()] += d.n })
    return names.map((n, i) => ({ label: n, value: c[i] }))
  }, [a])
  const labels = days.map((d) => d.slice(5))
  return (
    <>
      <div className="grid grid-cols-2 gap-2 md:grid-cols-3 xl:grid-cols-6">
        <MetricCard label="Alerts" value={num(a.alerts_total)} />
        <MetricCard label="Confirmed objects" value={num(a.confirmed_objects)} tone="ok" />
        <MetricCard label="Raw detections" value={num(detTotal)} />
        <MetricCard label="Confirmed detections" value={num(confirmed)} hint={detTotal ? `${pct(confirmed / detTotal)} of raw` : undefined} />
        <MetricCard label="Rejected detections" value={num(rejected)} hint={detTotal ? `${pct(rejected / detTotal)} of raw` : undefined} tone={rejected ? 'warn' : undefined} />
        <MetricCard label="Most common class" value={a.by_class[0] ? cls(a.by_class[0].class) : '—'} hint={a.by_class[0] ? `${a.by_class[0].n} alerts` : undefined} />
      </div>
      <Panel title="Litter over time" subtitle="Alerts and confirmed objects per day">
        <LineChart ariaLabel="Alerts and confirmed objects per day" labels={labels} area height={210}
          series={[{ name: 'Alerts', values: days.map((d) => byDay.get(d) ?? 0), color: 'var(--crit)' }, { name: 'Confirmed objects', values: days.map((d) => confByDay.get(d) ?? 0), color: 'var(--info)' }]} />
        <Legend items={[{ name: 'Alerts', color: 'var(--crit)' }, { name: 'Confirmed objects', color: 'var(--info)' }]} />
      </Panel>
      <div className="grid gap-3 lg:grid-cols-3">
        <Panel title="Litter by camera">
          {a.by_camera.length ? <BarList data={a.by_camera.map((r) => ({ label: r.camera_id, value: r.n }))} /> : <EmptyState compact title="No alerts" />}
        </Panel>
        <Panel title="Litter by class">
          {a.by_class.length ? <BarList data={a.by_class.map((r, i) => ({ label: cls(r.class), value: r.n, color: SERIES[i % SERIES.length] }))} /> : <EmptyState compact title="No alerts" />}
        </Panel>
        <Panel title="Severity & status">
          {a.alerts_total ? (
            <div className="flex items-center gap-4">
              <Donut ariaLabel="Alerts by severity" size={120}
                data={a.by_severity.map((s) => ({ label: s.severity, value: s.n, color: s.severity === 'HIGH' ? 'var(--crit)' : s.severity === 'MEDIUM' ? 'var(--warn)' : 'var(--info)' }))}
                center={<><span className="tabular text-lg font-semibold">{a.alerts_total}</span><span className="text-[10px] text-fg-3">alerts</span></>} />
              <div className="flex flex-1 flex-col gap-2 text-[12px]">
                {a.by_severity.map((s) => <div key={s.severity} className="flex justify-between"><span className="text-fg-2">{s.severity}</span><span className="tabular font-mono">{s.n}</span></div>)}
                <div className="my-1 border-t border-line" />
                {a.by_status.map((s) => <div key={s.status} className="flex justify-between"><span className="text-fg-2">{s.status.replace('_', ' ')}</span><span className="tabular font-mono">{s.n}</span></div>)}
              </div>
            </div>
          ) : <EmptyState compact title="No alerts" />}
        </Panel>
      </div>
      <div className="grid gap-3 lg:grid-cols-3">
        <Panel title="Hourly distribution" subtitle="Alerts by hour of day (UTC)">
          <Columns ariaLabel="Alerts by hour" data={Array.from({ length: 24 }, (_, h) => ({ label: String(h).padStart(2, '0'), value: a.by_hour.find((x) => x.hour === String(h).padStart(2, '0'))?.n ?? 0 }))} />
        </Panel>
        <Panel title="Daily distribution" subtitle="Alerts by day of week">
          <Columns ariaLabel="Alerts by day of week" data={dow} color="var(--exp)" />
        </Panel>
        <Panel title="Detection confidence" subtitle="Confidence of alerts (bins of 0.1)">
          <Columns ariaLabel="Alert confidence histogram" color="var(--ok)"
            data={Array.from({ length: 10 }, (_, i) => ({ label: (i / 10).toFixed(1), value: a.confidence_histogram.find((b) => Math.round(b.bin * 10) === i)?.n ?? 0 }))} />
        </Panel>
      </div>
      <div className="grid gap-3 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
        <Panel title="Confirmed vs rejected detections" subtitle="Raw detection rows per day by final status">
          <LineChart ariaLabel="Detections per day by status" labels={labels} height={180}
            series={['confirmed', 'candidate', 'weak', 'rejected'].map((s) => ({ name: s, color: STATUS_COLORS[s], values: days.map((d) => detDaily.get(d)?.[s] ?? 0) }))} />
          <Legend items={Object.entries(STATUS_COLORS).map(([name, color]) => ({ name, color }))} />
        </Panel>
        <Panel title="Why detections were rejected">
          {a.rejection_reasons.length ? (
            <>
              <StackBar parts={a.detections_by_status.map((s) => ({ label: s.status, value: s.n, color: STATUS_COLORS[s.status] }))} />
              <div className="mt-3"><BarList color="var(--crit)" data={a.rejection_reasons.map((r) => ({ label: (r.reason ?? 'unknown').replace(/_/g, ' '), value: r.n }))} /></div>
            </>
          ) : <EmptyState compact title="No rejected detections" />}
        </Panel>
      </div>
      <Panel title="Alert frequency" subtitle="Alerts per day">
        <Columns ariaLabel="Alerts per day" color="var(--crit)" data={days.map((d) => ({ label: d.slice(5), value: byDay.get(d) ?? 0 }))} />
      </Panel>
    </>
  )
}
