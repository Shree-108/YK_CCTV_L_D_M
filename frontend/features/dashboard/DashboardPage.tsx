'use client'

import Link from 'next/link'
import { useMemo } from 'react'
import {
  AlertTriangle, ArrowRight, BellRing, CheckCircle2, FileVideo, ImageIcon, Images, ScanSearch, ShieldAlert, Sparkles, Trash2,
} from 'lucide-react'
import { url } from '../../api/client'
import { Api } from '../../api/endpoints'
import type { Alert } from '../../api/types'
import { useQuery } from '../../hooks/useQuery'
import { ago, className as cls, dateTime, dayStartIso, num, pct, todayLocal } from '../../lib/format'
import { Badge, Button, PageHeader, Panel, StatusBadge } from '../../components/ui/primitives'
import { BarList, LineChart } from '../../components/ui/charts'
import { KpiCard } from '../../components/ui/kpi'
import { EmptyState, QueryView } from '../../components/ui/states'

const DAYS = 14
const JOB_LIMIT = 500   // backend maximum for /api/jobs; a full page means "at least this many"

const sevTone = (s: Alert['severity']) => (s === 'HIGH' ? 'crit' : s === 'MEDIUM' ? 'warn' : 'info')
const sevLabel = (s: Alert['severity']) => s[0] + s.slice(1).toLowerCase()

export default function DashboardPage() {
  const start = todayLocal(-(DAYS - 1))
  const a = useQuery(`analytics:overview:${start}`, () => Api.analytics({ start: dayStartIso(start) }), { poll: 30000 })
  const recent = useQuery('alerts:recent', () => Api.alerts({ limit: 8 }), { poll: 15000 })
  const pending = useQuery('alerts:newcount', () => Api.alerts({ status: 'new', limit: 1 }), { poll: 15000 })
  const health = useQuery('health', Api.health, { poll: 30000 })
  const jobs = useQuery('jobs:overview', () => Api.jobs(JOB_LIMIT), { poll: 30000 })
  const days = useMemo(() => Array.from({ length: DAYS }, (_, i) => todayLocal(i - DAYS + 1)), [])
  const byDay = useMemo(() => new Map(a.data?.by_day.map((d) => [d.day, d.n])), [a.data])
  const high = a.data?.by_severity.find((s) => s.severity === 'HIGH')?.n ?? 0
  const top = a.data?.by_class[0]
  const videoJobs = useMemo(() => (jobs.data?.jobs ?? []).filter((j) => j.kind === 'video'), [jobs.data])
  const videosDone = videoJobs.filter((j) => j.status === 'done').length
  const capped = (jobs.data?.jobs.length ?? 0) >= JOB_LIMIT
  const evidence = (recent.data?.alerts ?? []).filter((al) => al.snapshot_url || al.crop_url).slice(0, 6)

  return (
    <div className="flex flex-col gap-5">
      <PageHeader title="Overview" description={`Litter detected in CCTV images, videos and live cameras over the last ${DAYS} days.`}
        actions={<>
          <Link href="/detect/image"><Button icon={ImageIcon}>Analyse image</Button></Link>
          <Link href="/detect/video"><Button variant="primary" icon={FileVideo}>Analyse video</Button></Link>
        </>} />

      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 2xl:grid-cols-6">
        <KpiCard label="Total litter detections" icon={Trash2} tone="info" loading={a.loading && !a.data} value={a.data ? num(a.data.alerts_total) : null} sub={`incidents · last ${DAYS} days`} />
        <KpiCard label="Pending alerts" icon={BellRing} tone={pending.data?.total ? 'warn' : 'ok'} loading={pending.loading && !pending.data}
          value={pending.data ? num(pending.data.total) : null} sub="awaiting review" />
        <KpiCard label="High-priority incidents" icon={ShieldAlert} tone="crit" loading={a.loading && !a.data} value={a.data ? num(high) : null} sub={`last ${DAYS} days`} />
        <KpiCard label="Videos processed" icon={FileVideo} tone="neutral" loading={jobs.loading && !jobs.data}
          value={jobs.data ? `${num(videosDone)}${capped ? '+' : ''}` : null} sub={jobs.data ? `${videoJobs.length - videosDone} queued, running or failed` : jobs.error ? 'unavailable' : ''} />
        <KpiCard label="Most frequent type" icon={ScanSearch} loading={a.loading && !a.data} value={top ? cls(top.class) : a.data ? '—' : null} sub={top ? `${num(top.n)} incidents` : 'no data yet'} />
        <KpiCard label="Active detection model" icon={Sparkles} tone="exp" loading={health.loading && !health.data} value={health.data?.model?.version ?? (health.data ? '—' : null)}
          sub={health.data?.model ? `${health.data.model.device.toUpperCase()} · ${health.data.model.classes.length} classes` : health.error ? 'backend offline' : health.data ? 'no model loaded' : ''} />
      </div>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
        <Panel title="Detection activity" subtitle={`Incidents per day · last ${DAYS} days`}>
          <QueryView q={a} isEmpty={(d) => d.alerts_total === 0}
            empty={<EmptyState compact title="No detections yet" body="Upload a CCTV image or video to start." />}>
            {() => <LineChart ariaLabel="Detections per day" area height={230} labels={days.map((d) => d.slice(5))}
              series={[{ name: 'Detections', values: days.map((d) => byDay.get(d) ?? 0), color: 'var(--info)' }]} />}
          </QueryView>
        </Panel>
        <Panel title="Top litter categories" subtitle="By number of incidents">
          <QueryView q={a} isEmpty={(d) => d.by_class.length === 0} empty={<EmptyState compact title="No data" />}>
            {(d) => <BarList data={d.by_class.slice(0, 7).map((r) => ({ label: cls(r.class), value: r.n }))} />}
          </QueryView>
        </Panel>
      </div>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <Panel title="Recent alerts" noPad icon={AlertTriangle}
          actions={<Link href="/alerts" className="inline-flex items-center gap-1 text-[12px] font-medium text-fg-2 hover:text-fg">View all <ArrowRight className="size-3.5" aria-hidden /></Link>}>
          <QueryView q={recent} isEmpty={(d) => d.alerts.length === 0}
            empty={<EmptyState icon={CheckCircle2} compact title="No alerts" body="No litter incidents have been recorded." />}>
            {(d) => <AlertMiniList alerts={d.alerts} />}
          </QueryView>
        </Panel>
        <Panel title="Latest evidence" subtitle="Snapshots from the most recent alerts" icon={Images}>
          <QueryView q={recent} isEmpty={() => evidence.length === 0}
            empty={<EmptyState icon={Images} compact title="No evidence images yet" body="Alerts store a snapshot and a crop of the detected litter." />}>
            {() => (
              <ul className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                {evidence.map((al) => (
                  <li key={al.id}>
                    <Link href={`/alerts?id=${al.id}`} className="group block overflow-hidden rounded-lg border border-line bg-panel-2 hover:border-line-strong">
                      <div className="relative aspect-[4/3] bg-video">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img src={url((al.crop_url ?? al.snapshot_url)!)} alt={`${cls(al.class)} evidence, alert ${al.id}`} loading="lazy"
                          className="size-full object-cover transition-opacity group-hover:opacity-90" />
                        <Badge tone={sevTone(al.severity)} className="absolute left-1.5 top-1.5 bg-black/60 backdrop-blur">{sevLabel(al.severity)}</Badge>
                      </div>
                      <div className="px-2.5 py-2">
                        <div className="truncate text-[12.5px] font-medium text-fg">{cls(al.class)} <span className="font-normal text-fg-3">· {pct(al.confidence, 0)}</span></div>
                        <div className="truncate text-[11px] text-fg-3">{al.camera_id} · {ago(al.ts)}</div>
                      </div>
                    </Link>
                  </li>
                ))}
              </ul>
            )}
          </QueryView>
        </Panel>
      </div>

      <Panel title="Recent video analyses" noPad icon={FileVideo}
        actions={<Link href="/detect/video" className="inline-flex items-center gap-1 text-[12px] font-medium text-fg-2 hover:text-fg">Video Analysis <ArrowRight className="size-3.5" aria-hidden /></Link>}>
        <QueryView q={jobs} isEmpty={() => videoJobs.length === 0}
          empty={<EmptyState icon={FileVideo} compact title="No videos processed yet" body="Upload a CCTV video in Video Analysis." />}>
          {() => (
            <ul className="divide-y divide-line">
              {videoJobs.slice(0, 5).map((j) => (
                <li key={j.id} className="flex items-center gap-3 px-4 py-2.5">
                  <span className="flex size-9 shrink-0 items-center justify-center rounded-md border border-line bg-panel-2"><FileVideo className="size-4 text-fg-3" aria-hidden /></span>
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-[13px] font-medium text-fg">{j.input_name}</div>
                    <div className="truncate text-[11px] text-fg-3">{j.camera_id ?? 'upload'} · {dateTime(j.created_at)}</div>
                  </div>
                  {j.status === 'running' && <span className="tabular text-[12px] text-fg-2">{pct(j.progress, 0)}</span>}
                  <StatusBadge status={j.status} />
                </li>
              ))}
            </ul>
          )}
        </QueryView>
      </Panel>
    </div>
  )
}

export function AlertMiniList({ alerts }: { alerts: Alert[] }) {
  return (
    <ul className="divide-y divide-line">
      {alerts.map((al) => (
        <li key={al.id}>
          <Link href={`/alerts?id=${al.id}`} className="flex items-center gap-3 px-4 py-2.5 hover:bg-hover/60">
            {al.crop_url ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={url(al.crop_url)} alt="" loading="lazy" className="size-10 shrink-0 rounded-md border border-line bg-video object-cover" />
            ) : <div className="flex size-10 shrink-0 items-center justify-center rounded-md border border-line bg-panel-2"><AlertTriangle className="size-4 text-fg-3" /></div>}
            <div className="min-w-0 flex-1">
              <div className="truncate text-[13px] font-medium text-fg">{cls(al.class)} <span className="font-normal text-fg-3">· {pct(al.confidence, 0)}</span></div>
              <div className="truncate text-[11px] text-fg-3">{al.camera_id} · {ago(al.ts)}</div>
            </div>
            <Badge tone={sevTone(al.severity)} dot>{sevLabel(al.severity)}</Badge>
            <StatusBadge status={al.status} />
          </Link>
        </li>
      ))}
    </ul>
  )
}
