'use client'

import Link from 'next/link'
import { useMemo } from 'react'
import { AlertTriangle, CheckCircle2, FileVideo, ImageIcon, ScanSearch, ShieldAlert, Sparkles, Trash2 } from 'lucide-react'
import { url } from '../../api/client'
import { Api } from '../../api/endpoints'
import type { Alert } from '../../api/types'
import { useQuery } from '../../hooks/useQuery'
import { ago, className as cls, dayStartIso, num, pct, todayLocal } from '../../lib/format'
import { Badge, Button, PageHeader, Panel, StatusBadge } from '../../components/ui/primitives'
import { BarList, LineChart } from '../../components/ui/charts'
import { KpiCard } from '../../components/ui/kpi'
import { EmptyState, QueryView } from '../../components/ui/states'

const DAYS = 14

export default function DashboardPage() {
  const start = todayLocal(-(DAYS - 1))
  const a = useQuery(`analytics:overview:${start}`, () => Api.analytics({ start: dayStartIso(start) }), { poll: 30000 })
  const recent = useQuery('alerts:recent', () => Api.alerts({ limit: 8 }), { poll: 15000 })
  const health = useQuery('health', Api.health, { poll: 30000 })
  const days = useMemo(() => Array.from({ length: DAYS }, (_, i) => todayLocal(i - DAYS + 1)), [])
  const byDay = useMemo(() => new Map(a.data?.by_day.map((d) => [d.day, d.n])), [a.data])
  const high = a.data?.by_severity.find((s) => s.severity === 'HIGH')?.n ?? 0
  const top = a.data?.by_class[0]

  return (
    <div className="flex flex-col gap-4">
      <PageHeader title="Overview" description={`Litter detected in uploaded CCTV images and videos, last ${DAYS} days.`}
        actions={<>
          <Link href="/detect/image"><Button icon={ImageIcon}>Analyse image</Button></Link>
          <Link href="/detect/video"><Button variant="primary" icon={FileVideo}>Analyse video</Button></Link>
        </>} />

      <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
        <KpiCard label="Total litter detections" icon={Trash2} tone="info" loading={a.loading} value={a.data ? num(a.data.alerts_total) : null} sub="confirmed incidents" />
        <KpiCard label="High priority" icon={ShieldAlert} tone="crit" loading={a.loading} value={a.data ? num(high) : null} sub="incidents" />
        <KpiCard label="Most frequent type" icon={ScanSearch} loading={a.loading} value={top ? cls(top.class) : a.data ? '—' : null} sub={top ? `${top.n} incidents` : 'no data yet'} />
        <KpiCard label="Detection model" icon={Sparkles} tone="exp" loading={health.loading} value={health.data?.model?.version ?? (health.data ? '—' : null)} sub={health.data?.model ? `${health.data.model.device.toUpperCase()} · ${health.data.model.classes.length} classes` : health.error ? 'backend offline' : ''} />
      </div>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
        <Panel title="Detections by date">
          <QueryView q={a} isEmpty={(d) => d.alerts_total === 0}
            empty={<EmptyState compact title="No detections yet" body="Upload a CCTV image or video to start." />}>
            {() => <LineChart ariaLabel="Detections per day" area height={220} labels={days.map((d) => d.slice(5))}
              series={[{ name: 'Detections', values: days.map((d) => byDay.get(d) ?? 0), color: 'var(--info)' }]} />}
          </QueryView>
        </Panel>
        <Panel title="Top litter types">
          <QueryView q={a} isEmpty={(d) => d.by_class.length === 0} empty={<EmptyState compact title="No data" />}>
            {(d) => <BarList data={d.by_class.slice(0, 6).map((r) => ({ label: cls(r.class), value: r.n }))} />}
          </QueryView>
        </Panel>
      </div>

      <Panel title="Recent alerts" noPad actions={<Link href="/alerts" className="text-[12px] text-info hover:underline">View all →</Link>}>
        <QueryView q={recent} isEmpty={(d) => d.alerts.length === 0}
          empty={<EmptyState icon={CheckCircle2} compact title="No alerts" body="No litter incidents have been recorded." />}>
          {(d) => <AlertMiniList alerts={d.alerts} />}
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
            <Badge tone={al.severity === 'HIGH' ? 'crit' : al.severity === 'MEDIUM' ? 'warn' : 'info'}>{al.severity[0] + al.severity.slice(1).toLowerCase()}</Badge>
            <StatusBadge status={al.status} />
          </Link>
        </li>
      ))}
    </ul>
  )
}
