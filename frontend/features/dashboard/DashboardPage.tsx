'use client'

// Overview: litter events and reporting from GET /api/litter/analytics and /api/litter/analyses. "Litter events"
// are unique objects; "frame observations" are sightings in analysed frames - always shown apart, never summed.
import Link from 'next/link'
import { useMemo } from 'react'
import { ArrowRight, CalendarClock, Eye, FileVideo, ImageIcon, ListChecks, ScanSearch, Timer, Trash2, XCircle } from 'lucide-react'
import { Api } from '../../api/endpoints'
import { useQuery } from '../../hooks/useQuery'
import { className as cls, duration, num, todayLocal } from '../../lib/format'
import { seriesForDays } from '../../lib/litter'
import { Button, PageHeader, Panel } from '../../components/ui/primitives'
import { Legend, LineChart } from '../../components/ui/charts'
import { KpiCard } from '../../components/ui/kpi'
import { EmptyState, ErrorState, LoadingState, QueryView } from '../../components/ui/states'
import { ByClassPanel, CameraTable, HourlyTodayPanel, ProcessingPanel, RecentAnalysesTable, StoreTable } from '../analytics/litterCharts'

const DAYS = 14

export default function DashboardPage() {
  const a = useQuery('litter:analytics:overview', () => Api.litterAnalytics({ granularity: 'day' }), { poll: 30000 })
  const recent = useQuery('litter:analyses:recent', () => Api.litterAnalyses(10), { poll: 30000 })
  const days = useMemo(() => Array.from({ length: DAYS }, (_, i) => todayLocal(i - DAYS + 1)), [])
  const trend = useMemo(() => seriesForDays(a.data?.series ?? [], days), [a.data, days])
  const t = a.data?.totals
  const loading = a.loading && !a.data

  return (
    <div className="flex flex-col gap-5">
      <PageHeader title="Overview" description="Litter detected in CCTV images and videos. A litter event is one unique object; frame observations count every analysed frame it was seen in."
        actions={<>
          <Link href="/detect/image"><Button icon={ImageIcon}>Analyse image</Button></Link>
          <Link href="/detect/video"><Button variant="primary" icon={FileVideo}>Analyse video</Button></Link>
        </>} />

      {a.error && !a.data && <Panel><ErrorState error={a.error} onRetry={a.refetch} title="Litter analytics unavailable" compact /></Panel>}
      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 2xl:grid-cols-6">
        <KpiCard label="Litter events today" icon={CalendarClock} tone="crit" loading={loading} value={t ? num(t.events_today) : null} sub="unique objects" />
        <KpiCard label="Total litter events" icon={Trash2} tone="info" loading={loading} value={t ? num(t.unique_events) : null} sub="unique objects, all time" />
        <KpiCard label="Frame observations" icon={Eye} loading={loading} value={t ? num(t.frame_observations) : null} sub="sightings, not objects" />
        <KpiCard label="Most frequent class" icon={ScanSearch} loading={loading} value={a.data ? (a.data.most_frequent_class ? cls(a.data.most_frequent_class) : '—') : null}
          sub={a.data?.by_class[0] ? `${num(a.data.by_class[0].events)} events` : 'no data yet'} />
        <KpiCard label="Average visible time" icon={Timer} loading={loading} value={t ? (t.avg_visible_sec != null ? duration(t.avg_visible_sec) : '—') : null} sub="per litter event" />
        <KpiCard label="Analyses failed" icon={XCircle} tone={t?.analyses_failed ? 'warn' : 'neutral'} loading={loading} value={t ? num(t.analyses_failed) : null}
          sub={t ? `of ${num(t.analyses)} analyses` : ''} />
      </div>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
        <Panel title="Daily trend" subtitle={`Last ${DAYS} days`}>
          <QueryView q={a} isEmpty={(d) => d.totals.unique_events === 0 && d.totals.frame_observations === 0}
            empty={<EmptyState compact title="No litter events yet" body="Analyse a CCTV image or video to start." />}>
            {() => (
              <>
                <LineChart ariaLabel="Litter events and frame observations per day" height={230} labels={days.map((d) => d.slice(5))}
                  series={[{ name: 'Litter events', values: trend.events, color: 'var(--crit)' }, { name: 'Frame observations', values: trend.observations, color: 'var(--info)', dashed: true }]} />
                <Legend items={[{ name: 'Litter events (unique)', color: 'var(--crit)' }, { name: 'Frame observations', color: 'var(--info)' }]} />
              </>
            )}
          </QueryView>
        </Panel>
        {a.data ? <ByClassPanel a={a.data} /> : <Panel title="Litter events by class"><QueryView q={a}>{() => null}</QueryView></Panel>}
      </div>

      {a.data && (
        <>
          <div className="grid gap-4 xl:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
            <HourlyTodayPanel a={a.data} />
            <ProcessingPanel a={a.data} />
          </div>
          <div className="grid gap-4 xl:grid-cols-2">
            <StoreTable a={a.data} />
            <CameraTable a={a.data} />
          </div>
        </>
      )}

      <Panel title="Recent analyses" noPad icon={ListChecks}
        actions={<Link href="/analytics" className="inline-flex items-center gap-1 text-[12px] font-medium text-fg-2 hover:text-fg">Analytics <ArrowRight className="size-3.5" aria-hidden /></Link>}>
        {recent.error && !recent.data ? <ErrorState error={recent.error} onRetry={recent.refetch} compact />
          : !recent.data ? <div className="p-4"><LoadingState rows={3} /></div> : <RecentAnalysesTable rows={recent.data.analyses} />}
      </Panel>
    </div>
  )
}
