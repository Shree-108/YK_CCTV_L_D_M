'use client'

// Litter events over time (GET /api/litter/analytics) with granularity and store / camera filters.
import { useState } from 'react'
import { CalendarClock, Eye, ScanSearch, Trash2 } from 'lucide-react'
import { Api } from '../../api/endpoints'
import type { Granularity } from '../../api/types'
import { useQuery } from '../../hooks/useQuery'
import { className as cls, num } from '../../lib/format'
import { bucketLabel } from '../../lib/litter'
import { Panel, Segmented } from '../../components/ui/primitives'
import { Legend, LineChart } from '../../components/ui/charts'
import { KpiCard } from '../../components/ui/kpi'
import { EmptyState, ErrorState, LoadingState } from '../../components/ui/states'
import { StoreCameraSelect, type StoreCamera } from '../stores/StoreCameraSelect'
import { ByClassPanel, CameraTable, HourlyTodayPanel, StoreTable } from './litterCharts'

export function LitterAnalyticsSection({ start, end }: { start: string; end: string }) {
  const [who, setWho] = useState<StoreCamera>({ store_id: '', camera_id: '' })
  const [granularity, setGranularity] = useState<Granularity>('day')
  const q = { start, end, granularity, store_id: who.store_id || undefined, camera_id: who.camera_id || undefined }
  const a = useQuery(`litter:analytics:page:${JSON.stringify(q)}`, () => Api.litterAnalytics(q), { poll: 60000 })
  const d = a.data
  return (
    <section aria-labelledby="litter-h" className="flex flex-col gap-4">
      <Panel title={<span id="litter-h">Litter events</span>} subtitle="Unique litter events and frame observations (operator-rejected events excluded)"
        actions={<Segmented size="sm" label="Granularity" value={granularity} onChange={setGranularity}
          options={[{ value: 'hour', label: 'Hour' }, { value: 'day', label: 'Day' }, { value: 'week', label: 'Week' }]} />}>
        <StoreCameraSelect columns={2} value={who} onChange={setWho} />
      </Panel>
      {a.error && !d ? <Panel><ErrorState error={a.error} onRetry={a.refetch} compact /></Panel> : !d ? <Panel><LoadingState rows={4} /></Panel> : (
        <>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <KpiCard label="Litter events" icon={Trash2} tone="info" value={num(d.totals.unique_events)} sub="unique objects in the period" />
            <KpiCard label="Frame observations" icon={Eye} value={num(d.totals.frame_observations)} sub="sightings, not objects" />
            <KpiCard label="Events today" icon={CalendarClock} tone="crit" value={num(d.totals.events_today)} />
            <KpiCard label="Most frequent class" icon={ScanSearch} value={d.most_frequent_class ? cls(d.most_frequent_class) : '—'} />
          </div>
          <Panel title="Trend" subtitle={`Per ${granularity}`}>
            {d.series.length === 0 ? <EmptyState compact title="No litter events in this period" /> : (
              <>
                <LineChart ariaLabel={`Litter events and frame observations per ${granularity}`} height={210} labels={d.series.map((p) => bucketLabel(p.bucket))}
                  series={[{ name: 'Litter events', values: d.series.map((p) => p.events), color: 'var(--crit)' },
                    { name: 'Frame observations', values: d.series.map((p) => p.observations), color: 'var(--info)', dashed: true }]} />
                <Legend items={[{ name: 'Litter events (unique)', color: 'var(--crit)' }, { name: 'Frame observations', color: 'var(--info)' }]} />
              </>
            )}
          </Panel>
          <div className="grid gap-4 lg:grid-cols-2">
            <ByClassPanel a={d} />
            <HourlyTodayPanel a={d} />
          </div>
          <div className="grid gap-4 lg:grid-cols-2">
            <StoreTable a={d} />
            <CameraTable a={d} />
          </div>
        </>
      )}
    </section>
  )
}
