'use client'

import { useRouter, useSearchParams } from 'next/navigation'
import { useEffect, useState } from 'react'
import { Check, CheckCheck, Download, ShieldAlert, ShieldCheck, XCircle } from 'lucide-react'
import { url } from '../../api/client'
import { Api, type AlertQuery } from '../../api/endpoints'
import type { Alert, AlertStatus } from '../../api/types'
import { invalidate, useQuery } from '../../hooks/useQuery'
import { ago, className as cls, dateTime, dayEndIso, dayStartIso, num, pct } from '../../lib/format'
import { useApp } from '../../stores/app'
import { Badge, Button, Field, Input, PageHeader, Panel, Select, StatusBadge, Tabs } from '../../components/ui/primitives'
import { Drawer } from '../../components/ui/overlay'
import { DataTable } from '../../components/ui/table'
import { EmptyState, ErrorState } from '../../components/ui/states'
import { AspectBox, ImageOverlay } from '../../components/vision/Overlay'

const STATUS_TABS: { value: AlertStatus | 'all'; label: string }[] = [
  { value: 'new', label: 'New' }, { value: 'acknowledged', label: 'Acknowledged' }, { value: 'resolved', label: 'Resolved' },
  { value: 'false_positive', label: 'Rejected' }, { value: 'all', label: 'All' },
]
const PAGE = 50

export default function AlertsPage() {
  const params = useSearchParams()
  const router = useRouter()
  const openId = params.get('id')
  const [tab, setTab] = useState<AlertStatus | 'all'>('new')
  const [f, setF] = useState({ camera_id: '', class: '', severity: '', min_confidence: '', start: '', end: '' })
  const [offset, setOffset] = useState(0)
  const [selected, setSelected] = useState<Set<string | number>>(new Set())
  const cams = useQuery('cameras:list', Api.cameras)
  const classes = useQuery('classes', Api.classes)
  const q: AlertQuery = {
    limit: PAGE, offset, status: tab === 'all' ? undefined : tab, camera_id: f.camera_id || undefined, class: f.class || undefined,
    severity: f.severity || undefined, min_confidence: f.min_confidence ? Number(f.min_confidence) : undefined,
    start: f.start ? dayStartIso(f.start) : undefined, end: f.end ? dayEndIso(f.end) : undefined,
  }
  const key = `alerts:list:${JSON.stringify(q)}`
  const alerts = useQuery(key, () => Api.alerts(q), { poll: 10000 })
  const counts = Object.fromEntries(STATUS_TABS.map((t) => [t.value, useQuery(`alerts:count:${t.value}`, () => Api.alerts({ status: t.value === 'all' ? undefined : t.value, limit: 1 }), { poll: 15000 }).data?.total])) // eslint-disable-line react-hooks/rules-of-hooks
  const { toast } = useApp()
  useEffect(() => setOffset(0), [tab, JSON.stringify(f)]) // eslint-disable-line react-hooks/exhaustive-deps

  const bulk = async (status: AlertStatus) => {
    const ids = [...selected] as number[]
    const res = await Promise.allSettled(ids.map((id) => Api.updateAlert(id, { status })))
    const ok = res.filter((r) => r.status === 'fulfilled').length
    toast({ tone: ok === ids.length ? 'ok' : 'warn', title: `${ok}/${ids.length} alerts → ${status.replace('_', ' ')}` })
    setSelected(new Set())
    invalidate('alerts')
  }
  const exportQs = new URLSearchParams(Object.entries({ camera_id: f.camera_id, start: q.start ?? '', end: q.end ?? '' }).filter(([, v]) => v)).toString()

  return (
    <div className="flex flex-col gap-3">
      <PageHeader title="Alerts" description="Incidents are created only for confirmed litter (passed filters, temporal confirmation, deduplication). One physical object produces one alert; repeats increase “occurrences”."
        actions={<a href={url(`/api/export/alerts.csv${exportQs ? `?${exportQs}` : ''}`)}><Button icon={Download}>Export CSV</Button></a>} />
      <Panel noPad>
        <Tabs value={tab} onChange={setTab} tabs={STATUS_TABS.map((t) => ({ ...t, count: counts[t.value] }))} />
        <div className="grid grid-cols-2 gap-2 border-b border-line p-3 md:grid-cols-3 xl:grid-cols-6">
          <Field label="Camera"><Select value={f.camera_id} onChange={(e) => setF({ ...f, camera_id: e.target.value })}>
            <option value="">All</option>{cams.data?.cameras.map((c) => <option key={c.id} value={c.id}>{c.id}</option>)}<option value="upload">upload</option></Select></Field>
          <Field label="Class"><Select value={f.class} onChange={(e) => setF({ ...f, class: e.target.value })}>
            <option value="">All</option>{classes.data?.classes.filter((c) => c.in_active_model).map((c) => <option key={c.name} value={c.name}>{cls(c.name)}</option>)}</Select></Field>
          <Field label="Priority"><Select value={f.severity} onChange={(e) => setF({ ...f, severity: e.target.value })}>
            <option value="">All</option><option>HIGH</option><option>MEDIUM</option><option>LOW</option></Select></Field>
          <Field label="Min confidence"><Select value={f.min_confidence} onChange={(e) => setF({ ...f, min_confidence: e.target.value })}>
            <option value="">Any</option>{[0.5, 0.6, 0.7, 0.8, 0.9].map((v) => <option key={v} value={v}>≥ {v * 100}%</option>)}</Select></Field>
          <Field label="From"><Input type="date" value={f.start} onChange={(e) => setF({ ...f, start: e.target.value })} /></Field>
          <Field label="To"><Input type="date" value={f.end} onChange={(e) => setF({ ...f, end: e.target.value })} /></Field>
        </div>
        {alerts.error && !alerts.data ? <ErrorState error={alerts.error} onRetry={alerts.refetch} compact /> : (
          <DataTable<Alert> rows={alerts.data?.alerts ?? []} rowKey={(a) => a.id} selectedKey={openId ? Number(openId) : null}
            onRowClick={(a) => router.replace(`/alerts?id=${a.id}`, { scroll: false })} selectable selected={selected} onSelectedChange={setSelected}
            serverPage={{ offset, limit: PAGE, total: alerts.data?.total ?? 0, onChange: setOffset }}
            toolbar={selected.size > 0 ? <>
              <span className="text-[12px] text-fg-2">{selected.size} selected</span>
              <Button size="sm" icon={Check} onClick={() => bulk('acknowledged')}>Acknowledge</Button>
              <Button size="sm" icon={CheckCheck} onClick={() => bulk('resolved')}>Resolve</Button>
              <Button size="sm" icon={XCircle} variant="danger" onClick={() => bulk('false_positive')}>Reject</Button>
            </> : <span className="text-[12px] text-fg-3">{alerts.data ? `${num(alerts.data.total)} alerts` : 'Loading…'}</span>}
            empty={alerts.data ? <EmptyState icon={ShieldCheck} compact title={tab === 'new' ? 'No new alerts' : 'No alerts match'} body="No confirmed litter events have been recorded for this filter." /> : <div className="p-3"><div className="skeleton h-24 rounded-md" /></div>}
            columns={[
              { key: 'sev', header: 'Priority', cell: (a) => <Badge tone={a.severity === 'HIGH' ? 'crit' : a.severity === 'MEDIUM' ? 'warn' : 'info'} dot>{a.severity}</Badge>, sortValue: (a) => ({ HIGH: 3, MEDIUM: 2, LOW: 1 })[a.severity] },
              { key: 'snap', header: 'Snapshot', cell: (a) => a.crop_url ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={url(a.crop_url)} alt={`${a.class} crop`} loading="lazy" className="h-9 w-12 rounded-sm border border-line bg-video object-cover" />
              ) : <span className="text-fg-3">—</span> },
              { key: 'class', header: 'Object', cell: (a) => <span className="font-medium uppercase text-fg">{cls(a.class)}</span>, sortValue: (a) => a.class },
              { key: 'conf', header: 'Confidence', align: 'right', cell: (a) => <span className="font-mono">{pct(a.confidence, 0)}</span>, sortValue: (a) => a.confidence },
              { key: 'cam', header: 'Location / Camera', cell: (a) => <span className="font-mono">{a.camera_id}</span>, sortValue: (a) => a.camera_id },
              { key: 'ts', header: 'Timestamp', cell: (a) => <span title={dateTime(a.ts)}>{dateTime(a.ts)} <span className="text-fg-3">· {ago(a.ts)}</span></span>, sortValue: (a) => a.ts },
              { key: 'track', header: 'Track ID', cell: (a) => <span className="font-mono">{a.track_id ?? '—'}</span> },
              { key: 'occ', header: 'Seen', align: 'right', cell: (a) => `${a.occurrences}×`, hideable: true },
              { key: 'src', header: 'Source', cell: (a) => a.source_type ?? '—', hideable: true, defaultHidden: true },
              { key: 'model', header: 'Model', cell: (a) => <span className="font-mono text-[11px]">{a.model_version}</span>, hideable: true, defaultHidden: true },
              { key: 'status', header: 'Status', cell: (a) => <StatusBadge status={a.status} />, sortValue: (a) => a.status },
            ]} />
        )}
      </Panel>
      <AlertDrawer id={openId ? Number(openId) : null} onClose={() => router.replace('/alerts', { scroll: false })} />
    </div>
  )
}

export function AlertDrawer({ id, onClose }: { id: number | null; onClose: () => void }) {
  const { toast } = useApp()
  const a = useQuery(id ? `alerts:one:${id}` : null, () => Api.alert(id!))
  const tl = useQuery(id ? `alerts:timeline:${id}` : null, () => Api.alertTimeline(id!))
  const [notes, setNotes] = useState('')
  const [busy, setBusy] = useState<string | null>(null)
  const [dims, setDims] = useState<[number, number]>([0, 0])
  useEffect(() => { setNotes(a.data?.notes ?? ''); setDims([0, 0]) }, [a.data?.id]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!a.data?.snapshot_url) return
    const img = new Image()
    img.onload = () => setDims([img.naturalWidth, img.naturalHeight])
    img.src = url(a.data.snapshot_url)
  }, [a.data?.snapshot_url])

  const update = async (body: { status?: AlertStatus; notes?: string }, label: string) => {
    if (!id) return
    setBusy(label)
    try {
      await Api.updateAlert(id, body)
      toast({ tone: 'ok', title: `Alert #${id}: ${label}` })
      invalidate('alerts')
    } catch (e) {
      toast({ tone: 'crit', title: `Could not update alert #${id}`, body: (e as Error).message })
    } finally {
      setBusy(null)
    }
  }
  const al = a.data
  return (
    <Drawer open={id !== null} onClose={onClose} width="w-[720px]"
      title={al ? `Alert #${al.id} · ${cls(al.class).toUpperCase()}` : `Alert #${id}`}
      subtitle={al && <span className="flex items-center gap-2"><Badge tone={al.severity === 'HIGH' ? 'crit' : al.severity === 'MEDIUM' ? 'warn' : 'info'} dot>{al.severity}</Badge><StatusBadge status={al.status} /><span>{dateTime(al.ts)}</span></span>}
      footer={al && <>
        <Button icon={XCircle} variant="danger" loading={busy === 'rejected'} disabled={al.status === 'false_positive'} onClick={() => update({ status: 'false_positive' }, 'rejected')}>Reject (false positive)</Button>
        <Button icon={Check} loading={busy === 'acknowledged'} disabled={al.status !== 'new'} onClick={() => update({ status: 'acknowledged' }, 'acknowledged')}>Acknowledge</Button>
        <Button icon={CheckCheck} variant="primary" loading={busy === 'resolved'} disabled={al.status === 'resolved'} onClick={() => update({ status: 'resolved' }, 'resolved')}>Resolve</Button>
      </>}>
      {a.error ? <ErrorState error={a.error} onRetry={a.refetch} compact /> : !al ? <div className="skeleton aspect-video rounded-md" /> : (
        <div className="flex flex-col gap-4">
          {al.snapshot_url ? (
            <AspectBox width={dims[0] || 16} height={dims[1] || 9} className="overflow-hidden rounded-md border border-line">
              <ImageOverlay src={url(al.snapshot_url)} width={dims[0]} height={dims[1]} alt={`Snapshot of alert ${al.id}`} className="absolute inset-0"
                boxes={dims[0] ? [{ key: 1, bbox: al.bbox, color: 'var(--crit)', label: cls(al.class).toUpperCase(), sub: `${pct(al.confidence, 0)}  ID:${al.track_id ?? '—'}` }] : []} />
            </AspectBox>
          ) : <EmptyState compact icon={ShieldAlert} title="No snapshot stored" />}
          <div className="grid gap-4 md:grid-cols-[160px_1fr]">
            <div>
              <h3 className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-fg-3">Crop</h3>
              {al.crop_url ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={url(al.crop_url)} alt="Cropped litter object" className="w-full rounded-md border border-line bg-video object-contain" />
              ) : <span className="text-fg-3">—</span>}
            </div>
            <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-[13px]">
              {([
                ['Camera', <span key="c" className="font-mono">{al.camera_id}</span>], ['Location', al.location || '—'],
                ['Class', cls(al.class)], ['Confidence', pct(al.confidence, 1)],
                ['Track ID', al.track_id ?? '—'], ['Occurrences', `${al.occurrences}× (last ${ago(al.last_seen_ts)})`],
                ['Model', <span key="m" className="font-mono text-[12px]">{al.model_version}</span>], ['Source', al.source_type ?? '—'],
                ['Video time', al.video_time_sec != null ? `${al.video_time_sec.toFixed(2)} s` : '—'], ['Bounding box', <span key="b" className="font-mono text-[12px]">{al.bbox.map((v) => Math.round(v)).join(', ')}</span>],
              ] as [string, React.ReactNode][]).map(([k, v]) => (
                <div key={k} className="min-w-0"><dt className="text-[11px] uppercase tracking-wide text-fg-3">{k}</dt><dd className="truncate text-fg">{v}</dd></div>
              ))}
            </dl>
          </div>
          <div>
            <h3 className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-fg-3">Timeline (track observations)</h3>
            {tl.data?.observations.length ? (
              <div className="flex h-10 items-end gap-px rounded-md border border-line bg-panel-2 p-1" role="img" aria-label="Confidence of each observation">
                {tl.data.observations.map((o, i) => (
                  <span key={i} className="flex-1 rounded-t-[1px]" title={`${o.video_time_sec != null ? o.video_time_sec.toFixed(1) + 's' : o.ts} · ${o.status} · ${pct(o.confidence, 0)}`}
                    style={{ height: `${o.confidence * 100}%`, background: o.status === 'confirmed' ? 'var(--ok)' : o.status === 'candidate' ? 'var(--warn)' : 'var(--fg-3)' }} />
                ))}
              </div>
            ) : <p className="text-[12px] text-fg-3">{tl.data?.note ?? 'Loading…'}</p>}
          </div>
          <div>
            <label htmlFor="alert-notes" className="mb-1.5 block text-[11px] font-semibold uppercase tracking-wide text-fg-3">Operator notes</label>
            <textarea id="alert-notes" value={notes} onChange={(e) => setNotes(e.target.value)} rows={3} maxLength={2000}
              placeholder="e.g. cleaned by staff at 14:40; wind-blown from bin area"
              className="w-full rounded-md border border-line bg-panel-2 p-2 text-[13px] text-fg placeholder:text-fg-3 focus:border-info focus:outline-none" />
            <div className="mt-1.5 flex justify-end">
              <Button size="sm" disabled={notes === (al.notes ?? '')} loading={busy === 'notes saved'} onClick={() => update({ notes }, 'notes saved')}>Save notes</Button>
            </div>
          </div>
        </div>
      )}
    </Drawer>
  )
}
