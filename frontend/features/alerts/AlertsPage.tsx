'use client'

import { useRouter, useSearchParams } from 'next/navigation'
import { useEffect, useState } from 'react'
import { BellRing, Check, CheckCheck, CircleSlash, Eye, ShieldAlert, ShieldCheck, XCircle } from 'lucide-react'
import { url } from '../../api/client'
import { Api, type AlertQuery } from '../../api/endpoints'
import type { Alert, AlertStatus, DismissReason } from '../../api/types'
import { invalidate, useQuery } from '../../hooks/useQuery'
import { ago, className as cls, dateTime, dayEndIso, dayStartIso, num, pct } from '../../lib/format'
import { lockedFor, useApp } from '../../stores/app'
import { Badge, Button, Field, Input, PageHeader, Panel, Select, StatusBadge, Tabs } from '../../components/ui/primitives'
import { Drawer } from '../../components/ui/overlay'
import { DataTable } from '../../components/ui/table'
import { KpiCard } from '../../components/ui/kpi'
import { EmptyState, ErrorState } from '../../components/ui/states'
import { ExportCsvButton } from './ExportCsvButton'
import { AspectBox, ImageOverlay } from '../../components/vision/Overlay'

// Pilot mode (phase 2): the CCTV room Confirms or Dismisses each new alert; LOW severity goes to the review list.
type Tab = AlertStatus | 'all' | 'log'
const STATUS_TABS: { value: Tab; label: string }[] = [
  { value: 'new', label: 'CCTV room (new)' }, { value: 'review', label: 'Review list (low)' }, { value: 'confirmed', label: 'Confirmed' },
  { value: 'dismissed', label: 'Dismissed' }, { value: 'all', label: 'All' }, { value: 'log', label: 'Dismissal log' },
]
export const DISMISS_REASONS: { value: DismissReason; label: string }[] = [
  { value: 'shadow', label: 'Shadow' }, { value: 'carton', label: 'Carton / box' }, { value: 'stock', label: 'Shop stock / display' },
  { value: 'puddle', label: 'Puddle / stain' }, { value: 'person_vehicle', label: 'Person / vehicle' }, { value: 'duplicate', label: 'Duplicate' },
  { value: 'already_cleaned', label: 'Already cleaned' }, { value: 'other', label: 'Other' },
]
const PAGE = 50

export default function AlertsPage() {
  const params = useSearchParams()
  const router = useRouter()
  const openId = params.get('id')
  const [tab, setTab] = useState<Tab>('new')
  const [bulkReason, setBulkReason] = useState<DismissReason>('shadow')
  const [f, setF] = useState({ camera_id: '', class: '', severity: '', min_confidence: '', start: '', end: '' })
  const [offset, setOffset] = useState(0)
  const [selected, setSelected] = useState<Set<string | number>>(new Set())
  const cams = useQuery('cameras:list', Api.cameras)
  const classes = useQuery('classes', Api.classes)
  const q: AlertQuery = {
    limit: PAGE, offset, status: tab === 'all' || tab === 'log' ? undefined : tab, camera_id: f.camera_id || undefined, class: f.class || undefined,
    severity: f.severity || undefined, min_confidence: f.min_confidence ? Number(f.min_confidence) : undefined,
    start: f.start ? dayStartIso(f.start) : undefined, end: f.end ? dayEndIso(f.end) : undefined,
  }
  const key = `alerts:list:${JSON.stringify(q)}`
  const alerts = useQuery(tab === 'log' ? null : key, () => Api.alerts(q), { poll: 10000 })
  const counts = Object.fromEntries(STATUS_TABS.filter((t) => t.value !== 'log').map((t) => [t.value, useQuery(`alerts:count:${t.value}`, () => Api.alerts({ status: t.value === 'all' ? undefined : t.value, limit: 1 }), { poll: 15000 }).data?.total])) // eslint-disable-line react-hooks/rules-of-hooks
  const log = useQuery(tab === 'log' ? `alerts:dismissals:${offset}` : null, () => Api.dismissals(PAGE, offset), { poll: 15000 })
  const { toast, canOperate, role, loginOn } = useApp()
  const opLock = lockedFor('operator', role, loginOn)
  useEffect(() => setOffset(0), [tab, JSON.stringify(f)]) // eslint-disable-line react-hooks/exhaustive-deps

  const bulk = async (action: 'confirm' | 'dismiss') => {
    const ids = [...selected] as number[]
    const res = await Promise.allSettled(ids.map((id) => action === 'confirm' ? Api.confirmAlert(id) : Api.dismissAlert(id, bulkReason)))
    const ok = res.filter((r) => r.status === 'fulfilled').length
    toast({ tone: ok === ids.length ? 'ok' : 'warn', title: `${ok}/${ids.length} alerts ${action === 'confirm' ? 'confirmed' : `dismissed (${bulkReason.replace('_', ' ')})`}` })
    setSelected(new Set())
    invalidate('alerts')
  }
  const exportQs = new URLSearchParams(Object.entries({ camera_id: f.camera_id, start: q.start ?? '', end: q.end ?? '' }).filter(([, v]) => v)).toString()

  return (
    <div className="flex flex-col gap-4">
      <PageHeader title="Alerts" description="Pilot mode: an alert reaches the CCTV room only for confirmed litter inside the monitored region with severity MEDIUM or above, at most once per location per 10 minutes (video / live: after the object has stayed still for 30 s). LOW severity goes to the review list. Owner notifications are off. Confirm or Dismiss each alert; every dismissal is logged with its crop, full frame, reason and time."
        actions={<ExportCsvButton table="alerts" query={exportQs} />} />
      <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
        <KpiCard label="CCTV room" icon={BellRing} tone="crit" value={counts.new !== undefined ? num(counts.new) : null} loading={counts.new === undefined} sub="awaiting Confirm / Dismiss" onClick={() => setTab('new')} />
        <KpiCard label="Review list" icon={Eye} tone="info" value={counts.review !== undefined ? num(counts.review) : null} loading={counts.review === undefined} sub="low severity, no alert" onClick={() => setTab('review')} />
        <KpiCard label="Confirmed" icon={ShieldCheck} tone="ok" value={counts.confirmed !== undefined ? num(counts.confirmed) : null} loading={counts.confirmed === undefined} sub="real litter" onClick={() => setTab('confirmed')} />
        <KpiCard label="Dismissed" icon={CircleSlash} tone="neutral" value={counts.dismissed !== undefined ? num(counts.dismissed) : null} loading={counts.dismissed === undefined} sub="logged with reason" onClick={() => setTab('dismissed')} />
      </div>
      <Panel noPad>
        <Tabs value={tab} onChange={setTab} tabs={STATUS_TABS.map((t) => ({ ...t, count: counts[t.value] }))} />
        <div className="grid grid-cols-2 gap-3 border-b border-line bg-panel-2/40 px-4 py-3 md:grid-cols-3 xl:grid-cols-6">
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
        {tab === 'log' ? <DismissalLog data={log.data} error={log.error} offset={offset} onPage={setOffset} /> : alerts.error && !alerts.data ? <ErrorState error={alerts.error} onRetry={alerts.refetch} compact /> : (
          <DataTable<Alert> rows={alerts.data?.alerts ?? []} rowKey={(a) => a.id} selectedKey={openId ? Number(openId) : null}
            onRowClick={(a) => router.replace(`/alerts?id=${a.id}`, { scroll: false })} selectable selected={selected} onSelectedChange={setSelected}
            serverPage={{ offset, limit: PAGE, total: alerts.data?.total ?? 0, onChange: setOffset }}
            toolbar={selected.size > 0 ? <>
              <span className="text-[12px] text-fg-2">{selected.size} selected</span>
              <Button size="sm" icon={CheckCheck} variant="primary" disabled={!canOperate} title={opLock} onClick={() => bulk('confirm')}>Confirm</Button>
              <Select value={bulkReason} onChange={(e) => setBulkReason(e.target.value as DismissReason)} aria-label="Dismissal reason" className="w-44">
                {DISMISS_REASONS.map((r) => <option key={r.value} value={r.value}>{r.label}</option>)}
              </Select>
              <Button size="sm" icon={XCircle} variant="danger" disabled={!canOperate} title={opLock} onClick={() => bulk('dismiss')}>Dismiss</Button>
            </> : <span className="text-[12px] text-fg-3">{alerts.data ? `${num(alerts.data.total)} alerts` : 'Loading…'}</span>}
            empty={alerts.data ? <EmptyState icon={ShieldCheck} compact title={tab === 'new' ? 'No new alerts' : 'No alerts match'} body="No confirmed litter events have been recorded for this filter." /> : <div className="p-3"><div className="skeleton h-24 rounded-md" /></div>}
            columns={[
              { key: 'sev', header: 'Priority', cell: (a) => <Badge tone={a.severity === 'HIGH' ? 'crit' : a.severity === 'MEDIUM' ? 'warn' : 'info'} dot>{a.severity}</Badge>, sortValue: (a) => ({ HIGH: 3, MEDIUM: 2, LOW: 1 })[a.severity] },
              { key: 'snap', header: 'Snapshot', cell: (a) => a.crop_url ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={url(a.crop_url)} alt={`${a.class} crop`} loading="lazy" className="h-11 w-16 rounded-md border border-line bg-video object-cover" />
              ) : <span className="text-fg-3">—</span> },
              { key: 'class', header: 'Litter category', cell: (a) => <span className="inline-flex rounded-full border border-line bg-panel-2 px-2.5 py-0.5 text-[12.5px] font-medium capitalize text-fg">{cls(a.class)}</span>, sortValue: (a) => a.class },
              { key: 'conf', header: 'Confidence', align: 'right', cell: (a) => <span className="font-mono">{pct(a.confidence, 0)}</span>, sortValue: (a) => a.confidence },
              { key: 'cam', header: 'Location / Camera', cell: (a) => <span className="font-mono">{a.camera_id}</span>, sortValue: (a) => a.camera_id },
              { key: 'ts', header: 'Timestamp', cell: (a) => <span className="flex flex-col leading-tight"><span className="text-fg">{dateTime(a.ts)}</span><span className="text-[11px] text-fg-3">{ago(a.ts)}</span></span>, sortValue: (a) => a.ts },
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
  const { toast, canOperate, role, loginOn } = useApp()
  const opLock = lockedFor('operator', role, loginOn)
  const a = useQuery(id ? `alerts:one:${id}` : null, () => Api.alert(id!))
  const tl = useQuery(id ? `alerts:timeline:${id}` : null, () => Api.alertTimeline(id!))
  const [notes, setNotes] = useState('')
  const [reason, setReason] = useState<DismissReason>('shadow')
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
  const decide = async (action: 'confirm' | 'dismiss') => {
    if (!id) return
    setBusy(action)
    try {
      if (action === 'confirm') await Api.confirmAlert(id)
      else await Api.dismissAlert(id, reason, notes || undefined)
      toast({ tone: 'ok', title: `Alert #${id} ${action === 'confirm' ? 'confirmed' : `dismissed (${reason.replace('_', ' ')})`}` })
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
        <Select value={reason} onChange={(e) => setReason(e.target.value as DismissReason)} aria-label="Dismissal reason" className="w-44">
          {DISMISS_REASONS.map((r) => <option key={r.value} value={r.value}>{r.label}</option>)}
        </Select>
        <Button icon={XCircle} variant="danger" loading={busy === 'dismiss'} disabled={!canOperate || al.status === 'dismissed'} title={opLock} onClick={() => decide('dismiss')}>Dismiss</Button>
        <Button icon={Check} variant="primary" loading={busy === 'confirm'} disabled={!canOperate || al.status === 'confirmed'} title={opLock} onClick={() => decide('confirm')}>Confirm</Button>
        <Button icon={CheckCheck} variant="ghost" loading={busy === 'resolved'} disabled={!canOperate || al.status === 'resolved'} title={opLock} onClick={() => update({ status: 'resolved' }, 'resolved')}>Cleaned up</Button>
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
                ['Decision', al.decided_at ? `${al.status}${al.decision_reason ? ` (${al.decision_reason.replace('_', ' ')})` : ''} · ${dateTime(al.decided_at)}${al.decided_by ? ` by ${al.decided_by}` : ''}` : '—'],
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
              <Button size="sm" disabled={!canOperate || notes === (al.notes ?? '')} title={opLock} loading={busy === 'notes saved'} onClick={() => update({ notes }, 'notes saved')}>Save notes</Button>
            </div>
          </div>
        </div>
      )}
    </Drawer>
  )
}


/** Pilot dismissal log: every Dismiss with its crop, full frame (kept copies), reason, time and who. */
function DismissalLog({ data, error, offset, onPage }: {
  data?: { dismissals: import('../../api/types').AlertDismissal[]; total: number }; error: unknown; offset: number; onPage: (o: number) => void
}) {
  if (error && !data) return <ErrorState error={error as Error} compact />
  return (
    <DataTable rows={data?.dismissals ?? []} rowKey={(d) => d.id} dense
      serverPage={{ offset, limit: PAGE, total: data?.total ?? 0, onChange: onPage }}
      toolbar={<span className="text-[12px] text-fg-3">{data ? `${num(data.total)} dismissals` : 'Loading…'}</span>}
      empty={data ? <EmptyState icon={ShieldCheck} compact title="No dismissals yet" body="Each Dismiss is logged here with its crop, full frame, reason and time." /> : <div className="p-3"><div className="skeleton h-24 rounded-md" /></div>}
      columns={[
        { key: 'ts', header: 'Dismissed', cell: (d) => <span className="flex flex-col leading-tight"><span className="text-fg">{dateTime(d.ts)}</span><span className="text-[11px] text-fg-3">{d.actor ?? '—'}</span></span>, sortValue: (d) => d.ts },
        { key: 'crop', header: 'Crop', cell: (d) => d.crop_url ? (
          // eslint-disable-next-line @next/next/no-img-element
          <a href={url(d.crop_url)} target="_blank" rel="noreferrer"><img src={url(d.crop_url)} alt={`${d.class} crop`} loading="lazy" className="h-11 w-16 rounded-md border border-line bg-video object-cover" /></a>
        ) : <span className="text-fg-3">—</span> },
        { key: 'full', header: 'Full frame', cell: (d) => d.full_url ? <a href={url(d.full_url)} target="_blank" rel="noreferrer" className="text-info underline-offset-2 hover:underline">open</a> : <span className="text-fg-3">—</span> },
        { key: 'reason', header: 'Reason', cell: (d) => <Badge tone="neutral">{DISMISS_REASONS.find((r) => r.value === d.reason)?.label ?? d.reason}</Badge>, sortValue: (d) => d.reason },
        { key: 'note', header: 'Note', cell: (d) => <span className="text-fg-2">{d.note ?? ''}</span> },
        { key: 'alert', header: 'Alert', cell: (d) => <span className="font-mono">#{d.alert_id}</span>, sortValue: (d) => d.alert_id },
        { key: 'class', header: 'Class', cell: (d) => cls(d.class), sortValue: (d) => d.class },
        { key: 'conf', header: 'Confidence', align: 'right', cell: (d) => <span className="font-mono">{pct(d.confidence, 0)}</span>, sortValue: (d) => d.confidence },
        { key: 'cam', header: 'Camera', cell: (d) => <span className="font-mono">{d.camera_id}</span> },
        { key: 'alert_ts', header: 'Alert time', cell: (d) => dateTime(d.alert_ts), hideable: true },
      ]} />
  )
}
