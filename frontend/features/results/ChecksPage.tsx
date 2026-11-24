'use client'

// Results: ONE ROW = ONE CHECK of one camera (one screenshot of a four-screenshot analysis), with exactly the fields of
// the importer CSV (backend/services/garbage_export.py, via /api/results/checks). Clean and view-blocked checks are
// listed as well as litter. The recording time is shown as recorded (+05:30); a row without a trustworthy recording
// time says "time unknown" - a processing time is never shown in its place.
import { useEffect, useState } from 'react'
import { AlertTriangle, CameraOff, Clock, EyeOff, ListChecks, ShieldCheck } from 'lucide-react'
import { url } from '../../api/client'
import { Api } from '../../api/endpoints'
import type { ResultCheckRow, ResultChecksQuery } from '../../api/resultsTypes'
import { useQuery } from '../../hooks/useQuery'
import { className as cls, dayEndIso, dayStartIso, num } from '../../lib/format'
import { bundleBadge, cameraText, checkStatusInfo, recordingTimeText, resultsQuery } from '../../lib/results'
import { Badge, Field, Input, PageHeader, Panel, Select } from '../../components/ui/primitives'
import { DataTable } from '../../components/ui/table'
import { KpiCard } from '../../components/ui/kpi'
import { EmptyState, ErrorState } from '../../components/ui/states'
import { ExportCsvButton } from '../alerts/ExportCsvButton'

const PAGE = 50
const EMPTY = { store: '', camera: '', status: '', litter_class: '', min_confidence: '', date_from: '', date_to: '' }

export default function ChecksPage() {
  const [f, setF] = useState(EMPTY)
  const [offset, setOffset] = useState(0)
  useEffect(() => setOffset(0), [JSON.stringify(f)]) // eslint-disable-line react-hooks/exhaustive-deps
  const q: ResultChecksQuery = {
    limit: PAGE, offset, store: f.store || undefined, camera: f.camera || undefined, status: f.status || undefined,
    litter_class: f.litter_class || undefined, min_confidence: f.min_confidence ? Number(f.min_confidence) : undefined,
    date_from: f.date_from || undefined, date_to: f.date_to || undefined,
  }
  const res = useQuery(`results:checks:${JSON.stringify(q)}`, () => Api.resultChecks(q), { poll: 15000 })
  const s = res.data?.summary
  const opts = res.data?.options
  const importerQs = resultsQuery({ store: f.store, camera: f.camera, status: f.status, litter_class: f.litter_class,
    min_confidence: f.min_confidence, date_from: f.date_from, date_to: f.date_to })
  const legacyQs = resultsQuery({ camera_id: f.camera, start: f.date_from ? dayStartIso(f.date_from) : '', end: f.date_to ? dayEndIso(f.date_to) : '' })
  const pick = (status: string) => setF({ ...f, status: f.status === status ? '' : status })

  return (
    <div className="flex flex-col gap-4">
      <PageHeader title="Results"
        description="One row per check of one camera (one screenshot of a four-screenshot analysis), with exactly the columns of the importer CSV. Clean and view-blocked checks are listed too. Times are recording times in India time (+05:30)."
        actions={<ExportCsvButton table="detections" query={legacyQs} importerQuery={importerQs} />} />
      <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
        <KpiCard label="Total checks" icon={ListChecks} tone="info" value={s ? num(s.total) : null} loading={!s} sub={s ? `${num(s.import_ready)} import-ready` : undefined} onClick={() => setF({ ...f, status: '' })} />
        <KpiCard label="litter_detected" icon={AlertTriangle} tone="crit" value={s ? num(s.litter_detected) : null} loading={!s} sub="confirmed litter in the frame" onClick={() => pick('litter_detected')} />
        <KpiCard label="clean" icon={ShieldCheck} tone="ok" value={s ? num(s.clean) : null} loading={!s} sub="no confirmed litter" onClick={() => pick('clean')} />
        <KpiCard label="view_blocked" icon={EyeOff} tone="warn" value={s ? num(s.view_blocked) : null} loading={!s} sub="picture not usable" onClick={() => pick('view_blocked')} />
      </div>
      {s && (s.time_unknown > 0 || s.no_camera > 0) && (
        <div role="status" className="flex flex-col gap-1 rounded-lg border border-warn/40 bg-warn/10 px-4 py-2.5 text-[13px] text-fg">
          {s.time_unknown > 0 && <span className="flex items-center gap-2"><Clock className="size-4 text-warn" aria-hidden />{num(s.time_unknown)} check(s) have no trustworthy recording time: shown as “time unknown”, left out of the importer CSV. Start the analysis with a recording start (or use an NVR file name) to fix this.</span>}
          {s.no_camera > 0 && <span className="flex items-center gap-2"><CameraOff className="size-4 text-warn" aria-hidden />{num(s.no_camera)} check(s) have no camera assigned (camera “upload” or empty): left out of the importer CSV.</span>}
        </div>
      )}
      <Panel noPad>
        <div className="grid grid-cols-2 gap-3 border-b border-line bg-panel-2/40 px-4 py-3 md:grid-cols-4 xl:grid-cols-7">
          <Field label="Store"><Select value={f.store} onChange={(e) => setF({ ...f, store: e.target.value })}>
            <option value="">All</option>{opts?.stores.map((v) => <option key={v} value={v}>{v}</option>)}</Select></Field>
          <Field label="Camera"><Select value={f.camera} onChange={(e) => setF({ ...f, camera: e.target.value })}>
            <option value="">All</option>{opts?.cameras.map((v) => <option key={v} value={v}>{v}</option>)}</Select></Field>
          <Field label="Status"><Select value={f.status} onChange={(e) => setF({ ...f, status: e.target.value })}>
            <option value="">All</option><option value="litter_detected">litter_detected</option><option value="clean">clean</option><option value="view_blocked">view_blocked</option></Select></Field>
          <Field label="Class"><Select value={f.litter_class} onChange={(e) => setF({ ...f, litter_class: e.target.value })}>
            <option value="">All</option>{opts?.classes.map((v) => <option key={v} value={v}>{cls(v)}</option>)}</Select></Field>
          <Field label="Min confidence (0-100)"><Input type="number" min={0} max={100} step={1} value={f.min_confidence} placeholder="any" onChange={(e) => setF({ ...f, min_confidence: e.target.value })} /></Field>
          <Field label="From (recording date)"><Input type="date" value={f.date_from} onChange={(e) => setF({ ...f, date_from: e.target.value })} /></Field>
          <Field label="To (recording date)"><Input type="date" value={f.date_to} onChange={(e) => setF({ ...f, date_to: e.target.value })} /></Field>
        </div>
        {res.error && !res.data ? <ErrorState error={res.error} onRetry={res.refetch} compact /> : (
          <DataTable<ResultCheckRow> rows={res.data?.rows ?? []} rowKey={(r) => `${r.job_id}:${r.idx}`}
            serverPage={{ offset, limit: PAGE, total: res.data?.total ?? 0, onChange: setOffset }}
            toolbar={<span className="text-[12px] text-fg-3">{res.data ? `${num(res.data.total)} checks${(f.date_from || f.date_to) ? ' (checks without a recording time never match a date filter)' : ''}` : 'Loading…'}</span>}
            empty={res.data ? <EmptyState icon={ListChecks} compact title="No checks match" body="Each screenshot of a finished four-screenshot analysis appears here as one check." /> : <div className="p-3"><div className="skeleton h-24 rounded-md" /></div>}
            columns={[
              { key: 'thumb', header: 'Image', cell: (r) => (
                <a href={url(r.image_url)} target="_blank" rel="noreferrer" title="Open the annotated image">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={url(r.image_url)} alt={`Annotated screenshot ${r.idx} of ${r.job_id}`} loading="lazy" className="h-11 w-16 rounded-md border border-line bg-video object-cover" />
                </a>) },
              { key: 'timestamp', header: 'timestamp', sortValue: (r) => r.timestamp ?? '', cell: (r) => {
                const t = recordingTimeText(r.timestamp)
                return t.unknown
                  ? <span title={r.time_warning ?? undefined}><Badge tone="warn" dot>time unknown</Badge></span>
                  : <span className="whitespace-nowrap font-mono text-[12px]">{t.text}</span>
              } },
              { key: 'camera_id', header: 'camera_id', sortValue: (r) => r.camera_id, cell: (r) => {
                const c = cameraText(r.camera_id)
                return c.missing ? <Badge tone="warn" title={`camera on the analysis: ${r.camera_raw ?? '(empty)'}`}>no camera assigned</Badge> : <span className="font-mono">{c.text}</span>
              } },
              { key: 'image_name', header: 'image_name', cell: (r) => <a href={url(r.image_url)} target="_blank" rel="noreferrer" className="font-mono text-[11.5px] text-info underline-offset-2 hover:underline">{r.image_name}</a> },
              { key: 'litter_class', header: 'litter_class', sortValue: (r) => r.litter_class, cell: (r) => r.litter_class ? <span className="capitalize">{cls(r.litter_class)}</span> : <span className="text-fg-3" aria-label="empty">—</span> },
              { key: 'confidence_score', header: 'confidence_score', align: 'right', sortValue: (r) => r.confidence_score ? Number(r.confidence_score) : -1, cell: (r) => r.confidence_score ? <span className="font-mono">{r.confidence_score}</span> : <span className="text-fg-3" aria-label="empty">—</span> },
              { key: 'detection_status', header: 'detection_status', sortValue: (r) => r.detection_status, cell: (r) => {
                const i = checkStatusInfo(r.detection_status)
                return <Badge tone={i.tone} dot title={i.hint}>{i.label}</Badge>
              } },
              { key: 'remark', header: 'remark', cell: (r) => <span className="block max-w-[340px] truncate text-[12px] text-fg-2" title={r.remark}>{r.remark}</span> },
              { key: 'cart_id', header: 'cart_id', cell: (r) => r.cart_id ? <span className="font-mono">{r.cart_id}</span> : <span className="text-fg-3" title={r.store_flag ?? undefined}>—</span> },
              { key: 'store_name', header: 'store_name', sortValue: (r) => r.store_name, cell: (r) => r.store_name || <span className="text-[12px] text-warn" title={r.store_flag ?? undefined}>{r.job_store_id ? `${r.job_store_id} (not in store config)` : 'no store'}</span> },
              { key: 'bundle', header: 'Job / bundle', cell: (r) => {
                const b = bundleBadge(r.bundle)
                return <span className="flex flex-col gap-0.5 leading-tight">
                  <span className="max-w-[150px] break-all font-mono text-[11px] text-fg-2">{r.job_id} · s{r.idx}</span>
                  <span title={b.title}><Badge tone={b.tone}>{b.label}</Badge></span>
                </span>
              } },
              { key: 'import', header: 'In importer CSV', hideable: true, cell: (r) => r.import_ready ? <Badge tone="ok">yes</Badge> : <span title={r.import_issue ?? undefined}><Badge tone="neutral">no</Badge></span> },
            ]} />
        )}
      </Panel>
    </div>
  )
}
