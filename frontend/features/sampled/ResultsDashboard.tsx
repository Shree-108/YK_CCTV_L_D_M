'use client'

// Results Dashboard for the four-screenshot video analysis. Every number comes from /api/sampled/reports/*.
import Link from 'next/link'
import { useEffect, useState } from 'react'
import {
  BarChart3, CalendarDays, CheckCircle2, Clock, Download, FileSpreadsheet, FileVideo, History, Images, ScanSearch, Trash2, XCircle,
} from 'lucide-react'
import { downloadAuthed, url } from '../../api/client'
import { Api } from '../../api/endpoints'
import type { SampledAnalysisRow, SampledDetectionRow, SampledHourlyRow } from '../../api/sampledTypes'
import { useQuery } from '../../hooks/useQuery'
import { dateTime, dayEndIso, dayStartIso, duration, num, pct } from '../../lib/format'
import { lockedFor, useApp } from '../../stores/app'
import { Badge, Button, Field, Input, PageHeader, Panel, Segmented, Select, Tabs } from '../../components/ui/primitives'
import { BarList } from '../../components/ui/charts'
import { KpiCard, MetricCard } from '../../components/ui/kpi'
import { DataTable } from '../../components/ui/table'
import { EmptyState, ErrorState, LoadingState, QueryView } from '../../components/ui/states'
import { litterName } from '../detection/common'
import { CsvPreview, FrameCard, STATE } from './shared'

export default function ResultsDashboard() {
  const ov = useQuery('sampled:overview', Api.sampledOverview, { poll: 10000 })
  const s = ov.data?.summary
  const loading = ov.loading && !ov.data
  return (
    <div className="flex flex-col gap-5">
      <PageHeader title="Results Dashboard" description="Four-screenshot video analyses: summary, latest result, CSV reports, monthly totals and history."
        actions={<Link href="/detect/video"><Button variant="primary" icon={FileVideo}>Analyse a video</Button></Link>} />
      {ov.error && !ov.data && <Panel><ErrorState error={ov.error} onRetry={ov.refetch} /></Panel>}

      {/* A. Summary */}
      <section aria-labelledby="sum-h" className="flex flex-col gap-3">
        <h2 id="sum-h" className="text-[13px] font-semibold uppercase tracking-[0.08em] text-fg-3">A · Summary</h2>
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4 2xl:grid-cols-7">
          <KpiCard label="Videos received" icon={FileVideo} loading={loading} value={s ? num(s.videos_received) : null} />
          <KpiCard label="Analyses completed" icon={CheckCircle2} tone="ok" loading={loading} value={s ? num(s.analyses_completed) : null}
            sub={s ? `${s.analyses_in_progress} in progress${s.awaiting_roi ? ` · ${s.awaiting_roi} awaiting ROI` : ''}` : ''} />
          <KpiCard label="Screenshots extracted" icon={Images} loading={loading} value={s ? num(s.screenshots_extracted) : null} />
          <KpiCard label="Screenshots analysed" icon={ScanSearch} loading={loading} value={s ? num(s.screenshots_analysed) : null} />
          <KpiCard label="Total litter detections" icon={Trash2} tone="crit" loading={loading} value={s ? num(s.total_litter) : null} />
          <KpiCard label="Detection categories" icon={BarChart3} tone="info" loading={loading} value={s ? num(Object.keys(s.categories).length) : null}
            sub={s && Object.keys(s.categories)[0] ? `top: ${litterName(Object.keys(s.categories)[0])}` : 'none yet'} />
          <KpiCard label="Failed analyses" icon={XCircle} tone={s?.analyses_failed ? 'crit' : 'neutral'} loading={loading} value={s ? num(s.analyses_failed) : null}
            sub={s?.analyses_cancelled ? `${s.analyses_cancelled} cancelled` : undefined} />
        </div>
      </section>

      {/* B + C. Latest analysis and its CSV */}
      <div className="grid gap-4 2xl:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]">
        <Panel title="B · Latest video analysis" icon={FileVideo}
          actions={ov.data?.latest && <Link href={`/detect/video?job=${ov.data.latest.id}`} className="text-[12px] font-medium text-fg-2 hover:text-fg">Open →</Link>}>
          <QueryView q={ov} isEmpty={(d) => !d.latest} empty={<EmptyState icon={FileVideo} compact title="No completed analysis yet" body="Analyse a video to see its four screenshots here." />}>
            {(d) => {
              const l = d.latest!
              return (
                <div className="flex flex-col gap-3">
                  {d.newest_other && (
                    <p className="rounded-md border border-warn/30 bg-warn-bg px-3 py-2 text-[12.5px] text-warn">
                      Newer analysis “{d.newest_other.video_name}” is {STATE[d.newest_other.state].label.toLowerCase()}.{' '}
                      <Link className="underline" href={`/detect/video?job=${d.newest_other.id}`}>Open it</Link>
                    </p>
                  )}
                  <div className="grid grid-cols-2 gap-2.5 md:grid-cols-4">
                    <MetricCard label="Video" value={<span className="text-[14px]">{l.video_name}</span>} hint={l.media ? duration(l.media.duration_sec) : undefined} />
                    <MetricCard label="Store · camera" value={<span className="text-[14px]">{[l.store_id, l.camera_id].filter(Boolean).join(' · ') || '—'}</span>} />
                    <MetricCard label="Status" value={<Badge tone={STATE[l.state].tone} dot>{STATE[l.state].label}</Badge>} hint={dateTime(l.finished_at)} />
                    <MetricCard label="Litter detected" value={l.counts?.litter ?? 0} tone={l.counts?.litter ? 'crit' : 'ok'}
                      hint={Object.entries(l.counts?.by_class ?? {}).map(([k, v]) => `${v}× ${litterName(k)}`).join(', ') || 'no litter'} />
                  </div>
                  <div className="grid gap-3 md:grid-cols-2">{l.frames.map((f) => <FrameCard key={f.idx} frame={f} compact />)}</div>
                </div>
              )
            }}
          </QueryView>
        </Panel>
        <Panel title="C · Latest analysis CSV" icon={FileSpreadsheet}>
          {ov.data?.latest?.csv_url ? <CsvPreview path={ov.data.latest.csv_url} filename={`litter_report_${ov.data.latest.id}.csv`} max={10} />
            : loading ? <LoadingState rows={4} /> : <EmptyState compact icon={FileSpreadsheet} title="No CSV yet" />}
        </Panel>
      </div>

      {/* D. Monthly */}
      <MonthlyReports months={ov.data?.months} rules={ov.data?.grouping_rules} categories={s?.categories} />

      {/* E. History */}
      <HistorySection />

      {/* F. Hourly */}
      <HourlySection />
    </div>
  )
}

// ---------------------------------------------------------------- D
function MonthlyReports({ months, rules, categories }: { months?: { completion: string[]; recording: string[] }; rules?: { completion: string; recording: string }; categories?: Record<string, number> }) {
  const { role, loginOn, toast } = useApp()
  const [by, setBy] = useState<'completion' | 'recording'>('completion')
  const list = months?.[by] ?? []
  const [month, setMonth] = useState('')
  useEffect(() => { if (!list.includes(month)) setMonth(list[0] ?? '') }, [by, list.join()]) // eslint-disable-line react-hooks/exhaustive-deps
  const q = useQuery(month ? `sampled:month:${month}:${by}` : null, () => Api.sampledMonth(month, by))
  const [busy, setBusy] = useState(false)
  const locked = loginOn ? lockedFor('admin', role, loginOn) : undefined
  return (
    <Panel title="D · Monthly reports" icon={CalendarDays} subtitle={rules?.[by]}
      actions={<>
        <Segmented size="sm" label="Group by" value={by} onChange={setBy} options={[{ value: 'completion', label: 'Analysis month' }, { value: 'recording', label: 'Recording month' }]} />
        <Select aria-label="Month" value={month} onChange={(e) => setMonth(e.target.value)} className="h-7 w-32" disabled={!list.length}>
          {list.length ? list.map((m) => <option key={m}>{m}</option>) : <option value="">—</option>}
        </Select>
        <Button size="sm" icon={Download} disabled={!month || !!locked} title={locked ?? 'Download this month as CSV'} loading={busy}
          onClick={async () => { setBusy(true); try { await downloadAuthed(`/api/sampled/reports/months/${month}.csv?by=${by}`, `litter_report_${month}${by === 'recording' ? '_by_recording_date' : ''}.csv`) } catch (e) { toast({ tone: 'crit', title: 'Download failed', body: (e as Error).message }) } finally { setBusy(false) } }}>
          Month CSV
        </Button>
      </>}>
      {!list.length ? <EmptyState compact icon={CalendarDays} title="No monthly data yet" body={by === 'recording' ? 'Only analyses with a recording start are grouped by recording month.' : 'Completed or failed analyses appear here by month.'} /> :
        q.error ? <ErrorState error={q.error} onRetry={q.refetch} compact /> : !q.data ? <LoadingState rows={3} /> : (
          <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_320px]">
            <div className="flex flex-col gap-3">
              <div className="grid grid-cols-2 gap-2.5 md:grid-cols-5">
                <MetricCard label="Analyses" value={q.data.analyses} />
                <MetricCard label="Completed" value={q.data.analyses_completed} tone="ok" />
                <MetricCard label="Failed" value={q.data.analyses_failed} tone={q.data.analyses_failed ? 'crit' : undefined} />
                <MetricCard label="Screenshots analysed" value={q.data.screenshots_analysed} hint={`of ${q.data.screenshots_extracted} extracted`} />
                <MetricCard label="Litter" value={q.data.total_litter} tone={q.data.total_litter ? 'crit' : undefined} />
              </div>
              <AnalysesTable rows={q.data.analyses_list} />
            </div>
            <div>
              <h3 className="mb-2 text-[12px] font-semibold text-fg-2">Litter categories in {q.data.month}</h3>
              {Object.keys(q.data.categories).length ? <BarList data={Object.entries(q.data.categories).map(([k, v]) => ({ label: litterName(k), value: v }))} />
                : <EmptyState compact title="No litter this month" />}
              {categories && Object.keys(categories).length > 0 && <p className="mt-3 text-[11.5px] text-fg-3">All time: {Object.entries(categories).map(([k, v]) => `${litterName(k)} ${v}`).join(' · ')}</p>}
            </div>
          </div>
        )}
    </Panel>
  )
}

function Thumbs({ r }: { r: SampledAnalysisRow }) {
  return (
    <span className="flex gap-1">
      {r.thumbs.slice(0, 4).map((t) => (
        // eslint-disable-next-line @next/next/no-img-element
        <img key={t} src={url(t)} alt="" loading="lazy" className="h-9 w-14 rounded-sm border border-line bg-video object-cover" />
      ))}
      {!r.thumbs.length && <span className="text-fg-3">—</span>}
    </span>
  )
}

function AnalysesTable({ rows, serverPage }: { rows: SampledAnalysisRow[]; serverPage?: { offset: number; total: number; limit: number; onChange: (o: number) => void } }) {
  return (
    <div className="overflow-hidden rounded-lg border border-line">
      <DataTable<SampledAnalysisRow> rows={rows} rowKey={(r) => r.id} serverPage={serverPage} dense
        onRowClick={(r) => { window.location.href = `/detect/video?job=${encodeURIComponent(r.id)}` }}
        empty={<EmptyState compact title="No analyses match" />}
        columns={[
          { key: 'v', header: 'Video', cell: (r) => <span className="flex flex-col"><span className="max-w-[220px] truncate text-fg">{r.video_name}</span><span className="text-[11px] text-fg-3">{dateTime(r.created_at)}</span></span>, sortValue: (r) => r.created_at },
          { key: 'sc', header: 'Store · camera', cell: (r) => [r.store_id, r.camera_id].filter(Boolean).join(' · ') || <span className="text-fg-3">—</span> },
          { key: 'ev', header: 'Evidence', cell: (r) => <Thumbs r={r} /> },
          { key: 'ts', header: 'Timestamps', cell: (r) => <span className="font-mono text-[11px]">{r.timestamps.join(' ')}</span>, hideable: true },
          { key: 'n', header: 'Litter', align: 'right', cell: (r) => <span className="tabular text-fg">{r.litter}</span>, sortValue: (r) => r.litter },
          { key: 'a', header: 'Analysed', align: 'right', cell: (r) => `${r.screenshots_analysed}/4`, hideable: true },
          { key: 's', header: 'Status', cell: (r) => <Badge tone={STATE[r.state].tone} dot title={r.error ?? undefined}>{STATE[r.state].label}</Badge>, sortValue: (r) => r.state },
        ]} />
    </div>
  )
}

// ---------------------------------------------------------------- E
const LIMIT = 25

function HistorySection() {
  const [tab, setTab] = useState<'videos' | 'detections'>('videos')
  const [f, setF] = useState({ store: '', camera: '', start: '', end: '', state: '', q: '', cls: '' })
  const [offset, setOffset] = useState(0)
  useEffect(() => setOffset(0), [tab, JSON.stringify(f)]) // eslint-disable-line react-hooks/exhaustive-deps
  const range = { start: f.start ? dayStartIso(f.start) : undefined, end: f.end ? dayEndIso(f.end) : undefined }
  const vq = { store: f.store, camera: f.camera, state: f.state, q: f.q, ...range, limit: LIMIT, offset }
  const dq = { store: f.store, camera: f.camera, class: f.cls, ...range, limit: LIMIT, offset }
  const videos = useQuery(tab === 'videos' ? `sampled:history:${JSON.stringify(vq)}` : null, () => Api.sampledHistory(vq), { poll: 15000 })
  const dets = useQuery(tab === 'detections' ? `sampled:dets:${JSON.stringify(dq)}` : null, () => Api.sampledDetections(dq), { poll: 15000 })
  const stores = videos.data?.stores ?? [], cams = videos.data?.cameras ?? []
  return (
    <Panel title="E · All historical data" icon={History} noPad>
      <Tabs value={tab} onChange={setTab} tabs={[{ value: 'videos', label: 'Video history', count: videos.data?.total }, { value: 'detections', label: 'Detection history', count: dets.data?.total }]} />
      <div className="grid grid-cols-2 gap-3 border-b border-line bg-panel-2/40 px-4 py-3 md:grid-cols-3 xl:grid-cols-6">
        <Field label="Store"><Input list="hist-stores" value={f.store} onChange={(e) => setF({ ...f, store: e.target.value })} placeholder="All" /></Field>
        <datalist id="hist-stores">{stores.map((x) => <option key={x} value={x} />)}</datalist>
        <Field label="Camera"><Input list="hist-cams" value={f.camera} onChange={(e) => setF({ ...f, camera: e.target.value })} placeholder="All" /></Field>
        <datalist id="hist-cams">{cams.map((x) => <option key={x} value={x} />)}</datalist>
        <Field label="From"><Input type="date" value={f.start} onChange={(e) => setF({ ...f, start: e.target.value })} /></Field>
        <Field label="To"><Input type="date" value={f.end} onChange={(e) => setF({ ...f, end: e.target.value })} /></Field>
        {tab === 'videos' ? <>
          <Field label="Status"><Select value={f.state} onChange={(e) => setF({ ...f, state: e.target.value })}>
            <option value="">All</option>{Object.entries(STATE).map(([k, v]) => <option key={k} value={k}>{v.label}</option>)}</Select></Field>
          <Field label="Video name"><Input value={f.q} maxLength={80} onChange={(e) => setF({ ...f, q: e.target.value })} placeholder="Search…" /></Field>
        </> : (
          <Field label="Category"><Select value={f.cls} onChange={(e) => setF({ ...f, cls: e.target.value })}>
            <option value="">All</option>{dets.data?.classes.map((c) => <option key={c} value={c}>{litterName(c)}</option>)}</Select></Field>
        )}
      </div>
      {tab === 'videos' ? (
        videos.error && !videos.data ? <ErrorState error={videos.error} onRetry={videos.refetch} compact /> :
          <div className="p-4"><AnalysesTable rows={videos.data?.analyses ?? []} serverPage={{ offset, limit: LIMIT, total: videos.data?.total ?? 0, onChange: setOffset }} /></div>
      ) : (
        dets.error && !dets.data ? <ErrorState error={dets.error} onRetry={dets.refetch} compact /> : (
          <DataTable<SampledDetectionRow> rows={dets.data?.detections ?? []} rowKey={(r) => r.id}
            serverPage={{ offset, limit: LIMIT, total: dets.data?.total ?? 0, onChange: setOffset }}
            empty={<EmptyState compact icon={ScanSearch} title="No litter detections match" />}
            columns={[
              { key: 'ev', header: 'Evidence', cell: (r) => r.evidence_url ? (
                <a href={`/detect/video?job=${encodeURIComponent(r.job_id)}`}>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={url(r.evidence_url)} alt={`${r.class} evidence`} loading="lazy" className="h-11 w-16 rounded-md border border-line bg-video object-cover" />
                </a>) : '—' },
              { key: 'c', header: 'Litter category', cell: (r) => <span className="inline-flex rounded-full border border-line bg-panel-2 px-2.5 py-0.5 text-[12.5px] font-medium text-fg">{litterName(r.class)}</span>, sortValue: (r) => r.class },
              { key: 'p', header: 'Confidence', align: 'right', cell: (r) => <span className="tabular">{pct(r.confidence, 1)}</span>, sortValue: (r) => r.confidence },
              { key: 'v', header: 'Video', cell: (r) => <span className="max-w-[200px] truncate">{r.video_name}</span> },
              { key: 'sc', header: 'Store · camera', cell: (r) => [r.store_id, r.camera_id].filter(Boolean).join(' · ') || '—' },
              { key: 't', header: 'Video time', cell: (r) => <span className="font-mono text-[12px]">{r.video_timestamp}</span> },
              { key: 'rt', header: 'Recording time', cell: (r) => r.recording_timestamp?.replace('T', ' ') ?? <span className="text-fg-3">unknown</span>, hideable: true },
              { key: 'ts', header: 'Analysed', cell: (r) => dateTime(r.ts), sortValue: (r) => r.ts, hideable: true },
            ]} />
        )
      )}
    </Panel>
  )
}

// ---------------------------------------------------------------- F
function HourlySection() {
  const [f, setF] = useState({ store: '', camera: '', date: '' })
  const q = useQuery(`sampled:hourly:${JSON.stringify(f)}`, () => Api.sampledHourly(f), { poll: 30000 })
  return (
    <Panel title="F · Hourly summary" icon={Clock} subtitle="Real recording time only — video positions are never shown as clock times">
      <div className="mb-3 grid grid-cols-2 gap-3 md:grid-cols-4">
        <Field label="Date"><Input type="date" value={f.date} onChange={(e) => setF({ ...f, date: e.target.value })} /></Field>
        <Field label="Store"><Input value={f.store} onChange={(e) => setF({ ...f, store: e.target.value })} placeholder="All" /></Field>
        <Field label="Camera"><Input value={f.camera} onChange={(e) => setF({ ...f, camera: e.target.value })} placeholder="All" /></Field>
      </div>
      {q.error && !q.data ? <ErrorState error={q.error} onRetry={q.refetch} compact /> : !q.data ? <LoadingState rows={3} /> : (
        <>
          <p className="mb-3 text-[12px] text-fg-3">
            {q.data.note}{q.data.excluded_analyses > 0 && <> <span className="text-warn">{q.data.excluded_analyses} completed analys{q.data.excluded_analyses === 1 ? 'is has' : 'es have'} no recording start and {q.data.excluded_analyses === 1 ? 'is' : 'are'} not included.</span></>}
          </p>
          {!q.data.available || !q.data.rows.length ? <EmptyState compact icon={Clock} title="No hourly data"
            body="Enter the recording start when starting an analysis (it is pre-filled from NVR file names) to get an hourly summary." /> : (
            <div className="overflow-hidden rounded-lg border border-line">
              <DataTable<SampledHourlyRow> rows={q.data.rows} rowKey={(r) => `${r.hour_start}-${r.store_id}-${r.camera_id}`} dense
                columns={[
                  { key: 'h', header: 'Hour', cell: (r) => <span className="font-mono text-[12px]">{r.hour_start.replace('T', ' ')}–{r.hour_end.slice(11)}</span>, sortValue: (r) => r.hour_start },
                  { key: 'sc', header: 'Store · camera', cell: (r) => [r.store_id, r.camera_id].filter(Boolean).join(' · ') || '—' },
                  { key: 'a', header: 'Analyses', align: 'right', cell: (r) => r.analyses },
                  { key: 's', header: 'Screenshots', align: 'right', cell: (r) => `${r.analysed}/${r.screenshots}` },
                  { key: 'w', header: 'With litter', align: 'right', cell: (r) => r.with_litter, sortValue: (r) => r.with_litter },
                  { key: 'n', header: 'Litter', align: 'right', cell: (r) => <span className="tabular font-medium text-fg">{r.litter}</span>, sortValue: (r) => r.litter },
                  { key: 'c', header: 'Categories', cell: (r) => Object.entries(r.by_class).map(([k, v]) => `${v}× ${litterName(k)}`).join(', ') || '—' },
                ]} />
            </div>
          )}
        </>
      )}
    </Panel>
  )
}
