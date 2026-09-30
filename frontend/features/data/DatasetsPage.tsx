'use client'

import Link from 'next/link'
import { useRouter, useSearchParams } from 'next/navigation'
import { useEffect, useState } from 'react'
import { AlertTriangle, Copy, Database, Download, FolderOpen, GitCompare, ImageIcon } from 'lucide-react'
import { downloadAuthed, url } from '../../api/client'
import { Api } from '../../api/endpoints'
import type { DatasetImage, DatasetStats, DatasetSummary } from '../../api/types'
import { invalidate, useQuery } from '../../hooks/useQuery'
import { className as cls, cx, dateTime, num } from '../../lib/format'
import { useApp } from '../../stores/app'
import { Badge, Button, Field, Input, PageHeader, Panel, Select, Tabs } from '../../components/ui/primitives'
import { Modal } from '../../components/ui/overlay'
import { BarList, Columns, SERIES, StackBar } from '../../components/ui/charts'
import { MetricCard } from '../../components/ui/kpi'
import { DataTable } from '../../components/ui/table'
import { EmptyState, ErrorState, LoadingState } from '../../components/ui/states'
import { ImageOverlay } from '../../components/vision/Overlay'

type Tab = 'overview' | 'inspect' | 'validation' | 'compare'

export default function DatasetsPage() {
  const params = useSearchParams()
  const router = useRouter()
  const list = useQuery('datasets', Api.datasets)
  const name = params.get('name') ?? list.data?.datasets[0]?.name ?? null
  const [tab, setTab] = useState<Tab>('overview')
  const [versionOf, setVersionOf] = useState<string | null>(null)
  const { isAdmin } = useApp()
  const stats = useQuery(name ? `dataset:${name}` : null, () => Api.dataset(name!))

  return (
    <div className="flex flex-col gap-3">
      <PageHeader title="Dataset Lab" description="YOLO datasets found in the dataset folder, with statistics and integrity validation computed from the files on disk (cached until files change)." />
      {list.error && !list.data ? <ErrorState error={list.error} onRetry={list.refetch} /> : !list.data ? <LoadingState /> : list.data.datasets.length === 0 ? (
        <EmptyState icon={Database} title="No datasets found" body="Add a YOLO dataset folder (with data.yaml) under dataset/." />
      ) : (
        <>
          <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-4">
            {list.data.datasets.map((d) => (
              <DatasetCard key={d.name} d={d} active={d.name === name} onOpen={() => router.replace(`/datasets?name=${encodeURIComponent(d.name)}`)}
                onVersion={isAdmin ? () => setVersionOf(d.name) : undefined} />
            ))}
          </div>
          {name && (
            <Panel noPad>
              <Tabs value={tab} onChange={setTab} tabs={[
                { value: 'overview', label: 'Overview' }, { value: 'inspect', label: 'Inspect images', count: stats.data?.totals.images },
                { value: 'validation', label: 'Validation', count: stats.data ? stats.data.issues.length : undefined }, { value: 'compare', label: 'Compare' },
              ]} />
              <div className="p-3">
                {stats.error ? <ErrorState error={stats.error} onRetry={stats.refetch} compact /> : !stats.data ? <LoadingState rows={6} label="Computing dataset statistics" /> : (
                  tab === 'overview' ? <Overview s={stats.data} /> :
                  tab === 'inspect' ? <Inspector s={stats.data} /> :
                  tab === 'validation' ? <Validation s={stats.data} /> :
                  <Compare base={stats.data} all={list.data.datasets} />
                )}
              </div>
            </Panel>
          )}
        </>
      )}
      <VersionDialog source={versionOf} onClose={() => setVersionOf(null)} />
    </div>
  )
}

function DatasetCard({ d, active, onOpen, onVersion }: { d: DatasetSummary; active: boolean; onOpen: () => void; onVersion?: () => void }) {
  const issues = Object.entries(d.issue_counts ?? {}).filter(([k]) => k !== 'tiny_object' && k !== 'class_without_labels').reduce((a, [, v]) => a + v, 0)
  const synthetic = /synthetic/i.test(d.readme ?? '')
  return (
    <section className={cx('flex flex-col rounded-lg border bg-panel', active ? 'border-info' : 'border-line')}>
      <div className="flex items-start justify-between gap-2 p-3">
        <div className="min-w-0">
          <h3 className="truncate font-mono text-[13px] font-semibold uppercase text-fg">{d.name}</h3>
          <p className="text-[11px] text-fg-3">{d.created ? dateTime(d.created) : ''}</p>
        </div>
        <div className="flex flex-col items-end gap-1">
          {synthetic && <Badge tone="warn">synthetic</Badge>}
          {issues > 0 && <Badge tone="crit">{issues} issues</Badge>}
        </div>
      </div>
      {d.error ? <p className="px-3 pb-3 text-[12px] text-crit">{d.error}</p> : (
        <div className="grid grid-cols-3 gap-x-2 gap-y-1 px-3 pb-3 text-[12px]">
          {[['Images', d.totals.images], ['Objects', d.totals.objects], ['Classes', d.classes.length],
            ['Train', d.splits.train?.images ?? 0], ['Val', d.splits.val?.images ?? 0], ['Test', d.splits.test?.images ?? 0]].map(([k, v]) => (
            <div key={k as string}><div className="text-[10px] uppercase tracking-wide text-fg-3">{k}</div><div className="tabular font-mono text-fg">{num(v as number)}</div></div>
          ))}
        </div>
      )}
      <div className="mt-auto flex gap-1 border-t border-line p-2">
        <Button size="sm" icon={FolderOpen} onClick={onOpen}>Open</Button>
        <Button size="sm" variant="ghost" icon={Copy} disabled={!onVersion} title={!onVersion ? 'Admin sign-in required' : 'Copy into a new version'} onClick={onVersion}>New version</Button>
        <ExportButton name={d.name} enabled={!!onVersion} />
      </div>
    </section>
  )
}

function ExportButton({ name, enabled }: { name: string; enabled: boolean }) {
  const { toast } = useApp()
  const [busy, setBusy] = useState(false)
  return (
    <Button size="sm" variant="ghost" icon={Download} disabled={!enabled} loading={busy} title={!enabled ? 'Admin sign-in required' : 'Download as ZIP'}
      onClick={async () => {
        setBusy(true)
        try { await downloadAuthed(`/api/datasets/${encodeURIComponent(name)}/export`, `${name}.zip`) }
        catch (e) { toast({ tone: 'crit', title: 'Export failed', body: (e as Error).message }) }
        finally { setBusy(false) }
      }}>Export</Button>
  )
}

function Overview({ s }: { s: DatasetStats }) {
  const perClass = s.classes.map((c, i) => ({ label: cls(c), value: s.objects_per_class[c] ?? 0, color: SERIES[i % SERIES.length] }))
  const sd = s.size_distribution
  const neg = s.totals.negatives, pos = s.totals.images - neg
  return (
    <div className="flex flex-col gap-3">
      {s.readme && /synthetic/i.test(s.readme) && (
        <p className="flex items-start gap-2 rounded-md border border-warn/30 bg-warn-bg p-2.5 text-[12px] text-warn"><AlertTriangle className="mt-0.5 size-3.5 shrink-0" />{s.readme.split('\n').filter(Boolean).slice(0, 4).join(' ')}</p>
      )}
      <div className="grid grid-cols-2 gap-2 md:grid-cols-4 xl:grid-cols-8">
        <MetricCard label="Images" value={num(s.totals.images)} />
        <MetricCard label="Objects" value={num(s.totals.objects)} />
        <MetricCard label="Classes" value={s.classes.length} hint={`${s.classes.filter((c) => !s.objects_per_class[c]).length} without labels`} />
        <MetricCard label="Positive : negative" value={`${pos} : ${neg}`} hint={neg ? `${(pos / neg).toFixed(1)} : 1` : 'no negatives'} />
        <MetricCard label="Avg box" value={s.box_area_px ? `${Math.round(Math.sqrt(s.box_area_px.mean))} px` : '—'} hint="√area, original pixels" />
        <MetricCard label="Median box" value={s.box_area_px ? `${Math.round(Math.sqrt(s.box_area_px.median))} px` : '—'} />
        <MetricCard label="Imbalance" value={s.class_imbalance_ratio ? `${s.class_imbalance_ratio}×` : '—'} hint="max/min labelled class" tone={(s.class_imbalance_ratio ?? 0) > 10 ? 'warn' : undefined} />
        <MetricCard label="Resolutions" value={Object.keys(s.resolutions).length} hint={Object.keys(s.resolutions).slice(0, 2).join(', ')} />
      </div>
      <div className="grid gap-3 lg:grid-cols-3">
        <Panel title="Objects per class"><BarList data={perClass} /></Panel>
        <Panel title="Images per class"><BarList color="var(--exp)" data={s.classes.map((c) => ({ label: cls(c), value: s.images_per_class[c] ?? 0 }))} /></Panel>
        <div className="flex flex-col gap-3">
          <Panel title="Splits">
            <DataTable rows={Object.entries(s.splits)} rowKey={([k]) => k} dense columns={[
              { key: 's', header: 'Split', cell: ([k]) => <span className="text-fg">{k}</span> },
              { key: 'i', header: 'Images', align: 'right', cell: ([, v]) => num(v.images) },
              { key: 'o', header: 'Objects', align: 'right', cell: ([, v]) => num(v.objects) },
              { key: 'n', header: 'Negatives', align: 'right', cell: ([, v]) => num(v.negatives) },
              { key: 'e', header: 'Empty labels', align: 'right', cell: ([, v]) => num(v.empty_labels ?? 0) },
            ]} />
          </Panel>
          <Panel title="Object size (COCO)">
            <StackBar parts={[{ label: 'small', value: sd.small, color: 'var(--crit)' }, { label: 'medium', value: sd.medium, color: 'var(--warn)' }, { label: 'large', value: sd.large, color: 'var(--ok)' }]} />
            <div className="mt-2 grid grid-cols-3 text-[12px]">
              <span className="text-fg-2">small <b className="font-mono text-fg">{sd.small}</b></span>
              <span className="text-fg-2">medium <b className="font-mono text-fg">{sd.medium}</b></span>
              <span className="text-fg-2">large <b className="font-mono text-fg">{sd.large}</b></span>
            </div>
          </Panel>
        </div>
      </div>
      {s.box_area_px && (
        <Panel title="Box size distribution" subtitle="Objects by √(box area) in original-image pixels">
          <Columns ariaLabel="Box size histogram" height={120}
            data={s.box_area_px.histogram.map((v, i) => ({ label: i < s.box_area_px!.histogram_edges_side_px.length - 1 ? `${s.box_area_px!.histogram_edges_side_px[i]}–${s.box_area_px!.histogram_edges_side_px[i + 1]}` : `${s.box_area_px!.histogram_edges_side_px[i]}+`, value: v, color: i < 4 ? 'var(--crit)' : 'var(--info)' }))} />
        </Panel>
      )}
    </div>
  )
}

function Inspector({ s }: { s: DatasetStats }) {
  const [f, setF] = useState({ split: '', class: '', size: '', status: '' })
  const [offset, setOffset] = useState(0)
  const LIMIT = 24
  const q = useQuery(`ds:inspect:${s.name}:${JSON.stringify(f)}:${offset}`, () => Api.datasetImages(s.name, { ...f, offset, limit: LIMIT }))
  useEffect(() => setOffset(0), [JSON.stringify(f)]) // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-2 gap-2 md:grid-cols-5">
        <Field label="Split"><Select value={f.split} onChange={(e) => setF({ ...f, split: e.target.value })}><option value="">All</option>{['train', 'val', 'test'].map((x) => <option key={x}>{x}</option>)}</Select></Field>
        <Field label="Class"><Select value={f.class} onChange={(e) => setF({ ...f, class: e.target.value })}><option value="">All</option>{s.classes.map((c) => <option key={c} value={c}>{cls(c)}</option>)}</Select></Field>
        <Field label="Object size"><Select value={f.size} onChange={(e) => setF({ ...f, size: e.target.value })}><option value="">Any</option>{['small', 'medium', 'large'].map((x) => <option key={x}>{x}</option>)}</Select></Field>
        <Field label="Annotation status"><Select value={f.status} onChange={(e) => setF({ ...f, status: e.target.value })}><option value="">Any</option>{['annotated', 'empty', 'negative', 'issues'].map((x) => <option key={x}>{x}</option>)}</Select></Field>
        <Field label="Camera / source"><Select disabled><option>not recorded in dataset</option></Select></Field>
      </div>
      {q.error ? <ErrorState error={q.error} compact onRetry={q.refetch} /> : !q.data ? <LoadingState rows={4} /> : q.data.total === 0 ? <EmptyState compact icon={ImageIcon} title="No images match these filters" /> : (
        <>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4 2xl:grid-cols-6">
            {q.data.items.map((it) => <Thumb key={it.id} ds={s.name} it={it} />)}
          </div>
          <div className="flex items-center justify-between text-[12px] text-fg-3">
            <span>{offset + 1}–{Math.min(offset + LIMIT, q.data.total)} of {q.data.total}</span>
            <div className="flex gap-1">
              <Button size="sm" disabled={offset === 0} onClick={() => setOffset(offset - LIMIT)}>Previous</Button>
              <Button size="sm" disabled={offset + LIMIT >= q.data.total} onClick={() => setOffset(offset + LIMIT)}>Next</Button>
            </div>
          </div>
        </>
      )}
    </div>
  )
}

function Thumb({ ds, it }: { ds: string; it: DatasetImage }) {
  const ann = useQuery(`ann:${ds}:${it.id}`, () => Api.annotation(ds, it.id))
  const a = ann.data
  return (
    <Link href={`/ground-truth?dataset=${encodeURIComponent(ds)}&id=${encodeURIComponent(it.id)}`} className="group overflow-hidden rounded-md border border-line bg-panel-2 hover:border-info">
      <div className="relative aspect-video">
        <ImageOverlay src={url(`/api/datasets/${encodeURIComponent(ds)}/image?id=${encodeURIComponent(it.id)}`)} width={a?.width ?? 0} height={a?.height ?? 0} className="absolute inset-0" showLabels={false}
          alt={it.file} boxes={(a?.ground_truth ?? []).map((g) => ({ key: g.index, bbox: g.bbox, color: 'var(--info)' }))} />
      </div>
      <div className="flex items-center justify-between gap-1 px-2 py-1 text-[11px]">
        <span className="truncate text-fg-2 group-hover:text-fg">{it.file}</span>
        <span className="shrink-0 text-fg-3">{it.split} · {it.status === 'negative' ? 'neg' : `${it.objects}`}</span>
      </div>
    </Link>
  )
}

function Validation({ s }: { s: DatasetStats }) {
  const [type, setType] = useState('')
  const rows = s.issues.filter((i) => !type || i.type === type)
  const severe = new Set(['corrupt_image', 'missing_label', 'label_error', 'orphan_label', 'duplicate_image'])
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap gap-1.5">
        <button onClick={() => setType('')} className={cx('rounded-md border px-2 py-1 text-[12px]', !type ? 'border-info text-fg' : 'border-line text-fg-2')}>All ({s.issues.length})</button>
        {Object.entries(s.issue_counts).map(([k, v]) => (
          <button key={k} onClick={() => setType(k)} className={cx('rounded-md border px-2 py-1 text-[12px]', type === k ? 'border-info text-fg' : 'border-line text-fg-2')}>
            <span className={severe.has(k) ? 'text-crit' : 'text-warn'}>●</span> {k.replace(/_/g, ' ')} ({v})
          </button>
        ))}
      </div>
      {s.issues.length === 0 ? <EmptyState compact title="No integrity problems found" body="Checked: corrupt images, missing/orphan labels, invalid class ids, coordinates, zero-area and duplicate boxes, duplicate and near-duplicate images, tiny objects, classes without labels." /> : (
        <DataTable rows={rows} rowKey={(r) => `${r.type}:${r.image}:${r.detail}`} dense pageSize={50}
          columns={[
            { key: 't', header: 'Check', cell: (r) => <Badge tone={severe.has(r.type) ? 'crit' : 'warn'}>{r.type.replace(/_/g, ' ')}</Badge>, sortValue: (r) => r.type },
            { key: 'i', header: 'File', cell: (r) => r.image && !r.image.endsWith('.txt') ? <Link className="font-mono text-[12px] text-info hover:underline" href={`/ground-truth?dataset=${encodeURIComponent(s.name)}&id=${encodeURIComponent(r.image)}`}>{r.image}</Link> : <span className="font-mono text-[12px]">{r.image || '—'}</span> },
            { key: 'd', header: 'Detail', cell: (r) => r.detail },
          ]} />
      )}
    </div>
  )
}

function Compare({ base, all }: { base: DatasetStats; all: DatasetSummary[] }) {
  const others = all.filter((d) => d.name !== base.name)
  const [other, setOther] = useState(others[0]?.name ?? '')
  const o = useQuery(other ? `dataset:${other}` : null, () => Api.dataset(other))
  if (!others.length) return <EmptyState compact icon={GitCompare} title="Only one dataset" body="Create a new version to compare statistics side by side." />
  const b = o.data
  const rows = b ? [
    ['Images', base.totals.images, b.totals.images], ['Objects', base.totals.objects, b.totals.objects], ['Negatives', base.totals.negatives, b.totals.negatives],
    ['Small objects', base.size_distribution.small, b.size_distribution.small], ['Issues', base.issues.length, b.issues.length],
    ...base.classes.map((c) => [`· ${cls(c)}`, base.objects_per_class[c] ?? 0, b.objects_per_class[c] ?? 0]),
  ] as [string, number, number][] : []
  return (
    <div className="flex flex-col gap-3">
      <Field label="Compare with"><Select value={other} onChange={(e) => setOther(e.target.value)} className="max-w-xs">{others.map((d) => <option key={d.name}>{d.name}</option>)}</Select></Field>
      {!b ? <LoadingState /> : (
        <DataTable rows={rows} rowKey={(r) => r[0]} dense pageSize={100} columns={[
          { key: 'k', header: 'Metric', cell: (r) => r[0] },
          { key: 'a', header: base.name, align: 'right', cell: (r) => num(r[1]) },
          { key: 'b', header: b.name, align: 'right', cell: (r) => num(r[2]) },
          { key: 'd', header: 'Δ', align: 'right', cell: (r) => <span className={r[2] - r[1] > 0 ? 'text-ok' : r[2] - r[1] < 0 ? 'text-crit' : 'text-fg-3'}>{r[2] - r[1] > 0 ? '+' : ''}{r[2] - r[1]}</span> },
        ]} />
      )}
    </div>
  )
}

function VersionDialog({ source, onClose }: { source: string | null; onClose: () => void }) {
  const { toast } = useApp()
  const [name, setName] = useState('')
  const [pending, setPending] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  return (
    <Modal open={!!source} onClose={onClose} title="Create dataset version"
      footer={<><Button variant="ghost" onClick={onClose}>Cancel</Button><Button variant="primary" loading={pending} disabled={!/^[A-Za-z0-9_-]+$/.test(name)} onClick={async () => {
        setPending(true); setErr(null)
        try { await Api.createDatasetVersion(source!, name); toast({ tone: 'ok', title: `Dataset ${name} created` }); invalidate('datasets'); onClose() }
        catch (e) { setErr((e as Error).message) } finally { setPending(false) }
      }}>Create</Button></>}>
      <div className="flex flex-col gap-3">
        <p className="text-[12px] text-fg-3">Copies <span className="font-mono">{source}</span> (images, labels, data.yaml) into a new folder so it can be edited without changing the original.</p>
        <Field label="New dataset name"><Input value={name} onChange={(e) => setName(e.target.value)} placeholder="LITTER-CCTV-V002" data-autofocus /></Field>
        {err && <p className="text-[12px] text-crit">{err}</p>}
      </div>
    </Modal>
  )
}
