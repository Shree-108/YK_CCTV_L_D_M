'use client'

import Link from 'next/link'
import { useMemo, useState } from 'react'
import { Boxes, Target } from 'lucide-react'
import { Api } from '../../api/endpoints'
import type { Evaluation, ModelInfo } from '../../api/types'
import { useQuery } from '../../hooks/useQuery'
import { num, pct } from '../../lib/format'
import { Badge, Button, Field, PageHeader, Panel, Select, StatusBadge, Unavailable } from '../../components/ui/primitives'
import { DataTable } from '../../components/ui/table'
import { EmptyState, ErrorState } from '../../components/ui/states'

type Objective = 'recall' | 'precision' | 'f1' | 'small' | 'realtime'
const OBJ: Record<Objective, { label: string; key: string; note: string }> = {
  recall: { label: 'Maximum recall', key: 'recall', note: 'Fewest missed litter items; expect more false alarms.' },
  precision: { label: 'Maximum precision', key: 'precision', note: 'Fewest false alarms; expect more missed items.' },
  f1: { label: 'Balanced F1', key: 'f1', note: 'Balance of precision and recall at the operating threshold.' },
  small: { label: 'Small-object recall', key: 'small', note: 'Recall on objects < 32×32 px — the hard case for CCTV.' },
  realtime: { label: 'Real-time performance', key: 'fps', note: 'Frames per second measured on this server during evaluation.' },
}

type Row = { m: ModelInfo; ev?: Evaluation }

export default function ModelsPage() {
  const models = useQuery('models', Api.models)
  const evals = useQuery('evaluations', Api.evaluations)
  const [objective, setObjective] = useState<Objective>('f1')
  const [dataset, setDataset] = useState('')
  const list = models.data?.models ?? []
  const done = (evals.data?.evaluations ?? []).filter((e) => e.status === 'done' && e.summary)
  const datasets = [...new Set(done.map((e) => `${e.summary!.dataset}/${e.summary!.split}`))]
  const ds = dataset || datasets[0] || ''
  // latest evaluation per model on the selected dataset/split
  const evalIds = useMemo(() => {
    const m = new Map<string, string>()
    done.forEach((e) => { if (`${e.summary!.dataset}/${e.summary!.split}` === ds && !m.has(e.summary!.model_version)) m.set(e.summary!.model_version, e.id) })
    return m
  }, [done, ds])
  const ids = [...evalIds.values()]
  const full = useQuery(ids.length ? `evals:full:${ids.join(',')}` : null, () => Promise.all(ids.map((id) => Api.evaluation(id))))
  const rows: Row[] = list.map((m) => ({ m, ev: full.data?.find((e) => e.model_version === m.version && evalIds.get(m.version) === e.id) }))
  const val = (r: Row, k: string): number | null => {
    const e = r.ev
    if (!e) return null
    if (k === 'small') return e.by_size.small.recall
    if (k === 'fps') return e.latency_ms.mean ? 1000 / e.latency_ms.mean : null
    return (e.overall as unknown as Record<string, number | null>)[k] ?? null
  }
  const sorted = [...rows].sort((a, b) => (val(b, OBJ[objective].key) ?? -1) - (val(a, OBJ[objective].key) ?? -1))
  const hl = (k: string) => (OBJ[objective].key === k ? 'bg-info-bg/60' : undefined)

  return (
    <div className="flex flex-col gap-3">
      <PageHeader title="Model comparison" description="Measured results of each registered version on the same evaluation split. No model is labelled “best” — choose the objective that matters for your operation; the table sorts by it."
        actions={<Link href="/evaluation"><Button icon={Target}>Run evaluation</Button></Link>} />
      <Panel>
        <div className="grid gap-3 md:grid-cols-[260px_260px_1fr] md:items-end">
          <Field label="Optimization objective">
            <Select value={objective} onChange={(e) => setObjective(e.target.value as Objective)}>
              {Object.entries(OBJ).map(([k, v]) => <option key={k} value={k}>{v.label}</option>)}
            </Select>
          </Field>
          <Field label="Evaluation split">
            <Select value={ds} onChange={(e) => setDataset(e.target.value)} disabled={!datasets.length}>
              {datasets.length === 0 && <option value="">No evaluations yet</option>}
              {datasets.map((d) => <option key={d}>{d}</option>)}
            </Select>
          </Field>
          <p className="text-[12px] text-fg-3">{OBJ[objective].note}</p>
        </div>
      </Panel>
      <Panel noPad>
        {models.error && !models.data ? <ErrorState error={models.error} onRetry={models.refetch} compact /> : (
          <DataTable<Row> rows={sorted} rowKey={(r) => r.m.version}
            empty={<EmptyState icon={Boxes} compact title="No models registered" />}
            columns={[
              { key: 'model', header: 'Model', cell: (r) => <span className="text-fg">{r.m.architecture}</span>, sortValue: (r) => r.m.architecture },
              { key: 'version', header: 'Version', cell: (r) => <span className="flex items-center gap-1.5 font-mono text-fg">{r.m.version}{r.m.active && <StatusBadge status="active" />}</span> },
              { key: 'p', header: 'Precision', align: 'right', className: hl('precision'), cell: (r) => pct(val(r, 'precision')), sortValue: (r) => val(r, 'precision') },
              { key: 'r', header: 'Recall', align: 'right', className: hl('recall'), cell: (r) => pct(val(r, 'recall')), sortValue: (r) => val(r, 'recall') },
              { key: 'f1', header: 'F1', align: 'right', className: hl('f1'), cell: (r) => pct(val(r, 'f1')), sortValue: (r) => val(r, 'f1') },
              { key: 'm50', header: 'mAP50', align: 'right', cell: (r) => pct(val(r, 'mAP50')), sortValue: (r) => val(r, 'mAP50') },
              { key: 'm95', header: 'mAP50-95', align: 'right', cell: (r) => pct(val(r, 'mAP50_95')), sortValue: (r) => val(r, 'mAP50_95') },
              { key: 'small', header: 'Small recall', align: 'right', className: hl('small'), cell: (r) => r.ev ? `${pct(r.ev.by_size.small.recall)}${r.ev.by_size.small.instances ? '' : ' (n=0)'}` : '—', sortValue: (r) => val(r, 'small') },
              { key: 'fps', header: 'FPS', align: 'right', className: hl('fps'), cell: (r) => num(val(r, 'fps'), 1), sortValue: (r) => val(r, 'fps') },
              { key: 'lat', header: 'Latency', align: 'right', cell: (r) => r.ev?.latency_ms.mean ? `${num(r.ev.latency_ms.mean, 0)} ms` : '—', sortValue: (r) => r.ev?.latency_ms.mean ?? null },
              { key: 'gpu', header: 'GPU memory', align: 'right', cell: () => <Unavailable>not measured (CPU)</Unavailable>, hideable: true },
              { key: 'thr', header: 'Op. threshold', align: 'right', cell: (r) => r.ev ? r.ev.operating_threshold.toFixed(2) : '—', hideable: true },
              { key: 'ev', header: '', cell: (r) => r.ev ? <Link className="text-info hover:underline" href={`/evaluation?id=${r.ev.id}`}>details</Link> : <Link className="text-fg-3 hover:text-info" href={`/evaluation?model=${encodeURIComponent(r.m.version)}`}>evaluate</Link> },
            ]} />
        )}
      </Panel>
      {ds && <p className="text-[11px] text-fg-3">Numbers are from the platform’s own evaluation on <b>{ds}</b>. <Badge tone="warn">note</Badge> If this split is synthetic, the results do not predict real CCTV performance.</p>}
    </div>
  )
}
