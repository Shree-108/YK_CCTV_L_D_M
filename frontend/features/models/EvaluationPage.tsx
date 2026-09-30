'use client'

import Link from 'next/link'
import { useRouter, useSearchParams } from 'next/navigation'
import { useEffect, useState } from "react"
import { AlertTriangle, Play, SlidersHorizontal, Target } from 'lucide-react'
import { Api } from '../../api/endpoints'
import type { Evaluation, EvaluationListItem } from '../../api/types'
import { invalidate, useQuery } from '../../hooks/useQuery'
import { className as cls, dateTime, num, pct } from '../../lib/format'
import { useApp } from '../../stores/app'
import { Badge, Button, Field, PageHeader, Panel, Select, StatusBadge, Toggle } from '../../components/ui/primitives'
import { ConfirmDialog } from '../../components/ui/overlay'
import { Legend, LineChart, XYChart } from '../../components/ui/charts'
import { MetricCard } from '../../components/ui/kpi'
import { DataTable } from '../../components/ui/table'
import { EmptyState, ErrorState, LoadingState } from '../../components/ui/states'

export default function EvaluationPage() {
  const params = useSearchParams()
  const router = useRouter()
  const { toast } = useApp()
  const models = useQuery('models', Api.models)
  const datasets = useQuery('datasets', Api.datasets)
  const list = useQuery('evaluations', Api.evaluations, { poll: 3000 })
  const [cfg, setCfg] = useState({ model: params.get('model') ?? '', dataset: '', split: 'test', compare: true })
  const [starting, setStarting] = useState(false)
  const id = params.get('id')
  const evals = list.data?.evaluations ?? []
  const selectedId = id ?? evals.find((e) => e.status === 'done')?.id ?? null
  const ev = useQuery(selectedId ? `eval:${selectedId}` : null, () => Api.evaluation(selectedId!))
  const running = evals.filter((e) => e.status === 'running' || e.status === 'queued')
  useEffect(() => { if (!cfg.dataset && datasets.data?.datasets[0]) setCfg((c) => ({ ...c, dataset: datasets.data!.datasets[0].name })) }, [datasets.data, cfg.dataset])
  useEffect(() => { if (!running.length) invalidate('models') }, [running.length])

  const start = async () => {
    setStarting(true)
    try {
      const r = await Api.startEvaluation({ model_version: cfg.model || undefined, dataset: cfg.dataset, split: cfg.split, compare_tiling: cfg.compare })
      toast({ tone: 'info', title: 'Evaluation started', body: r.job_id })
      invalidate('evaluations')
      router.replace(`/evaluation?id=${r.job_id}`)
    } catch (e) {
      toast({ tone: 'crit', title: 'Could not start evaluation', body: (e as Error).message })
    } finally {
      setStarting(false)
    }
  }
  const current = evals.find((e) => e.id === selectedId)
  useEffect(() => { if (current?.status === 'done' && ev.error) invalidate(`eval:${current.id}`) }, [current?.status]) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="flex flex-col gap-3">
      <PageHeader title="Evaluation" description="Runs the production detector over a labelled split and measures it against ground truth: overall and per-class metrics, recall by object size, a confidence-threshold sweep, and (optionally) full-frame vs tiled inference." />
      <Panel title="Run evaluation" icon={Target}>
        <div className="grid items-end gap-3 md:grid-cols-[1fr_1fr_120px_auto_auto]">
          <Field label="Model"><Select value={cfg.model} onChange={(e) => setCfg({ ...cfg, model: e.target.value })}>
            <option value="">Active ({models.data?.active ?? '—'})</option>
            {models.data?.models.filter((m) => !m.active).map((m) => <option key={m.version}>{m.version}</option>)}</Select></Field>
          <Field label="Evaluation dataset"><Select value={cfg.dataset} onChange={(e) => setCfg({ ...cfg, dataset: e.target.value })}>
            {datasets.data?.datasets.map((d) => <option key={d.name}>{d.name}</option>)}</Select></Field>
          <Field label="Split"><Select value={cfg.split} onChange={(e) => setCfg({ ...cfg, split: e.target.value })}>{['test', 'val', 'train'].map((s) => <option key={s}>{s}</option>)}</Select></Field>
          <div className="pb-1"><Toggle label="Compare tiled vs full" checked={cfg.compare} onChange={(v) => setCfg({ ...cfg, compare: v })} /></div>
          <Button variant="primary" icon={Play} loading={starting} disabled={!cfg.dataset} onClick={start}>Evaluate</Button>
        </div>
        {cfg.split === 'train' && <p className="mt-2 flex items-center gap-1.5 text-[12px] text-warn"><AlertTriangle className="size-3.5" />Evaluating on the training split overstates performance.</p>}
      </Panel>

      <div className="grid gap-3 xl:grid-cols-[320px_minmax(0,1fr)]">
        <Panel title="Evaluations" noPad>
          {list.error && !list.data ? <ErrorState error={list.error} compact onRetry={list.refetch} /> : (
            <ul className="max-h-[70vh] divide-y divide-line overflow-y-auto">
              {evals.length === 0 && <li><EmptyState compact title="No evaluations yet" body="Run an evaluation to generate model performance metrics." /></li>}
              {evals.map((e) => <EvalListItem key={e.id} e={e} active={e.id === selectedId} onClick={() => router.replace(`/evaluation?id=${e.id}`)} />)}
            </ul>
          )}
        </Panel>
        <div className="min-w-0">
          {current && (current.status === 'running' || current.status === 'queued') ? (
            <Panel><div className="py-10 text-center"><p className="mb-3 text-[13px] text-fg">Evaluating {current.input_name}… {pct(current.progress, 0)}</p>
              <div className="mx-auto h-1.5 max-w-md overflow-hidden rounded-full bg-panel-2"><div className="h-full bg-info transition-all" style={{ width: `${current.progress * 100}%` }} /></div></div></Panel>
          ) : current?.status === 'failed' ? <Panel><ErrorState error={new Error(current.error ?? 'failed')} title="Evaluation failed" /></Panel>
            : !selectedId ? <Panel><EmptyState icon={Target} title="No evaluation selected" body="Run an evaluation to generate model performance metrics." /></Panel>
            : ev.error ? <Panel><ErrorState error={ev.error} onRetry={ev.refetch} /></Panel>
            : !ev.data ? <Panel><LoadingState rows={8} /></Panel>
            : <EvalReport e={ev.data} />}
        </div>
      </div>
    </div>
  )
}

function EvalListItem({ e, active, onClick }: { e: EvaluationListItem; active: boolean; onClick: () => void }) {
  return (
    <li>
      <button onClick={onClick} className={`flex w-full flex-col gap-0.5 px-3 py-2 text-left hover:bg-hover ${active ? 'bg-info-bg' : ''}`}>
        <div className="flex items-center justify-between gap-2">
          <span className="truncate font-mono text-[12px] text-fg">{e.summary?.model_version ?? e.input_name}</span>
          <StatusBadge status={e.status} />
        </div>
        <div className="flex justify-between text-[11px] text-fg-3">
          <span>{e.input_name}</span>
          <span>{e.summary ? `mAP50 ${pct(e.summary.overall.mAP50, 1)}` : pct(e.progress, 0)}</span>
        </div>
        <span className="text-[11px] text-fg-3">{dateTime(e.created_at)}</span>
      </button>
    </li>
  )
}

function EvalReport({ e }: { e: Evaluation }) {
  const { isAdmin, toast } = useApp()
  const sweep = e.threshold_sweep
  const [thrIdx, setThrIdx] = useState(() => Math.max(0, sweep.findIndex((s) => Math.abs(s.threshold - e.operating_threshold) < 1e-6)))
  const [confirm, setConfirm] = useState(false)
  const [saving, setSaving] = useState(false)
  const cur = sweep[thrIdx]
  const o = e.overall
  const smallWarn = e.by_size.small.instances > 0 && (e.by_size.small.recall ?? 0) < (o.recall - 0.1)

  const apply = async () => {
    setSaving(true)
    try {
      const s = await Api.settings()
      const t = cur.threshold
      await Api.saveSettings({ confidence: { confirmed_threshold: t, candidate_threshold: Math.min(s.settings.confidence.candidate_threshold, t), weak_threshold: Math.min(s.settings.confidence.weak_threshold, t) } })
      toast({ tone: 'ok', title: `Confirmed threshold set to ${t.toFixed(2)}`, body: 'Applies to all cameras without an override.' })
      invalidate('settings')
    } catch (err) {
      toast({ tone: 'crit', title: 'Could not apply threshold', body: (err as Error).message })
    } finally {
      setSaving(false)
      setConfirm(false)
    }
  }

  return (
    <div className="flex flex-col gap-3">
      <Panel title={`${e.model_version} on ${e.dataset}/${e.split}`} subtitle={`${e.images} images · ${e.gt_objects} labelled objects · operating threshold ${e.operating_threshold.toFixed(2)} · IoU 0.5 · ${dateTime(e.created_at)}`}>
        <div className="grid grid-cols-2 gap-2 md:grid-cols-4 xl:grid-cols-7">
          <MetricCard label="Precision" value={pct(o.precision)} />
          <MetricCard label="Recall" value={pct(o.recall)} />
          <MetricCard label="F1" value={pct(o.f1)} />
          <MetricCard label="mAP50" value={pct(o.mAP50)} />
          <MetricCard label="mAP50-95" value={pct(o.mAP50_95)} />
          <MetricCard label="FP on negatives" value={num(e.negatives.false_positives)} hint={e.negatives.images ? `${e.negatives.images} clean images` : 'no negatives'} tone={e.negatives.false_positives ? 'warn' : undefined} />
          <MetricCard label="Latency" value={`${num(e.latency_ms.mean, 0)} ms`} hint={`p95 ${num(e.latency_ms.p95, 0)} ms`} />
        </div>
        {e.gt_objects < 200 && <p className="mt-2 flex items-center gap-1.5 text-[12px] text-warn"><AlertTriangle className="size-3.5" />Only {e.gt_objects} labelled objects — metrics have wide uncertainty. Check whether this split is synthetic before drawing conclusions.</p>}
      </Panel>

      <div className="grid gap-3 lg:grid-cols-2">
        <Panel title="Precision–recall curve" subtitle="From the confidence sweep (IoU 0.5)">
          <XYChart ariaLabel="Precision recall curve" xLabel="Recall" yLabel="Precision"
            series={[{ name: 'PR', color: 'var(--info)', points: sweep.map((s) => ({ x: s.recall, y: s.precision, label: `t=${s.threshold} P=${pct(s.precision)} R=${pct(s.recall)}` })) }]}
            marker={{ x: cur.recall, y: cur.precision, label: `threshold ${cur.threshold}` }} />
        </Panel>
        <Panel title="Metrics vs threshold">
          <LineChart ariaLabel="Precision, recall and F1 versus confidence threshold" labels={sweep.map((s) => s.threshold.toFixed(2))} height={220} yMax={1} yFormat={(v) => pct(v, 0)} markerX={thrIdx}
            series={[{ name: 'Precision', values: sweep.map((s) => s.precision), color: 'var(--ok)' }, { name: 'Recall', values: sweep.map((s) => s.recall), color: 'var(--warn)' }, { name: 'F1', values: sweep.map((s) => s.f1), color: 'var(--info)' }]} />
          <Legend items={[{ name: 'Precision', color: 'var(--ok)' }, { name: 'Recall', color: 'var(--warn)' }, { name: 'F1', color: 'var(--info)' }]} />
        </Panel>
      </div>

      <Panel title="Threshold lab" icon={SlidersHorizontal} subtitle="Move the threshold to see the measured trade-off on this split. Higher = fewer false alarms but more missed litter.">
        <div className="grid gap-4 lg:grid-cols-[1fr_auto]">
          <div>
            <input type="range" min={0} max={sweep.length - 1} step={1} value={thrIdx} onChange={(ev) => setThrIdx(Number(ev.target.value))}
              aria-label="Confidence threshold" aria-valuetext={cur.threshold.toFixed(2)} className="w-full" />
            <div className="flex justify-between font-mono text-[10px] text-fg-3"><span>{sweep[0].threshold.toFixed(2)}</span><span className="text-[13px] font-semibold text-fg">{cur.threshold.toFixed(2)}</span><span>{sweep[sweep.length - 1].threshold.toFixed(2)}</span></div>
            <div className="mt-3 grid grid-cols-5 gap-2">
              <MetricCard label="Precision" value={pct(cur.precision)} />
              <MetricCard label="Recall" value={pct(cur.recall)} />
              <MetricCard label="F1" value={pct(cur.f1)} />
              <MetricCard label="False positives" value={cur.fp} tone={cur.fp ? 'crit' : undefined} />
              <MetricCard label="False negatives" value={cur.fn} tone={cur.fn ? 'warn' : undefined} />
            </div>
            <p className="mt-2 text-[11px] text-fg-3">Best F1 on this split at {e.best_f1_threshold.toFixed(2)}. This is a measurement, not a recommendation — confirm on real CCTV data.</p>
          </div>
          <div className="flex flex-col justify-center gap-2">
            <Button variant="ghost" onClick={() => setThrIdx(sweep.findIndex((s) => s.threshold === e.best_f1_threshold))}>Jump to best F1</Button>
            <Button variant="primary" disabled={!isAdmin} title={!isAdmin ? 'Admin sign-in required' : undefined} onClick={() => setConfirm(true)}>Apply as system threshold</Button>
          </div>
        </div>
      </Panel>

      <div className="grid gap-3 lg:grid-cols-3">
        <Panel title="Recall by object size" subtitle="COCO buckets on original-image pixels">
          <div className="flex flex-col gap-2">
            {(['small', 'medium', 'large'] as const).map((k) => (
              <div key={k} className="flex items-center justify-between rounded-md border border-line bg-panel-2 px-2.5 py-1.5 text-[12.5px]">
                <span className="text-fg-2">{k} <span className="text-fg-3">{k === 'small' ? '< 32²' : k === 'medium' ? '32²–96²' : '> 96²'} px</span></span>
                <span className="tabular font-mono text-fg">{pct(e.by_size[k].recall)} <span className="text-fg-3">({e.by_size[k].detected}/{e.by_size[k].instances})</span></span>
              </div>
            ))}
            <div className="flex items-center justify-between px-2.5 text-[12px] text-fg-3"><span>tiny (&lt; 16² px)</span><span className="font-mono">{pct(e.tiny_recall.recall)} ({e.tiny_recall.detected}/{e.tiny_recall.instances})</span></div>
            {smallWarn && <p className="text-[12px] text-warn">Small-object recall is well below overall recall — overall mAP hides this weakness.</p>}
          </div>
        </Panel>
        <Panel title="Priority litter recall">
          <div className="flex flex-col gap-2">
            {Object.entries(e.report_groups).map(([k, v]) => (
              <div key={k} className="flex items-center justify-between rounded-md border border-line bg-panel-2 px-2.5 py-1.5 text-[12.5px]">
                <span className="text-fg-2">{cls(k)}</span>
                <span className="tabular font-mono text-fg">{v.instances ? `${pct(v.recall)} (${v.detected}/${v.instances})` : 'no labels'}</span>
              </div>
            ))}
          </div>
        </Panel>
        <Panel title="Tiled vs full-frame inference">
          {e.tiling_comparison && e.modes ? (
            <div className="flex flex-col gap-2 text-[12.5px]">
              {(['full_frame', 'tiled'] as const).map((k) => (
                <div key={k} className="grid grid-cols-4 gap-1 rounded-md border border-line bg-panel-2 px-2.5 py-1.5">
                  <span className="text-fg-2">{k.replace('_', ' ')}</span>
                  <span className="tabular font-mono">mAP {pct(e.modes![k].overall.mAP50, 0)}</span>
                  <span className="tabular font-mono">small {pct(e.modes![k].by_size.small.recall, 0)}</span>
                  <span className="tabular font-mono">{num(e.modes![k].latency_ms.mean, 0)} ms</span>
                </div>
              ))}
              <p className={e.tiling_comparison.recommend_tiling ? 'text-ok' : 'text-fg-2'}>
                {e.tiling_comparison.recommend_tiling ? 'Tiling improved results on this split' : 'Tiling did not measurably improve results on this split'} (ΔmAP50 {(e.tiling_comparison.delta_mAP50 * 100).toFixed(1)} pts, Δsmall recall {(e.tiling_comparison.delta_small_recall * 100).toFixed(1)} pts, {e.tiling_comparison.latency_factor}× latency).
              </p>
              <p className="text-[11px] text-fg-3">{e.tiling_comparison.note}</p>
            </div>
          ) : <EmptyState compact title="Not measured" body="Re-run with “Compare tiled vs full” enabled." />}
        </Panel>
      </div>

      <Panel title="Per-class metrics" subtitle={`At threshold ${e.operating_threshold.toFixed(2)}, IoU 0.5`} noPad>
        <DataTable rows={e.per_class} rowKey={(r) => r.class} dense pageSize={50}
          columns={[
            { key: 'c', header: 'Class', cell: (r) => <span className="text-fg">{cls(r.class)}</span>, sortValue: (r) => r.class },
            { key: 'n', header: 'Instances', align: 'right', cell: (r) => r.instances, sortValue: (r) => r.instances },
            { key: 'ap', header: 'AP50', align: 'right', cell: (r) => pct(r.ap50), sortValue: (r) => r.ap50 },
            { key: 'ap95', header: 'AP50-95', align: 'right', cell: (r) => pct(r.ap50_95), sortValue: (r) => r.ap50_95 },
            { key: 'p', header: 'Precision', align: 'right', cell: (r) => pct(r.precision), sortValue: (r) => r.precision },
            { key: 'r', header: 'Recall', align: 'right', cell: (r) => <span className={r.instances && r.recall < 0.5 ? 'text-crit' : undefined}>{r.instances ? pct(r.recall) : '—'}</span>, sortValue: (r) => r.recall },
            { key: 'f1', header: 'F1', align: 'right', cell: (r) => pct(r.f1), sortValue: (r) => r.f1 },
            { key: 'fp', header: 'FP', align: 'right', cell: (r) => r.fp },
            { key: 'fn', header: 'FN', align: 'right', cell: (r) => r.fn },
          ]} />
      </Panel>
      <Panel title="Hardest images" subtitle="Most misses + false positives" noPad
        actions={<Link href="/hard-examples" className="text-[12px] text-info hover:underline">{e.hard_examples_found} hard examples mined →</Link>}>
        <DataTable rows={[...e.per_image].sort((a, b) => (b.fn + b.fp) - (a.fn + a.fp)).slice(0, 15)} rowKey={(r) => r.id} dense
          columns={[
            { key: 'id', header: 'Image', cell: (r) => <Link className="font-mono text-info hover:underline" href={`/ground-truth?dataset=${encodeURIComponent(e.dataset)}&id=${encodeURIComponent(r.id)}`}>{r.id}</Link> },
            { key: 'gt', header: 'GT', align: 'right', cell: (r) => r.gt },
            { key: 'tp', header: 'TP', align: 'right', cell: (r) => r.tp },
            { key: 'fp', header: 'FP', align: 'right', cell: (r) => <span className={r.fp ? 'text-crit' : undefined}>{r.fp}</span> },
            { key: 'fn', header: 'FN', align: 'right', cell: (r) => <span className={r.fn ? 'text-warn' : undefined}>{r.fn}</span> },
            { key: 'neg', header: '', cell: (r) => r.negative ? <Badge>negative</Badge> : null },
          ]} />
      </Panel>
      <ConfirmDialog open={confirm} onClose={() => setConfirm(false)} onConfirm={apply} pending={saving} title="Apply confirmed threshold?" confirmLabel="Apply"
        body={<>Set the global confirmed threshold to <b className="font-mono">{cur.threshold.toFixed(2)}</b>. On this split that gives precision {pct(cur.precision)}, recall {pct(cur.recall)} ({cur.fp} FP / {cur.fn} FN). Cameras with their own override keep it.</>} />
    </div>
  )
}

