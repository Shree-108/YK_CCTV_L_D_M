'use client'

import { useRouter, useSearchParams } from 'next/navigation'
import { useState } from 'react'
import { FlaskConical, PackagePlus } from 'lucide-react'
import { url } from '../../api/client'
import { Api } from '../../api/endpoints'
import type { Experiment } from '../../api/types'
import { invalidate, useQuery } from '../../hooks/useQuery'
import { dateTime, pct } from '../../lib/format'
import { useApp } from '../../stores/app'
import { Badge, Button, Field, Input, KeyValue, PageHeader, Panel } from '../../components/ui/primitives'
import { Drawer, Modal } from '../../components/ui/overlay'
import { Legend, LineChart } from '../../components/ui/charts'
import { DataTable } from '../../components/ui/table'
import { EmptyState, ErrorState, LoadingState } from '../../components/ui/states'

export default function ExperimentsPage() {
  const params = useSearchParams()
  const router = useRouter()
  const { isAdmin } = useApp()
  const [register, setRegister] = useState<string | null>(null)
  const id = params.get('id')
  return (
    <div className="flex flex-col gap-3">
      <PageHeader title="Experiments" description="Training runs discovered on disk (Ultralytics results.csv + args.yaml). Click a run for its configuration, curves and artifacts." />
      <Panel noPad><ExperimentsTable onOpen={(e) => router.replace(`/experiments?id=${encodeURIComponent(e.id)}`)} onRegister={isAdmin ? setRegister : undefined} selected={id} /></Panel>
      <ExperimentDrawer id={id} onClose={() => router.replace('/experiments')} onRegister={isAdmin ? setRegister : undefined} />
      <RegisterDialog experimentId={register} onClose={() => setRegister(null)} />
    </div>
  )
}

export function ExperimentsTable({ onOpen, onRegister, selected }: { onOpen?: (e: Experiment) => void; onRegister?: (id: string) => void; selected?: string | null }) {
  const router = useRouter()
  const q = useQuery('experiments', Api.experiments)
  if (q.error && !q.data) return <ErrorState error={q.error} onRetry={q.refetch} compact />
  if (!q.data) return <div className="p-3"><LoadingState /></div>
  return (
    <DataTable<Experiment> rows={q.data.experiments} rowKey={(e) => e.id} selectedKey={selected}
      onRowClick={(e) => (onOpen ? onOpen(e) : router.push(`/experiments?id=${encodeURIComponent(e.id)}`))}
      empty={<EmptyState icon={FlaskConical} compact title="No training runs found" body="Runs saved under runs/ or backend/runs/ are listed automatically." />}
      columns={[
        { key: 'id', header: 'Experiment', cell: (e) => <span className="font-mono text-fg">{e.name}</span>, sortValue: (e) => e.name },
        { key: 'model', header: 'Model', cell: (e) => e.model },
        { key: 'ds', header: 'Dataset', cell: (e) => e.dataset },
        { key: 'date', header: 'Date', cell: (e) => dateTime(e.date), sortValue: (e) => e.date },
        { key: 'status', header: 'Status', cell: (e) => <Badge tone={e.status === 'complete' ? 'ok' : 'warn'}>{e.status}</Badge> },
        { key: 'ep', header: 'Epochs', align: 'right', cell: (e) => `${e.epochs_run}${e.epochs_planned ? ` / ${e.epochs_planned}` : ''}` },
        { key: 'sz', header: 'imgsz', align: 'right', cell: (e) => e.imgsz ?? '—' },
        { key: 'm', header: 'mAP50 (val)', align: 'right', cell: (e) => pct(e.best?.mAP50), sortValue: (e) => e.best?.mAP50 ?? null },
        { key: 'm95', header: 'mAP50-95', align: 'right', cell: (e) => pct(e.best?.mAP50_95), sortValue: (e) => e.best?.mAP50_95 ?? null },
        { key: 'reg', header: '', cell: (e) => onRegister && e.has_best_weights ? (
          <Button size="sm" variant="ghost" icon={PackagePlus} onClick={(ev) => { ev.stopPropagation(); onRegister(e.id) }}>Register</Button>) : null },
      ]} />
  )
}

function ExperimentDrawer({ id, onClose, onRegister }: { id: string | null; onClose: () => void; onRegister?: (id: string) => void }) {
  const q = useQuery(id ? `experiment:${id}` : null, () => Api.experiment(id!))
  const e = q.data
  const ep = e?.epochs ?? []
  const labels = ep.map((r) => String(r.epoch ?? ''))
  const s = (k: string) => ep.map((r) => r[k] ?? null)
  return (
    <Drawer open={!!id} onClose={onClose} title={e?.name ?? 'Experiment'} subtitle={e && <span className="font-mono">{e.id}</span>} width="w-[760px]"
      footer={e && onRegister && e.has_best_weights && <Button variant="primary" icon={PackagePlus} onClick={() => onRegister(e.id)}>Register as model version</Button>}>
      {q.error ? <ErrorState error={q.error} compact /> : !e ? <LoadingState rows={6} /> : (
        <div className="flex flex-col gap-4">
          <KeyValue cols={3} items={[
            ['Model', e.model], ['Dataset', e.dataset], ['Status', e.status], ['Epochs', `${e.epochs_run} / ${e.epochs_planned ?? '—'}`],
            ['Image size', e.imgsz ?? '—'], ['Date', dateTime(e.date)], ['Best epoch', e.best?.epoch ?? '—'],
            ['Best mAP50', pct(e.best?.mAP50)], ['Best mAP50-95', pct(e.best?.mAP50_95)],
          ]} />
          {ep.length < 2 && <p className="text-[12px] text-warn">Only {ep.length} epoch(s) recorded — curves need at least two points.</p>}
          <Panel title="Loss">
            <LineChart ariaLabel="Training and validation losses" labels={labels} height={170} yFormat={(v) => v.toFixed(2)}
              series={[
                { name: 'train box', values: s('train/box_loss'), color: 'var(--info)' }, { name: 'val box', values: s('val/box_loss'), color: 'var(--info)', dashed: true },
                { name: 'train cls', values: s('train/cls_loss'), color: 'var(--exp)' }, { name: 'val cls', values: s('val/cls_loss'), color: 'var(--exp)', dashed: true },
              ]} />
            <Legend items={[{ name: 'box (solid train / dashed val)', color: 'var(--info)' }, { name: 'cls', color: 'var(--exp)' }]} />
          </Panel>
          <Panel title="Validation metrics">
            <LineChart ariaLabel="Precision, recall and mAP per epoch" labels={labels} height={170} yMax={1} yFormat={(v) => pct(v, 0)}
              series={[
                { name: 'precision', values: s('metrics/precision(B)'), color: 'var(--ok)' }, { name: 'recall', values: s('metrics/recall(B)'), color: 'var(--warn)' },
                { name: 'mAP50', values: s('metrics/mAP50(B)'), color: 'var(--info)' }, { name: 'mAP50-95', values: s('metrics/mAP50-95(B)'), color: 'var(--exp)' },
              ]} />
            <Legend items={[{ name: 'precision', color: 'var(--ok)' }, { name: 'recall', color: 'var(--warn)' }, { name: 'mAP50', color: 'var(--info)' }, { name: 'mAP50-95', color: 'var(--exp)' }]} />
          </Panel>
          <div className="grid gap-3 md:grid-cols-2">
            <Panel title="Training configuration"><KeyValue items={Object.entries(e.config).map(([k, v]) => [k, <span key={k} className="font-mono text-[12px]">{String(v)}</span>])} /></Panel>
            <Panel title="Augmentation"><KeyValue items={Object.entries(e.augmentation).map(([k, v]) => [k, <span key={k} className="font-mono text-[12px]">{String(v)}</span>])} /></Panel>
          </div>
          <Panel title="Artifacts">
            {e.artifacts.length === 0 ? <p className="text-[12px] text-fg-3">No plots were saved for this run.</p> : (
              <div className="grid gap-2 sm:grid-cols-2">
                {e.artifacts.map((a) => (
                  <a key={a} href={url(`/api/experiments/${encodeURIComponent(e.id)}/artifacts/${a}`)} target="_blank" rel="noreferrer" className="group rounded-md border border-line p-1.5 hover:border-line-strong">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={url(`/api/experiments/${encodeURIComponent(e.id)}/artifacts/${a}`)} alt={a} loading="lazy" className="aspect-video w-full rounded-sm bg-white object-contain" />
                    <div className="mt-1 truncate text-[11px] text-fg-3 group-hover:text-fg">{a}</div>
                  </a>
                ))}
              </div>
            )}
          </Panel>
          <p className="text-[11px] text-fg-3">Hardware and full training logs are not recorded in the run folder; only what Ultralytics saved is shown.</p>
        </div>
      )}
    </Drawer>
  )
}

export function RegisterDialog({ experimentId, onClose }: { experimentId: string | null; onClose: () => void }) {
  const { toast } = useApp()
  const [version, setVersion] = useState('')
  const [notes, setNotes] = useState('')
  const [pending, setPending] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const submit = async () => {
    if (!experimentId) return
    setPending(true)
    setErr(null)
    try {
      await Api.registerModel({ experiment_id: experimentId, version, notes })
      toast({ tone: 'ok', title: `Registered ${version}`, body: 'Evaluate it before deploying.' })
      invalidate('models')
      setVersion('')
      setNotes('')
      onClose()
    } catch (e) {
      setErr((e as Error).message)
    } finally {
      setPending(false)
    }
  }
  return (
    <Modal open={!!experimentId} onClose={onClose} title="Register model version"
      footer={<><Button variant="ghost" onClick={onClose}>Cancel</Button><Button variant="primary" loading={pending} disabled={!/^[A-Za-z0-9_.-]+$/.test(version)} onClick={submit}>Register</Button></>}>
      <div className="flex flex-col gap-3">
        <p className="text-[12px] text-fg-3">Copies <span className="font-mono">weights/best.pt</span> of <span className="font-mono">{experimentId}</span> into a new immutable version. It is not activated.</p>
        <Field label="Version id" hint="letters, digits, . _ -"><Input value={version} onChange={(e) => setVersion(e.target.value)} placeholder="v2_litter_tune" data-autofocus /></Field>
        <Field label="Notes"><Input value={notes} onChange={(e) => setNotes(e.target.value)} placeholder="Dataset, caveats…" /></Field>
        {err && <p className="text-[12px] text-crit">{err}</p>}
      </div>
    </Modal>
  )
}

