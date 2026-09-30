'use client'

import Link from 'next/link'
import { useRouter, useSearchParams } from 'next/navigation'
import { useState } from 'react'
import { Archive, BarChart3, Boxes, FlaskConical, GitCompare, History, Rocket, Sparkles, Target, Undo2 } from 'lucide-react'
import { Api } from '../../api/endpoints'
import type { ModelInfo } from '../../api/types'
import { invalidate, useQuery } from '../../hooks/useQuery'
import { dateTime, num, pct } from '../../lib/format'
import { useApp } from '../../stores/app'
import { Badge, Button, KeyValue, PageHeader, Panel, StatusBadge, Tabs, Unavailable } from '../../components/ui/primitives'
import { ConfirmDialog, Drawer } from '../../components/ui/overlay'
import { DataTable } from '../../components/ui/table'
import { EmptyState, ErrorState, LoadingState } from '../../components/ui/states'
import { ExperimentsTable, RegisterDialog } from './ExperimentsPage'

type Tab = 'installed' | 'training' | 'evaluation' | 'deployment' | 'versions'

export default function ModelLabPage() {
  const params = useSearchParams()
  const router = useRouter()
  const { isAdmin, toast } = useApp()
  const models = useQuery('models', Api.models)
  const events = useQuery('events:models', () => Api.events(300))
  const [tab, setTab] = useState<Tab>('installed')
  const [deploy, setDeploy] = useState<ModelInfo | null>(null)
  const [busy, setBusy] = useState(false)
  const [register, setRegister] = useState<string | null>(null)
  const open = params.get('v')
  const list = models.data?.models ?? []
  const openModel = list.find((m) => m.version === open) ?? null
  const history = (events.data?.events ?? []).filter((e) => e.source === 'models')

  const doDeploy = async () => {
    if (!deploy) return
    setBusy(true)
    try {
      await Api.activateModel(deploy.version)
      toast({ tone: 'ok', title: `Deployed ${deploy.version}`, body: 'Live cameras switch on their next frame; the previous model is kept.' })
      invalidate('models')
      invalidate('health')
      invalidate('events')
    } catch (e) {
      toast({ tone: 'crit', title: 'Deployment failed — previous model still active', body: (e as Error).message })
    } finally {
      setBusy(false)
      setDeploy(null)
    }
  }

  return (
    <div className="flex flex-col gap-3">
      <PageHeader title="Model Lab" badge={<Badge tone="exp">AI</Badge>}
        description="Registered YOLO model versions, their measured metrics, deployment and rollback. Versions are immutable and never deleted automatically."
        actions={<>
          <Link href="/models"><Button icon={GitCompare}>Compare</Button></Link>
          <Link href="/evaluation"><Button icon={Target}>Evaluate</Button></Link>
        </>} />
      <Tabs value={tab} onChange={setTab} tabs={[
        { value: 'installed', label: 'Installed models', count: list.length }, { value: 'training', label: 'Training' },
        { value: 'evaluation', label: 'Evaluation' }, { value: 'deployment', label: 'Deployment' }, { value: 'versions', label: 'Versions', count: history.length || null },
      ]} />
      {models.error && !models.data ? <ErrorState error={models.error} onRetry={models.refetch} /> : !models.data ? <LoadingState rows={4} /> : (
        <>
          {tab === 'installed' && (list.length === 0 ? (
            <EmptyState icon={Boxes} title="No models registered" body="Register a training run from the Training tab, or place a model folder in models/litter_yolo/." />
          ) : (
            <div className="grid gap-3 lg:grid-cols-2 2xl:grid-cols-3">
              {list.map((m) => <ModelCard key={m.version} m={m} isAdmin={isAdmin} onOpen={() => router.replace(`/model-lab?v=${encodeURIComponent(m.version)}`)} onDeploy={() => setDeploy(m)} />)}
            </div>
          ))}
          {tab === 'training' && (
            <Panel title="Training runs" icon={FlaskConical} noPad
              actions={<Link href="/experiments" className="text-[12px] text-info hover:underline">Experiments →</Link>}>
              <div className="border-b border-line p-3 text-[12px] text-fg-3">
                Training runs are executed with <code className="font-mono text-fg-2">training/train.py</code> (GPU recommended, e.g. Colab) and appear here automatically.
                Starting training from the UI is not available: this machine has no GPU and training must not compete with live inference.
                Register a finished run to make it a deployable model version.
              </div>
              <ExperimentsTable onRegister={isAdmin ? setRegister : undefined} />
            </Panel>
          )}
          {tab === 'evaluation' && (
            <Panel title="Latest evaluation per model" noPad actions={<Link href="/evaluation"><Button size="sm" icon={Target}>Run evaluation</Button></Link>}>
              <DataTable rows={list} rowKey={(m) => m.version} dense
                columns={[
                  { key: 'v', header: 'Version', cell: (m) => <span className="font-mono text-fg">{m.version}</span> },
                  { key: 'd', header: 'Evaluated on', cell: (m) => m.latest_evaluation ? `${m.latest_evaluation.dataset} / ${m.latest_evaluation.split}` : <Unavailable>not evaluated</Unavailable> },
                  { key: 'p', header: 'Precision', align: 'right', cell: (m) => pct(m.latest_evaluation?.overall.precision) },
                  { key: 'r', header: 'Recall', align: 'right', cell: (m) => pct(m.latest_evaluation?.overall.recall) },
                  { key: 'm', header: 'mAP50', align: 'right', cell: (m) => pct(m.latest_evaluation?.overall.mAP50) },
                  { key: 'when', header: 'When', cell: (m) => dateTime(m.latest_evaluation?.finished_at) },
                  { key: 'o', header: '', cell: (m) => m.latest_evaluation ? <Link className="text-info hover:underline" href={`/evaluation?id=${m.latest_evaluation.id}`}>Open</Link> : null },
                ]} />
            </Panel>
          )}
          {tab === 'deployment' && (
            <Panel title="Deployment" icon={Rocket}>
              <p className="mb-3 text-[12.5px] text-fg-2">The active model serves every camera, video job and upload. Switching is a hot swap: in-flight frames finish on the old model; if the new weights fail to load, the previous model stays active.</p>
              <DataTable rows={list} rowKey={(m) => m.version} dense
                columns={[
                  { key: 'v', header: 'Version', cell: (m) => <span className="font-mono text-fg">{m.version}</span> },
                  { key: 's', header: 'State', cell: (m) => m.active ? <StatusBadge status="active" /> : <Badge>standby</Badge> },
                  { key: 'reg', header: 'Registered', cell: (m) => dateTime(m.registered_at) },
                  { key: 'a', header: 'Action', cell: (m) => m.active ? <span className="text-fg-3">serving</span> : (
                    <Button size="sm" icon={m.registered_at < (list.find((x) => x.active)?.registered_at ?? '') ? Undo2 : Rocket} disabled={!isAdmin || !m.file_exists}
                      title={!isAdmin ? 'Admin sign-in required' : undefined} onClick={() => setDeploy(m)}>
                      {m.registered_at < (list.find((x) => x.active)?.registered_at ?? '') ? 'Rollback to' : 'Deploy'}
                    </Button>) },
                ]} />
              {!isAdmin && <p className="mt-2 text-[11px] text-fg-3">Sign in as admin (Settings → Security) to deploy or roll back.</p>}
            </Panel>
          )}
          {tab === 'versions' && (
            <Panel title="Deployment history" icon={History} noPad>
              <DataTable rows={history} rowKey={(e) => e.id} dense empty={<EmptyState compact title="No model changes recorded yet" />}
                columns={[
                  { key: 't', header: 'Time', cell: (e) => dateTime(e.ts) },
                  { key: 'm', header: 'Event', cell: (e) => <span className="text-fg">{e.message}</span> },
                ]} />
            </Panel>
          )}
        </>
      )}
      <ConfirmDialog open={!!deploy} onClose={() => setDeploy(null)} onConfirm={doDeploy} pending={busy}
        title={`Deploy ${deploy?.version}?`} confirmLabel="Deploy"
        body={<>This makes <b className="font-mono">{deploy?.version}</b> the active model for all cameras and new jobs. The currently active model is kept and can be restored at any time.
          {!deploy?.latest_evaluation && <p className="mt-2 text-warn">This version has not been evaluated yet.</p>}</>} />
      <RegisterDialog experimentId={register} onClose={() => setRegister(null)} />
      <Drawer open={!!openModel} onClose={() => router.replace('/model-lab')} title={openModel?.name ?? ''} subtitle={<span className="font-mono">{openModel?.version}</span>} width="w-[560px]">
        {openModel && <ModelDetail m={openModel} />}
      </Drawer>
    </div>
  )
}

function ModelCard({ m, onOpen, onDeploy, isAdmin }: { m: ModelInfo; onOpen: () => void; onDeploy: () => void; isAdmin: boolean }) {
  const ev = m.latest_evaluation?.overall
  const full = useQuery(m.latest_evaluation ? `eval:${m.latest_evaluation.id}` : null, () => Api.evaluation(m.latest_evaluation!.id))
  const metric = (label: string, v: number | null | undefined, val?: number | null) => (
    <div className="min-w-0">
      <div className="text-[10px] uppercase tracking-wide text-fg-3">{label}</div>
      <div className="tabular font-mono text-[14px] text-fg">{pct(v)}</div>
      {val !== undefined && <div className="tabular font-mono text-[10px] text-fg-3" title="validation during training">val {pct(val)}</div>}
    </div>
  )
  return (
    <section className="flex flex-col rounded-lg border border-line bg-panel">
      <header className="flex items-start justify-between gap-2 border-b border-line p-3.5">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <Sparkles className="size-4 text-exp" aria-hidden />
            <h3 className="truncate text-[14px] font-semibold text-fg">{m.name}</h3>
          </div>
          <div className="mt-0.5 truncate font-mono text-[11px] text-fg-3">{m.version}</div>
        </div>
        {m.active ? <StatusBadge status="active" /> : !m.file_exists ? <Badge tone="crit">weights missing</Badge> : <Badge>standby</Badge>}
      </header>
      <div className="flex flex-col gap-3 p-3.5">
        <KeyValue cols={3} items={[
          ['Architecture', <span key="a" className="font-mono">{m.architecture}</span>], ['Train size', m.train_imgsz ? `${m.train_imgsz}px` : '—'],
          ['Classes', m.classes.length], ['Dataset', m.dataset ?? '—'], ['Trained', m.trained_at ? dateTime(m.trained_at) : '—'], ['Weights', m.size_mb ? `${m.size_mb} MB` : '—'],
        ]} />
        <div className="rounded-md border border-line bg-panel-2 p-2.5">
          <div className="mb-1.5 flex items-center justify-between text-[11px] text-fg-3">
            <span>{ev ? `Evaluation · ${m.latest_evaluation!.dataset}/${m.latest_evaluation!.split}` : 'Not evaluated with the platform yet'}</span>
            {!ev && <Link href={`/evaluation?model=${encodeURIComponent(m.version)}`} className="text-info hover:underline">Evaluate →</Link>}
          </div>
          <div className="grid grid-cols-4 gap-2">
            {metric('mAP50', ev?.mAP50, m.metrics.mAP50)}
            {metric('mAP50-95', ev?.mAP50_95, m.metrics.mAP50_95)}
            {metric('Precision', ev?.precision, m.metrics.precision)}
            {metric('Recall', ev?.recall, m.metrics.recall)}
            {metric('F1', ev?.f1, m.metrics.f1)}
            {metric('Small recall', full.data?.by_size.small.recall)}
            <div><div className="text-[10px] uppercase tracking-wide text-fg-3">FPS</div><div className="tabular font-mono text-[14px] text-fg">{full.data?.latency_ms.mean ? num(1000 / full.data.latency_ms.mean, 1) : '—'}</div><div className="text-[10px] text-fg-3">{full.data ? 'this machine' : ''}</div></div>
          </div>
        </div>
        {m.notes && <p className="rounded-md border border-warn/30 bg-warn-bg px-2 py-1.5 text-[11.5px] text-warn">{m.notes}</p>}
      </div>
      <footer className="mt-auto flex flex-wrap gap-1.5 border-t border-line p-2.5">
        <Button size="sm" onClick={onOpen}>Open</Button>
        <Link href={`/evaluation?model=${encodeURIComponent(m.version)}`}><Button size="sm" icon={BarChart3}>Evaluate</Button></Link>
        <Link href="/models"><Button size="sm" icon={GitCompare}>Compare</Button></Link>
        <Button size="sm" variant="primary" icon={Rocket} disabled={m.active || !isAdmin || !m.file_exists} title={!isAdmin ? 'Admin sign-in required' : undefined} onClick={onDeploy}>{m.active ? 'Deployed' : 'Deploy'}</Button>
        <Button size="sm" variant="ghost" icon={Archive} disabled title="Archiving is not supported by the backend; versions are kept on disk">Archive</Button>
      </footer>
    </section>
  )
}

function ModelDetail({ m }: { m: ModelInfo }) {
  return (
    <div className="flex flex-col gap-4 text-[13px]">
      <KeyValue items={[
        ['Version', <span key="v" className="font-mono">{m.version}</span>], ['State', m.active ? 'active' : 'standby'],
        ['Architecture', m.architecture], ['Train image size', m.train_imgsz ? `${m.train_imgsz}px` : '—'],
        ['Registered', dateTime(m.registered_at)], ['Trained', dateTime(m.trained_at)],
        ['Source experiment', m.experiment_id ? <Link key="e" className="text-info hover:underline" href={`/experiments?id=${encodeURIComponent(m.experiment_id)}`}>{m.experiment_id}</Link> : '—'],
        ['Weights file', <span key="p" className="font-mono text-[11px]" title={m.path}>{m.path.split(/[\\/]/).slice(-3).join('/')}</span>],
      ]} />
      <div>
        <h3 className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-fg-3">Validation metrics recorded during training</h3>
        <KeyValue cols={3} items={[['Precision', pct(m.metrics.precision)], ['Recall', pct(m.metrics.recall)], ['F1', pct(m.metrics.f1)],
          ['mAP50', pct(m.metrics.mAP50)], ['mAP50-95', pct(m.metrics.mAP50_95)], ['Best epoch', m.metrics.epoch ?? '—']]} />
        {m.metrics.source && <p className="mt-1 text-[11px] text-fg-3">Source: {m.metrics.source}</p>}
      </div>
      <div>
        <h3 className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-fg-3">Classes ({m.classes.length})</h3>
        <div className="flex flex-wrap gap-1">{m.classes.map((c) => <Badge key={c}>{c}</Badge>)}</div>
      </div>
      {m.notes && <p className="rounded-md border border-warn/30 bg-warn-bg p-2 text-[12px] text-warn">{m.notes}</p>}
    </div>
  )
}
