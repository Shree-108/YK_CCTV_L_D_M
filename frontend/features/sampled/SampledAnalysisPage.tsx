'use client'

// Four-screenshot video analysis.
// Default (zero-click): drop a video -> upload + automatic analysis -> results (AutoAnalysis.tsx).
// ?flow=manual (operator review): Upload -> Preview -> Store & camera -> Analyse -> Results.
// URL state: ?media=<id> (uploaded, not started) then ?job=<id> (survives reloads; jobs run on the server).
import Link from 'next/link'
import { usePathname, useRouter, useSearchParams } from 'next/navigation'
import { useEffect, useState } from 'react'
import {
  AlertTriangle, BarChart3, Camera, FileVideo, Info, Layers, Play, RotateCcw, ScanSearch, X,
} from 'lucide-react'
import { ApiError, url } from '../../api/client'
import { Api } from '../../api/endpoints'
import type { SampledJob, SampledMedia } from '../../api/sampledTypes'
import { invalidate, useQuery } from '../../hooks/useQuery'
import { dateTime, duration, num } from '../../lib/format'
import { lockedFor, useApp } from '../../stores/app'
import { Badge, Button, Field, Input, Panel, Toggle } from '../../components/ui/primitives'
import { MetricCard } from '../../components/ui/kpi'
import { EmptyState, ErrorState, LoadingState } from '../../components/ui/states'
import { StoreCameraSelect } from '../stores/StoreCameraSelect'
import { DropZone, FileInfo, Steps, fileSize, litterName } from '../detection/common'
import { CsvPreview, FrameCard, STATE, StageList, isActive } from './shared'
import { AutoJobView, AutoUpload } from './AutoAnalysis'

const ACCEPT = 'video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,video/webm,.mkv,.avi,.m4v'
const STEPS = ['Upload', 'Preview', 'Store & camera', 'Analyse', 'Results']

export default function SampledAnalysisPage() {
  const params = useSearchParams()
  const router = useRouter()
  const pathname = usePathname()
  const mediaId = params.get('media')
  const jobId = params.get('job')
  const manual = params.get('flow') === 'manual'
  const setFlow = (toManual: boolean) => {
    const p = new URLSearchParams()
    if (params.get('mode')) p.set('mode', params.get('mode')!)
    if (toManual) p.set('flow', 'manual')
    router.replace(`${pathname}?${p.toString()}`, { scroll: false })
  }
  const [local, setLocal] = useState<{ file: File; src: string } | null>(null)
  const go = (q: Record<string, string>) => {
    const p = new URLSearchParams(params.toString())
    p.delete('media'); p.delete('job')
    Object.entries(q).forEach(([k, v]) => p.set(k, v))
    router.replace(`${pathname}?${p.toString()}`, { scroll: false })
  }
  const restart = () => { if (local) URL.revokeObjectURL(local.src); setLocal(null); go({}) }
  const [polling, setPolling] = useState(true)
  const job = useQuery(jobId ? `sampled:job:${jobId}` : null, () => Api.sampledJob(jobId!), { poll: polling ? 1500 : undefined })
  const j = job.data?.id === jobId ? job.data : undefined
  useEffect(() => {
    setPolling(!j || isActive(j.state))
    if (j && !isActive(j.state)) invalidate('sampled:overview')
  }, [j?.state]) // eslint-disable-line react-hooks/exhaustive-deps
  const step = !mediaId && !jobId ? 0 : !jobId ? 2 : j?.state === 'done' ? 5 : 3

  const auto = jobId ? j?.mode === 'auto' : !manual && !mediaId
  return (
    <div className="flex flex-col gap-4">
      {!auto && !(jobId && !j) && (
        <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-line bg-panel px-4 py-3">
          <Steps steps={STEPS} current={step} />
          {!jobId && <Button size="sm" variant="ghost" onClick={() => setFlow(false)}>Back to automatic analysis</Button>}
        </div>
      )}
      {jobId ? (
        j ? (j.mode === 'auto' ? <AutoJobView job={j} onNew={restart} /> : <JobView job={j} onNew={restart} />) : job.error ? <Panel><ErrorState error={job.error} onRetry={job.refetch} action={<Button onClick={restart}>New analysis</Button>} /></Panel>
          : <Panel><LoadingState rows={5} /></Panel>
      ) : mediaId ? (
        <Configure mediaId={mediaId} local={local} onStarted={(id) => go({ job: id })} onBack={restart} />
      ) : manual ? (
        <UploadStep onUploaded={(m, file) => { setLocal({ file, src: URL.createObjectURL(file) }); go({ media: m.id }) }} />
      ) : (
        <AutoUpload onJob={(id) => go({ job: id })} onAdvanced={() => setFlow(true)} />
      )}
    </div>
  )
}

// ---------------------------------------------------------------- 1. upload
function UploadStep({ onUploaded }: { onUploaded: (m: SampledMedia, f: File) => void }) {
  const { toast, canOperate, role, loginOn } = useApp()
  const lock = lockedFor('operator', role, loginOn)
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<Error | null>(null)
  const upload = async (f: File) => {
    if (!canOperate) { toast({ tone: 'warn', title: 'Not allowed', body: lock }); return }
    setBusy(f.name); setError(null)
    try {
      const form = new FormData()
      form.append('file', f)
      onUploaded(await Api.sampledUpload(form), f)
    } catch (e) {
      setError(e as Error)
    } finally {
      setBusy(null)
    }
  }
  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_340px]">
      <Panel title="Upload video" icon={FileVideo}>
        {busy ? (
          <div className="flex min-h-[320px] flex-col items-center justify-center gap-3 rounded-xl border border-line bg-panel-2/50 text-center" aria-live="polite">
            <div className="h-1.5 w-56 overflow-hidden rounded-full bg-panel"><div className="h-full w-1/3 animate-pulse rounded-full bg-info" /></div>
            <p className="text-[13px] font-medium text-fg">Uploading and validating “{busy}”…</p>
            <p className="text-[12px] text-fg-3">The server checks the video is readable and measures its real length.</p>
          </div>
        ) : <DropZone accept={ACCEPT} onFile={upload} label="Drag and drop a CCTV video here" hint="MP4, MOV, AVI, MKV, M4V or WebM · the server enforces the size limit" />}
        {error && <div className="mt-3"><ErrorState error={error} title="The video was not accepted" compact /></div>}
      </Panel>
      <Panel title="How it works" icon={Info}>
        <ol className="flex flex-col gap-3 text-[13px] text-fg-2">
          {[
            'Exactly four screenshots are taken at random positions, one in each quarter of the video.',
            'You choose the store, camera and recording start before the analysis starts.',
            'The litter model analyses the whole of each screenshot.',
            'You get annotated screenshots, detections and summary CSV reports and a results entry.',
          ].map((t, i) => (
            <li key={i} className="flex gap-3"><span className="flex size-5 shrink-0 items-center justify-center rounded-full border border-line-strong text-[11px] font-semibold text-fg">{i + 1}</span>{t}</li>
          ))}
        </ol>
      </Panel>
    </div>
  )
}

// ---------------------------------------------------------------- 2-3. preview + camera / store
function toLocalInput(iso: string | null | undefined) {
  return iso ? iso.slice(0, 19) : ''
}

function Configure({ mediaId, local, onStarted, onBack }: { mediaId: string; local: { file: File; src: string } | null; onStarted: (id: string) => void; onBack: () => void }) {
  const { toast, canOperate, role, loginOn } = useApp()
  const lock = lockedFor('operator', role, loginOn)
  const m = useQuery(`sampled:media:${mediaId}`, () => Api.sampledMedia(mediaId))
  const health = useQuery('health', Api.health)
  const [f, setF] = useState({ store_id: '', camera_id: '', recording_start: '', create_alerts: true, seed: '' })
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => { if (m.data?.suggested_recording_start) setF((x) => ({ ...x, recording_start: x.recording_start || toLocalInput(m.data!.suggested_recording_start) })) }, [m.data])

  const start = async () => {
    setBusy(true); setError(null)
    try {
      const r = await Api.sampledCreate({
        media_id: mediaId, store_id: f.store_id || undefined, camera_id: f.camera_id || undefined,
        recording_start: f.recording_start || undefined, create_alerts: f.create_alerts, seed: f.seed ? Number(f.seed) : undefined,
      })
      if (r.reused) toast({ tone: 'ok', title: 'Same analysis already completed', body: 'Showing the earlier result (same video, seed and settings).' })
      invalidate('sampled:')
      onStarted(r.job_id)
    } catch (e) {
      const d = e instanceof ApiError ? (e.detail as { job_id?: string } | undefined) : undefined
      if (e instanceof ApiError && e.status === 409 && d?.job_id) { toast({ tone: 'warn', title: 'Already in progress', body: 'Opening the running analysis of this video.' }); onStarted(d.job_id); return }
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  if (m.error) return <Panel><ErrorState error={m.error} onRetry={m.refetch} action={<Button onClick={onBack}>Upload another video</Button>} /></Panel>
  if (!m.data) return <Panel><LoadingState rows={6} /></Panel>
  const md = m.data
  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_380px]">
      <div className="flex flex-col gap-4">
        <Panel title="Preview" subtitle={md.filename} icon={FileVideo}
          actions={<Button size="sm" variant="ghost" icon={RotateCcw} onClick={onBack}>Choose another video</Button>}>
          <div className="flex flex-col gap-3">
            {local ? (
              // eslint-disable-next-line jsx-a11y/media-has-caption
              <video src={local.src} controls muted playsInline preload="metadata" className="aspect-video w-full rounded-lg bg-video" />
            ) : (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={url(`/api/sampled/media/${md.id}/preview.jpg`)} alt="First readable frame of the video" className="aspect-video w-full rounded-lg bg-video object-contain" />
            )}
            <FileInfo items={[['File name', md.filename], ['File size', fileSize(md.size_bytes)], ['Resolution', `${md.width} × ${md.height}px`], ['Duration', duration(md.duration_sec)]]} />
            <FileInfo items={[['Frame rate', `${num(md.fps, 2)} fps`], ['Total frames', num(md.frames)], ['Received', dateTime(md.created_at)], ['Screenshots', '4 (random)']]} />
            {md.previously_received && md.previously_received.length > 0 && (
              <p className="flex gap-2 rounded-lg border border-warn/30 bg-warn-bg px-3 py-2 text-[12.5px] text-warn">
                <AlertTriangle className="mt-0.5 size-3.5 shrink-0" aria-hidden />
                This exact file was received before ({md.previously_received.map((p) => dateTime(p.received_at)).join(', ')}). A new analysis draws four new random screenshots.
              </p>
            )}
          </div>
        </Panel>
      </div>
      <Panel title="Store & camera" icon={Camera}>
        <div className="flex flex-col gap-3.5">
          <StoreCameraSelect value={{ store_id: f.store_id, camera_id: f.camera_id }} onChange={(v) => setF({ ...f, ...v })} />
          <Field label="Recording start" hint={md.suggested_recording_start ? 'Pre-filled from the file name — please check it.' : 'Optional. Needed for real recording times and the hourly summary.'}>
            <Input type="datetime-local" step={1} value={f.recording_start} onChange={(e) => setF({ ...f, recording_start: e.target.value })} />
          </Field>
          <Toggle label="Create alerts" checked={f.create_alerts} onChange={(v) => setF({ ...f, create_alerts: v })}
            description="Litter detected with high confidence appears on the Alerts page (one alert per object, even if seen in several screenshots)." />
          <details className="rounded-lg border border-line px-3 py-2">
            <summary className="cursor-pointer text-[12.5px] text-fg-2">Advanced</summary>
            <div className="mt-2">
              <Field label="Random seed" hint="Optional. The same seed picks the same four frames again (for verification).">
                <Input inputMode="numeric" value={f.seed} onChange={(e) => setF({ ...f, seed: e.target.value.replace(/\D/g, '').slice(0, 10) })} placeholder="random" />
              </Field>
            </div>
          </details>
          {health.data && !health.data.model && <p className="text-[12px] text-crit">No detection model is loaded on the server; the analysis cannot run.</p>}
          {error && <p className="rounded-md bg-crit-bg px-3 py-2 text-[12.5px] text-crit">{error}</p>}
          <Button variant="primary" icon={Play} loading={busy} disabled={!canOperate} title={lock} onClick={start} className="h-9">Start analysis</Button>
        </div>
      </Panel>
    </div>
  )
}

// ---------------------------------------------------------------- 4-5. processing, results
function JobView({ job, onNew }: { job: SampledJob; onNew: () => void }) {
  const { toast, canOperate, role, loginOn } = useApp()
  const lock = lockedFor('operator', role, loginOn)
  const [cancelling, setCancelling] = useState(false)
  const s = STATE[job.state]
  const c = job.counts ?? {}
  const timestamps = job.frames.map((f) => f.video_timestamp.slice(0, 8))
  const cancel = async () => {
    setCancelling(true)
    try { await Api.sampledCancel(job.id); invalidate(`sampled:job:${job.id}`) } catch (e) { toast({ tone: 'crit', title: 'Could not cancel', body: (e as Error).message }) } finally { setCancelling(false) }
  }
  return (
    <>
      <Panel title={job.video_name ?? 'Video analysis'} icon={FileVideo}
        subtitle={[job.store_id && `Store ${job.store_id}`, job.camera_id && `Camera ${job.camera_id}`, job.media && duration(job.media.duration_sec)].filter(Boolean).join(' · ') || 'Store and camera not given'}
        actions={<>
          <Badge tone={s.tone} dot>{s.label}</Badge>
          {isActive(job.state) && <Button size="sm" variant="ghost" icon={X} loading={cancelling} disabled={!canOperate} title={lock} onClick={cancel}>Cancel</Button>}
          <Button size="sm" icon={RotateCcw} onClick={onNew}>New analysis</Button>
        </>}>
        <div className="grid gap-5 lg:grid-cols-[260px_minmax(0,1fr)]">
          <StageList job={job} />
          <div className="flex flex-col gap-3">
            <div className="grid grid-cols-2 gap-2.5 md:grid-cols-4">
              <MetricCard label="Screenshots extracted" value={`${job.frames.filter((f) => f.status === 'saved').length} / ${job.sample_count}`} />
              <MetricCard label="Screenshots analysed" value={c.screenshots_analysed ?? (job.state === 'done' ? 0 : '—')} />
              <MetricCard label="Total litter" value={c.litter ?? '—'} tone={c.litter ? 'crit' : undefined} />
              <MetricCard label="Alerts created" value={c.alerts ?? '—'} hint={job.create_alerts ? undefined : 'alerts off'} />
            </div>
            <dl className="grid grid-cols-1 gap-x-6 gap-y-2 text-[12.5px] sm:grid-cols-2">
              {([
                ['Job ID', <span key="j" className="font-mono">{job.id}</span>],
                ['Selected timestamps', <span key="t" className="font-mono">{timestamps.join(' · ')}</span>],
                ['Random seed', <span key="s" className="font-mono">{job.seed}</span>],
                ['Model', job.model_version ?? 'active model'],
                ['Started', dateTime(job.created_at)],
              ] as [string, React.ReactNode][]).map(([k, v]) => (
                <div key={k} className="flex min-w-0 justify-between gap-3 border-b border-line/60 pb-1.5"><dt className="text-fg-3">{k}</dt><dd className="truncate text-right text-fg">{v}</dd></div>
              ))}
            </dl>
            {job.error && <p className="flex gap-2 rounded-md bg-crit-bg px-3 py-2 text-[12.5px] text-crit"><AlertTriangle className="mt-0.5 size-3.5 shrink-0" aria-hidden />{job.error}</p>}
          </div>
        </div>
      </Panel>

      {job.frames.some((f) => f.status !== 'planned') && (
        <Panel title="The four screenshots" icon={ScanSearch}
          subtitle={job.state === 'done' ? 'Annotated with litter boxes and confidence' : 'Analysis results appear here as each screenshot is processed'}>
          <div className="grid gap-3 md:grid-cols-2">{job.frames.map((f) => <FrameCard key={f.idx} frame={f} />)}</div>
        </Panel>
      )}

      {job.state === 'done' && (
        <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_340px]">
          <Panel title="CSV report" subtitle="One row per litter detection; screenshots without litter keep one row each">
            {job.csv_url ? <CsvPreview path={job.csv_url} filename={`litter_report_${job.id}.csv`} /> : <EmptyState compact title="No CSV report" />}
          </Panel>
          <Panel title="Detection summary" icon={Layers}
            actions={<Link href="/results" className="inline-flex items-center gap-1 text-[12px] font-medium text-fg-2 hover:text-fg"><BarChart3 className="size-3.5" aria-hidden />Results Dashboard</Link>}>
            {Object.keys(c.by_class ?? {}).length === 0 ? <EmptyState compact title="No litter detected by the model" body="Nothing reached the confirmed threshold in the analysed screenshots. This is not a guarantee that the area is litter-free." /> : (
              <ul className="flex flex-col gap-2">
                {Object.entries(c.by_class ?? {}).sort((a, b) => b[1] - a[1]).map(([k, v]) => (
                  <li key={k} className="flex items-center justify-between rounded-lg border border-line bg-panel-2 px-3 py-2 text-[13px]"><span className="text-fg">{litterName(k)}</span><span className="tabular font-semibold text-fg">{v}</span></li>
                ))}
              </ul>
            )}
          </Panel>
        </div>
      )}
    </>
  )
}
