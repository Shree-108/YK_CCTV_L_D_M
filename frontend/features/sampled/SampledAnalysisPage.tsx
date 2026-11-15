'use client'

// Four-screenshot video analysis: Upload -> Preview -> Select camera -> Store ROI -> Analyse -> Results.
// URL state: ?media=<id> (uploaded, not started) then ?job=<id> (survives reloads; jobs run on the server).
import Link from 'next/link'
import { usePathname, useRouter, useSearchParams } from 'next/navigation'
import { useEffect, useMemo, useState } from 'react'
import {
  AlertTriangle, BarChart3, Camera, FileVideo, Info, Layers, Play, RotateCcw, ScanSearch, SkipForward, Trash2, Undo2, X,
} from 'lucide-react'
import { ApiError, url } from '../../api/client'
import { Api } from '../../api/endpoints'
import type { RoiZone } from '../../api/types'
import type { SampledJob, SampledMedia } from '../../api/sampledTypes'
import { invalidate, useQuery } from '../../hooks/useQuery'
import { cx, dateTime, duration, num, pct } from '../../lib/format'
import { lockedFor, useApp } from '../../stores/app'
import { Badge, Button, Field, Input, Panel, Segmented, Toggle } from '../../components/ui/primitives'
import { MetricCard } from '../../components/ui/kpi'
import { EmptyState, ErrorState, LoadingState } from '../../components/ui/states'
import { RoiEditor } from '../../components/vision/Overlay'
import { DropZone, FileInfo, Steps, fileSize, litterName } from '../detection/common'
import { CsvPreview, FrameCard, STATE, StageList, isActive } from './shared'

const ACCEPT = 'video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,video/webm,.mkv,.avi,.m4v'
const STEPS = ['Upload', 'Preview', 'Select camera', 'Store ROI', 'Analyse', 'Results']

export default function SampledAnalysisPage() {
  const params = useSearchParams()
  const router = useRouter()
  const pathname = usePathname()
  const mediaId = params.get('media')
  const jobId = params.get('job')
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
  const step = !mediaId && !jobId ? 0 : !jobId ? 2 : !j ? 3
    : j.state === 'done' ? 6 : ['analysing', 'generating_images', 'generating_csv'].includes(j.stage) ? 4 : 3

  return (
    <div className="flex flex-col gap-4">
      <div className="rounded-xl border border-line bg-panel px-4 py-3">
        <Steps steps={STEPS} current={step} />
      </div>
      {jobId ? (
        j ? <JobView job={j} onNew={restart} /> : job.error ? <Panel><ErrorState error={job.error} onRetry={job.refetch} action={<Button onClick={restart}>New analysis</Button>} /></Panel>
          : <Panel><LoadingState rows={5} /></Panel>
      ) : mediaId ? (
        <Configure mediaId={mediaId} local={local} onStarted={(id) => go({ job: id })} onBack={restart} />
      ) : (
        <UploadStep onUploaded={(m, file) => { setLocal({ file, src: URL.createObjectURL(file) }); go({ media: m.id }) }} />
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
            'The outdoor area outside the store is identified on the screenshots and the shop interior is left out.',
            'You check the proposed ROI, adjust it if needed, and confirm it.',
            'The litter model analyses only the outdoor area of the four screenshots.',
            'You get annotated screenshots, a CSV report and a Results Dashboard entry.',
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
  const cams = useQuery('cameras:list', Api.cameras)
  const health = useQuery('health', Api.health)
  const [f, setF] = useState({ store_id: '', camera_id: '', recording_start: '', roi_mode: 'auto' as 'auto' | 'camera' | 'manual', auto_accept: false, create_alerts: true, seed: '' })
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => { if (m.data?.suggested_recording_start) setF((x) => ({ ...x, recording_start: x.recording_start || toLocalInput(m.data!.suggested_recording_start) })) }, [m.data])

  const start = async () => {
    setBusy(true); setError(null)
    try {
      const r = await Api.sampledCreate({
        media_id: mediaId, store_id: f.store_id || undefined, camera_id: f.camera_id || undefined,
        recording_start: f.recording_start || undefined, roi_mode: f.roi_mode, auto_accept: f.roi_mode === 'auto' && f.auto_accept,
        create_alerts: f.create_alerts, seed: f.seed ? Number(f.seed) : undefined,
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
      <Panel title="Select camera" icon={Camera}>
        <div className="flex flex-col gap-3.5">
          <Field label="Store ID" hint="Optional. Left blank, the report leaves it blank.">
            <Input value={f.store_id} maxLength={64} placeholder="e.g. Store 12" onChange={(e) => setF({ ...f, store_id: e.target.value })} />
          </Field>
          <Field label="Camera ID" hint="Optional. A saved ROI for this camera can be reused.">
            <Input value={f.camera_id} maxLength={40} list="sampled-cams" placeholder="e.g. CAM-07" onChange={(e) => setF({ ...f, camera_id: e.target.value })} />
            <datalist id="sampled-cams">{cams.data?.cameras.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}</datalist>
          </Field>
          <Field label="Recording start" hint={md.suggested_recording_start ? 'Pre-filled from the file name — please check it.' : 'Optional. Needed for real recording times and the hourly summary.'}>
            <Input type="datetime-local" step={1} value={f.recording_start} onChange={(e) => setF({ ...f, recording_start: e.target.value })} />
          </Field>
          <Field label="Outdoor ROI">
            <Segmented label="ROI method" value={f.roi_mode} onChange={(v) => setF({ ...f, roi_mode: v })} options={[
              { value: 'auto', label: 'Automatic' }, { value: 'camera', label: 'Saved for camera' }, { value: 'manual', label: 'Draw myself' }]} />
          </Field>
          <p className="-mt-1.5 text-[12px] text-fg-3">
            {f.roi_mode === 'auto' ? 'The outdoor ground is identified on the four screenshots; you confirm it before analysis.'
              : f.roi_mode === 'camera' ? 'The ROI saved for this camera is pre-filled; you confirm it before analysis.'
              : 'You draw the outdoor area on a screenshot before analysis.'}
          </p>
          {f.roi_mode === 'auto' && (
            <Toggle label="Skip confirmation when confident" checked={f.auto_accept} onChange={(v) => setF({ ...f, auto_accept: v })}
              description="Only when the outdoor area is identified with high confidence and consistently in all screenshots. Otherwise you are always asked." />
          )}
          <Toggle label="Create alerts" checked={f.create_alerts} onChange={(v) => setF({ ...f, create_alerts: v })}
            description="Confirmed litter appears on the Alerts page (one alert per object, even if seen in several screenshots)." />
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

// ---------------------------------------------------------------- 4-6. processing, ROI confirmation, results
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
                ['ROI', job.roi_status === 'confirmed' ? `${job.roi_source === 'whole_frame' ? 'whole frame' : job.roi_source} · confirmed` : job.roi_status],
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

      {job.state === 'awaiting_roi' && <RoiConfirm job={job} />}

      {job.frames.some((f) => f.status !== 'planned') && job.state !== 'awaiting_roi' && (
        <Panel title="The four screenshots" icon={ScanSearch}
          subtitle={job.state === 'done' ? 'Annotated with the confirmed ROI, litter boxes and confidence' : 'Analysis results appear here as each screenshot is processed'}>
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
            {Object.keys(c.by_class ?? {}).length === 0 ? <EmptyState compact title="No litter detected" body="Nothing inside the confirmed outdoor area reached the confirmed threshold in the analysed screenshots." /> : (
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

// ---------------------------------------------------------------- ROI confirmation
function RoiConfirm({ job }: { job: SampledJob }) {
  const { toast, canOperate, role, loginOn } = useApp()
  const lock = lockedFor('operator', role, loginOn)
  const p = job.proposal?.proposal
  const perFrame = p?.available && p.static_view === false
  const saved = job.frames.filter((f) => f.status === 'saved')
  const [sel, setSel] = useState(saved[0]?.idx ?? 1)
  const initial = useMemo(() => job.proposal?.prefill ?? [], [job.proposal])
  const [zones, setZones] = useState<RoiZone[]>(initial)
  const [frameZones, setFrameZones] = useState<Record<number, RoiZone[]>>(() =>
    perFrame ? Object.fromEntries((p?.frames ?? []).map((f) => [f.idx, f.zones])) : {})
  const [skip, setSkip] = useState<Set<number>>(new Set())
  const [wholeFrame, setWholeFrame] = useState(false)
  const [saveCam, setSaveCam] = useState(false)
  const [busy, setBusy] = useState(false)
  const frame = job.frames.find((f) => f.idx === sel)
  const current = perFrame ? frameZones[sel] ?? [] : zones
  const setCurrent = (z: RoiZone[]) => (perFrame ? setFrameZones((m) => ({ ...m, [sel]: z })) : setZones(z))
  const hasInclude = (zs: RoiZone[]) => zs.some((z) => z.mode === 'include')
  const missing = wholeFrame ? [] : saved.filter((f) => !skip.has(f.idx) && !hasInclude(perFrame ? frameZones[f.idx] ?? [] : zones)).map((f) => f.idx)
  const conf = p?.confidence

  const confirm = async () => {
    setBusy(true)
    try {
      await Api.sampledConfirmRoi(job.id, {
        zones: perFrame ? [] : zones, frame_zones: perFrame ? Object.fromEntries(Object.entries(frameZones).filter(([k]) => !skip.has(Number(k)))) : undefined,
        skip_frames: [...skip], whole_frame: wholeFrame, save_for_camera: saveCam,
      })
      invalidate(`sampled:job:${job.id}`)
      toast({ tone: 'ok', title: 'ROI confirmed', body: 'Analysing the screenshots…' })
    } catch (e) {
      toast({ tone: 'crit', title: 'ROI not accepted', body: (e as Error).message })
    } finally {
      setBusy(false)
    }
  }

  return (
    <Panel title="Store ROI — confirm the outdoor area" icon={Layers}
      subtitle="Litter is counted only where the bottom-centre of an object lies inside the outdoor area">
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_320px]">
        <div className="flex flex-col gap-3">
          <div className="flex flex-wrap gap-2" role="tablist" aria-label="Screenshot">
            {job.frames.map((f) => (
              <button key={f.idx} role="tab" aria-selected={sel === f.idx} disabled={f.status !== 'saved'} onClick={() => setSel(f.idx)}
                className={cx('flex items-center gap-2 rounded-lg border px-2.5 py-1.5 text-[12px] disabled:opacity-40',
                  sel === f.idx ? 'border-info bg-info-bg text-fg' : 'border-line text-fg-2 hover:border-line-strong')}>
                #{f.idx} <span className="font-mono">{f.video_timestamp.slice(0, 8)}</span>
                {f.status !== 'saved' ? <Badge>{f.status}</Badge> : skip.has(f.idx) ? <Badge tone="warn">skipped</Badge>
                  : hasInclude(perFrame ? frameZones[f.idx] ?? [] : zones) || wholeFrame ? <Badge tone="ok">ROI</Badge> : <Badge tone="warn">no ROI</Badge>}
              </button>
            ))}
          </div>
          {frame && frame.status === 'saved' && job.media ? (
            wholeFrame ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={url(frame.original_url!)} alt={`Screenshot ${frame.idx}`} className="w-full rounded-md border border-line bg-video" />
            ) : (
              <RoiEditor key={`${sel}-${perFrame}`} src={url(frame.original_url!)} width={job.media.width} height={job.media.height} zones={current} onChange={setCurrent} />
            )
          ) : <EmptyState compact title="This screenshot could not be read" body={frame?.error ?? undefined} />}
        </div>
        <div className="flex flex-col gap-3">
          <div className={cx('rounded-lg border px-3 py-2.5 text-[12.5px]',
            !p?.available ? 'border-line bg-panel-2 text-fg-2' : p.status === 'proposed' ? 'border-ok/30 bg-ok-bg text-ok' : 'border-warn/30 bg-warn-bg text-warn')}>
            <div className="font-medium">
              {!p?.available ? 'No automatic proposal' : p.status === 'proposed' ? 'Outdoor area identified' : 'Please check the outdoor area'}
              {conf != null && <span className="font-normal"> · confidence {pct(conf, 0)}</span>}
            </div>
            <p className="mt-1 opacity-90">{p?.note ?? (job.proposal?.prefill_source === 'camera' ? 'Pre-filled with the ROI saved for this camera.' : 'Draw the outdoor ground area.')}</p>
            {job.proposal?.prefill_source && <p className="mt-1 opacity-80">Pre-filled from: {job.proposal.prefill_source === 'auto' ? 'automatic identification' : 'the ROI saved for this camera'}.</p>}
            {perFrame && <p className="mt-1 opacity-90">The camera view changes between screenshots: each screenshot has its own ROI.</p>}
          </div>
          <div className="flex flex-wrap gap-2">
            <Button size="sm" icon={Undo2} disabled={wholeFrame} onClick={() => (perFrame ? setFrameZones(Object.fromEntries((p?.frames ?? []).map((f) => [f.idx, f.zones]))) : setZones(initial))}>Reset to proposal</Button>
            <Button size="sm" icon={Trash2} disabled={wholeFrame || !current.length} onClick={() => setCurrent([])}>Clear ROI</Button>
            <Button size="sm" icon={SkipForward} disabled={!frame || frame.status !== 'saved'}
              onClick={() => setSkip((s) => { const n = new Set(s); if (n.has(sel)) n.delete(sel); else n.add(sel); return n })}>
              {skip.has(sel) ? 'Analyse this screenshot' : 'Skip this screenshot'}
            </Button>
          </div>
          <Toggle label="Analyse the whole frame (no ROI)" checked={wholeFrame} onChange={setWholeFrame}
            description="Only if the camera shows nothing but the outdoor area. Shop interior would otherwise be analysed too." />
          <Toggle label="Save this ROI for the camera" checked={saveCam} onChange={setSaveCam} disabled={!job.camera_id || wholeFrame || perFrame}
            description={job.camera_id ? `Reused for later analyses of ${job.camera_id}.` : 'Enter a Camera ID when starting the analysis to save the ROI.'} />
          {missing.length > 0 && <p className="text-[12px] text-warn">Draw the outdoor area (an include zone) for screenshot {missing.map((i) => `#${i}`).join(', ')}, or skip it.</p>}
          <Button variant="primary" icon={Play} loading={busy} disabled={!canOperate || missing.length > 0 || skip.size >= saved.length} title={lock} onClick={confirm} className="h-9">
            Confirm ROI and analyse
          </Button>
        </div>
      </div>
    </Panel>
  )
}
