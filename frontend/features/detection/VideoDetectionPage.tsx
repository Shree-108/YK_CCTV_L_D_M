'use client'

// Full-video tracking analysis (the primary video workflow): every sampled frame is analysed, objects are tracked,
// and each DETECTED track becomes ONE litter event. Unique events and frame observations are always shown apart.
import { useSearchParams } from 'next/navigation'
import { useEffect, useRef, useState } from 'react'
import { Download, FileVideo, Layers, Play, RotateCcw } from 'lucide-react'
import { url } from '../../api/client'
import { Api } from '../../api/endpoints'
import type { Job, VideoResult } from '../../api/types'
import { invalidate, useQuery } from '../../hooks/useQuery'
import { dateTime, duration, num, pct } from '../../lib/format'
import { INTERVAL_OPTIONS, intervalSeconds, recordingStartValue, type IntervalChoice } from '../../lib/litter'
import { lockedFor, useApp } from '../../stores/app'
import { Badge, Button, Field, Input, PageHeader, Panel, Select, StatusBadge } from '../../components/ui/primitives'
import { MetricCard } from '../../components/ui/kpi'
import { DataTable } from '../../components/ui/table'
import { EmptyState } from '../../components/ui/states'
import { StoreCameraSelect, type StoreCamera } from '../stores/StoreCameraSelect'
import { ProcessingBanner } from './AnalysisOutputs'
import { DetectionSettingsForm, DropZone, FileInfo, Steps, fileExt, fileSize, toForm, type DetectConfig } from './common'
import { VideoLitterResults, VideoTechnicalDetails, jobFile } from './VideoResults'
import { AnalysisInfo, LitterReportPanel, ScreenshotGallery } from './VideoEvidence'

const ACCEPT = 'video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,video/webm,.mkv,.avi,.m4v'

export default function VideoDetectionPage({ embedded = false }: { embedded?: boolean } = {}) {
  const { toast, canOperate, role, loginOn } = useApp()
  const opLock = lockedFor('operator', role, loginOn)
  const settings = useQuery('settings', Api.settings)
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [cfg, setCfg] = useState<DetectConfig>({ model_version: '', confidence: 0.6, image_size: '', tiling: 'auto' })
  const [who, setWho] = useState<StoreCamera>({ store_id: '', camera_id: '' })
  const [recStart, setRecStart] = useState('')
  const [interval, setIntervalChoice] = useState<IntervalChoice>('')
  const [customInterval, setCustomInterval] = useState('')
  const [win, setWin] = useState('')
  const [minConf, setMinConf] = useState('')
  // ?video_job=<id> opens an earlier full-video analysis (links from the Overview)
  const params = useSearchParams()
  const [jobId, setJobId] = useState<string | null>(() => params.get('video_job'))
  const [finished, setFinished] = useState(false)
  const [submitError, setSubmitError] = useState<Error | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const s = settings.data?.settings
  const iv = intervalSeconds(interval, customInterval)
  const recBad = !!recStart && !recordingStartValue(recStart)

  useEffect(() => { if (s) setCfg((c) => ({ ...c, confidence: s.confidence.confirmed_threshold })) }, [s])
  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview) }, [preview])

  // poll the job until it is done/failed, then stop (the page switches to the processed video by itself)
  const job = useQuery<Job<VideoResult>>(jobId ? `job:${jobId}` : null, () => Api.job<VideoResult>(jobId!), { poll: jobId && !finished ? 1000 : undefined })
  const j = job.data?.id === jobId ? job.data : undefined
  const done = j?.status === 'done'
  const failed = j?.status === 'failed'
  const jobs = useQuery('jobs', () => Api.jobs(20), { poll: 10000 })
  useEffect(() => { setFinished(j?.status === 'done' || j?.status === 'failed') }, [j?.status])
  useEffect(() => {
    if (j?.status === 'done' || j?.status === 'failed') { invalidate('alerts'); invalidate('jobs'); invalidate('litter') }
    if (j?.status === 'failed') toast({ tone: 'crit', title: 'Video analysis FAILED', body: j.error ?? undefined })
  }, [j?.status]) // eslint-disable-line react-hooks/exhaustive-deps

  const onFile = (f: File) => {
    setFile(f)
    setJobId(null)
    setPreview(URL.createObjectURL(f))
  }

  const submit = async () => {
    if (!file || iv.error || recBad) return
    setSubmitting(true)
    setSubmitError(null)
    try {
      const extra: Record<string, string> = {}
      const rs = recordingStartValue(recStart)
      if (rs) extra.recording_start = rs
      if (iv.value !== null) extra.sample_interval_sec = String(iv.value)
      if (win) extra.temporal_window = win
      if (minConf) extra.min_confirmations = minConf
      const r = await Api.detectVideo(toForm(file, cfg, who, extra))
      setFinished(false)
      setJobId(r.job_id)
      if (r.identity?.warnings?.length) toast({ tone: 'warn', title: 'Identity warning', body: r.identity.warnings.join(' ') })
    } catch (e) {
      setSubmitError(e as Error)
    } finally {
      setSubmitting(false)
    }
  }

  const running = !!jobId && (!j || j.status === 'queued' || j.status === 'running')
  const reset = () => { setFile(null); setJobId(null); setPreview(null) }
  return (
    <div className="flex flex-col gap-4">
      {!embedded && <PageHeader title="Video Analysis" description="Every sampled frame is analysed and litter is tracked through the video; each object is counted once as a litter event." />}
      <div className="rounded-xl border border-line bg-panel px-4 py-3">
        <Steps steps={['Upload video', 'Store, camera & sampling', 'Process', 'Review results']}
          current={done ? 3 : running || submitting ? 2 : file || jobId ? 1 : 0} />
      </div>
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_340px]">
        <Panel title={j ? j.input_name : file ? file.name : 'Upload video'} icon={FileVideo}
          actions={<>
            <AnalysisStatus running={running} done={done} failed={failed} />
            {(file || jobId) && <Button size="sm" variant="ghost" icon={RotateCcw} disabled={running} onClick={reset}>New analysis</Button>}
          </>}>
          {!file && !jobId ? (
            <DropZone accept={ACCEPT} onFile={onFile} label="Drag and drop a CCTV video here" hint="MP4, MOV, AVI, MKV, M4V or WebM" />
          ) : running ? (
            <Progress job={j} />
          ) : failed ? (
            <div className="flex flex-col gap-3">
              <ProcessingBanner status="FAILED" detail={j!.error ?? 'The analysis did not complete.'} />
              <p className="text-[12.5px] text-fg-3">No detection or summary CSV was produced. Nothing about litter can be concluded from this video.</p>
              {file && <Button icon={Play} className="self-start" disabled={!canOperate} title={opLock} loading={submitting} onClick={submit}>Try again</Button>}
            </div>
          ) : done ? (
            <ProcessedVideo jobId={j!.id} src={jobFile(j!, 'video')} />
          ) : jobId ? (
            <div className="py-16 text-center text-[13px] text-fg-3">Loading analysis…</div>
          ) : (
            <div className="flex flex-col gap-3">
              <FileInfo items={[['File name', file!.name], ['Size', fileSize(file!.size)], ['Format', fileExt(file!)], ['Analysis', 'Full video, tracking']]} />
              {preview && (
                // eslint-disable-next-line jsx-a11y/media-has-caption
                <video src={preview} controls muted playsInline preload="metadata" className="aspect-video w-full rounded-md bg-video" />
              )}
              {submitError && <p className="rounded-md bg-crit-bg px-3 py-2 text-[12.5px] text-crit">{submitError.message}</p>}
            </div>
          )}
        </Panel>
        <Panel title="Analysis settings" icon={Layers}>
          <div className="flex flex-col gap-3.5">
            <StoreCameraSelect value={who} onChange={setWho} disabled={running} />
            <Field label="Recording start (optional)" hint={recBad ? 'Invalid date and time.' : 'Local date and time of the first frame. Without it, event times are processing times.'}>
              <Input type="datetime-local" step={1} value={recStart} onChange={(e) => setRecStart(e.target.value)} aria-invalid={recBad} />
            </Field>
            <Field label="Sampling interval" hint={iv.error ?? (interval === '' ? `Server default: ${s ? `${num(1 / s.video.sample_fps, 2)} s` : '—'} between analysed frames.` : 'One frame is analysed per interval.')}>
              <Select value={interval} onChange={(e) => setIntervalChoice(e.target.value as IntervalChoice)}>
                {INTERVAL_OPTIONS.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
              </Select>
              {interval === 'custom' && (
                <Input className="mt-1.5" inputMode="decimal" value={customInterval} placeholder="seconds, 0.1 – 60" aria-label="Custom sampling interval in seconds"
                  aria-invalid={!!iv.error} onChange={(e) => setCustomInterval(e.target.value)} />
              )}
            </Field>
            <DetectionSettingsForm cfg={cfg} setCfg={setCfg}>
              <details className="rounded-lg border border-line px-3 py-2">
                <summary className="cursor-pointer text-[12.5px] text-fg-2">Temporal confirmation</summary>
                <div className="mt-2 grid grid-cols-2 gap-2">
                  <Field label="Window"><Select value={win} onChange={(e) => setWin(e.target.value)}>
                    <option value="">{s ? s.tracking.temporal_window : 'default'}</option>{[3, 5, 7, 10, 15].map((v) => <option key={v} value={v}>{v}</option>)}</Select></Field>
                  <Field label="Min hits"><Select value={minConf} onChange={(e) => setMinConf(e.target.value)}>
                    <option value="">{s ? s.tracking.min_confirmations : 'default'}</option>{[1, 2, 3, 4, 5, 7].map((v) => <option key={v} value={v}>{v}</option>)}</Select></Field>
                </div>
                <p className="mt-2 text-[11px] text-fg-3">A track becomes a litter event (DETECTED) after “min hits” detections within the last “window” analysed frames.</p>
              </details>
              <Button variant="primary" icon={Play} disabled={!file || running || !canOperate || !!iv.error || recBad} title={opLock} loading={submitting} onClick={submit} className="w-full">Analyze Video</Button>
            </DetectionSettingsForm>
          </div>
        </Panel>
      </div>
      {done && j && <AnalysisInfo job={j} />}
      {done && j && <ScreenshotGallery job={j} />}
      {done && j && <LitterReportPanel job={j} />}
      {done && j && <VideoLitterResults job={j} />}
      {done && j?.result?.stats && <VideoTechnicalDetails job={j} />}
      <Panel title="Recent video analyses" noPad>
        <DataTable rows={(jobs.data?.jobs ?? []).filter((x) => x.kind === 'video')} rowKey={(x) => x.id} dense onRowClick={(x) => { setFinished(false); setJobId(x.id) }} selectedKey={jobId}
          empty={<EmptyState compact title="No videos processed yet" />}
          columns={[
            { key: 'name', header: 'Video', cell: (x) => <span className="text-fg">{x.input_name}</span> },
            { key: 'cam', header: 'Camera', cell: (x) => <span className="font-mono">{x.camera_id ?? 'UNKNOWN'}</span> },
            { key: 'created', header: 'Submitted', cell: (x) => dateTime(x.created_at), sortValue: (x) => x.created_at },
            { key: 'status', header: 'Status', cell: (x) => <StatusBadge status={x.status} /> },
            { key: 'p', header: 'Progress', align: 'right', cell: (x) => pct(x.progress, 0) },
          ]} />
      </Panel>
    </div>
  )
}

function AnalysisStatus({ running, done, failed }: { running: boolean; done: boolean; failed: boolean }) {
  if (running) return <Badge tone="warn" dot>Processing video…</Badge>
  if (done) return <Badge tone="ok" dot>Analysis complete</Badge>
  if (failed) return <Badge tone="crit" dot>FAILED</Badge>
  return null
}

/** The annotated MP4, loaded and started as soon as the job is done (muted: browsers only autoplay muted video). */
function ProcessedVideo({ jobId, src: path }: { jobId: string; src: string | null }) {
  const ref = useRef<HTMLVideoElement>(null)
  const [error, setError] = useState(false)
  const src = path ? url(path) : null
  useEffect(() => {
    setError(false)
    const v = ref.current
    if (v) void v.play().catch(() => { /* autoplay blocked: the controls stay visible */ })
  }, [jobId])
  if (!src || error) {
    return (
      <div className="flex aspect-video flex-col items-center justify-center gap-2 rounded-md border border-line bg-video text-center">
        <p className="text-[13px] text-fg">{src ? 'The annotated video could not be played in this browser.' : 'No annotated video was produced.'}</p>
        {src && <a href={src} download><Button size="sm" icon={Download}>Download annotated video</Button></a>}
      </div>
    )
  }
  return (
    // eslint-disable-next-line jsx-a11y/media-has-caption
    <video key={jobId} ref={ref} data-job={jobId} src={src} controls autoPlay muted playsInline preload="auto" onError={() => setError(true)}
      onLoadedData={(e) => { void e.currentTarget.play().catch(() => {}) }} className="aspect-video w-full rounded-md bg-video" />
  )
}

function Progress({ job }: { job?: Job<VideoResult> }) {
  if (!job) return <div className="py-16 text-center text-[13px] text-fg-2" aria-live="polite">Uploading and starting analysis…</div>
  const p = job.result?.progress
  return (
    <div className="flex flex-col gap-4 py-6" aria-live="polite">
      <div className="flex items-center justify-between text-[13px]">
        <span className="flex items-center gap-2 text-fg"><StatusBadge status={job.status} /> {job.status === 'queued' ? 'Waiting for the processing worker…' : 'Analysing every sampled frame and tracking litter…'}</span>
        <span className="tabular font-mono text-fg">{pct(job.progress, 0)}</span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-panel-2" role="progressbar" aria-valuenow={Math.round(job.progress * 100)} aria-valuemin={0} aria-valuemax={100}>
        <div className="h-full bg-info transition-[width] duration-500" style={{ width: `${job.progress * 100}%` }} />
      </div>
      <div className="grid grid-cols-2 gap-2 md:grid-cols-5">
        <MetricCard label="Frames read" value={p ? `${num(p.frames_read)}${p.frames_total ? ` / ${num(p.frames_total)}` : ''}` : '—'} />
        <MetricCard label="Frames analysed" value={num(p?.frames_analysed)} />
        <MetricCard label="Processing FPS" value={num(p?.processing_fps, 1)} />
        <MetricCard label="Estimated time left" value={p?.eta_sec != null ? duration(p.eta_sec) : '—'} />
        <MetricCard label="Litter events so far" value={num(p?.confirmed_objects)} hint={p?.detections != null ? `${num(p.detections)} raw detections so far` : undefined} />
      </div>
    </div>
  )
}
