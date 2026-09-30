'use client'

import Link from 'next/link'
import { useEffect, useMemo, useRef, useState } from 'react'
import { Download, FileVideo, Layers, Play, RotateCcw, ScanSearch, Timer } from 'lucide-react'
import { url } from '../../api/client'
import { Api } from '../../api/endpoints'
import type { Job, LitterSummary, VideoResult } from '../../api/types'
import { invalidate, useQuery } from '../../hooks/useQuery'
import { className as cls, dateTime, duration, num, pct } from '../../lib/format'
import { useApp } from '../../stores/app'
import { Badge, Button, Field, PageHeader, Panel, Select, StatusBadge } from '../../components/ui/primitives'
import { BarList } from '../../components/ui/charts'
import { MetricCard } from '../../components/ui/kpi'
import { DataTable } from '../../components/ui/table'
import { EmptyState, ErrorState } from '../../components/ui/states'
import { DetectionSettingsForm, DropZone, RoiSetup, grabVideoFrame, litterName, toForm, useRoi, type DetectConfig } from './common'

export default function VideoDetectionPage() {
  const { toast } = useApp()
  const settings = useQuery('settings', Api.settings)
  const [file, setFile] = useState<File | null>(null)
  const [frame, setFrame] = useState<{ src: string; w: number; h: number } | null>(null)
  const [frameError, setFrameError] = useState<string | null>(null)
  const [cfg, setCfg] = useState<DetectConfig>({ model_version: '', confidence: 0.6, image_size: '', tiling: 'auto', camera_id: 'upload', use_roi: true })
  const [sampleFps, setSampleFps] = useState('')
  const [win, setWin] = useState('')
  const [minConf, setMinConf] = useState('')
  const [jobId, setJobId] = useState<string | null>(null)
  const [finished, setFinished] = useState(false)
  const [submitError, setSubmitError] = useState<Error | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const s = settings.data?.settings
  const roi = useRoi(cfg.camera_id)

  useEffect(() => { if (s) setCfg((c) => ({ ...c, confidence: s.confidence.confirmed_threshold })) }, [s])

  // poll the job until it is done/failed, then stop (the page switches to the processed video by itself)
  const job = useQuery<Job<VideoResult>>(jobId ? `job:${jobId}` : null, () => Api.job<VideoResult>(jobId!), { poll: jobId && !finished ? 1000 : undefined })
  const j = job.data?.id === jobId ? job.data : undefined
  const done = j?.status === 'done'
  const jobs = useQuery('jobs', () => Api.jobs(20), { poll: 10000 })
  useEffect(() => { setFinished(j?.status === 'done' || j?.status === 'failed') }, [j?.status])
  useEffect(() => {
    if (j?.status === 'done') { invalidate('alerts'); invalidate('jobs') }
    if (j?.status === 'failed') toast({ tone: 'crit', title: 'Video processing failed', body: j.error ?? undefined })
  }, [j?.status]) // eslint-disable-line react-hooks/exhaustive-deps

  const onFile = (f: File) => {
    setFile(f)
    setJobId(null)
    setFrame(null)
    setFrameError(null)
    grabVideoFrame(f).then(setFrame, (e: Error) => setFrameError(e.message))
  }

  const submit = async () => {
    if (!file) return
    setSubmitting(true)
    setSubmitError(null)
    try {
      const extra: Record<string, string> = {}
      if (sampleFps) extra.sample_fps = sampleFps
      if (win) extra.temporal_window = win
      if (minConf) extra.min_confirmations = minConf
      const r = await Api.detectVideo(toForm(file, cfg, extra))
      setFinished(false)
      setJobId(r.job_id)
    } catch (e) {
      setSubmitError(e as Error)
    } finally {
      setSubmitting(false)
    }
  }

  const running = !!jobId && (!j || j.status === 'queued' || j.status === 'running')
  const reset = () => { setFile(null); setJobId(null); setFrame(null) }
  return (
    <div className="flex flex-col gap-3">
      <PageHeader title="Video detection" description="Upload Video → Define ROI (the outdoor ground area) → Analyze Video. When analysis is complete the processed video, with the ROI, litter boxes and confidence scores, opens and plays automatically." />
      <div className="grid gap-3 xl:grid-cols-[minmax(0,1fr)_320px]">
        <Panel title={j ? j.input_name : file ? file.name : 'Upload Video'} icon={FileVideo}
          actions={<>
            <AnalysisStatus running={running} done={done} failed={j?.status === 'failed'} />
            {(file || jobId) && <Button size="sm" variant="ghost" icon={RotateCcw} disabled={running} onClick={reset}>New analysis</Button>}
          </>}>
          {!file && !jobId ? (
            <DropZone accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,video/webm,.mkv,.avi,.m4v" onFile={onFile}
              label="Upload Video — drop a CCTV video or click to browse" hint="MP4, MOV, AVI, MKV, M4V or WebM" />
          ) : running ? (
            <Progress job={j} />
          ) : j?.status === 'failed' ? (
            <ErrorState error={new Error(j.error ?? 'failed')} title="Processing failed" onRetry={file ? submit : undefined} />
          ) : done ? (
            <ProcessedVideo jobId={j!.id} />
          ) : jobId ? (
            <div className="py-16 text-center text-[13px] text-fg-3">Loading analysis…</div>
          ) : (
            <div className="flex flex-col gap-3">
              {frame ? <RoiSetup cameraId={cfg.camera_id} src={frame.src} width={frame.w} height={frame.h} /> : (
                <div className="flex aspect-video items-center justify-center rounded-md border border-line bg-video text-[12px] text-fg-3">
                  {frameError ?? 'Loading first frame…'}
                </div>
              )}
              <div className="flex flex-wrap items-center gap-3">
                <p className="text-[12px] text-fg-3">
                  {file!.name} · {(file!.size / 2 ** 20).toFixed(1)} MB ·{' '}
                  {roi.active && cfg.use_roi ? 'litter will be detected only inside the ROI' : 'no ROI: the whole frame will be analysed'}
                </p>
                <Button variant="primary" icon={Play} className="ml-auto" loading={submitting} onClick={submit}>Analyze Video</Button>
              </div>
              {submitError && <p className="text-[12px] text-crit">{submitError.message}</p>}
            </div>
          )}
        </Panel>
        <Panel title="Analysis settings" icon={Layers}>
          <DetectionSettingsForm cfg={cfg} setCfg={setCfg}>
            <div className="grid grid-cols-3 gap-2">
              <Field label="Sample fps"><Select value={sampleFps} onChange={(e) => setSampleFps(e.target.value)}>
                <option value="">{s ? s.video.sample_fps : 'def'}</option>{[1, 2, 5, 10, 15].map((v) => <option key={v} value={v}>{v}</option>)}</Select></Field>
              <Field label="Window"><Select value={win} onChange={(e) => setWin(e.target.value)}>
                <option value="">{s ? s.tracking.temporal_window : 'def'}</option>{[3, 5, 7, 10, 15].map((v) => <option key={v} value={v}>{v}</option>)}</Select></Field>
              <Field label="Min hits"><Select value={minConf} onChange={(e) => setMinConf(e.target.value)}>
                <option value="">{s ? s.tracking.min_confirmations : 'def'}</option>{[1, 2, 3, 4, 5, 7].map((v) => <option key={v} value={v}>{v}</option>)}</Select></Field>
            </div>
            <p className="text-[11px] text-fg-3">Frames are sampled at “Sample fps”. An object counts as litter after “min hits” detections within the last “window” sampled frames; its box then follows it through the video.</p>
            <Button variant="primary" icon={Play} disabled={!file || running} loading={submitting} onClick={submit} className="w-full">Analyze Video</Button>
          </DetectionSettingsForm>
        </Panel>
      </div>
      {done && j?.result?.stats && <LitterResults job={j} />}
      {done && j?.result?.stats && (
        <details className="rounded-lg border border-line bg-panel">
          <summary className="cursor-pointer select-none px-3 py-2 text-[12px] font-medium text-fg-2 hover:text-fg">Technical details (pipeline statistics, events, downloads)</summary>
          <div className="flex flex-col gap-3 p-3 pt-0"><Results job={j} /></div>
        </details>
      )}
      <Panel title="Recent video jobs" noPad>
        <DataTable rows={(jobs.data?.jobs ?? []).filter((x) => x.kind === 'video')} rowKey={(x) => x.id} dense onRowClick={(x) => { setFinished(false); setJobId(x.id) }} selectedKey={jobId}
          empty={<EmptyState compact title="No videos processed yet" />}
          columns={[
            { key: 'name', header: 'Video', cell: (x) => <span className="text-fg">{x.input_name}</span> },
            { key: 'cam', header: 'Camera', cell: (x) => <span className="font-mono">{x.camera_id}</span> },
            { key: 'created', header: 'Submitted', cell: (x) => dateTime(x.created_at), sortValue: (x) => x.created_at },
            { key: 'status', header: 'Status', cell: (x) => <StatusBadge status={x.status} /> },
            { key: 'p', header: 'Progress', align: 'right', cell: (x) => pct(x.progress, 0) },
          ]} />
      </Panel>
    </div>
  )
}

function AnalysisStatus({ running, done, failed }: { running: boolean; done: boolean; failed: boolean }) {
  if (running) return <Badge tone="info" dot className="animate-pulse">Processing Video…</Badge>
  if (done) return <Badge tone="ok" dot>Analysis Complete</Badge>
  if (failed) return <Badge tone="crit" dot>Failed</Badge>
  return null
}

/** The annotated MP4, loaded and started as soon as the job is done (muted: browsers only autoplay muted video). */
function ProcessedVideo({ jobId }: { jobId: string }) {
  const ref = useRef<HTMLVideoElement>(null)
  const [error, setError] = useState(false)
  const src = url(`/api/jobs/${jobId}/files/video`)
  useEffect(() => {
    setError(false)
    const v = ref.current
    if (v) void v.play().catch(() => { /* autoplay blocked: the controls stay visible */ })
  }, [jobId])
  if (error) {
    return (
      <div className="flex aspect-video flex-col items-center justify-center gap-2 rounded-md border border-line bg-video text-center">
        <p className="text-[13px] text-fg">The processed video could not be played in this browser.</p>
        <a href={src} download><Button size="sm" icon={Download}>Download processed video</Button></a>
      </div>
    )
  }
  return (
    // eslint-disable-next-line jsx-a11y/media-has-caption
    <video key={jobId} ref={ref} data-job={jobId} src={src} controls autoPlay muted playsInline preload="auto" onError={() => setError(true)}
      onLoadedData={(e) => { void e.currentTarget.play().catch(() => {}) }} className="aspect-video w-full rounded-md bg-video" />
  )
}

function LitterResults({ job }: { job: Job<VideoResult> }) {
  const st = job.result!.stats!
  const lit: LitterSummary = st.litter ?? fromEvents(job.result!.events ?? [], st.rejection_reasons?.outside_roi ?? 0)
  const seek = (t: number) => {
    const v = document.querySelector<HTMLVideoElement>(`video[data-job="${job.id}"]`)
    if (v) { v.currentTime = Math.max(0, t - 0.5); void v.play().catch(() => {}); v.scrollIntoView({ block: 'nearest', behavior: 'smooth' }) }
  }
  return (
    <Panel title="Detection results" icon={ScanSearch}
      subtitle={lit.roi_active ? 'ROI active — only litter inside the outdoor ROI is counted' : 'No ROI — the whole frame was analysed'}
      actions={<a href={url(`/api/jobs/${job.id}/files/video`)} download><Button size="sm" icon={Download}>Download processed video</Button></a>}>
      <div className="grid grid-cols-2 gap-2 md:grid-cols-5">
        <MetricCard label="Total litter detected" value={lit.total} tone={lit.total ? 'crit' : 'ok'} hint="unique objects" />
        <MetricCard label="Litter types" value={lit.classes.length} />
        <MetricCard label="Highest confidence" value={lit.highest_confidence != null ? pct(lit.highest_confidence, 0) : '—'} />
        <MetricCard label="Average confidence" value={lit.average_confidence != null ? pct(lit.average_confidence, 0) : '—'} />
        <MetricCard label="Detections inside ROI" value={lit.roi_active ? lit.inside_roi : '—'}
          hint={lit.roi_active ? `${num(lit.outside_roi_ignored)} outside-ROI detections ignored` : 'ROI not set'} />
      </div>
      <div className="mt-3">
        {lit.classes.length === 0 ? <EmptyState compact title="No litter detected" body={lit.roi_active ? 'Nothing inside the ROI was confirmed as litter.' : undefined} /> : (
          <DataTable rows={lit.classes} rowKey={(c) => c.class} dense
            columns={[
              { key: 'cls', header: 'Litter type', cell: (c) => <span className="font-medium text-fg">{litterName(c.class)}</span>, sortValue: (c) => c.class },
              { key: 'conf', header: 'Confidence', cell: (c) => <span className="tabular text-fg">{pct(c.max_confidence, 0)}</span>, sortValue: (c) => c.max_confidence },
              { key: 'avg', header: 'Average', cell: (c) => <span className="tabular text-fg-2">{pct(c.avg_confidence, 0)}</span>, sortValue: (c) => c.avg_confidence },
              { key: 'n', header: 'Count', align: 'right', cell: (c) => <span className="tabular text-fg">{c.count}</span>, sortValue: (c) => c.count },
              { key: 't', header: 'Detected at (click to jump)', cell: (c) => (
                <span className="flex flex-wrap gap-1">
                  {c.timestamps_sec.map((t, i) => (
                    <button key={i} onClick={() => seek(t)} className="rounded border border-line bg-panel-2 px-1.5 font-mono text-[11px] text-info hover:border-info">{clock(t)}</button>
                  ))}
                </span>) },
            ]} />
        )}
      </div>
    </Panel>
  )
}

/** Jobs processed before the litter summary existed: rebuild it from the confirmation events. */
function fromEvents(events: NonNullable<VideoResult['events']>, ignored: number): LitterSummary {
  const conf = events.filter((e) => e.event === 'confirmed')
  const by = new Map<string, LitterSummary['classes'][number]>()
  for (const e of conf) {
    const b = by.get(e.class) ?? { class: e.class, count: 0, max_confidence: 0, avg_confidence: 0, timestamps_sec: [] }
    b.avg_confidence = (b.avg_confidence * b.count + e.confidence) / (b.count + 1)
    b.count += 1
    b.max_confidence = Math.max(b.max_confidence, e.confidence)
    b.timestamps_sec.push(e.video_time_sec)
    by.set(e.class, b)
  }
  return {
    total: conf.length, roi_active: ignored > 0, inside_roi: conf.length, outside_roi_ignored: ignored,
    highest_confidence: conf.length ? Math.max(...conf.map((e) => e.confidence)) : null,
    average_confidence: conf.length ? conf.reduce((a, e) => a + e.confidence, 0) / conf.length : null,
    classes: [...by.values()].sort((a, b) => b.count - a.count),
    objects: conf.map((e) => ({ track_id: e.track_id, class: e.class, first_seen_sec: e.video_time_sec, max_confidence: e.confidence, avg_confidence: e.confidence })),
  }
}

const clock = (t: number) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, '0')}`

function Progress({ job }: { job?: Job<VideoResult> }) {
  if (!job) return <div className="py-16 text-center text-[13px] text-fg-2" aria-live="polite">Uploading and starting analysis…</div>
  const p = job.result?.progress
  return (
    <div className="flex flex-col gap-4 py-6" aria-live="polite">
      <div className="flex items-center justify-between text-[13px]">
        <span className="flex items-center gap-2 text-fg"><StatusBadge status={job.status} /> {job.status === 'queued' ? 'Waiting for the processing worker…' : 'Processing Video… detecting litter inside the ROI'}</span>
        <span className="tabular font-mono text-fg">{pct(job.progress, 0)}</span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-panel-2" role="progressbar" aria-valuenow={Math.round(job.progress * 100)} aria-valuemin={0} aria-valuemax={100}>
        <div className="h-full bg-info transition-[width] duration-500" style={{ width: `${job.progress * 100}%` }} />
      </div>
      <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
        <MetricCard label="Frames read" value={p ? `${num(p.frames_read)}${p.frames_total ? ` / ${num(p.frames_total)}` : ''}` : '—'} />
        <MetricCard label="Processing FPS" value={num(p?.processing_fps, 1)} />
        <MetricCard label="Estimated time left" value={p?.eta_sec != null ? duration(p.eta_sec) : '—'} />
        <MetricCard label="Litter found so far" value={num(p?.confirmed_objects)} tone="ok" />
      </div>
    </div>
  )
}

function Results({ job }: { job: Job<VideoResult> }) {
  const st = job.result!.stats!
  const events = job.result!.events ?? []
  const confirmed = events.filter((e) => e.event === 'confirmed')
  const classes = useQuery('classes', Api.classes)
  const prio = useMemo(() => {
    const p = Object.fromEntries((classes.data?.classes ?? []).map((c) => [c.name, c.priority]))
    const out = { high: 0, medium: 0, low: 0 }
    confirmed.forEach((e) => {
      const v = p[e.class] ?? 'medium'
      out[v === 'critical' || v === 'high' ? 'high' : v === 'low' ? 'low' : 'medium'] += 1
    })
    return out
  }, [classes.data, confirmed])
  const dur = st.video.duration_sec || 1
  const timeline = useMemo(() => confirmed.map((e) => ({ ...e, left: `${(e.video_time_sec / dur) * 100}%` })), [confirmed, dur])
  const seek = (t: number) => {
    const v = document.querySelector<HTMLVideoElement>(`video[data-job="${job.id}"]`)
    if (v) { v.currentTime = t; void v.play() }
  }
  return (
    <>
      <Panel title="Statistics" subtitle={`${st.video.width}×${st.video.height} · ${num(st.video.fps, 1)} fps · ${duration(st.video.duration_sec)} · model ${st.model_version} · output ${st.video.output_codec}`}
        actions={<>
          {(['detections', 'events', 'stats'] as const).map((k) => (
            <a key={k} href={url(`/api/jobs/${job.id}/files/${k}`)} download><Button size="sm" variant="ghost" icon={Download}>{k}.json</Button></a>
          ))}
        </>}>
        <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
          <MetricCard label="Total detections" value={confirmed.length} tone="info" hint="unique litter objects" />
          <MetricCard label="High priority" value={prio.high} tone={prio.high ? 'crit' : undefined} />
          <MetricCard label="Medium priority" value={prio.medium} tone={prio.medium ? 'warn' : undefined} />
          <MetricCard label="Low priority" value={prio.low} />
          <MetricCard label="Video duration" value={duration(st.video.duration_sec)} hint={`${num(st.video.fps, 0)} fps · ${st.video.width}×${st.video.height}`} />
          <MetricCard label="Frames analysed" value={num(st.frames_analysed)} hint={`of ${num(st.video.frames)} · model ran on ${num(st.model_runs ?? st.frames_analysed)}`} />
          <MetricCard label="Processing time" value={duration(st.wall_time_sec)} hint={st.device ? `on ${String(st.device).toUpperCase()}` : undefined} />
          <MetricCard label="Average confidence" value={confirmed.length ? pct(confirmed.reduce((x, e) => x + e.confidence, 0) / confirmed.length, 0) : '—'} />
        </div>
        {st.new_alerts > 0 && <Link href="/alerts" className="mt-2 inline-block text-[12px] text-info hover:underline">Review {st.new_alerts} alert(s) →</Link>}
      </Panel>
      <Panel title="Detection timeline" subtitle="Confirmation events on the video timeline — click to seek" icon={Timer}>
        {timeline.length === 0 ? <EmptyState compact title="No litter was confirmed in this video" body="Candidates that never met the temporal confirmation rule are counted in the statistics but produce no event." /> : (
          <div className="relative h-10 rounded-md border border-line bg-panel-2">
            {timeline.map((e) => (
              <button key={`${e.track_id}-${e.frame_idx}`} onClick={() => seek(e.video_time_sec)} title={`${cls(e.class)} ID ${e.track_id} @ ${e.video_time_sec.toFixed(1)}s`}
                className="absolute top-1.5 h-7 w-1.5 -translate-x-1/2 rounded-sm bg-ok hover:w-2.5" style={{ left: e.left }} aria-label={`Seek to ${cls(e.class)} at ${e.video_time_sec.toFixed(1)} seconds`} />
            ))}
          </div>
        )}
      </Panel>
      <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_340px]">
        <Panel title="Event list" noPad>
          <DataTable rows={events} rowKey={(e) => `${e.event}-${e.track_id}-${e.frame_idx}`} dense onRowClick={(e) => seek(e.video_time_sec)}
            empty={<EmptyState compact title="No events" />}
            columns={[
              { key: 't', header: 'Time', cell: (e) => <span className="font-mono">{e.video_time_sec.toFixed(2)}s</span>, sortValue: (e) => e.video_time_sec },
              { key: 'ev', header: 'Event', cell: (e) => <Badge tone={e.event === 'confirmed' ? 'ok' : 'neutral'}>{e.event}</Badge> },
              { key: 'cls', header: 'Class', cell: (e) => cls(e.class), sortValue: (e) => e.class },
              { key: 'id', header: 'Track', cell: (e) => <span className="font-mono">ID {e.track_id}</span> },
              { key: 'conf', header: 'Mean conf', align: 'right', cell: (e) => pct(e.confidence, 0) },
              { key: 'alert', header: 'Alert', cell: (e) => e.alert_id ? <Link className="text-info hover:underline" href={`/alerts?id=${e.alert_id}`} onClick={(ev) => ev.stopPropagation()}>#{e.alert_id}</Link> : '—' },
            ]} />
        </Panel>
        <Panel title="Confirmed by class">
          {Object.keys(st.confirmed_by_class).length === 0 ? <EmptyState compact title="None" /> :
            <BarList data={Object.entries(st.confirmed_by_class).sort((a, b) => b[1] - a[1]).map(([k, v]) => ({ label: cls(k), value: v }))} />}
          {Object.keys(st.rejection_reasons).length > 0 && (
            <div className="mt-4">
              <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-fg-3">Rejection reasons</h3>
              <BarList color="var(--fg-3)" data={Object.entries(st.rejection_reasons).map(([k, v]) => ({ label: k.replace(/_/g, ' '), value: v }))} />
            </div>
          )}
        </Panel>
      </div>
    </>
  )
}
