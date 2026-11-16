'use client'

// Zero-click video analysis: drop a video -> upload + automatic analysis (one request) -> live progress ->
// four screenshots (original + annotated), summary and reports. Nothing to click after the drop.
// Logic (validation, duplicate-drop guard, lifecycle, labels) lives in frontend/lib/autoAnalysis.ts (unit-tested).
import { useEffect, useRef, useState, type ReactNode } from 'react'
import { AlertTriangle, Check, CircleSlash, Download, FileText, FileVideo, Info, Layers, Loader2, RotateCcw, ScanSearch, SlidersHorizontal, X } from 'lucide-react'
import { downloadAuthed, url } from '../../api/client'
import { Api } from '../../api/endpoints'
import type { SampledFrame, SampledJob, SampledStage } from '../../api/sampledTypes'
import type { RoiZone } from '../../api/types'
import { invalidate, useQuery } from '../../hooks/useQuery'
import { cx, dateTime, duration, num, pct } from '../../lib/format'
import {
  DEFAULTS_KEY, ID_RE, LIFECYCLE_STEPS, STORE_RE, cleanDefaults, createAutoUploader, isTerminal, overallLabel, resultsReady,
  roiLabel, screenshotFacts, stepIndex, validateVideo, type Lifecycle, type UploadDefaults,
} from '../../lib/autoAnalysis'
import { lockedFor, useApp } from '../../stores/app'
import { Badge, Button, Field, Input, Panel, type Tone } from '../../components/ui/primitives'
import { MetricCard } from '../../components/ui/kpi'
import { EmptyState, ErrorState } from '../../components/ui/states'
import { DropZone, litterName } from '../detection/common'
import { CsvPreview, OUTCOME } from './shared'

const ACCEPT = 'video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,video/webm,.mkv,.avi,.m4v,.mov,.mp4,.webm'
const SEVERITY_TONE: Record<string, Tone> = { HIGH: 'crit', MEDIUM: 'warn', LOW: 'neutral' }
const LIFECYCLE_TONE: Record<Lifecycle, Tone> = {
  UPLOADING: 'info', QUEUED: 'neutral', PROCESSING: 'warn', EXTRACTING_SCREENSHOTS: 'warn', RUNNING_DETECTION: 'warn',
  GENERATING_REPORT: 'warn', COMPLETED: 'ok', FAILED: 'crit', CANCELLED: 'neutral', AWAITING_ROI: 'warn',
}
const REPORT_COLS = ['record_type', 'screenshot_number', 'screenshot_timestamp_seconds', 'litter_class', 'confidence_score',
  'severity', 'object_track_id', 'final_classification', 'review_required']
const PAVEMENT_COLS = ['record_type', 'screenshot_number', 'litter_class', 'confidence_score', 'pavement_outcome',
  'pavement_reason', 'size_check', 'review_required']

/** Where a stopped job got to (for the progress strip of a FAILED / CANCELLED job). */
function stageLifecycle(stage: SampledStage): Lifecycle {
  return stage === 'sampling' ? 'EXTRACTING_SCREENSHOTS'
    : stage === 'analysing' || stage === 'generating_images' ? 'RUNNING_DETECTION'
    : stage === 'generating_csv' ? 'GENERATING_REPORT' : 'PROCESSING'
}

function readDefaults(): UploadDefaults {
  try { return cleanDefaults(JSON.parse(window.localStorage.getItem(DEFAULTS_KEY) ?? 'null')) } catch { return cleanDefaults(null) }
}

function saveDefaults(d: UploadDefaults) {
  try { window.localStorage.setItem(DEFAULTS_KEY, JSON.stringify(d)) } catch { /* private window: not remembered */ }
}

export function LifecycleSteps({ current, stopped }: { current: number; stopped?: 'FAILED' | 'CANCELLED' }) {
  return (
    <ol className="flex flex-wrap items-center gap-x-2 gap-y-1.5" aria-label="Analysis progress">
      {LIFECYCLE_STEPS.map((s, i) => {
        const done = i < current
        const now = i === current
        const halted = now && stopped
        return (
          <li key={s.key} className="flex items-center gap-2" aria-current={now ? 'step' : undefined}>
            <span className={cx('flex size-6 items-center justify-center rounded-full border text-[11px] font-semibold',
              done ? 'border-ok/40 bg-ok-bg text-ok' : halted ? 'border-crit/40 bg-crit-bg text-crit'
                : now ? 'border-info bg-info text-on-accent' : 'border-line-strong text-fg-3')}>
              {done ? <Check className="size-3.5" aria-hidden /> : halted ? <X className="size-3.5" aria-hidden />
                : now ? <Loader2 className="size-3.5 animate-spin" aria-hidden /> : i + 1}
            </span>
            <span className={cx('text-[12.5px]', done || now ? 'font-medium text-fg' : 'text-fg-3')}>{s.label}</span>
            {i < LIFECYCLE_STEPS.length - 1 && <span className="mx-1 hidden h-px w-6 bg-line-strong sm:block" aria-hidden />}
          </li>
        )
      })}
    </ol>
  )
}

// ---------------------------------------------------------------- drop -> upload -> job
export function AutoUpload({ onJob, onAdvanced }: { onJob: (jobId: string) => void; onAdvanced: () => void }) {
  const { toast, canOperate, role, loginOn } = useApp()
  const lock = lockedFor('operator', role, loginOn)
  const health = useQuery('health', Api.health)
  const cams = useQuery('cameras:list', Api.cameras)
  const [defaults, setDefaults] = useState<UploadDefaults>({ camera_id: '', store_id: '' })
  const [uploading, setUploading] = useState<string | null>(null)
  const [error, setError] = useState<Error | null>(null)
  const latest = useRef({ defaults, limits: health.data?.limits ?? null, onJob })
  latest.current = { defaults, limits: health.data?.limits ?? null, onJob }
  useEffect(() => { setDefaults(readDefaults()) }, [])

  // one uploader for the page's lifetime: it is what makes a doubled drop event upload once
  const uploader = useRef<ReturnType<typeof createAutoUploader<File>>>(null)
  if (!uploader.current) {
    uploader.current = createAutoUploader<File>({
      validate: (f) => validateVideo(f, latest.current.limits),
      upload: (f) => Api.sampledAutoUpload(f, cleanDefaults(latest.current.defaults)),
      onStart: (f) => { setError(null); setUploading(f.name) },
      onJob: (id, r) => {
        setUploading(null)
        invalidate('sampled:')
        if (r.duplicate_of_active) toast({ tone: 'warn', title: 'Already being analysed', body: 'This video is already being analysed. Showing that analysis.' })
        latest.current.onJob(id)
      },
      onError: (e) => { setUploading(null); setError(e) },
    })
  }

  const drop = (f: File) => {
    if (!canOperate) { toast({ tone: 'warn', title: 'Not allowed', body: lock }); return }
    void uploader.current!.drop(f)
  }
  const update = (patch: Partial<UploadDefaults>) => {
    const next = { ...defaults, ...patch }
    setDefaults(next)
    saveDefaults(cleanDefaults(next))
  }
  const camBad = !!defaults.camera_id && !ID_RE.test(defaults.camera_id.trim())
  const storeBad = !!defaults.store_id && !STORE_RE.test(defaults.store_id.trim())
  const noModel = health.data && !health.data.model
  const exts = (health.data?.limits?.video_extensions ?? ['.mp4', '.mov', '.avi', '.mkv', '.m4v', '.webm']).map((e) => e.slice(1).toUpperCase()).join(', ')

  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_340px]">
      <Panel title="Drop a CCTV video" subtitle="The analysis starts as soon as the video is dropped" icon={FileVideo}>
        <div className="flex flex-col gap-3">
          {uploading ? (
            <div className="flex min-h-[320px] flex-col items-center justify-center gap-4 rounded-xl border border-line bg-panel-2/50 p-6 text-center" aria-live="polite">
              <LifecycleSteps current={0} />
              <p className="text-[13px] font-medium text-fg">Uploading “{uploading}”…</p>
              <p className="text-[12px] text-fg-3">The analysis starts on the server as soon as the upload finishes. Keep this page open until then.</p>
            </div>
          ) : (
            <DropZone accept={ACCEPT} onFile={drop} label="Drag and drop a CCTV video here"
              hint={`${exts}${health.data?.limits ? ` · up to ${health.data.limits.max_video_mb} MB` : ''} · analysis starts automatically`} />
          )}
          {error && <ErrorState error={error} title="Analysis not started" compact />}
          {noModel && <p className="rounded-md bg-crit-bg px-3 py-2 text-[12.5px] text-crit">No detection model is loaded on the server; videos cannot be analysed.</p>}
          <details className="rounded-lg border border-line px-3 py-2" open={!!(defaults.camera_id || defaults.store_id)}>
            <summary className="cursor-pointer text-[12.5px] text-fg-2">Upload defaults (optional, applied to every dropped video)</summary>
            <div className="mt-3 grid gap-3 sm:grid-cols-2">
              <Field label="Camera ID" hint={camBad ? 'Letters, digits, - and _ only (max 40).' : 'A camera with a saved ROI is analysed inside that ROI.'}>
                <Input value={defaults.camera_id} maxLength={40} list="auto-cams" placeholder="none" onChange={(e) => update({ camera_id: e.target.value })} />
                <datalist id="auto-cams">{cams.data?.cameras.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}</datalist>
              </Field>
              <Field label="Store ID" hint={storeBad ? 'Letters, digits, spaces, . - _ only (max 64).' : 'Recorded in the report; left blank when not given.'}>
                <Input value={defaults.store_id} maxLength={64} placeholder="none" onChange={(e) => update({ store_id: e.target.value })} />
              </Field>
            </div>
            <p className="mt-2 text-[12px] text-fg-3">Without a camera that has a saved ROI, the whole frame is analysed and the ROI status says NOT_CONFIGURED.</p>
          </details>
        </div>
      </Panel>
      <Panel title="What happens automatically" icon={Info}>
        <ol className="flex flex-col gap-3 text-[13px] text-fg-2">
          {[
            'The video is checked and uploaded; the analysis is created at the same time.',
            'Four screenshots are taken at 10 %, 35 %, 65 % and 90 % of the video. An unreadable frame is replaced by the nearest readable one (recorded).',
            'The ROI saved for the camera is applied. Without one, the whole frame is analysed and marked NOT_CONFIGURED.',
            'The litter model analyses the four screenshots. The same object seen in several screenshots is counted once.',
            'Annotated screenshots, a summary and a CSV report appear here automatically.',
          ].map((t, i) => (
            <li key={i} className="flex gap-3"><span className="flex size-5 shrink-0 items-center justify-center rounded-full border border-line-strong text-[11px] font-semibold text-fg">{i + 1}</span>{t}</li>
          ))}
        </ol>
        <p className="mt-3 text-[12px] text-fg-3">Four representative screenshots are analysed, not every frame. For every frame, use Full video (tracking).</p>
        <Button size="sm" variant="ghost" icon={SlidersHorizontal} className="mt-3" onClick={onAdvanced}>Review settings and ROI before analysis</Button>
      </Panel>
    </div>
  )
}

// ---------------------------------------------------------------- job: progress + results
export function AutoJobView({ job, onNew }: { job: SampledJob; onNew: () => void }) {
  const { toast, canOperate, role, loginOn } = useApp()
  const lock = lockedFor('operator', role, loginOn)
  const [cancelling, setCancelling] = useState(false)
  const l = job.lifecycle
  const stopped = l === 'FAILED' || l === 'CANCELLED' ? l : undefined
  const current = stepIndex(l, stopped ? stageLifecycle(job.stage) : undefined)
  const ready = resultsReady(job)
  const s = job.summary
  const roi = roiLabel(job.roi_policy, job.camera_id)
  const overall = overallLabel(s?.overall_analysis_status)
  const cancel = async () => {
    setCancelling(true)
    try { await Api.sampledCancel(job.id); invalidate(`sampled:job:${job.id}`) } catch (e) { toast({ tone: 'crit', title: 'Could not cancel', body: (e as Error).message }) } finally { setCancelling(false) }
  }
  const download = async (path: string, name: string) => {
    try { await downloadAuthed(path, name) } catch (e) { toast({ tone: 'crit', title: 'Download failed', body: (e as Error).message }) }
  }
  return (
    <>
      <Panel title={job.video_name ?? 'Video analysis'} icon={FileVideo}
        subtitle={[job.store_id && `Store ${job.store_id}`, job.camera_id && `Camera ${job.camera_id}`, job.media && duration(job.media.duration_sec), `started ${dateTime(job.created_at)}`].filter(Boolean).join(' · ')}
        actions={<>
          <Badge tone={LIFECYCLE_TONE[l]} dot>{l.replace(/_/g, ' ')}</Badge>
          {!isTerminal(l) && <Button size="sm" variant="ghost" icon={X} loading={cancelling} disabled={!canOperate} title={lock} onClick={cancel}>Cancel</Button>}
          <Button size="sm" icon={RotateCcw} onClick={onNew}>Analyse another video</Button>
        </>}>
        <div className="flex flex-col gap-3">
          <LifecycleSteps current={current} stopped={stopped} />
          {job.error && (
            <p className="flex gap-2 rounded-md bg-crit-bg px-3 py-2 text-[12.5px] text-crit"><AlertTriangle className="mt-0.5 size-3.5 shrink-0" aria-hidden />
              <span>{job.error}{l === 'FAILED' ? ' This video has not been reported as clean.' : ''}</span></p>
          )}
          {job.roi_policy && (
            <p className={cx('flex gap-2 rounded-md px-3 py-2 text-[12.5px]', roi.tone === 'ok' ? 'bg-ok-bg text-ok' : 'bg-warn-bg text-warn')}>
              <Layers className="mt-0.5 size-3.5 shrink-0" aria-hidden /><span><b className="font-medium">ROI: {roi.text}.</b> {job.roi_policy.note}</span>
            </p>
          )}
        </div>
      </Panel>

      {ready && s && (
        <Panel title="Summary" icon={Layers} subtitle={s.note}>
          <div className="grid grid-cols-2 gap-2.5 md:grid-cols-4">
            <MetricCard label="Total detected objects" value={num(s.total_objects_detected)} tone={s.total_objects_detected ? 'crit' : undefined} hint="confirmed, all screenshots" />
            <MetricCard label="Estimated unique objects" value={num(s.unique_objects_estimated)} hint="same object counted once" />
            <MetricCard label="High severity" value={num(s.total_high_severity)} tone={s.total_high_severity ? 'crit' : undefined} hint="per unique object" />
            <MetricCard label="Medium severity" value={num(s.total_medium_severity)} tone={s.total_medium_severity ? 'warn' : undefined} />
            <MetricCard label="Low severity" value={num(s.total_low_severity)} />
            <MetricCard label="ROI status" value={job.roi_policy?.status.replace(/_/g, ' ') ?? '—'} tone={roi.tone === 'ok' ? 'ok' : 'warn'} />
            <MetricCard label="Analysis duration" value={typeof s.processing_duration_seconds === 'number' ? `${num(s.processing_duration_seconds, 1)} s` : '—'} />
            <MetricCard label="Result" value={overall.text} tone={overall.tone} hint={`${s.screenshots_analysed} of ${s.screenshots_planned} screenshots analysed`} />
          </div>
          {(s.review_required === 'TRUE' || s.possible_litter_objects > 0 || s.sampling_adjustments > 0) && (
            <ul className="mt-3 flex flex-col gap-1 text-[12.5px] text-fg-2">
              {s.review_required === 'TRUE' && <li>• Review needed: see the <span className="font-mono">review_required</span> column of the CSV.</li>}
              {s.possible_litter_objects > 0 && <li>• {s.possible_litter_objects} possible litter object(s) below the confirmed threshold (not counted above).</li>}
              {s.sampling_adjustments > 0 && <li>• {s.sampling_adjustments} screenshot(s) taken from the nearest readable frame (planned frame unreadable).</li>}
              {s.alerts_created > 0 && <li>• {s.alerts_created} alert(s) created on the Alerts page.</li>}
            </ul>
          )}
        </Panel>
      )}

      {job.frames.some((f) => f.status !== 'planned') && (
        <Panel title="Four screenshots" icon={ScanSearch} subtitle={ready ? 'Original and annotated; boxes show the model’s detections and confidence' : 'Screenshots appear as they are extracted and analysed'}>
          <div className="grid gap-3 lg:grid-cols-2">{job.frames.map((f) => <AutoFrameCard key={f.idx} frame={f} done={isTerminal(l)}
            zones={f.roi ?? job.roi_policy?.zones ?? job.roi ?? []} />)}</div>
        </Panel>
      )}

      {ready && job.reports && (
        <Panel title="CSV report" icon={FileText} subtitle="One row per screenshot plus one per detected object; totals are in the summary file"
          actions={<>
            <Button size="sm" icon={Download} onClick={() => download(job.reports!.summary_csv, `${job.id}_analysis_summary.csv`)}>Summary CSV</Button>
            <Button size="sm" icon={Download} onClick={() => download(job.reports!.summary_json, `${job.id}_analysis_summary.json`)}>Summary JSON</Button>
          </>}>
          <CsvPreview path={job.reports.detection_csv} filename={`${job.id}_detection_report.csv`}
            cols={job.frames.some((f) => f.pavement) ? PAVEMENT_COLS : REPORT_COLS} />
        </Panel>
      )}
    </>
  )
}

/** The camera's saved ROI drawn over the untouched original (green = pavement, red dashed = exclusion). The SVG uses
 * the frame's own pixel size as viewBox with "meet", which is exactly how object-contain places the picture. */
function RoiOverlay({ zones, size }: { zones: RoiZone[]; size: [number, number] }) {
  const [w, h] = size
  return (
    <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="xMidYMid meet" className="pointer-events-none absolute inset-0 size-full" aria-hidden>
      {zones.map((z, i) => (
        <polygon key={i} points={z.polygon.map(([x, y]) => `${x * w},${y * h}`).join(' ')}
          fill={z.mode === 'include' ? 'rgba(60,200,60,0.12)' : 'rgba(220,40,40,0.16)'}
          stroke={z.mode === 'include' ? 'rgb(60,200,60)' : 'rgb(220,40,40)'} strokeWidth={Math.max(2, w / 640)}
          strokeDasharray={z.mode === 'include' ? undefined : `${w / 160} ${w / 240}`} vectorEffect="non-scaling-stroke" />
      ))}
    </svg>
  )
}

function AutoFrameCard({ frame, done, zones }: { frame: SampledFrame; done: boolean; zones: RoiZone[] }) {
  const facts = screenshotFacts(frame)
  const o = frame.outcome ? OUTCOME[frame.outcome] : null
  const pending = !done && !frame.outcome
  const [showDiag, setShowDiag] = useState(false)
  const pav = facts.pavement
  const overlay = pav?.active && frame.image_size && zones.length ? <RoiOverlay zones={zones} size={frame.image_size} /> : null
  const image = (src: string | null, label: string, dl: string | null, extra?: ReactNode) => (
    <figure className="flex min-w-0 flex-col gap-1">
      <div className="relative aspect-video overflow-hidden rounded-md bg-video">
        {src ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={url(src)} alt={`Screenshot ${frame.idx}, ${label.toLowerCase()}, at ${frame.video_timestamp}`} loading="lazy" className="size-full object-contain" />
        ) : (
          <div className="flex size-full flex-col items-center justify-center gap-1.5 p-3 text-center text-[12px] text-fg-3">
            {pending ? <Loader2 className="size-4 animate-spin" aria-hidden /> : <CircleSlash className="size-4" aria-hidden />}
            {pending ? 'Processing…' : 'Not available'}
          </div>
        )}
        {src && extra}
      </div>
      <figcaption className="flex items-center justify-between text-[11.5px] text-fg-3">
        {label}
        {dl && <a href={url(dl)} className="inline-flex items-center gap-1 text-fg-2 hover:text-fg"><Download className="size-3" aria-hidden />JPG</a>}
      </figcaption>
    </figure>
  )
  return (
    <article className="flex min-w-0 flex-col gap-2.5 rounded-xl border border-line bg-panel p-3">
      <header className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-[13px] font-medium text-fg">Screenshot {frame.idx}
          <span className="ml-2 font-mono text-[12px] font-normal text-fg-3">{frame.video_timestamp} · frame {frame.frame_idx}</span>
        </div>
        <div className="flex items-center gap-1.5">
          {facts.adjusted && <Badge tone="warn" title={frame.sampling_note ?? undefined}>nearest readable frame</Badge>}
          {o ? <Badge tone={o.tone} dot>{o.label}</Badge> : pending ? <Badge>processing</Badge> : null}
        </div>
      </header>
      <div className="grid grid-cols-2 gap-2">
        {image(frame.original_url, overlay ? 'Original + approved ROI' : 'Original', frame.download_original_url, overlay)}
        {image(showDiag && frame.diagnostic_url ? frame.diagnostic_url : frame.annotated_url,
          showDiag && frame.diagnostic_url ? 'Diagnostic (incl. rejected)' : 'Annotated', frame.download_annotated_url)}
      </div>
      {pav && (
        <div className="flex flex-wrap items-center gap-1.5 text-[12px]">
          <Badge tone={pav.active ? 'ok' : pav.general ? 'neutral' : 'warn'}>{pav.active ? 'Camera ROI mode · pavement-validated'
            : pav.general ? 'General image mode · not pavement-validated'
              : `${pav.status.replace(/_/g, ' ')} · pavement-only detection unavailable`}</Badge>
          {frame.pavement?.camera_id && <Badge>Camera {frame.pavement.camera_id}</Badge>}
          {pav.active && <Badge tone={facts.count ? 'crit' : 'neutral'}>{facts.count} confirmed</Badge>}
          {pav.active && <Badge tone={facts.possible ? 'warn' : 'neutral'}>{facts.possible} candidate{facts.possible === 1 ? '' : 's'} (review)</Badge>}
          {pav.rejected > 0 && <Badge>{pav.rejected} rejected</Badge>}
          {pav.unverified > 0 && <Badge tone="warn">{pav.unverified} unverified (no ROI)</Badge>}
          {frame.diagnostic_url && (
            <label className="ml-auto inline-flex cursor-pointer items-center gap-1.5 text-fg-3">
              <input type="checkbox" checked={showDiag} onChange={(e) => setShowDiag(e.target.checked)} />Diagnostic view
            </label>
          )}
        </div>
      )}
      {facts.roiUnavailable && o && <p className="flex gap-1.5 text-[12px] text-warn"><AlertTriangle className="mt-0.5 size-3.5 shrink-0" aria-hidden />{o.hint}</p>}
      {facts.analysed && (
        facts.count === 0 ? <EmptyState compact title="No litter detected" body={facts.possible ? `${facts.possible} possible object(s) below the confirmed threshold.` : undefined} /> : (
          <div className="flex flex-col gap-1.5">
            <p className="text-[12px] text-fg-3">{facts.count} litter object{facts.count === 1 ? '' : 's'} detected{facts.possible ? ` · ${facts.possible} possible` : ''}</p>
            <ul className="flex flex-wrap gap-1.5">
              {facts.items.map((d, i) => (
                <li key={i} className="inline-flex items-center gap-1.5 rounded-full border border-line bg-panel-2 px-2 py-0.5 text-[12px]" title={d.object ? `Estimated object ${d.object}` : undefined}>
                  <span className="text-fg">{litterName(d.cls)}</span>
                  <span className="tabular text-fg-3">{pct(d.confidence, 0)}</span>
                  {d.severity && <Badge tone={SEVERITY_TONE[d.severity] ?? 'neutral'}>{d.severity.toLowerCase()}</Badge>}
                </li>
              ))}
            </ul>
          </div>
        )
      )}
      {(frame.error || frame.sampling_note) && (
        <p className="flex gap-1.5 text-[12px] text-fg-3"><AlertTriangle className="mt-0.5 size-3.5 shrink-0" aria-hidden />{frame.error ?? frame.sampling_note}</p>
      )}
    </article>
  )
}
