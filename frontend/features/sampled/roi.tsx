'use client'

// ROI calibration and display for the automatic four-screenshot analysis.
// - RoiOverlay: a normalised polygon drawn over a screenshot. The SVG uses the screenshot's own pixel size as viewBox
//   with "meet", exactly how object-contain places the picture, so the outline stays aligned at any display size.
// - RoiLegend: what the colours mean.
// - RoiCalibration: the operator picks the clearest screenshot, clicks the boundary of the monitored ground, edits
//   it (drag a point, undo, reset) and approves it. The server validates it again, stores it on this analysis only
//   and re-analyses the same four screenshots with it. Nothing is approved without the button press.
import { useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react'
import { AlertTriangle, Check, Layers, RotateCcw, Undo2 } from 'lucide-react'
import { ApiError, url } from '../../api/client'
import { Api } from '../../api/endpoints'
import type { SampledJob } from '../../api/sampledTypes'
import type { RoiZone } from '../../api/types'
import { invalidate } from '../../hooks/useQuery'
import { cx } from '../../lib/format'
import { labelAnchor, normalisePoint, polygonProblem, svgPoints, type Pt } from '../../lib/roiPolygon'
import { lockedFor, useApp } from '../../stores/app'
import { Badge, Button, Panel, Toggle } from '../../components/ui/primitives'

export const ROI_GREEN = 'rgb(46,204,64)'

/** Approved (solid green) or proposed (dashed amber, explicitly "not approved") ROI over a screenshot. */
export function RoiOverlay({ zones, size, kind = 'approved' }: { zones: RoiZone[]; size: [number, number]; kind?: 'approved' | 'proposed' }) {
  const [w, h] = size
  const fs = Math.max(11, w / 55)
  const color = kind === 'approved' ? ROI_GREEN : 'rgb(255,190,40)'
  return (
    <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="xMidYMid meet" className="pointer-events-none absolute inset-0 size-full" aria-hidden>
      {zones.map((z, i) => {
        const pts = z.polygon as Pt[]
        const include = z.mode === 'include'
        const [lx, ly] = labelAnchor(pts, w, h)
        const text = include ? (kind === 'approved' ? 'APPROVED ROI' : 'PROPOSED ROI · NOT APPROVED') : `EXCLUDED · ${z.name}`
        const tw = text.length * fs * 0.62 + fs * 0.8
        const tx = Math.min(Math.max(0, lx - tw / 2), w - tw)
        const ty = Math.min(Math.max(fs * 1.5, ly - fs * 0.4), h)
        return (
          <g key={i}>
            <polygon points={svgPoints(pts, w, h)}
              fill={include ? (kind === 'approved' ? 'rgba(46,204,64,0.14)' : 'rgba(255,190,40,0.12)') : 'rgba(220,40,40,0.16)'}
              stroke={include ? color : 'rgb(220,40,40)'} strokeWidth={2.5} strokeLinejoin="round"
              strokeDasharray={include && kind === 'approved' ? undefined : '8 5'} vectorEffect="non-scaling-stroke" />
            <rect x={tx} y={ty - fs * 1.25} width={tw} height={fs * 1.5} rx={fs * 0.25} fill={include ? color : 'rgb(220,40,40)'} opacity={0.92} />
            <text x={tx + fs * 0.4} y={ty - fs * 0.1} fontSize={fs} fontWeight={700} fontFamily="var(--font-mono, monospace)" fill="#0b0d10">{text}</text>
          </g>
        )
      })}
    </svg>
  )
}

export function RoiLegend({ className }: { className?: string }) {
  return (
    <ul className={cx('flex flex-wrap items-center gap-x-4 gap-y-1.5 text-[12px] text-fg-2', className)} aria-label="Legend">
      <li className="inline-flex items-center gap-1.5"><span className="h-3 w-5 rounded-sm border-2" style={{ borderColor: ROI_GREEN, background: 'rgba(46,204,64,0.18)' }} aria-hidden />Green polygon = approved monitoring ROI</li>
      <li className="inline-flex items-center gap-1.5"><span className="h-3 w-5 rounded-sm border-2" style={{ borderColor: 'rgb(255,90,0)' }} aria-hidden />Detection box = detected litter (confirmed, inside the ROI)</li>
      <li className="inline-flex items-center gap-1.5"><span className="h-3 w-5 rounded-sm border-2 border-dashed border-fg-3" aria-hidden />Outside ROI = excluded from pavement-litter confirmation (kept in the CSV for diagnostics)</li>
    </ul>
  )
}

/** Screenshot size: the analysed picture's size, else the video's. */
export function frameSize(job: SampledJob, idx?: number): [number, number] | null {
  const f = job.frames.find((x) => x.idx === idx) ?? job.frames.find((x) => x.image_size)
  if (f?.image_size) return f.image_size
  return job.media ? [job.media.width, job.media.height] : null
}

// ---------------------------------------------------------------- drawing
function PolygonDrawer({ src, size, points, onChange, onMove }: { src: string; size: [number, number]; points: Pt[]; onChange: (p: Pt[]) => void; onMove: (p: Pt[]) => void }) {
  const [w, h] = size
  const svg = useRef<SVGSVGElement>(null)
  const drag = useRef<number | null>(null)
  const toImage = (clientX: number, clientY: number): Pt | null => {
    const el = svg.current
    const ctm = el?.getScreenCTM()
    if (!el || !ctm) return null
    const p = el.createSVGPoint()
    p.x = clientX; p.y = clientY
    const q = p.matrixTransform(ctm.inverse())       // screen -> screenshot pixels, whatever the display size
    return normalisePoint(q.x, q.y, w, h)
  }
  const r = Math.max(5, w / 150)
  const problem = points.length >= 3 ? polygonProblem(points, w, h) : null
  return (
    <div className="relative w-full overflow-hidden rounded-md border border-line bg-video" style={{ aspectRatio: `${w} / ${h}` }}>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={src} alt="Reference screenshot for the ROI" draggable={false} className="absolute inset-0 size-full select-none object-contain" />
      <svg ref={svg} viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="xMidYMid meet" className="absolute inset-0 size-full cursor-crosshair touch-none"
        role="application" aria-label="Click to add boundary points of the monitored ground area; drag a point to move it"
        onClick={(e) => { if (drag.current !== null) return; const p = toImage(e.clientX, e.clientY); if (p) onChange([...points, p]) }}
        onPointerMove={(e) => {
          if (drag.current === null) return
          const p = toImage(e.clientX, e.clientY)
          if (p) onMove(points.map((q, i) => (i === drag.current ? p : q)))
        }}
        onPointerUp={() => { setTimeout(() => { drag.current = null }, 0) }}>
        {points.length >= 3 && (
          <polygon points={svgPoints(points, w, h)} fill={problem ? 'rgba(220,40,40,0.14)' : 'rgba(46,204,64,0.16)'}
            stroke={problem ? 'rgb(220,40,40)' : ROI_GREEN} strokeWidth={2.5} vectorEffect="non-scaling-stroke" strokeLinejoin="round" />
        )}
        {points.length === 2 && <polyline points={svgPoints(points, w, h)} fill="none" stroke={ROI_GREEN} strokeWidth={2.5} vectorEffect="non-scaling-stroke" />}
        {points.map(([x, y], i) => (
          <circle key={i} cx={x * w} cy={y * h} r={r} fill={i === points.length - 1 ? '#fff' : ROI_GREEN} stroke="#0b0d10" strokeWidth={1.5}
            vectorEffect="non-scaling-stroke" className="cursor-move"
            onClick={(e) => e.stopPropagation()}
            onPointerDown={(e: ReactPointerEvent<SVGCircleElement>) => {
              e.stopPropagation()
              drag.current = i
              onChange([...points])            // one undo step per drag
              svg.current?.setPointerCapture(e.pointerId)
            }}>
            <title>{`Point ${i + 1} - drag to move`}</title>
          </circle>
        ))}
      </svg>
    </div>
  )
}

// ---------------------------------------------------------------- calibration panel
export function RoiCalibration({ job, onDone }: { job: SampledJob; onDone?: () => void }) {
  const { toast, canOperate, role, loginOn } = useApp()
  const lock = lockedFor('operator', role, loginOn)
  const saved = job.frames.filter((f) => f.status === 'saved' && f.original_url)
  const policy = job.roi_policy
  const editing = policy?.approved && policy.source === 'OPERATOR_APPROVED_FOR_JOB'
  const initial = useMemo<Pt[]>(() => (editing ? ((policy!.zones.find((z) => z.mode === 'include')?.polygon ?? []) as Pt[]) : []), [editing, policy])
  const [ref, setRef] = useState<number>(policy?.reference_idx ?? saved[0]?.idx ?? 1)
  const [history, setHistory] = useState<Pt[][]>([initial])
  const points = history[history.length - 1]
  const set = (p: Pt[]) => setHistory((h) => [...h, p])
  const move = (p: Pt[]) => setHistory((h) => [...h.slice(0, -1), p])
  const [saveCam, setSaveCam] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const size = frameSize(job, ref)
  const refFrame = saved.find((f) => f.idx === ref)
  const problem = size ? polygonProblem(points, size[0], size[1]) : 'The screenshot size is unknown.'
  const draft: RoiZone[] = points.length >= 3 ? [{ name: 'monitored-ground', type: 'ground', mode: 'include', polygon: points }] : []

  const approve = async () => {
    setBusy(true); setError(null)
    try {
      await Api.sampledApproveRoi(job.id, { zones: draft, reference_idx: ref, save_for_camera: saveCam })
      invalidate(`sampled:job:${job.id}`)
      invalidate('sampled:')
      toast({ tone: 'ok', title: 'ROI approved', body: 'The four screenshots are being analysed again with the approved ROI.' })
      onDone?.()
    } catch (e) {
      setError(e instanceof ApiError ? e.message : (e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  if (!saved.length || !size) return null
  return (
    <Panel title={editing ? 'Edit the approved monitoring ROI' : 'Define the monitoring ROI'} icon={Layers}
      subtitle="Outline only the ground in front of the shop where litter collects. Leave out walls, shutters, signboards, the shop interior, road traffic and vehicles.">
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_300px]">
        <div className="flex min-w-0 flex-col gap-3">
          <div className="flex flex-wrap gap-2" role="radiogroup" aria-label="Reference screenshot">
            {saved.map((f) => (
              <button key={f.idx} role="radio" aria-checked={ref === f.idx} onClick={() => setRef(f.idx)}
                className={cx('flex items-center gap-2 rounded-lg border p-1 pr-2.5 text-[12px]', ref === f.idx ? 'border-info bg-info-bg text-fg' : 'border-line text-fg-2 hover:border-line-strong')}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={url(f.original_url!)} alt="" className="h-9 w-14 rounded-sm bg-video object-cover" />
                #{f.idx} <span className="font-mono">{f.video_timestamp.slice(0, 8)}</span>
              </button>
            ))}
          </div>
          {refFrame && <PolygonDrawer src={url(refFrame.original_url!)} size={size} points={points} onChange={set} onMove={move} />}
          <div className="flex flex-col gap-1.5">
            <p className="text-[12px] text-fg-3">Preview: the same polygon on all four screenshots (fixed camera view). It is a proposal until you approve it.</p>
            <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
              {saved.map((f) => (
                <div key={f.idx} className="relative overflow-hidden rounded-md border border-line bg-video" style={{ aspectRatio: `${size[0]} / ${size[1]}` }}>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={url(f.original_url!)} alt={`Screenshot ${f.idx} with the proposed ROI`} className="absolute inset-0 size-full object-contain" />
                  {draft.length > 0 && <RoiOverlay zones={draft} size={frameSize(job, f.idx) ?? size} kind="proposed" />}
                  <span className="absolute left-1 top-1 rounded bg-black/70 px-1 font-mono text-[10px] text-white">#{f.idx}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="flex flex-col gap-3">
          <ol className="flex flex-col gap-1.5 text-[12.5px] text-fg-2">
            <li>1. Choose the clearest screenshot above.</li>
            <li>2. Click the boundary of the monitored ground, point by point, following its real edge and perspective.</li>
            <li>3. Drag a point to adjust it. Undo removes the last change.</li>
            <li>4. Check the preview on all four screenshots, then approve.</li>
          </ol>
          <div className="flex flex-wrap items-center gap-2">
            <Badge tone={problem ? 'warn' : 'ok'}>{points.length} point{points.length === 1 ? '' : 's'}</Badge>
            <Button size="sm" icon={Undo2} disabled={history.length <= 1} onClick={() => setHistory((h) => h.slice(0, -1))}>Undo</Button>
            <Button size="sm" icon={RotateCcw} disabled={!points.length} onClick={() => set([])}>Reset</Button>
          </div>
          {points.length > 0 && problem && <p className="flex gap-1.5 text-[12px] text-warn"><AlertTriangle className="mt-0.5 size-3.5 shrink-0" aria-hidden />{problem}</p>}
          {job.camera_id ? (
            <Toggle label={`Also save for camera ${job.camera_id}`} checked={saveCam} onChange={setSaveCam}
              description="Later videos of this camera ID reuse it automatically. A video with a different frame shape asks for revalidation. Registered cameras need an administrator." />
          ) : (
            <p className="text-[12px] text-fg-3">No Camera ID was given: this ROI applies to this video only and is never reused for other videos.</p>
          )}
          <p className="text-[12px] text-fg-3">Approving re-runs detection on the same four screenshots with the existing model, thresholds and rules. Detections outside the ROI stay in the CSV as excluded.</p>
          {error && <p className="rounded-md bg-crit-bg px-3 py-2 text-[12.5px] text-crit">{error}</p>}
          <Button variant="primary" icon={Check} loading={busy} disabled={!canOperate || !!problem} title={lock ?? problem ?? undefined} onClick={approve} className="h-9">
            Approve ROI and re-analyse
          </Button>
          {onDone && editing && <Button size="sm" variant="ghost" onClick={onDone}>Cancel editing</Button>}
        </div>
      </div>
    </Panel>
  )
}
