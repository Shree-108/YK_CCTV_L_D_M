'use client'

// Ground ROI of one camera: overlay preview of the auto ROI + Compute / Approve / Edit / Clear.
// The polygon is drawn client-side over the reference (median) frame, so editing needs no server round trip.
import { useRef, useState, type PointerEvent as ReactPointerEvent } from 'react'
import { Check, Eraser, Pencil, Plus, ScanSearch, X } from 'lucide-react'
import { AutoRoiApi } from '../../api/autoRoi'
import { invalidate, useQuery } from '../../hooks/useQuery'
import { insertVertex, moveVertex, polygonValid, removeVertex, roiStatus, svgPoints, type Point } from '../../lib/autoRoi'
import { lockedFor, useApp } from '../../stores/app'
import { Badge, Button, Select } from '../../components/ui/primitives'
import { ConfirmDialog } from '../../components/ui/overlay'

export function AutoRoiPanel({ camId }: { camId: string }) {
  const { isAdmin, role, loginOn, toast } = useApp()
  const key = `cameras:autoroi:${camId}`
  const q = useQuery(key, () => AutoRoiApi.get(camId))
  const [frames, setFrames] = useState(5)
  const [busy, setBusy] = useState<string | null>(null)
  const [draft, setDraft] = useState<Point[] | null>(null)
  const [confirmClear, setConfirmClear] = useState(false)
  const [imgOk, setImgOk] = useState(true)
  const box = useRef<HTMLDivElement>(null)
  const drag = useRef<number | null>(null)
  const lock = lockedFor('admin', role, loginOn)

  const roi = q.data?.roi ?? null
  const own = q.data?.resolved_from === 'camera'
  const status = roiStatus(q.data)
  const poly = draft ?? roi?.polygon ?? null

  const run = async (label: string, fn: () => Promise<unknown>, ok: string) => {
    setBusy(label)
    try {
      await fn()
      toast({ tone: 'ok', title: ok })
      setImgOk(true)
      invalidate(key)
    } catch (e) { toast({ tone: 'crit', title: `${label} failed`, body: (e as Error).message }) } finally { setBusy(null) }
  }

  const toNorm = (e: ReactPointerEvent) => {
    const r = box.current!.getBoundingClientRect()
    return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height] as const
  }
  const onMove = (e: ReactPointerEvent) => {
    if (drag.current === null || !draft) return
    const [x, y] = toNorm(e)
    setDraft(moveVertex(draft, drag.current, x, y))
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2 text-[12px]">
        <Badge tone={status.tone} dot>{status.label}</Badge>
        {roi && own && <span className="text-fg-3">
          {roi.source === 'auto' ? `computed from ${roi.frames_used ?? '?'} frame(s) by ${roi.model ?? 'segmenter'}` : `source: ${roi.source}`}
          {roi.ground_share != null && ` · ground ${(roi.ground_share * 100).toFixed(0)}% of frame`}
          {roi.approved_at && ` · approved ${roi.approved_at.slice(0, 16).replace('T', ' ')}${roi.approved_by ? ` by ${roi.approved_by}` : ''}`}
        </span>}
      </div>

      <div ref={box} className="relative w-full select-none overflow-hidden rounded-md border border-line bg-panel-2"
        style={{ aspectRatio: roi?.frame_size ? `${roi.frame_size[0]} / ${roi.frame_size[1]}` : '16 / 9', touchAction: draft ? 'none' : undefined }}
        onPointerMove={onMove} onPointerUp={() => { drag.current = null }} onPointerLeave={() => { drag.current = null }}>
        {own && imgOk ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={AutoRoiApi.frameUrl(camId, roi?.updated_at ?? '')} alt={`Reference frame of ${camId}`} className="absolute inset-0 size-full object-fill" onError={() => setImgOk(false)} />
        ) : (
          <div className="absolute inset-0 grid place-items-center p-4 text-center text-[12px] text-fg-3">
            {own ? 'Reference frame not available.' : 'No ROI computed for this camera yet. Compute one from the camera’s frames.'}
          </div>
        )}
        {poly && (own || draft) && (
          <svg viewBox="0 0 1 1" preserveAspectRatio="none" className="pointer-events-none absolute inset-0 size-full" aria-hidden>
            <polygon points={svgPoints(poly)} fill="rgb(60 200 60 / 0.35)" stroke="rgb(40 255 40)" strokeWidth={2} vectorEffect="non-scaling-stroke" />
            {(roi?.holes ?? []).map((h, i) => (
              <polygon key={i} points={svgPoints(h)} fill="rgb(0 0 0 / 0.35)" stroke="rgb(255 60 60)" strokeWidth={2} vectorEffect="non-scaling-stroke" />
            ))}
          </svg>
        )}
        {draft?.map(([x, y], i) => (
          <button key={i} type="button" aria-label={`Vertex ${i + 1}: drag to move, right-click to remove`}
            className="absolute size-3.5 -translate-x-1/2 -translate-y-1/2 cursor-grab rounded-full border-2 border-white bg-ok shadow"
            style={{ left: `${x * 100}%`, top: `${y * 100}%` }}
            onPointerDown={(e) => { e.preventDefault(); drag.current = i; (e.target as HTMLElement).setPointerCapture?.(e.pointerId) }}
            onContextMenu={(e) => { e.preventDefault(); setDraft(removeVertex(draft, i)) }} />
        ))}
        {draft?.map(([x, y], i) => {
          const [x2, y2] = draft[(i + 1) % draft.length]
          return (
            <button key={`m${i}`} type="button" aria-label={`Add a vertex after ${i + 1}`} title="Add a vertex here"
              className="absolute grid size-3 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full border border-white/70 bg-black/40 text-white"
              style={{ left: `${((x + x2) / 2) * 100}%`, top: `${((y + y2) / 2) * 100}%` }}
              onClick={() => setDraft(insertVertex(draft, i))}><Plus className="size-2" /></button>
          )
        })}
      </div>

      {draft ? (
        <div className="flex flex-wrap items-center gap-2">
          <span className="mr-auto text-[11px] text-fg-3">Drag the white points; “+” adds a point; right-click a point to remove it.</span>
          <Button size="sm" variant="ghost" icon={X} onClick={() => setDraft(null)}>Cancel</Button>
          <Button size="sm" variant="primary" icon={Check} disabled={!polygonValid(draft)} loading={busy === 'Save'}
            onClick={() => run('Save', async () => { await AutoRoiApi.edit(camId, draft, roi?.holes ?? []); setDraft(null) }, 'ROI saved and approved')}>Save ROI</Button>
        </div>
      ) : (
        <div className="flex flex-wrap items-center gap-2">
          <label className="flex items-center gap-1.5 text-[12px] text-fg-3">Frames
            <Select value={frames} onChange={(e) => setFrames(Number(e.target.value))} className="w-16" aria-label="Frames for the median">
              {[1, 3, 5, 9].map((n) => <option key={n} value={n}>{n}</option>)}
            </Select>
          </label>
          <Button size="sm" icon={ScanSearch} disabled={!isAdmin || !!busy} title={lock ?? 'Median of the frames → floor/road/sidewalk segmentation → polygon. Takes 10–30 s on CPU.'}
            loading={busy === 'Compute'} onClick={() => run('Compute', () => AutoRoiApi.compute(camId, frames), 'Auto ROI computed: review it')}>{own ? 'Recompute' : 'Compute auto ROI'}</Button>
          <Button size="sm" variant="primary" icon={Check} disabled={!isAdmin || !own || !roi?.needs_review || !!busy} title={lock}
            loading={busy === 'Approve'} onClick={() => run('Approve', () => AutoRoiApi.approve(camId), 'ROI approved')}>Approve</Button>
          <Button size="sm" variant="ghost" icon={Pencil} disabled={!isAdmin || !own || !!busy} title={lock}
            onClick={() => roi && setDraft(roi.polygon.map((p) => [...p] as Point))}>Edit</Button>
          <Button size="sm" variant="ghost" icon={Eraser} className="ml-auto text-crit" disabled={!isAdmin || !own || !!busy} title={lock}
            onClick={() => setConfirmClear(true)}>Clear</Button>
        </div>
      )}
      {q.data?.note && <p className="text-[11px] text-fg-3">{q.data.note}</p>}
      <p className="text-[11px] text-fg-3">Precedence: an ROI sent with a request › this camera’s ROI › the global ROI › full frame. Shadows, puddles and cartons on the ground stay inside the ROI; those false positives are handled by negative mining, not by the ROI.</p>

      <ConfirmDialog open={confirmClear} onClose={() => setConfirmClear(false)} danger confirmLabel="Clear ROI" title={`Clear the ROI of ${camId}?`}
        body={<>The saved polygon and its reference frame are removed. The camera falls back to the global ROI, or the full frame.</>}
        onConfirm={async () => { setConfirmClear(false); await run('Clear', () => AutoRoiApi.clear(camId), 'ROI cleared') }} />
    </div>
  )
}
