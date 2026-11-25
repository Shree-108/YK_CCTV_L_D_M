'use client'

// Keyboard review of mined detections (phase 2 item 6). One crop at a time (a "group" = the same box in neighbouring
// frames, labelled with one key), autosaved on every key. Keys: 1 TP · 2 FP_shadow · 3 FP_carton · 4 FP_stock ·
// 5 FP_puddle · 6 FP_other · S / Space skip · Z / Backspace undo · ← → previous / next · D draw a missed box.
// Crops are at native pixel scale (the inference tile scale); the full frame shows where the box is.
import { useCallback, useEffect, useMemo, useRef, useState, type PointerEvent as RPointerEvent } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { ArrowLeft, ArrowRight, PenSquare, Tags, Trash2, Undo2 } from 'lucide-react'
import { LabelingApi, REVIEW_LABELS, type LabelItem, type LabelQueue, type ReviewLabel } from '../../api/labeling'
import { invalidate, useQuery } from '../../hooks/useQuery'
import { cx } from '../../lib/format'
import { lockedFor, useApp } from '../../stores/app'
import { Badge, Button, PageHeader, Panel, Select } from '../../components/ui/primitives'
import { EmptyState, ErrorState } from '../../components/ui/states'

const KEYS: Record<string, ReviewLabel> = { '1': 'TP', '2': 'FP_shadow', '3': 'FP_carton', '4': 'FP_stock', '5': 'FP_puddle', '6': 'FP_other' }
const LABEL_TONE: Record<string, 'ok' | 'warn' | 'neutral' | 'info'> = { TP: 'ok', FP_shadow: 'warn', FP_carton: 'warn', FP_stock: 'warn', FP_puddle: 'warn', FP_other: 'neutral' }
type Undo = { target: string; prev: string; index: number }

export default function LabelingPage() {
  const params = useSearchParams()
  const router = useRouter()
  const { toast, canOperate, role, loginOn } = useApp()
  const lock = lockedFor('operator', role, loginOn)
  const sessions = useQuery('labeling:videos', LabelingApi.videos)
  const video = params.get('video') ?? sessions.data?.videos[0]?.video_id ?? null
  const q = useQuery(video ? `labeling:queue:${video}` : null, () => LabelingApi.queue(video!))
  const [items, setItems] = useState<LabelItem[]>([])
  const [i, setI] = useState(0)
  const [undo, setUndo] = useState<Undo[]>([])
  const [saving, setSaving] = useState<string | null>(null)
  const [draw, setDraw] = useState(false)
  const [drawClass, setDrawClass] = useState<string>('')
  const [missed, setMissed] = useState<LabelQueue['missed']>([])

  useEffect(() => {
    if (!q.data) return
    // unlabelled first, then the rest, each in mining order; the position is kept across refreshes
    const un = q.data.items.filter((x) => !x.label), done = q.data.items.filter((x) => x.label)
    setItems([...un, ...done])
    setMissed(q.data.missed)
    setI(0)
    setUndo([])
    setDrawClass((c) => c || q.data!.manifest.class_names?.[0] || '')
  }, [q.data])

  const cur = items[i]
  const classes = q.data?.manifest.class_names ?? []
  const labelled = items.filter((x) => x.label).length
  const counts = useMemo(() => items.reduce<Record<string, number>>((m, x) => (x.label ? { ...m, [x.label]: (m[x.label] ?? 0) + 1 } : m), {}), [items])

  const save = useCallback(async (label: ReviewLabel | '', index = i, trueClass?: string) => {
    const it = items[index]
    if (!video || !it || !canOperate) return
    setSaving(it.group_id)
    try {
      await LabelingApi.label(video, it.group_id, label, true, trueClass)
      setUndo((u) => [...u.slice(-199), { target: it.group_id, prev: it.label, index }])
      setItems((xs) => xs.map((x, k) => (k === index ? { ...x, label } : x)))
      if (label && index === i) setI((n) => Math.min(items.length - 1, n + 1))
    } catch (e) {
      toast({ tone: 'crit', title: 'Not saved', body: (e as Error).message })
    } finally { setSaving(null) }
  }, [items, i, video, canOperate, toast])

  const doUndo = useCallback(async () => {
    const last = undo[undo.length - 1]
    if (!last || !video) return
    try {
      await LabelingApi.label(video, last.target, last.prev as ReviewLabel | '', true)
      setItems((xs) => xs.map((x, k) => (k === last.index ? { ...x, label: last.prev } : x)))
      setUndo((u) => u.slice(0, -1))
      setI(last.index)
    } catch (e) { toast({ tone: 'crit', title: 'Undo failed', body: (e as Error).message }) }
  }, [undo, video, toast])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement
      if (t && (t.tagName === 'INPUT' || t.tagName === 'SELECT' || t.tagName === 'TEXTAREA')) return
      if (draw) { if (e.key === 'Escape') setDraw(false); return }
      if (KEYS[e.key]) { e.preventDefault(); void save(KEYS[e.key]) }
      else if (e.key === 's' || e.key === 'S' || e.key === ' ') { e.preventDefault(); setI((n) => Math.min(items.length - 1, n + 1)) }
      else if (e.key === 'z' || e.key === 'Z' || e.key === 'Backspace') { e.preventDefault(); void doUndo() }
      else if (e.key === 'ArrowRight') setI((n) => Math.min(items.length - 1, n + 1))
      else if (e.key === 'ArrowLeft') setI((n) => Math.max(0, n - 1))
      else if (e.key === 'd' || e.key === 'D') setDraw(true)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [save, doUndo, items.length, draw])

  const addMissed = async (bbox: number[]) => {
    if (!video || !cur || !drawClass) return
    try {
      const m = await LabelingApi.addMissed(video, cur.frame_file, bbox, drawClass)
      setMissed((xs) => [...xs, m])
      toast({ tone: 'ok', title: `Missed ${drawClass} saved` })
    } catch (e) { toast({ tone: 'crit', title: 'Not saved', body: (e as Error).message }) }
  }
  const removeMissed = async (id: string) => {
    if (!video) return
    try { await LabelingApi.deleteMissed(video, id); setMissed((xs) => xs.filter((m) => m.missed_id !== id)) } catch (e) {
      toast({ tone: 'crit', title: 'Not deleted', body: (e as Error).message })
    }
  }

  const size = q.data?.manifest.frame_size ?? [2560, 1440]
  const frameMissed = missed.filter((m) => cur && m.frame_file === cur.frame_file)
  return (
    <div className="flex flex-col gap-4">
      <PageHeader title="Label mined detections"
        description="One crop at a time, autosaved. A crop stands for the same box in neighbouring frames (×N); one key labels them all. Crops are at native pixel scale, the scale the detector's tiles see. Press D to draw real litter the model missed."
        actions={<>
          <Select value={video ?? ''} onChange={(e) => router.replace(`/labeling?video=${encodeURIComponent(e.target.value)}`)} aria-label="Mined video" className="w-72">
            {(sessions.data?.videos ?? []).map((s) => <option key={s.video_id} value={s.video_id}>{s.video_file ?? s.video_id} · {s.labelled}/{s.offered}</option>)}
          </Select>
          <Button size="sm" variant="ghost" onClick={() => { invalidate('labeling'); }}>Refresh</Button>
        </>} />
      {sessions.data && !sessions.data.videos.length && (
        <Panel><EmptyState icon={Tags} title="Nothing mined yet" body="Run: python -m training.negative_mining.mine_tiles storage/media/<video>.mp4 --camera <CAM> --conf 0.10" /></Panel>
      )}
      {q.error && <Panel><ErrorState error={q.error} onRetry={q.refetch} compact /></Panel>}
      {q.data && (
        <>
          <div className="rounded-xl border border-line bg-panel px-4 py-3">
            <div className="mb-2 flex flex-wrap items-center gap-2 text-[12px]">
              <span className="font-medium text-fg">{labelled} / {items.length} labelled</span>
              <span className="text-fg-3">({items.reduce((n, x) => n + x.members, 0)} boxes · mined at conf ≥ {q.data.manifest.mining_conf ?? '?'} · cap {q.data.manifest.cap_per_class ?? '?'} crops/class)</span>
              {REVIEW_LABELS.map((l) => counts[l] ? <Badge key={l} tone={LABEL_TONE[l]}>{l} {counts[l]}</Badge> : null)}
              {missed.length > 0 && <Badge tone="info">missed {missed.length}</Badge>}
              {saving && <span className="ml-auto text-fg-3">saving…</span>}
            </div>
            <div className="h-1.5 overflow-hidden rounded bg-panel-2"><div className="h-full bg-ok transition-all" style={{ width: `${items.length ? (100 * labelled) / items.length : 0}%` }} /></div>
          </div>
          {!cur ? <Panel><EmptyState icon={Tags} title="No crops in this video" /></Panel> : (
            <div className="grid gap-4 xl:grid-cols-[440px_minmax(0,1fr)]">
              <Panel title={`Crop ${i + 1} of ${items.length}`} icon={Tags}
                subtitle={`${cur.class} · conf ${cur.conf.toFixed(2)}${cur.members > 1 ? ` (min ${cur.min_conf.toFixed(2)}) · ×${cur.members} frames` : ''} · t=${cur.timestamp_sec.toFixed(1)} s`}>
                <div className="flex flex-col gap-3">
                  <Crop video={video!} item={cur} />
                  <div className="flex flex-wrap items-center gap-1.5 text-[12px]">
                    {cur.below_030 && <Badge tone="warn" title="Below the app's 0.30 weak threshold: only seen by mining">below 0.30</Badge>}
                    {cur.status && <Badge tone="neutral">{cur.status}</Badge>}
                    {cur.label ? <Badge tone={LABEL_TONE[cur.label] ?? 'neutral'} dot>{cur.label}</Badge> : <Badge tone="neutral">unlabelled</Badge>}
                  </div>
                  <div className="grid grid-cols-3 gap-1.5">
                    {REVIEW_LABELS.map((l, k) => (
                      <Button key={l} size="sm" variant={cur.label === l ? 'primary' : 'ghost'} disabled={!canOperate} title={lock}
                        onClick={() => void save(l)}><span className="mr-1 font-mono text-fg-3">{k + 1}</span>{l}</Button>
                    ))}
                  </div>
                  {cur.label === 'TP' && classes.length > 0 && (
                    <label className="flex items-center gap-2 text-[12px] text-fg-3">True class
                      <Select className="w-48" defaultValue="" onChange={(e) => void save('TP', i, e.target.value)} aria-label="True class of this TP">
                        <option value="">{cur.class} (as detected)</option>
                        {classes.filter((c) => c !== cur.class).map((c) => <option key={c} value={c}>{c}</option>)}
                      </Select>
                    </label>
                  )}
                  <div className="flex flex-wrap items-center gap-1.5">
                    <Button size="sm" variant="ghost" icon={ArrowLeft} onClick={() => setI((n) => Math.max(0, n - 1))}>Prev</Button>
                    <Button size="sm" variant="ghost" icon={ArrowRight} onClick={() => setI((n) => Math.min(items.length - 1, n + 1))}>Skip</Button>
                    <Button size="sm" variant="ghost" icon={Undo2} disabled={!undo.length} onClick={() => void doUndo()}>Undo</Button>
                    <Button size="sm" variant={draw ? 'primary' : 'ghost'} icon={PenSquare} disabled={!canOperate} title={lock} onClick={() => setDraw((d) => !d)}>Draw missed box</Button>
                  </div>
                  <p className="text-[11px] text-fg-3">Keys: 1–6 label · S/Space skip · Z/Backspace undo · ←/→ move · D draw (Esc ends). Every key is saved at once.</p>
                </div>
              </Panel>
              <Panel title={draw ? 'Draw a box around real litter the model missed' : 'Full frame'} icon={PenSquare}
                subtitle={`${cur.frame_file} · frame ${cur.frame_idx}`}
                actions={draw ? <>
                  <Select value={drawClass} onChange={(e) => setDrawClass(e.target.value)} aria-label="Class of the drawn box" className="w-48">
                    {classes.map((c) => <option key={c} value={c}>{c}</option>)}
                  </Select>
                  <Button size="sm" onClick={() => setDraw(false)}>Done</Button>
                </> : undefined}>
                <FrameView video={video!} item={cur} size={size} draw={draw} missed={frameMissed} onBox={addMissed} />
                {frameMissed.length > 0 && (
                  <ul className="mt-2 flex flex-wrap gap-1.5 text-[12px]">
                    {frameMissed.map((m) => (
                      <li key={m.missed_id} className="flex items-center gap-1 rounded border border-line bg-panel-2 px-2 py-0.5">
                        <span className="text-fg">missed {m.class}</span>
                        <button type="button" aria-label={`Delete missed ${m.class}`} className="text-crit" onClick={() => void removeMissed(m.missed_id)}><Trash2 className="size-3" /></button>
                      </li>
                    ))}
                  </ul>
                )}
              </Panel>
            </div>
          )}
        </>
      )}
    </div>
  )
}

/** The crop (native pixels) with the box outlined. */
function Crop({ video, item }: { video: string; item: LabelItem }) {
  const cb = item.crop_box
  const path = `crops/${item.det_id}.jpg`
  return (
    <div className="relative mx-auto aspect-square w-full max-w-[400px] overflow-hidden rounded-md border border-line bg-video">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={LabelingApi.image(video, path)} alt={`Crop of ${item.class}`} className="absolute inset-0 size-full object-contain [image-rendering:pixelated]" />
      {cb && (
        <svg viewBox={`0 0 ${cb[2]} ${cb[3]}`} preserveAspectRatio="xMidYMid meet" className="pointer-events-none absolute inset-0 size-full" aria-hidden>
          <rect x={item.bbox[0] - cb[0]} y={item.bbox[1] - cb[1]} width={item.bbox[2] - item.bbox[0]} height={item.bbox[3] - item.bbox[1]}
            fill="none" stroke="#f5c542" strokeWidth={2} vectorEffect="non-scaling-stroke" strokeDasharray="4 3" />
        </svg>
      )}
    </div>
  )
}

/** Full frame with the current box; in draw mode, drag to draw a box (frame pixels). */
function FrameView({ video, item, size, draw, missed, onBox }: {
  video: string; item: LabelItem; size: [number, number]; draw: boolean; missed: LabelQueue['missed']; onBox: (b: number[]) => void
}) {
  const svg = useRef<SVGSVGElement>(null)
  const [start, setStart] = useState<[number, number] | null>(null)
  const [end, setEnd] = useState<[number, number] | null>(null)
  const [w, h] = size
  const pt = (e: RPointerEvent): [number, number] => {
    const s = svg.current!
    const p = s.createSVGPoint()
    p.x = e.clientX
    p.y = e.clientY
    const q = p.matrixTransform(s.getScreenCTM()!.inverse())
    return [Math.min(w, Math.max(0, q.x)), Math.min(h, Math.max(0, q.y))]
  }
  const rect = start && end ? [Math.min(start[0], end[0]), Math.min(start[1], end[1]), Math.max(start[0], end[0]), Math.max(start[1], end[1])] : null
  return (
    <div className="relative w-full overflow-hidden rounded-md border border-line bg-video" style={{ aspectRatio: `${w} / ${h}` }}>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={LabelingApi.image(video, item.frame_file)} alt={`Frame ${item.frame_idx}`} className="absolute inset-0 size-full object-fill" draggable={false} />
      <svg ref={svg} viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" className={cx('absolute inset-0 size-full', draw ? 'cursor-crosshair' : 'pointer-events-none')}
        onPointerDown={(e) => { if (!draw) return; e.currentTarget.setPointerCapture(e.pointerId); const p = pt(e); setStart(p); setEnd(p) }}
        onPointerMove={(e) => { if (draw && start) setEnd(pt(e)) }}
        onPointerUp={() => { if (draw && rect && rect[2] - rect[0] >= 3 && rect[3] - rect[1] >= 3) onBox(rect.map((v) => Math.round(v))); setStart(null); setEnd(null) }}>
        <rect x={item.bbox[0]} y={item.bbox[1]} width={item.bbox[2] - item.bbox[0]} height={item.bbox[3] - item.bbox[1]} fill="none" stroke="#f5c542" strokeWidth={3} vectorEffect="non-scaling-stroke" />
        {missed.map((m) => <rect key={m.missed_id} x={+m.x1} y={+m.y1} width={+m.x2 - +m.x1} height={+m.y2 - +m.y1} fill="none" stroke="rgb(63 185 80)" strokeWidth={3} vectorEffect="non-scaling-stroke" />)}
        {rect && <rect x={rect[0]} y={rect[1]} width={rect[2] - rect[0]} height={rect[3] - rect[1]} fill="rgb(63 185 80 / 0.15)" stroke="rgb(63 185 80)" strokeWidth={2} strokeDasharray="5 3" vectorEffect="non-scaling-stroke" />}
      </svg>
    </div>
  )
}
