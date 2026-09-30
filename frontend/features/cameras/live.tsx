'use client'

import Link from 'next/link'
import { useEffect, useMemo, useRef, useState } from 'react'
import { CameraOff, Download, Expand, History, Loader2, Pause, Pencil, Play, RefreshCw, Settings2, Sparkles } from 'lucide-react'
import { ApiError } from '../../api/client'
import { Api } from '../../api/endpoints'
import type { Camera, Detection, LiveFrame, RoiZone } from '../../api/types'
import { invalidate, useMutation, useQuery } from '../../hooks/useQuery'
import { cx, dayStartIso, downloadBlob, num, todayLocal } from '../../lib/format'
import { useApp } from '../../stores/app'
import { Button, Dot, KeyValue, Panel, StatusBadge, Unavailable } from '../../components/ui/primitives'
import { Drawer } from '../../components/ui/overlay'
import { ImageOverlay, STATUS_COLOR, type OverlayBox } from '../../components/vision/Overlay'

export interface OverlayOpts { detections: boolean; ids: boolean; confidence: boolean; roi: boolean; weak: boolean }
export const DEFAULT_OVERLAY: OverlayOpts = { detections: true, ids: true, confidence: true, roi: false, weak: false }

/** Polls the latest processed frame (raw image + detections of that same frame). */
export function useLiveFrame(camId: string | null, paused: boolean, intervalMs = 400) {
  const [frame, setFrame] = useState<LiveFrame | null>(null)
  const [image, setImage] = useState<string | null>(null)
  const [error, setError] = useState<Error | null>(null)
  const seq = useRef(0)
  useEffect(() => {
    seq.current = 0
    setFrame(null)
    setImage(null)
    setError(null)
  }, [camId])
  useEffect(() => {
    if (!camId || paused) return
    let stop = false
    let timer: ReturnType<typeof setTimeout>
    const ctrl = new AbortController()
    const tick = async () => {
      if (document.visibilityState === 'visible') {
        try {
          const r = await Api.live(camId, seq.current, ctrl.signal)
          if (stop) return
          setError(null)
          if (r.frame && r.image) {
            seq.current = r.frame.seq
            setImage(`data:image/jpeg;base64,${r.image}`)
            setFrame(r)
          } else setFrame((f) => (f ? { ...f, state: r.state, fps: r.fps, inference_ms: r.inference_ms, last_error: r.last_error, roi: r.roi } : r))
        } catch (e) {
          if ((e as Error).name !== 'AbortError' && !stop) setError(e as Error)
        }
      }
      if (!stop) timer = setTimeout(tick, intervalMs)
    }
    void tick()
    return () => { stop = true; ctrl.abort(); clearTimeout(timer) }
  }, [camId, paused, intervalMs])
  return { frame, image, error }
}

export function toBoxes(dets: Detection[], o: OverlayOpts): OverlayBox[] {
  if (!o.detections) return []
  return dets
    .filter((d) => d.status === 'confirmed' || d.status === 'candidate' || (o.weak && d.status === 'weak'))
    .map((d, i) => ({
      key: `${d.track_id ?? 'x'}-${i}`,
      bbox: d.bbox,
      color: STATUS_COLOR[d.status],
      thin: d.status === 'weak',
      dashed: d.status === 'candidate',
      label: d.status === 'weak' ? undefined : d.class.toUpperCase(),
      sub: [o.confidence ? `${Math.round(d.confidence * 100)}%` : null, o.ids && d.track_id != null ? `ID:${String(d.track_id).padStart(3, '0')}` : null].filter(Boolean).join('  ') || undefined,
    }))
}

/** Renders the current frame + overlay to a PNG for download. */
export async function snapshotPng(image: string, width: number, height: number, boxes: OverlayBox[], roi: RoiZone[] | null, name: string) {
  const img = new Image()
  img.src = image
  await img.decode()
  const c = document.createElement('canvas')
  c.width = width
  c.height = height
  const g = c.getContext('2d')!
  g.drawImage(img, 0, 0, width, height)
  const css = getComputedStyle(document.documentElement)
  const col = (v: string) => (v.startsWith('var(') ? css.getPropertyValue(v.slice(4, -1)).trim() : v)
  g.lineWidth = Math.max(2, width / 600)
  roi?.forEach((z) => {
    g.strokeStyle = z.mode === 'include' ? col('var(--info)') : col('var(--crit)')
    g.setLineDash([8, 5])
    g.beginPath()
    z.polygon.forEach(([x, y], i) => (i ? g.lineTo(x * width, y * height) : g.moveTo(x * width, y * height)))
    g.closePath()
    g.stroke()
  })
  g.setLineDash([])
  g.font = `600 ${Math.max(12, width / 70)}px monospace`
  boxes.forEach((b) => {
    g.strokeStyle = col(b.color)
    g.strokeRect(b.bbox[0], b.bbox[1], b.bbox[2] - b.bbox[0], b.bbox[3] - b.bbox[1])
    if (b.label) {
      const t = b.sub ? `${b.label}  ${b.sub}` : b.label
      const tw = g.measureText(t).width + 8, th = Math.max(12, width / 70) + 6
      g.fillStyle = col(b.color)
      g.fillRect(b.bbox[0], Math.max(0, b.bbox[1] - th), tw, th)
      g.fillStyle = '#0a0c0f'
      g.fillText(t, b.bbox[0] + 4, Math.max(th - 4, b.bbox[1] - 4))
    }
  })
  c.toBlob((blob) => blob && downloadBlob(blob, name), 'image/png')
}

// ---------------------------------------------------------------- camera tile
export function CameraTile({ cam, overlay, paused, onOpen, compact, heightClass = 'aspect-video' }: {
  cam: Camera; overlay: OverlayOpts; paused?: boolean; onOpen?: () => void; compact?: boolean; heightClass?: string
}) {
  const [localPause, setLocalPause] = useState(false)
  const running = cam.enabled && cam.status.state !== 'disabled' && cam.status.state !== 'stopped'
  const isPaused = !!paused || localPause
  const { frame, image, error } = useLiveFrame(running ? cam.id : null, isPaused)
  const tileRef = useRef<HTMLDivElement>(null)
  const f = frame?.frame
  const state = frame?.state ?? cam.status.state
  const boxes = useMemo(() => (f ? toBoxes(f.detections, overlay) : []), [f, overlay])
  const confirmed = f?.summary.confirmed_litter ?? 0
  const offline = state === 'reconnecting' || state === 'error'

  return (
    <div ref={tileRef} className={cx('group relative overflow-hidden rounded-md border bg-video', offline ? 'border-crit/50' : 'border-line', heightClass)}>
      {f && image ? (
        <ImageOverlay src={image} width={f.width} height={f.height} boxes={boxes} roi={overlay.roi ? frame?.roi : null}
          className="absolute inset-0" alt={`Live frame from ${cam.name}`} />
      ) : (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-center text-[12px] text-fg-3">
          {!cam.enabled ? <><CameraOff className="size-5" aria-hidden />Camera disabled</> :
            offline ? <><CameraOff className="size-5 text-crit" aria-hidden /><span className="text-fg-2">Unable to connect to {cam.id}</span><span className="max-w-[80%] truncate">{frame?.last_error ?? cam.status.last_error ?? 'Reconnecting with backoff…'}</span></> :
            error ? <><CameraOff className="size-5" aria-hidden />{error instanceof ApiError && error.status === 404 ? 'Stream not running' : error.message}</> :
            <><Loader2 className="size-5 animate-spin" aria-hidden />Connecting to stream…</>}
        </div>
      )}
      {/* top overlay */}
      <div className="pointer-events-none absolute inset-x-0 top-0 flex items-start justify-between gap-2 bg-gradient-to-b from-black/70 to-transparent p-2 text-white">
        <div className="min-w-0">
          <div className="flex items-center gap-1.5 text-[11px] font-semibold">
            {state === 'online' && !isPaused ? <><span className="live-dot size-1.5 rounded-full bg-[#f0524a]" aria-hidden />LIVE</> :
              isPaused ? <><Pause className="size-3" aria-hidden />PAUSED</> : <span className="uppercase text-[#d29922]">{state}</span>}
            <span className="font-mono text-white/90">{cam.id}</span>
          </div>
          {!compact && <div className="truncate text-[11px] text-white/70">{cam.name}{cam.location ? ` · ${cam.location}` : ''}</div>}
        </div>
        {f && (
          <div className="shrink-0 text-right font-mono text-[10.5px] leading-4 text-white/85">
            <div>AI {f.model_version}</div>
            <div>FPS {num(frame?.fps, 1)} · {num(f.inference_ms, 0)} ms</div>
          </div>
        )}
      </div>
      {/* bottom overlay */}
      {f && (
        <div className="pointer-events-none absolute inset-x-0 bottom-0 flex items-end justify-between bg-gradient-to-t from-black/70 to-transparent p-2 text-[11px] text-white">
          <span className={cx('rounded-sm px-1.5 py-px font-semibold', confirmed > 0 ? 'bg-[#f0524a]' : 'bg-white/15')}>CONFIRMED LITTER {confirmed}</span>
          <span className="font-mono text-white/75">raw {f.summary.raw_detections ?? 0} · tracks {f.summary.active_tracks ?? 0}</span>
        </div>
      )}
      {/* controls */}
      <div className="absolute right-2 top-10 flex flex-col gap-1 opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100">
        <TileBtn label={isPaused ? 'Resume' : 'Pause'} onClick={() => setLocalPause((p) => !p)}>{isPaused ? <Play className="size-3.5" /> : <Pause className="size-3.5" />}</TileBtn>
        <TileBtn label="Fullscreen" onClick={() => tileRef.current?.requestFullscreen?.()}><Expand className="size-3.5" /></TileBtn>
        <TileBtn label="Save snapshot" disabled={!f || !image}
          onClick={() => f && image && snapshotPng(image, f.width, f.height, boxes, overlay.roi ? frame?.roi ?? null : null, `${cam.id}_${f.seq}.png`)}>
          <Download className="size-3.5" />
        </TileBtn>
        {onOpen && <TileBtn label="Camera details" onClick={onOpen}><Settings2 className="size-3.5" /></TileBtn>}
      </div>
    </div>
  )
}

function TileBtn({ label, onClick, children, disabled }: { label: string; onClick: () => void; children: React.ReactNode; disabled?: boolean }) {
  return (
    <button aria-label={label} title={label} onClick={onClick} disabled={disabled}
      className="flex size-7 items-center justify-center rounded-md bg-black/60 text-white hover:bg-black/80 disabled:opacity-40">{children}</button>
  )
}

// ---------------------------------------------------------------- camera detail drawer
export function CameraDrawer({ cam, onClose }: { cam: Camera | null; onClose: () => void }) {
  const { isAdmin, toast } = useApp()
  const today = todayLocal()
  const stats = useQuery(cam ? `analytics:cam:${cam.id}:${today}` : null, () => Api.analytics({ camera_id: cam!.id, start: dayStartIso(today) }))
  const settings = useQuery('settings', Api.settings)
  const restart = useMutation(Api.restartCamera)
  const [thr, setThr] = useState<number | null>(null)
  const save = useMutation((id: string, v: number) => Api.updateCamera(id, { settings: { ...(cam?.settings ?? {}), confidence: { ...((cam?.settings as { confidence?: object })?.confidence ?? {}), confirmed_threshold: v } } }))
  useEffect(() => setThr(null), [cam?.id])
  if (!cam) return null
  const s = cam.status
  const camConf = (cam.settings as { confidence?: { confirmed_threshold?: number } })?.confidence?.confirmed_threshold
  const effective = camConf ?? settings.data?.settings.confidence.confirmed_threshold
  const detToday = stats.data?.detections_by_status.reduce((a, r) => a + r.n, 0)

  return (
    <Drawer open={!!cam} onClose={onClose} title={cam.name} width="w-[560px]"
      subtitle={<span className="flex items-center gap-2"><span className="font-mono">{cam.id}</span><StatusBadge status={s.state} /></span>}
      footer={<>
        <Link href={`/history?camera_id=${encodeURIComponent(cam.id)}`}><Button icon={History} variant="ghost">View history</Button></Link>
        <Link href={`/cameras?id=${encodeURIComponent(cam.id)}`}><Button icon={Pencil}>Edit camera</Button></Link>
        <Link href={`/live?focus=${encodeURIComponent(cam.id)}`}><Button icon={Expand} variant="primary">Open full screen</Button></Link>
      </>}>
      <div className="flex flex-col gap-4">
        <CameraTile cam={cam} overlay={{ ...DEFAULT_OVERLAY, roi: true }} compact />
        <KeyValue items={[
          ['Location', cam.location || <span className="text-fg-3">Location unavailable</span>],
          ['Stream source', <span key="s" className="font-mono text-[12px]" title={cam.source}>{cam.source}</span>],
          ['RTSP / stream status', s.state === 'online' ? `connected${s.resolution ? ` · ${s.resolution[0]}×${s.resolution[1]}` : ''}` : (s.last_error ?? s.state)],
          ['Reconnects', num(s.reconnects)],
          ['Processing FPS', num(s.fps, 1)],
          ['Inference latency', s.inference_ms ? `${num(s.inference_ms, 0)} ms` : '—'],
          ['Active tracks', num(s.summary?.active_tracks)],
          ['Confirmed litter (now)', num(s.summary?.confirmed_litter)],
          ["Today's detections", stats.loading ? '…' : num(detToday)],
          ["Today's alerts", stats.loading ? '…' : num(stats.data?.alerts_total)],
          ['Frames dropped', num(s.frames_dropped)],
          ['ROI zones', cam.roi.length ? cam.roi.map((z) => `${z.name} (${z.mode})`).join(', ') : 'none (whole frame)'],
        ]} />
        <Panel title="AI model" icon={Sparkles}>
          <div className="flex items-center justify-between gap-2 text-[13px]">
            <span className="font-mono text-fg">{s.state === 'online' ? 'shared active model' : '—'}</span>
            <Link href="/model-lab" className="text-[12px] text-info hover:underline">Change active model →</Link>
          </div>
          <p className="mt-1 text-[11px] text-fg-3">All cameras use the globally active model. Per-camera model selection is not supported by the backend.</p>
        </Panel>
        <Panel title="Confirmation threshold">
          <div className="flex items-center gap-3">
            <input type="range" min={0.2} max={0.95} step={0.05} aria-label="Confirmation threshold" disabled={!isAdmin}
              value={thr ?? effective ?? 0.6} onChange={(e) => setThr(Number(e.target.value))} className="flex-1" />
            <span className="tabular w-12 text-right font-mono">{(thr ?? effective ?? 0).toFixed(2)}</span>
            <Button size="sm" variant="primary" disabled={!isAdmin || thr === null} loading={save.pending}
              onClick={async () => {
                const r = await save.run(cam.id, thr!)
                if (r) { toast({ tone: 'ok', title: 'Threshold saved', body: `${cam.id}: ${thr!.toFixed(2)}` }); invalidate('cameras'); setThr(null) }
                else toast({ tone: 'crit', title: 'Could not save threshold', body: save.error?.message })
              }}>Save</Button>
          </div>
          <p className="mt-1 text-[11px] text-fg-3">{camConf !== undefined ? 'Camera override active.' : 'Using the global threshold.'} {!isAdmin && 'Sign in as admin (Settings → Security) to change it.'}</p>
        </Panel>
        <div className="flex flex-wrap gap-2">
          <Button icon={RefreshCw} loading={restart.pending} disabled={!isAdmin || !cam.enabled}
            title={!isAdmin ? 'Admin sign-in required' : undefined}
            onClick={async () => {
              const r = await restart.run(cam.id)
              toast(r ? { tone: 'ok', title: `Stream ${cam.id} restarted` } : { tone: 'crit', title: 'Restart failed', body: restart.error?.message })
              invalidate('cameras')
            }}>Restart stream</Button>
          {!isAdmin && <Unavailable>admin actions locked</Unavailable>}
        </div>
      </div>
    </Drawer>
  )
}

export function camTone(state: string) {
  return state === 'online' ? 'ok' : state === 'reconnecting' || state === 'error' ? 'crit' : state === 'connecting' ? 'warn' : 'neutral'
}

export function CameraHealthList({ cameras, onOpen }: { cameras: Camera[]; onOpen?: (c: Camera) => void }) {
  return (
    <ul className="flex flex-col divide-y divide-line">
      {cameras.map((c) => (
        <li key={c.id}>
          <button onClick={() => onOpen?.(c)} className="flex w-full items-center gap-2.5 px-1 py-2 text-left hover:bg-hover/50">
            <Dot tone={camTone(c.status.state)} pulse={c.status.state === 'online'} label={c.status.state} />
            <div className="min-w-0 flex-1">
              <div className="truncate text-[13px] text-fg">{c.name} <span className="font-mono text-[11px] text-fg-3">{c.id}</span></div>
              <div className="truncate text-[11px] text-fg-3">{c.status.state === 'online' ? `${num(c.status.fps, 1)} fps · ${num(c.status.inference_ms, 0)} ms` : c.status.last_error ?? c.status.state}</div>
            </div>
            <StatusBadge status={c.status.state} />
          </button>
        </li>
      ))}
    </ul>
  )
}
