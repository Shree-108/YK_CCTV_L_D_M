'use client'

import { useCallback, useRef, useState, type ReactNode } from 'react'
import { Check, MousePointerClick, PenLine, RotateCcw, Undo2, UploadCloud, X } from 'lucide-react'
import { Api } from '../../api/endpoints'
import type { RoiZone } from '../../api/types'
import { invalidate, useQuery } from '../../hooks/useQuery'
import { cx } from '../../lib/format'
import { useApp } from '../../stores/app'
import { Badge, Button, Field, Select, Slider, Toggle } from '../../components/ui/primitives'
import { AspectBox, ImageOverlay, type OverlayBox } from '../../components/vision/Overlay'

export function DropZone({ accept, onFile, label, hint, compact }: { accept: string; onFile: (f: File) => void; label: string; hint: ReactNode; compact?: boolean }) {
  const [over, setOver] = useState(false)
  const input = useRef<HTMLInputElement>(null)
  const pick = useCallback((f?: File | null) => f && onFile(f), [onFile])
  return (
    <div
      role="button" tabIndex={0} aria-label={label}
      onClick={() => input.current?.click()}
      onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && input.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setOver(true) }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); pick(e.dataTransfer.files?.[0]) }}
      className={cx('flex cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border border-dashed text-center transition-colors',
        compact ? 'p-5' : 'min-h-[300px] p-10', over ? 'border-info bg-info-bg' : 'border-line-strong bg-panel hover:border-info/60')}>
      <UploadCloud className="size-6 text-fg-3" aria-hidden />
      <p className="text-[13px] font-medium text-fg">{label}</p>
      <p className="text-[12px] text-fg-3">{hint}</p>
      <input ref={input} type="file" accept={accept} className="hidden" onChange={(e) => { pick(e.target.files?.[0]); e.target.value = '' }} />
    </div>
  )
}

export interface DetectConfig {
  model_version: string
  confidence: number
  image_size: number | ''
  tiling: 'auto' | 'on' | 'off'
  camera_id: string
  use_roi: boolean
}

/** Shared detection settings form (model, threshold, size, tiling, ROI). Values default to the live settings. */
export function DetectionSettingsForm({ cfg, setCfg, children }: { cfg: DetectConfig; setCfg: (f: (c: DetectConfig) => DetectConfig) => void; children?: ReactNode }) {
  const models = useQuery('models', Api.models)
  const cams = useQuery('cameras:list', Api.cameras)
  const settings = useQuery('settings', Api.settings)
  const s = settings.data?.settings
  return (
    <div className="flex flex-col gap-3.5">
      <Field label="Model version" hint={models.data?.active ? `Active: ${models.data.active}` : undefined}>
        <Select value={cfg.model_version} onChange={(e) => setCfg((c) => ({ ...c, model_version: e.target.value }))}>
          <option value="">Active model{models.data?.active ? ` (${models.data.active})` : ''}</option>
          {models.data?.models.filter((m) => !m.active && m.file_exists).map((m) => <option key={m.version} value={m.version}>{m.version}</option>)}
        </Select>
      </Field>
      <Slider label="Confirmed threshold" value={cfg.confidence} min={0.05} max={0.95} step={0.05} onChange={(v) => setCfg((c) => ({ ...c, confidence: v }))} />
      <p className="-mt-2 text-[11px] text-fg-3">System setting: {s ? s.confidence.confirmed_threshold.toFixed(2) : '—'}. Weak/candidate thresholds are capped to this value.</p>
      <Field label="Inference image size" hint={s ? `Default ${s.inference.image_size}px` : undefined}>
        <Select value={String(cfg.image_size)} onChange={(e) => setCfg((c) => ({ ...c, image_size: e.target.value ? Number(e.target.value) : '' }))}>
          <option value="">Default</option>
          {[480, 640, 800, 960, 1280, 1600].map((v) => <option key={v} value={v}>{v}px</option>)}
        </Select>
      </Field>
      <Field label="Tiled inference" hint={s ? `Tile ${s.tiling.tile_size}px · overlap ${Math.round(s.tiling.overlap * 100)}% · auto tiles frames > ${Math.round(s.tiling.tile_size * s.tiling.auto_min_ratio)}px` : undefined}>
        <Select value={cfg.tiling} onChange={(e) => setCfg((c) => ({ ...c, tiling: e.target.value as DetectConfig['tiling'] }))}>
          <option value="auto">Auto (by frame size)</option><option value="on">Always tile</option><option value="off">Off (full frame only)</option>
        </Select>
      </Field>
      <Field label="Camera context" hint="Uses that camera's ROI and settings overrides; alerts are attributed to it.">
        <Select value={cfg.camera_id} onChange={(e) => setCfg((c) => ({ ...c, camera_id: e.target.value }))}>
          <option value="upload">None (upload)</option>
          {cams.data?.cameras.map((c) => <option key={c.id} value={c.id}>{c.id} · {c.name}{c.roi.length ? ` (${c.roi.length} ROI)` : ''}</option>)}
        </Select>
      </Field>
      <Toggle label="Apply ROI" checked={cfg.use_roi} onChange={(v) => setCfg((c) => ({ ...c, use_roi: v }))}
        description="Only litter whose bottom-centre lies inside the ROI is detected; everything outside is ignored." />
      {children}
    </div>
  )
}

export function toForm(file: File, cfg: DetectConfig, extra: Record<string, string> = {}) {
  const f = new FormData()
  f.append('file', file)
  f.append('camera_id', cfg.camera_id)
  f.append('confidence', String(cfg.confidence))
  f.append('tiling', cfg.tiling)
  if (cfg.image_size) f.append('image_size', String(cfg.image_size))
  if (cfg.model_version) f.append('model_version', cfg.model_version)
  f.append('use_roi', String(cfg.use_roi))
  for (const [k, v] of Object.entries(extra)) f.append(k, v)
  return f
}

/** ``plastic_bag`` -> ``Plastic Bag`` */
export const litterName = (c: string) => c.replace(/_/g, ' ').replace(/\b\w/g, (m) => m.toUpperCase())

/** Saved ROI for a camera/upload context; "active" = at least one include zone. */
export function useRoi(cameraId: string) {
  const q = useQuery(`roi:${cameraId}`, () => Api.roi(cameraId))
  const zones = q.data?.zones ?? []
  return { zones, active: !!q.data?.active, loading: q.loading && !q.data }
}

/** First useful frame of a local video file (as a JPEG data URL) for ROI drawing. */
export function grabVideoFrame(file: File, atSec = 0.5): Promise<{ src: string; w: number; h: number }> {
  return new Promise((resolve, reject) => {
    const v = document.createElement('video')
    const u = URL.createObjectURL(file)
    v.muted = true
    v.preload = 'auto'
    v.src = u
    const done = () => URL.revokeObjectURL(u)
    v.onloadedmetadata = () => { v.currentTime = Math.min(atSec, (v.duration || 1) / 2) }
    v.onseeked = () => {
      const c = document.createElement('canvas')
      c.width = v.videoWidth
      c.height = v.videoHeight
      c.getContext('2d')!.drawImage(v, 0, 0)
      resolve({ src: c.toDataURL('image/jpeg', 0.9), w: v.videoWidth, h: v.videoHeight })
      done()
    }
    v.onerror = () => { done(); reject(new Error('This browser cannot preview the video; you can still analyse it.')) }
  })
}

/**
 * CCTV frame with the saved ROI drawn on it, plus the "Define ROI" workflow: click points on the
 * outdoor ground area, then Save ROI (or Reset ROI to analyse the whole frame again).
 */
export function RoiSetup({ cameraId, src, width, height, boxes, disabled }: {
  cameraId: string; src: string | null; width: number; height: number; boxes?: OverlayBox[]; disabled?: boolean
}) {
  const { toast } = useApp()
  const { zones, active } = useRoi(cameraId)
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState<[number, number][]>([])
  const [saving, setSaving] = useState(false)
  const shown: RoiZone[] = editing ? (draft.length ? [{ name: 'draft', type: 'sidewalk', mode: 'include', polygon: draft }] : []) : zones

  const save = async (next: RoiZone[]) => {
    setSaving(true)
    try {
      await Api.saveRoi(cameraId, next)
      invalidate(`roi:${cameraId}`)
      setEditing(false)
      setDraft([])
      toast({ tone: 'ok', title: next.length ? 'ROI saved' : 'ROI reset', body: next.length ? 'Litter detection is now limited to this area.' : 'The whole frame will be analysed.' })
    } catch (e) {
      toast({ tone: 'crit', title: 'Could not save ROI', body: (e as Error).message })
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="flex flex-col gap-2">
      <AspectBox width={width} height={height} className="overflow-hidden rounded-md border border-line">
        <ImageOverlay src={src} width={width} height={height} roi={shown} boxes={editing ? [] : boxes} className="absolute inset-0"
          alt="CCTV frame" onImageClick={editing ? (x, y) => setDraft((d) => [...d, [Math.round(x * 1e4) / 1e4, Math.round(y * 1e4) / 1e4]]) : undefined}>
          {editing && (
            <svg viewBox={`0 0 ${width} ${height}`} className="pointer-events-none absolute inset-0 size-full">
              {draft.map(([x, y], i) => <circle key={i} cx={x * width} cy={y * height} r={Math.max(3, width / 160)} fill="var(--warn)" stroke="#000" strokeWidth={1} />)}
            </svg>
          )}
          {editing && (
            <div className="pointer-events-none absolute left-2 top-2 flex items-center gap-1.5 rounded bg-black/70 px-2 py-1 text-[12px] text-white">
              <MousePointerClick className="size-3.5" aria-hidden /> Click the corners of the outdoor ground area · {draft.length} point{draft.length === 1 ? '' : 's'}
            </div>
          )}
        </ImageOverlay>
      </AspectBox>
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[12px] text-fg-2">ROI Status:</span>
        {active ? <Badge tone="ok" dot>Active</Badge> : <Badge tone="warn" dot>Not set — whole frame analysed</Badge>}
        <div className="ml-auto flex flex-wrap gap-1.5">
          {editing ? <>
            <Button size="sm" variant="ghost" icon={Undo2} disabled={!draft.length} onClick={() => setDraft((d) => d.slice(0, -1))}>Undo point</Button>
            <Button size="sm" variant="ghost" icon={X} onClick={() => { setEditing(false); setDraft([]) }}>Cancel</Button>
            <Button size="sm" variant="primary" icon={Check} disabled={draft.length < 3} loading={saving}
              onClick={() => save([{ name: 'outside-store', type: 'sidewalk', mode: 'include', polygon: draft }])}>Save ROI</Button>
          </> : <>
            <Button size="sm" variant="ghost" icon={RotateCcw} disabled={disabled || !zones.length} loading={saving} onClick={() => save([])}>Reset ROI</Button>
            <Button size="sm" icon={PenLine} disabled={disabled || !src} onClick={() => { setEditing(true); setDraft([]) }}>Define ROI</Button>
          </>}
        </div>
      </div>
    </div>
  )
}
