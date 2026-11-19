'use client'

import { useCallback, useRef, useState, type ReactNode } from 'react'
import { Check, UploadCloud } from 'lucide-react'
import { Api } from '../../api/endpoints'
import { useQuery } from '../../hooks/useQuery'
import { cx } from '../../lib/format'
import { Field, Select, Slider } from '../../components/ui/primitives'
import { appendIdentity, type StoreCamera } from '../stores/StoreCameraSelect'

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
      className={cx('flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed text-center transition-colors',
        compact ? 'p-5' : 'min-h-[320px] p-10', over ? 'border-info bg-info-bg' : 'border-line-strong bg-panel-2/50 hover:border-fg-3 hover:bg-panel-2')}>
      <span className={cx('flex items-center justify-center rounded-full border border-line bg-panel', compact ? 'size-10' : 'size-14')}>
        <UploadCloud className={cx(compact ? 'size-5' : 'size-6', over ? 'text-fg' : 'text-fg-2')} aria-hidden />
      </span>
      <p className="mt-1 text-[14px] font-medium text-fg">{label}</p>
      <p className="text-[12px] text-fg-3">{hint}</p>
      {!compact && <span className="mt-2 inline-flex h-8 items-center rounded-md border border-line bg-panel px-3 text-[12.5px] font-medium text-fg">Browse files</span>}
      <input ref={input} type="file" accept={accept} className="hidden" onChange={(e) => { pick(e.target.files?.[0]); e.target.value = '' }} />
    </div>
  )
}

/** Upload → Analyse → Results progress strip (presentation only). */
export function Steps({ steps, current }: { steps: string[]; current: number }) {
  return (
    <ol className="flex flex-wrap items-center gap-x-2 gap-y-1.5" aria-label="Analysis steps">
      {steps.map((s, i) => {
        const state = i < current ? 'done' : i === current ? 'now' : 'todo'
        return (
          <li key={s} className="flex items-center gap-2" aria-current={state === 'now' ? 'step' : undefined}>
            <span className={cx('flex size-6 items-center justify-center rounded-full border text-[11px] font-semibold',
              state === 'done' ? 'border-ok/40 bg-ok-bg text-ok' : state === 'now' ? 'border-info bg-info text-on-accent' : 'border-line-strong text-fg-3')}>
              {state === 'done' ? <Check className="size-3.5" aria-hidden /> : i + 1}
            </span>
            <span className={cx('text-[12.5px]', state === 'todo' ? 'text-fg-3' : 'font-medium text-fg')}>{s}</span>
            {i < steps.length - 1 && <span className="mx-1 hidden h-px w-8 bg-line-strong sm:block" aria-hidden />}
          </li>
        )
      })}
    </ol>
  )
}

/** Small cards describing the selected local file (name, size, type, resolution…). */
export function FileInfo({ items }: { items: [string, ReactNode][] }) {
  return (
    <dl className="grid grid-cols-2 gap-2 md:grid-cols-4">
      {items.map(([k, v]) => (
        <div key={k} className="min-w-0 rounded-lg border border-line bg-panel-2 px-3 py-2">
          <dt className="text-[11px] font-medium text-fg-3">{k}</dt>
          <dd className="truncate text-[13px] font-medium text-fg" title={typeof v === 'string' ? v : undefined}>{v}</dd>
        </div>
      ))}
    </dl>
  )
}

export const fileSize = (b: number) => (b >= 2 ** 20 ? `${(b / 2 ** 20).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1024))} KB`)
export const fileExt = (f: File) => (f.name.split('.').pop() ?? f.type.split('/').pop() ?? '—').toUpperCase()

export interface DetectConfig {
  model_version: string
  confidence: number
  image_size: number | ''
  tiling: 'auto' | 'on' | 'off'
}

/** Shared detection settings form (model, threshold, size, tiling). Values default to the live settings. */
export function DetectionSettingsForm({ cfg, setCfg, children }: {
  cfg: DetectConfig; setCfg: (f: (c: DetectConfig) => DetectConfig) => void; children?: ReactNode
}) {
  const models = useQuery('models', Api.models)
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
      <p className="-mt-2 text-[11px] text-fg-3">System setting: {s ? s.confidence.confirmed_threshold.toFixed(2) : '—'}. Detections at or above it are DETECTED; weaker ones are shown as UNDER REVIEW.</p>
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
      {children}
    </div>
  )
}

export function toForm(file: File, cfg: DetectConfig, who: StoreCamera, extra: Record<string, string> = {}) {
  const f = new FormData()
  f.append('file', file)
  appendIdentity(f, who)
  f.append('confidence', String(cfg.confidence))
  f.append('tiling', cfg.tiling)
  if (cfg.image_size) f.append('image_size', String(cfg.image_size))
  if (cfg.model_version) f.append('model_version', cfg.model_version)
  for (const [k, v] of Object.entries(extra)) f.append(k, v)
  return f
}

/** ``plastic_bag`` -> ``Plastic Bag`` */
export const litterName = (c: string) => c.replace(/_/g, ' ').replace(/\b\w/g, (m) => m.toUpperCase())
