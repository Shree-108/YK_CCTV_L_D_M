'use client'

import { useMemo, useRef, useSyncExternalStore, type ReactNode } from 'react'
import type { Bbox } from '../../api/types'
import { cx } from '../../lib/format'

// ------------------------------------------------------------------ "Show monitored region" (one switch, every page)
const REGION_KEY = 'ldm.showMonitoredRegion'
const regionListeners = new Set<() => void>()
function readShowRegion(): boolean {
  try { return window.localStorage.getItem(REGION_KEY) !== 'off' } catch { return true }
}
let showRegionValue: boolean | null = null
export function setShowRegion(on: boolean) {
  showRegionValue = on
  try { window.localStorage.setItem(REGION_KEY, on ? 'on' : 'off') } catch { /* private mode: in-memory only */ }
  regionListeners.forEach((l) => l())
}
/** On by default; remembered per browser; shared by every page (Image, Video, Live). */
export function useShowRegion(): boolean {
  return useSyncExternalStore(
    (l) => { regionListeners.add(l); return () => { regionListeners.delete(l) } },
    () => (showRegionValue ??= readShowRegion()),
    () => true,
  )
}
export function RegionToggle({ className }: { className?: string }) {
  const on = useShowRegion()
  return (
    <label className={cx('inline-flex cursor-pointer select-none items-center gap-1.5 text-[12px] text-fg-2', className)}>
      <input type="checkbox" checked={on} onChange={(e) => setShowRegion(e.target.checked)} />
      <span className="inline-block size-2.5 rounded-sm" style={{ background: 'rgb(135 206 250)' }} aria-hidden />
      Show monitored region
    </label>
  )
}

/** The region polygons (with holes) + "Monitored region" label, in a width x height pixel space. */
function RegionShapes({ regions, width, height, needsReview }: { regions?: OverlayRegion[]; width: number; height: number; needsReview?: boolean }) {
  if (!regions?.length || !width || !height) return null
  const fs = Math.max(12, Math.round(width / 55))                    // label scaled to the image width
  const lx = regions[0].points.reduce((m, p) => Math.min(m, p[0]), 1) * width + fs * 0.5
  const ly = regions[0].points.reduce((m, p) => Math.min(m, p[1]), 1) * height + fs * 1.4
  const label = `Monitored region${needsReview ? ' (needs review)' : ''}`
  return (
    <g>
      {regions.map((r, i) => {
        const path = [r.points, ...(r.holes ?? [])].map((ring) => 'M' + ring.map(([x, y]) => `${x * width} ${y * height}`).join('L') + 'Z').join(' ')
        return <path key={`roi${i}`} d={path} fillRule="evenodd" fill={REGION_COLOR} fillOpacity={0.2} stroke={REGION_COLOR} strokeWidth={3} strokeLinejoin="round" vectorEffect="non-scaling-stroke"><title>Monitored region</title></path>
      })}
      <rect x={lx - fs * 0.3} y={ly - fs * 1.05} width={label.length * fs * 0.56 + fs * 0.6} height={fs * 1.45} rx={fs * 0.2} fill={REGION_COLOR} />
      <text x={lx} y={ly} fontSize={fs} fill="#14202b" fontWeight={600}>{label}</text>
    </g>
  )
}

/** Region overlay for a plain <img>/<video> shown with object-contain in a box (pixel viewBox, so letterboxing matches). */
export function RegionOverlay({ regions, width, height, needsReview, className }: {
  regions?: OverlayRegion[] | null; width?: number | null; height?: number | null; needsReview?: boolean; className?: string
}) {
  const show = useShowRegion()
  const w = width || 1920, h = height || 1080
  if (!show || !regions?.length) return null
  return (
    <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="xMidYMid meet" className={cx('pointer-events-none absolute inset-0 size-full', className)} aria-hidden>
      <RegionShapes regions={regions} width={w} height={h} needsReview={needsReview} />
    </svg>
  )
}

export interface OverlayBox {
  key: string | number
  bbox: Bbox
  label?: string
  sub?: string
  color: string
  dashed?: boolean
  thin?: boolean
  onClick?: () => void
  selected?: boolean
}

export const STATUS_COLOR: Record<string, string> = {
  confirmed: 'var(--ok)', candidate: 'var(--warn)', weak: 'var(--fg-3)', rejected: 'var(--fg-3)',
  tp: 'var(--ok)', fp: 'var(--crit)', fn: 'var(--warn)', gt: 'var(--info)',
}

/**
 * Image + SVG overlay in the image's own pixel space (viewBox), so boxes stay
 * aligned at any display size. Labels avoid overlapping by stacking.
 */
/** Monitored region (automatic pavement ROI): normalised polygons with holes, drawn light blue (RGB 135,206,250). */
export type OverlayRegion = { points: [number, number][]; holes?: [number, number][][] }
export const REGION_COLOR = 'rgb(135 206 250)'

export function ImageOverlay({ src, width, height, boxes = [], showLabels = true, className, children, fit = 'contain', onImageClick, alt, regions }: {
  regions?: OverlayRegion[]
  src: string | null; width: number; height: number; boxes?: OverlayBox[]; showLabels?: boolean
  className?: string; children?: ReactNode; fit?: 'contain' | 'cover'; onImageClick?: (x: number, y: number) => void; alt: string
}) {
  const svgRef = useRef<SVGSVGElement>(null)
  const showRegion = useShowRegion()
  regions = showRegion ? regions : undefined
  const fontSize = Math.max(11, Math.round(Math.max(width, height) / 70))
  const labels = useMemo(() => {
    const placed: [number, number, number, number][] = []
    const hit = (r: number[], p: number[]) => !(r[2] <= p[0] || r[0] >= p[2] || r[3] <= p[1] || r[1] >= p[3])
    return boxes.map((b, bi) => {
      if (!b.label) return null
      const text = b.sub ? `${b.label}  ${b.sub}` : b.label
      const w = text.length * fontSize * 0.58 + 8, h = fontSize + 6
      const [x1, y1, x2, y2] = b.bbox
      // a label must not cover another label or another box: above, below, right-aligned, beside
      const others = boxes.filter((o, oi) => oi !== bi && o.label).map((o) => o.bbox)
      const tries = [[x1, y1 - h], [x1, y2 + 1], [x2 - w, y1 - h], [x2 - w, y2 + 1], [x2 + 2, y1], [x1 - w - 2, y1], [x1, y1 - 2 * h], [x1, y2 + h + 1]]
      for (const [tx, ty] of tries) {
        const x = Math.max(0, Math.min(tx, width - w)), y = Math.max(0, Math.min(ty, height - h))
        const r: [number, number, number, number] = [x, y, x + w, y + h]
        if (!placed.some((p) => hit(r, p)) && !others.some((o) => hit(r, o))) {
          placed.push(r)
          return { x, y, w, h, text }
        }
      }
      return null
    })
  }, [boxes, fontSize, width, height])

  return (
    <div className={cx('overflow-hidden bg-video', className ?? 'relative')}>
      {src ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={src} alt={alt} draggable={false} className={cx('absolute inset-0 size-full select-none', fit === 'cover' ? 'object-cover' : 'object-contain')} />
      ) : null}
      {width > 0 && height > 0 && (
        <svg ref={svgRef} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio={fit === 'cover' ? 'xMidYMid slice' : 'xMidYMid meet'}
          className={cx('absolute inset-0 size-full', onImageClick ? 'cursor-crosshair' : 'pointer-events-none')}
          onClick={(e) => {
            if (!onImageClick || !svgRef.current) return
            const pt = svgRef.current.createSVGPoint()
            pt.x = e.clientX
            pt.y = e.clientY
            const p = pt.matrixTransform(svgRef.current.getScreenCTM()!.inverse())
            // clicks right on the frame edge are clamped in
            const tol = Math.max(width, height) * 0.01
            if (p.x >= -tol && p.y >= -tol && p.x <= width + tol && p.y <= height + tol) {
              onImageClick(Math.min(1, Math.max(0, p.x / width)), Math.min(1, Math.max(0, p.y / height)))
            }
          }}>
          <RegionShapes regions={regions} width={width} height={height} />
          {showLabels && labels.map((l, i) => l && (
            <g key={i}>
              <rect x={l.x} y={l.y} width={l.w} height={l.h} fill={boxes[i].color} opacity={0.92} rx={2} />
              <text x={l.x + 4} y={l.y + l.h - 4} fontSize={fontSize} fontFamily="var(--font-mono)" fill="#0a0c0f" fontWeight={600}>{l.text}</text>
            </g>
          ))}
          {boxes.map((b) => (
            <rect key={b.key} x={b.bbox[0]} y={b.bbox[1]} width={Math.max(1, b.bbox[2] - b.bbox[0])} height={Math.max(1, b.bbox[3] - b.bbox[1])}
              fill={b.selected ? 'rgba(77,141,255,0.15)' : 'transparent'} stroke={b.color}
              strokeWidth={(b.thin ? 1 : 2) * Math.max(1, width / 900) * (b.selected ? 1.8 : 1)} strokeDasharray={b.dashed ? '5 3' : undefined}
              className={b.onClick ? 'pointer-events-auto cursor-pointer' : undefined} onClick={(e) => { if (b.onClick) { e.stopPropagation(); b.onClick() } }}>
              {b.label && <title>{b.sub ? `${b.label} ${b.sub}` : b.label}</title>}
            </rect>
          ))}
        </svg>
      )}
      {children}
    </div>
  )
}

/** Keeps a media container at the source aspect ratio. */
export function AspectBox({ width, height, className, children }: { width: number; height: number; className?: string; children: ReactNode }) {
  return <div className={cx('relative w-full', className)} style={{ aspectRatio: `${width || 16} / ${height || 9}` }}>{children}</div>
}
