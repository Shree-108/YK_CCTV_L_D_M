'use client'

import { useMemo, useRef, useState, type ReactNode } from 'react'
import type { Bbox, RoiZone } from '../../api/types'
import { cx } from '../../lib/format'

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
export function ImageOverlay({ src, width, height, boxes = [], roi, showLabels = true, className, children, fit = 'contain', onImageClick, alt }: {
  src: string | null; width: number; height: number; boxes?: OverlayBox[]; roi?: RoiZone[] | null; showLabels?: boolean
  className?: string; children?: ReactNode; fit?: 'contain' | 'cover'; onImageClick?: (x: number, y: number) => void; alt: string
}) {
  const svgRef = useRef<SVGSVGElement>(null)
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
            // clicks right on the frame edge are clamped in, so a polygon can reach the image border
            const tol = Math.max(width, height) * 0.01
            if (p.x >= -tol && p.y >= -tol && p.x <= width + tol && p.y <= height + tol) {
              onImageClick(Math.min(1, Math.max(0, p.x / width)), Math.min(1, Math.max(0, p.y / height)))
            }
          }}>
          {roi?.map((z, i) => (
            <polygon key={i} points={z.polygon.map(([x, y]) => `${x * width},${y * height}`).join(' ')}
              fill={z.mode === 'include' ? 'rgba(0,220,255,0.12)' : 'rgba(240,82,74,0.12)'}
              stroke={z.mode === 'include' ? '#00dcff' : 'var(--crit)'} strokeWidth={Math.max(2, width / 320)} strokeLinejoin="round"
              strokeDasharray={z.mode === 'include' ? undefined : '6 4'}>
              <title>{`${z.name} (${z.type}, ${z.mode})`}</title>
            </polygon>
          ))}
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

/** ROI polygon editor: click to add points; zones are normalized [0,1]. */
export function RoiEditor({ src, width, height, zones, onChange }: { src: string | null; width: number; height: number; zones: RoiZone[]; onChange: (z: RoiZone[]) => void }) {
  const [draft, setDraft] = useState<[number, number][]>([])
  const [mode, setMode] = useState<RoiZone['mode']>('include')
  const [type, setType] = useState<RoiZone['type']>('sidewalk')
  const previewZones: RoiZone[] = [...zones, ...(draft.length ? [{ name: 'draft', type, mode, polygon: draft }] : [])]
  return (
    <div className="flex flex-col gap-2">
      <AspectBox width={width} height={height} className="overflow-hidden rounded-md border border-line">
        <ImageOverlay src={src} width={width} height={height} roi={previewZones} className="absolute inset-0"
          alt="Camera frame for ROI drawing" onImageClick={(x, y) => setDraft((d) => [...d, [Math.round(x * 1e4) / 1e4, Math.round(y * 1e4) / 1e4]])}>
          <svg viewBox={`0 0 ${width} ${height}`} className="pointer-events-none absolute inset-0 size-full">
            {draft.map(([x, y], i) => <circle key={i} cx={x * width} cy={y * height} r={Math.max(3, width / 250)} fill="var(--warn)" />)}
          </svg>
        </ImageOverlay>
      </AspectBox>
      <div className="flex flex-wrap items-center gap-2 text-[12px]">
        <select aria-label="Zone type" value={type} onChange={(e) => setType(e.target.value as RoiZone['type'])} className="h-7 rounded-md border border-line bg-panel-2 px-2 text-fg">
          {['sidewalk', 'road_edge', 'ground', 'parking', 'entrance', 'custom'].map((t) => <option key={t} value={t}>{t.replace('_', ' ')}</option>)}
        </select>
        <select aria-label="Zone mode" value={mode} onChange={(e) => setMode(e.target.value as RoiZone['mode'])} className="h-7 rounded-md border border-line bg-panel-2 px-2 text-fg">
          <option value="include">include (monitor)</option><option value="exclude">exclude (ignore)</option>
        </select>
        <span className="text-fg-3">{draft.length} points — click on the frame to add</span>
        <div className="ml-auto flex gap-1.5">
          <button className="h-7 rounded-md border border-line px-2 text-fg-2 hover:bg-hover disabled:opacity-40" disabled={!draft.length} onClick={() => setDraft((d) => d.slice(0, -1))}>Undo point</button>
          <button className="h-7 rounded-md border border-line px-2 text-fg-2 hover:bg-hover disabled:opacity-40" disabled={!draft.length} onClick={() => setDraft([])}>Clear</button>
          <button className="h-7 rounded-md bg-info px-2 font-medium text-white disabled:opacity-40" disabled={draft.length < 3}
            onClick={() => { onChange([...zones, { name: `${type}-${zones.length + 1}`, type, mode, polygon: draft }]); setDraft([]) }}>Add zone</button>
        </div>
      </div>
      {zones.length > 0 && (
        <ul className="flex flex-col gap-1">
          {zones.map((z, i) => (
            <li key={i} className="flex items-center gap-2 rounded-md border border-line bg-panel-2 px-2 py-1 text-[12px]">
              <span className={cx('size-2 rounded-sm', z.mode === 'include' ? 'bg-info' : 'bg-crit')} aria-hidden />
              <span className="text-fg">{z.name}</span><span className="text-fg-3">{z.type} · {z.mode} · {z.polygon.length} pts</span>
              <button className="ml-auto text-fg-3 hover:text-crit" onClick={() => onChange(zones.filter((_, j) => j !== i))}>Remove</button>
            </li>
          ))}
        </ul>
      )}
      <p className="text-[11px] text-fg-3">The bottom-centre (foot point) of each detection is tested against the zones. With no include zone the whole frame is monitored; exclude zones always win.</p>
    </div>
  )
}
