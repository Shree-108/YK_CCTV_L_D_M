// Auto ROI display/edit rules (pure functions, unit-tested in frontend/tests/autoRoi.test.mjs).

export type Point = [number, number]

export interface AutoRoi {
  polygon: Point[]
  holes?: Point[][]
  source: 'auto' | 'auto_edited' | 'manual' | 'request'
  needs_review: boolean
  created_at?: string
  updated_at?: string
  approved_at?: string | null
  approved_by?: string | null
  frames_used?: number
  frame_size?: [number, number]
  ground_share?: number
  model?: string
  label_shares?: Record<string, number>
}

export interface AutoRoiResponse {
  key: string
  roi: AutoRoi | null
  resolved_from: 'request' | 'camera' | 'global' | 'none' | 'computed'
  computed?: boolean
  note?: string
}

export type Tone = 'ok' | 'warn' | 'crit' | 'info' | 'exp' | 'neutral'

/** Badge for the ROI state of one camera. */
export function roiStatus(r: Pick<AutoRoiResponse, 'roi' | 'resolved_from'> | undefined): { label: string; tone: Tone } {
  if (!r || !r.roi) return { label: 'No ROI · full frame', tone: 'neutral' }
  if (r.resolved_from === 'global') return { label: 'Global ROI (fallback)', tone: 'info' }
  if (r.roi.needs_review) return { label: 'Auto · needs review', tone: 'warn' }
  return { label: r.roi.source === 'manual' ? 'Manual · approved' : 'Approved', tone: 'ok' }
}

const clamp01 = (v: number) => Math.min(1, Math.max(0, v))
const r4 = (v: number) => Math.round(v * 10000) / 10000

/** Move vertex i to (x, y), clamped to the image and rounded like the backend. */
export function moveVertex(poly: Point[], i: number, x: number, y: number): Point[] {
  return poly.map((p, k) => (k === i ? [r4(clamp01(x)), r4(clamp01(y))] : p))
}

/** Insert a vertex in the middle of edge i -> i+1. */
export function insertVertex(poly: Point[], i: number): Point[] {
  const a = poly[i], b = poly[(i + 1) % poly.length]
  const mid: Point = [r4((a[0] + b[0]) / 2), r4((a[1] + b[1]) / 2)]
  return [...poly.slice(0, i + 1), mid, ...poly.slice(i + 1)]
}

/** Remove vertex i; a polygon never drops below 3 points. */
export function removeVertex(poly: Point[], i: number): Point[] {
  return poly.length <= 3 ? poly : poly.filter((_, k) => k !== i)
}

/** Shoelace area in normalised units (0..1). */
export function polygonArea(poly: Point[]): number {
  let s = 0
  for (let i = 0; i < poly.length; i++) {
    const [x1, y1] = poly[i], [x2, y2] = poly[(i + 1) % poly.length]
    s += x1 * y2 - x2 * y1
  }
  return Math.abs(s) / 2
}

/** Same check the API makes before saving: >= 3 points, inside 0..1, non-zero area. */
export function polygonValid(poly: Point[]): boolean {
  return poly.length >= 3 && poly.every(([x, y]) => x >= 0 && x <= 1 && y >= 0 && y <= 1) && polygonArea(poly) >= 1e-4
}

export const svgPoints = (poly: Point[]) => poly.map(([x, y]) => `${x},${y}`).join(' ')
