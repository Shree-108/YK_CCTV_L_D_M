/**
 * ROI calibration of an automatic video analysis: the operator draws the monitored ground polygon on one of the
 * four screenshots. Points are NORMALISED (0-1) to the screenshot, so the same polygon fits every screenshot of the
 * video and stays aligned at any display size. Same rules as the server (backend/sampling/roi_calibration.py),
 * which validates again on approval.
 *
 * Pure functions (no React) so they are unit-tested in frontend/tests/roiPolygon.test.mjs.
 */

export type Pt = [number, number]

export const MAX_POINTS = 64
export const MIN_AREA_FRAC = 0.002
export const MAX_AREA_FRAC = 0.98

/** Unsigned shoelace area, in the points' own units. */
export function polygonArea(pts: Pt[]): number {
  let s = 0
  for (let i = 0; i < pts.length; i++) {
    const [x1, y1] = pts[i]
    const [x2, y2] = pts[(i + 1) % pts.length]
    s += x1 * y2 - x2 * y1
  }
  return Math.abs(s) / 2
}

const cross = (o: Pt, a: Pt, b: Pt) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
const onSeg = (p: Pt, q: Pt, r: Pt) =>
  Math.min(p[0], q[0]) <= r[0] && r[0] <= Math.max(p[0], q[0]) && Math.min(p[1], q[1]) <= r[1] && r[1] <= Math.max(p[1], q[1])

export function segmentsIntersect(p1: Pt, p2: Pt, p3: Pt, p4: Pt): boolean {
  const d1 = cross(p3, p4, p1), d2 = cross(p3, p4, p2), d3 = cross(p1, p2, p3), d4 = cross(p1, p2, p4)
  if (d1 && d2 && d3 && d4 && (d1 > 0) !== (d2 > 0) && (d3 > 0) !== (d4 > 0)) return true
  return (d1 === 0 && onSeg(p3, p4, p1)) || (d2 === 0 && onSeg(p3, p4, p2)) ||
    (d3 === 0 && onSeg(p1, p2, p3)) || (d4 === 0 && onSeg(p1, p2, p4))
}

/** Does the closed polygon cross itself (edges that are not neighbours touch or cross)? */
export function selfIntersects(pts: Pt[]): boolean {
  const n = pts.length
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (j === i + 1 || (i === 0 && j === n - 1)) continue
      if (segmentsIntersect(pts[i], pts[(i + 1) % n], pts[j], pts[(j + 1) % n])) return true
    }
  }
  return false
}

/** Why this polygon (normalised points on a w x h screenshot) cannot be approved, or null. */
export function polygonProblem(norm: Pt[], w: number, h: number): string | null {
  if (norm.length < 3) return 'Click at least 3 points around the ground area.'
  if (norm.length > MAX_POINTS) return `At most ${MAX_POINTS} points are allowed.`
  if (norm.some(([x, y]) => !(x >= 0 && x <= 1 && y >= 0 && y <= 1))) return 'Every point must lie on the screenshot.'
  if (!(w > 0 && h > 0)) return 'The screenshot size is unknown.'
  const px: Pt[] = norm.map(([x, y]) => [x * w, y * h])
  for (let i = 0; i < px.length; i++) {
    const a = px[i], b = px[(i + 1) % px.length]
    if (Math.abs(a[0] - b[0]) < 1 && Math.abs(a[1] - b[1]) < 1) return 'Two points lie on top of each other. Undo the last point.'
  }
  if (selfIntersects(px)) return 'The boundary crosses itself. Click the points in order around the area.'
  const frac = polygonArea(px) / (w * h)
  if (frac < MIN_AREA_FRAC) return 'The area is too small to be a monitored ground region.'
  if (frac >= MAX_AREA_FRAC) return 'The area covers the whole screenshot. Draw only the ground in front of the shop.'
  return null
}

/** Normalised point for a click at (x, y) in screenshot pixels; clicks just outside the picture are clamped in. */
export function normalisePoint(x: number, y: number, w: number, h: number): Pt | null {
  const tol = Math.max(w, h) * 0.01
  if (x < -tol || y < -tol || x > w + tol || y > h + tol) return null
  const r = (v: number) => Math.round(v * 1e5) / 1e5
  return [r(Math.min(1, Math.max(0, x / w))), r(Math.min(1, Math.max(0, y / h)))]
}

/** SVG `points` attribute of a normalised polygon drawn in a viewBox of w x h (the screenshot's own pixels). */
export const svgPoints = (norm: Pt[], w: number, h: number) => norm.map(([x, y]) => `${x * w},${y * h}`).join(' ')

/** Position of the polygon's label: its top-most vertex (in viewBox units). */
export function labelAnchor(norm: Pt[], w: number, h: number): Pt {
  const top = norm.reduce((a, b) => (b[1] < a[1] ? b : a), norm[0] ?? [0, 0])
  return [top[0] * w, top[1] * h]
}
