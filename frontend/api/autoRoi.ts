// Auto ROI API (backend/api/routes_auto_roi.py). Kept out of endpoints.ts so it does not collide with other work there.
import { api, url } from './client'
import type { AutoRoiResponse, Point } from '../lib/autoRoi'

const cam = (id: string) => `/api/cameras/${encodeURIComponent(id)}/auto-roi`

export const AutoRoiApi = {
  get: (id: string) => api<AutoRoiResponse>(cam(id)),
  /** Median of `frames` frames -> segmentation -> polygon (needs_review). Slow on CPU: ~10-30 s. */
  compute: (id: string, frames = 5) => api<AutoRoiResponse>(cam(id), { method: 'POST', body: { frames } }),
  approve: (id: string) => api<AutoRoiResponse>(`${cam(id)}/approve`, { method: 'POST' }),
  edit: (id: string, polygon: Point[], holes: Point[][] = []) => api<AutoRoiResponse>(cam(id), { method: 'PUT', body: { polygon, holes } }),
  clear: (id: string) => api<{ cleared: boolean }>(cam(id), { method: 'DELETE' }),
  frameUrl: (id: string, version = '') => url(`${cam(id)}/frame.jpg${version ? `?v=${encodeURIComponent(version)}` : ''}`),
}
