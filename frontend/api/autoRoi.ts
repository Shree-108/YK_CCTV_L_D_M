// Auto ROI API (backend/api/routes_auto_roi.py). Kept out of endpoints.ts so it does not collide with other work there.
import { api, url } from './client'
import type { AutoRoiResponse, Point } from '../lib/autoRoi'

const cam = (id: string) => `/api/cameras/${encodeURIComponent(id)}/auto-roi`

export const AutoRoiApi = {
  get: (id: string) => api<AutoRoiResponse>(cam(id)),
  /** Re-run: `frames` frames (default settings.roi.vote_frames = 40; a file source is sampled over the whole video)
   * -> segmentation -> majority vote -> polygons (needs_review until approved). About 1-2 min on CPU. */
  compute: (id: string, frames?: number) => api<AutoRoiResponse>(cam(id), { method: 'POST', body: frames ? { frames } : {} }),
  /** Save every region (the edited first one + the others unchanged). Saved as needs_review until approved. */
  editAll: (id: string, polygons: { points: Point[]; holes?: Point[][] }[]) => api<AutoRoiResponse>(cam(id), { method: 'PUT', body: { polygons } }),
  approve: (id: string) => api<AutoRoiResponse>(`${cam(id)}/approve`, { method: 'POST' }),
  edit: (id: string, polygon: Point[], holes: Point[][] = []) => api<AutoRoiResponse>(cam(id), { method: 'PUT', body: { polygon, holes } }),
  clear: (id: string) => api<{ cleared: boolean }>(cam(id), { method: 'DELETE' }),
  frameUrl: (id: string, version = '') => url(`${cam(id)}/frame.jpg${version ? `?v=${encodeURIComponent(version)}` : ''}`),
}
