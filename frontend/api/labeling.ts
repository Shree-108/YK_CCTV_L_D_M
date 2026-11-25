// Keyboard review of mined detections (backend/api/routes_labeling.py, phase 2 item 6).
import { api, url } from './client'

export const REVIEW_LABELS = ['TP', 'FP_shadow', 'FP_carton', 'FP_stock', 'FP_puddle', 'FP_other'] as const
export type ReviewLabel = (typeof REVIEW_LABELS)[number]

export interface LabelSession {
  video_id: string; video_file?: string | null; frame_size?: [number, number] | null
  detections: number; offered: number; labelled: number; groups: number; missed: number; below_030: number
}
export interface LabelItem {
  group_id: string; det_id: string; members: number; frames: number[]; class: string; conf: number; min_conf: number
  below_030: boolean; bbox: [number, number, number, number]; crop_box: [number, number, number, number] | null
  frame_file: string; frame_idx: number; timestamp_sec: number; label: string; status: string; in_roi: string
}
export interface MissedBox {
  missed_id: string; frame_file: string; frame_idx: string; class: string; x1: string; y1: string; x2: string; y2: string
}
export interface LabelQueue {
  video_id: string; labels: string[]; items: LabelItem[]; missed: MissedBox[]
  manifest: { class_names?: string[]; frame_size?: [number, number]; mining_conf?: number; cap_per_class?: number; video_file?: string; roi?: { polygons?: { points: [number, number][]; holes?: [number, number][][] }[] } | null }
}

const v = (id: string) => `/api/labeling/videos/${encodeURIComponent(id)}`

export const LabelingApi = {
  videos: () => api<{ videos: LabelSession[] }>('/api/labeling/videos'),
  queue: (id: string) => api<LabelQueue>(v(id)),
  /** Autosave one label ('' = clear, used by undo); whole_group labels every frame of the same box. */
  label: (id: string, target: string, label: ReviewLabel | '', whole_group = true, true_class?: string) =>
    api<{ changed: string[] }>(`${v(id)}/labels/${encodeURIComponent(target)}`, { method: 'PUT', body: { label, whole_group, ...(true_class !== undefined ? { true_class } : {}) } }),
  addMissed: (id: string, frame_file: string, bbox: number[], cls: string) =>
    api<MissedBox>(`${v(id)}/missed`, { method: 'POST', body: { frame_file, bbox, class: cls } }),
  deleteMissed: (id: string, missedId: string) => api<{ deleted: string }>(`${v(id)}/missed/${encodeURIComponent(missedId)}`, { method: 'DELETE' }),
  image: (id: string, path: string) => url(`${v(id)}/image?path=${encodeURIComponent(path)}`),
}
