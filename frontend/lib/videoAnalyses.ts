/**
 * Overview page: "Videos processed" and "Recent video analyses" cover BOTH kinds of video analysis -
 * full-video tracking jobs (/api/jobs, kind 'video') and four-screenshot sampled analyses (/api/sampled/jobs,
 * the default Video Analysis mode). Counts are of completed analyses: a video analysed twice counts twice.
 * Pure functions (no React), unit-tested in frontend/tests/videoAnalyses.test.mjs.
 */

export interface FullVideoJob { id: string; status: string; camera_id: string | null; created_at: string; input_name: string; progress: number }
export interface SampledAnalysis { id: string; state: string; camera_id: string | null; created_at: string; video_name: string | null }

export interface RecentAnalysis {
  key: string
  kind: 'full' | 'sampled'
  name: string
  camera: string
  created: string
  status: string
  progress?: number
}

export interface VideoAnalysisCounts {
  done: number          // completed analyses of both kinds
  fullDone: number
  sampledDone: number
  notFinished: number   // queued, running, failed, cancelled or interrupted
}

/** ``sampledTotal`` / ``sampledDone``: totals reported by /api/sampled/reports/history (all states / state=done). */
export function videoAnalysisCounts(fullVideoJobs: FullVideoJob[], sampledTotal: number, sampledDone: number): VideoAnalysisCounts {
  const fullDone = fullVideoJobs.filter((j) => j.status === 'done').length
  return {
    done: fullDone + sampledDone,
    fullDone,
    sampledDone,
    notFinished: fullVideoJobs.length - fullDone + Math.max(0, sampledTotal - sampledDone),
  }
}

/** Newest ``n`` analyses of both kinds, newest first. */
export function recentAnalyses(fullVideoJobs: FullVideoJob[], sampled: SampledAnalysis[], n: number): RecentAnalysis[] {
  const rows: RecentAnalysis[] = [
    ...fullVideoJobs.map((j): RecentAnalysis => ({
      key: `v:${j.id}`, kind: 'full', name: j.input_name, camera: j.camera_id ?? 'upload', created: j.created_at, status: j.status,
      progress: j.status === 'running' ? j.progress : undefined,
    })),
    ...sampled.map((r): RecentAnalysis => ({
      key: `s:${r.id}`, kind: 'sampled', name: r.video_name ?? 'video', camera: r.camera_id ?? 'upload', created: r.created_at, status: r.state,
    })),
  ]
  return rows.sort((a, b) => b.created.localeCompare(a.created)).slice(0, n)
}
