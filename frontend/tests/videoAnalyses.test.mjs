// Unit tests: Overview "Videos processed" and "Recent video analyses" (frontend/lib/videoAnalyses.ts). Run: npm run test:frontend
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { recentAnalyses, videoAnalysisCounts } from '../lib/videoAnalyses.ts'

const full = (id, status, created) => ({ id, status, created_at: created, camera_id: null, input_name: `${id}.mp4`, progress: 0.4 })
const sampled = (id, state, created) => ({ id, state, created_at: created, camera_id: 'CAM-07', video_name: `${id}.mp4` })

test('completed four-screenshot analyses are counted with full-video jobs', () => {
  const jobs = [full('a', 'done', '2026-11-01T10:00:00'), full('b', 'failed', '2026-11-02T10:00:00'), full('c', 'running', '2026-11-03T10:00:00')]
  // 5 sampled analyses in all, 3 of them done (2 failed / cancelled)
  assert.deepEqual(videoAnalysisCounts(jobs, 5, 3), { done: 4, fullDone: 1, sampledDone: 3, notFinished: 4 })
})

test('only sampled analyses (the default mode) still give a non-zero count', () => {
  assert.deepEqual(videoAnalysisCounts([], 2, 2), { done: 2, fullDone: 0, sampledDone: 2, notFinished: 0 })
})

test('without sampled data the full-video figures are unchanged', () => {
  const jobs = [full('a', 'done', '2026-11-01T10:00:00'), full('b', 'queued', '2026-11-02T10:00:00')]
  assert.deepEqual(videoAnalysisCounts(jobs, 0, 0), { done: 1, fullDone: 1, sampledDone: 0, notFinished: 1 })
})

test('recent analyses mix both kinds, newest first, limited to n', () => {
  const rows = recentAnalyses(
    [full('a', 'done', '2026-11-01T10:00:00'), full('c', 'running', '2026-11-04T10:00:00')],
    [sampled('s1', 'running', '2026-11-03T10:00:00'), sampled('s2', 'done', '2026-11-05T10:00:00')], 3)
  assert.deepEqual(rows.map((r) => [r.key, r.kind, r.status]), [['s:s2', 'sampled', 'done'], ['v:c', 'full', 'running'], ['s:s1', 'sampled', 'running']])
  assert.equal(rows[1].progress, 0.4)                 // progress only for a running full-video job
  assert.equal(rows[0].progress, undefined)
  assert.equal(rows[0].camera, 'CAM-07')
})
