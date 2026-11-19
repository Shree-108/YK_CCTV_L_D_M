// Unit tests: zero-click video analysis logic (frontend/lib/autoAnalysis.ts). Run: npm run test:frontend
import assert from 'node:assert/strict'
import { test } from 'node:test'
import {
  cleanDefaults, createAutoUploader, isTerminal, overallLabel, resultsReady, screenshotFacts, shouldPoll,
  stepIndex, validateVideo, LIFECYCLE_STEPS,
} from '../lib/autoAnalysis.ts'

const LIMITS = { max_video_mb: 500, video_extensions: ['.avi', '.mov', '.mp4'] }
const file = (name, size = 10 * 1024 * 1024, lastModified = 1) => ({ name, size, lastModified })

function harness(upload) {
  const log = { uploads: [], jobs: [], errors: [], started: [] }
  const u = createAutoUploader({
    validate: (f) => validateVideo(f, LIMITS),
    upload: async (f) => { log.uploads.push(f.name); return upload(f) },
    onStart: (f) => log.started.push(f.name),
    onJob: (id) => log.jobs.push(id),
    onError: (e) => log.errors.push(e.message),
  })
  return { u, log }
}

test('a drop uploads the video and opens the analysis the server created - no further step', async () => {
  const { u, log } = harness(async () => ({ job_id: 'sv_1' }))
  assert.equal(await u.drop(file('cam.MP4')), 'sv_1')
  assert.deepEqual(log, { uploads: ['cam.MP4'], jobs: ['sv_1'], errors: [], started: ['cam.MP4'] })
  assert.equal(u.busy, false)
})

test('the same drop delivered twice uploads once; a second file during an upload is refused', async () => {
  let release
  const gate = new Promise((r) => { release = r })
  const { u, log } = harness(async () => { await gate; return { job_id: 'sv_2' } })
  const f = file('a.mp4')
  const first = u.drop(f)
  assert.equal(await u.drop(f), null)                         // duplicate event: silently ignored
  assert.equal(await u.drop(file('b.mp4', 5)), null)          // another file: told to wait
  assert.equal(u.busy, true)
  release()
  assert.equal(await first, 'sv_2')
  assert.deepEqual(log.uploads, ['a.mp4'])
  assert.deepEqual(log.jobs, ['sv_2'])
  assert.equal(log.errors.length, 1)
  assert.match(log.errors[0], /already being uploaded/)
  assert.equal(await u.drop(file('c.mp4')), 'sv_2')           // after it finished, the next drop works again
})

test('invalid, empty and oversized files never reach the server', async () => {
  const { u, log } = harness(async () => ({ job_id: 'x' }))
  for (const f of [file('notes.txt'), file('noext'), file('empty.mp4', 0), file('huge.mov', 501 * 1024 * 1024)]) {
    assert.equal(await u.drop(f), null)
  }
  assert.deepEqual(log.uploads, [])
  assert.equal(log.errors.length, 4)
  assert.match(log.errors[0], /not a supported video/)
  assert.match(log.errors[2], /empty/)
  assert.match(log.errors[3], /limit is 500 MB/)
  assert.equal(validateVideo(file('x.mkv'), null), null)       // limits unknown: default list, server enforces size
})

test('server refusal (corrupted video, too short) is reported, not swallowed', async () => {
  const { u, log } = harness(async () => { throw new Error('This video could not be opened (unsupported codec or damaged file).') })
  assert.equal(await u.drop(file('broken.mp4')), null)
  assert.deepEqual(log.jobs, [])
  assert.match(log.errors[0], /damaged file/)
  assert.equal(u.busy, false)
  const h2 = harness(async () => ({}))                          // accepted but no job: an error, never a silent stop
  await h2.u.drop(file('ok.mp4'))
  assert.match(h2.log.errors[0], /did not start an analysis/)
})

test('polling continues until the job finishes, and results appear only when the reports exist', () => {
  assert.equal(shouldPoll(undefined), true)
  for (const l of ['QUEUED', 'PROCESSING', 'EXTRACTING_SCREENSHOTS', 'RUNNING_DETECTION', 'GENERATING_REPORT']) {
    assert.equal(shouldPoll({ lifecycle: l }), true, l)
    assert.equal(resultsReady({ lifecycle: l, reports: null }), false)
  }
  for (const l of ['COMPLETED', 'FAILED', 'CANCELLED']) assert.equal(isTerminal(l), true)
  assert.equal(shouldPoll({ lifecycle: 'COMPLETED' }), false)
  assert.equal(resultsReady({ lifecycle: 'COMPLETED', reports: { detection_csv: '/x' } }), true)
  assert.equal(resultsReady({ lifecycle: 'FAILED', reports: null }), false)
  assert.equal(resultsReady({ lifecycle: 'COMPLETED', reports: null }), false)
})

test('progress steps follow the server lifecycle', () => {
  assert.deepEqual(LIFECYCLE_STEPS.map((s) => s.key), ['UPLOADING', 'QUEUED', 'PROCESSING', 'EXTRACTING_SCREENSHOTS',
    'RUNNING_DETECTION', 'GENERATING_REPORT', 'COMPLETED'])
  assert.equal(stepIndex('UPLOADING'), 0)
  assert.equal(stepIndex('RUNNING_DETECTION'), 4)
  assert.equal(stepIndex('COMPLETED'), LIFECYCLE_STEPS.length)            // all steps done
  assert.equal(stepIndex('FAILED', 'EXTRACTING_SCREENSHOTS'), 3)          // where it stopped
})

test('overall status labels: zero detections is never presented as clean', () => {
  assert.equal(overallLabel('NO_LITTER_OBSERVED').text, 'No litter detected by the model')
  assert.equal(overallLabel('NO_LITTER_OBSERVED').tone, 'neutral')
  assert.equal(overallLabel('LITTER_OBSERVED').tone, 'crit')
  assert.equal(overallLabel('INCOMPLETE_NO_LITTER_IN_ANALYSED_SCREENSHOTS').tone, 'warn')
  assert.equal(overallLabel('FAILED').tone, 'crit')
  for (const k of ['NO_LITTER_OBSERVED', 'INCOMPLETE_NO_LITTER_IN_ANALYSED_SCREENSHOTS']) assert.doesNotMatch(overallLabel(k).text, /clean/i)
})

test('a screenshot card lists confirmed litter with model confidence and severity', () => {
  const facts = screenshotFacts({
    idx: 1, frame_idx: 10, video_time_sec: 1, status: 'saved', outcome: 'LITTER_DETECTED', sampling_note: null,
    detections: [{ class: 'cup', confidence: 0.7, status: 'confirmed', severity: 'LOW', object_id: 'OBJ-002' },
      { class: 'plastic_bottle', confidence: 0.9, status: 'confirmed', severity: 'MEDIUM', object_id: 'OBJ-001' },
      { class: 'paper', confidence: 0.4, status: 'candidate' }, { class: 'cup', confidence: 0.8, status: 'rejected' }],
  })
  assert.equal(facts.count, 2)
  assert.deepEqual(facts.items.map((i) => [i.cls, i.confidence, i.severity]), [['plastic_bottle', 0.9, 'MEDIUM'], ['cup', 0.7, 'LOW']])
  assert.equal(facts.possible, 1)
  assert.equal(facts.analysed, true)
})

test('remembered upload defaults are validated', () => {
  assert.deepEqual(cleanDefaults({ camera_id: ' CAM-07 ', store_id: 'Store 12' }), { camera_id: 'CAM-07', store_id: 'Store 12' })
  assert.deepEqual(cleanDefaults({ camera_id: '../x', store_id: '=cmd()' }), { camera_id: '', store_id: '' })
  assert.deepEqual(cleanDefaults(null), { camera_id: '', store_id: '' })
})

test('a screenshot with NO_LITTER counts as analysed with zero litter', () => {
  const f = screenshotFacts({ idx: 2, frame_idx: 20, video_time_sec: 2, status: 'saved', outcome: 'NO_LITTER', sampling_note: null,
    detections: [{ class: 'paper', confidence: 0.3, status: 'weak' }] })
  assert.equal(f.count, 0)
  assert.equal(f.possible, 0)
  assert.equal(f.analysed, true)
  assert.equal(screenshotFacts({ idx: 3, frame_idx: 30, video_time_sec: 3, status: 'unreadable', outcome: 'UNREADABLE', detections: [] }).analysed, false)
})
