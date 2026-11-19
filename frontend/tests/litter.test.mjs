// Unit tests: litter-focus display rules (frontend/lib/litter.ts). Run: npm run test:frontend
import assert from 'node:assert/strict'
import { test } from 'node:test'
import {
  camerasForStore, confidenceText, detectionStatusInfo, detectionStatusOf, fileNameFromPath, intervalSeconds, processingStatusInfo,
  recordingStartValue, storeForCamera,
} from '../lib/litter.ts'

test('detection status badges: rejected shows its reason; legacy tiers map onto the contract', () => {
  assert.deepEqual(detectionStatusInfo('DETECTED').label, 'DETECTED')
  assert.equal(detectionStatusInfo('UNDER_REVIEW').label, 'UNDER REVIEW')
  assert.equal(detectionStatusInfo('UNDER_REVIEW').tone, 'warn')
  assert.equal(detectionStatusInfo('REJECTED', 'too_small').label, 'REJECTED · too small')
  assert.equal(detectionStatusInfo('REJECTED', 'some_new_reason').label, 'REJECTED · some new reason')
  assert.equal(detectionStatusInfo('SOMETHING_ELSE').tone, 'neutral')
  assert.equal(detectionStatusOf({ detection_status: 'TRACKED', status: 'confirmed' }), 'TRACKED')
  assert.equal(detectionStatusOf({ status: 'confirmed' }), 'DETECTED')
  assert.equal(detectionStatusOf({ status: 'weak' }), 'UNDER_REVIEW')
  assert.equal(detectionStatusOf({ status: 'rejected' }), 'REJECTED')
})

test('zero detections is never called "clean"', () => {
  const none = processingStatusInfo('COMPLETED_NO_LITTER_DETECTED')
  assert.equal(none.label, 'No litter detected by the model')
  assert.doesNotMatch(`${none.label} ${none.hint}`, /\bclean\b/i)
  assert.equal(processingStatusInfo('FAILED').tone, 'crit')
  assert.equal(processingStatusInfo('FAILED').label, 'FAILED')
  assert.equal(processingStatusInfo('COMPLETED_LITTER_DETECTED').label, 'Litter detected')
})

test('confidence is shown as a decimal and a percentage', () => {
  assert.equal(confidenceText(0.8731), '0.87 · 87%')
  assert.equal(confidenceText(0.05), '0.05 · 5%')
  assert.equal(confidenceText(null), '—')
})

test('sampling interval -> sample_interval_sec', () => {
  assert.deepEqual(intervalSeconds(''), { value: null, error: null })
  assert.deepEqual(intervalSeconds('1'), { value: 1, error: null })
  assert.deepEqual(intervalSeconds('5'), { value: 5, error: null })
  assert.deepEqual(intervalSeconds('custom', '0,5'), { value: 0.5, error: null })
  assert.deepEqual(intervalSeconds('custom', '60'), { value: 60, error: null })
  assert.match(intervalSeconds('custom', '0.05').error, /Between 0.1 and 60/)
  assert.match(intervalSeconds('custom', '61').error, /Between/)
  assert.match(intervalSeconds('custom', '').error, /Enter/)
  assert.match(intervalSeconds('custom', 'abc').error, /Enter/)
})

test('recording start is sent as a local ISO date-time with seconds', () => {
  assert.equal(recordingStartValue('2026-11-19T14:05'), '2026-11-19T14:05:00')
  assert.equal(recordingStartValue('2026-11-19T14:05:07'), '2026-11-19T14:05:07')
  assert.equal(recordingStartValue(''), null)
  assert.equal(recordingStartValue('yesterday'), null)
})

test('selecting a camera fills its store; the camera record wins', () => {
  const cams = [{ id: 'CAM_1', store_id: 'S1' }, { id: 'CAM_2', store_id: null }, { id: 'CAM_3', store_id: 'S2' }]
  const stores = [{ id: 'S1', cameras: [{ id: 'CAM_1' }] }, { id: 'S9', cameras: [{ id: 'CAM_X' }] }]
  assert.equal(storeForCamera('CAM_1', cams, stores), 'S1')
  assert.equal(storeForCamera('CAM_2', cams, stores), '')
  assert.equal(storeForCamera('CAM_X', cams, stores), 'S9')    // listed only under a store
  assert.equal(storeForCamera('', cams, stores), '')
  assert.deepEqual(camerasForStore('S2', cams).map((c) => c.id), ['CAM_3'])
  assert.equal(camerasForStore('', cams).length, 3)
})

test('download file names come from the URL', () => {
  assert.equal(fileNameFromPath('/api/files/S1_2026_11_19_CAM_1_img_1_detections.csv', 'x.csv'), 'S1_2026_11_19_CAM_1_img_1_detections.csv')
  assert.equal(fileNameFromPath('/api/jobs/abc/files/video', 'abc.mp4'), 'abc.mp4')
})

test('analytics series are mapped onto days and hours without inventing values', async () => {
  const { seriesForDays, bucketLabel, hours24 } = await import('../lib/litter.ts')
  const series = [{ bucket: '2026-11-18', events: 2, observations: 30 }, { bucket: '2026-11-19T14', events: 1, observations: 4 }, { bucket: '2026-11-19T15', events: 1, observations: 2 }]
  assert.deepEqual(seriesForDays(series, ['2026-11-17', '2026-11-18', '2026-11-19']), { events: [0, 2, 2], observations: [0, 30, 6] })
  assert.equal(bucketLabel('2026-11-19T14'), '19 14h')
  assert.equal(bucketLabel('2026-11-19'), '11-19')
  assert.equal(bucketLabel('2026-W47'), '2026-W47')
  const h = hours24([{ hour: 0, events: 1 }, { hour: 23, events: 4 }, { hour: 24, events: 9 }])
  assert.equal(h.length, 24)
  assert.equal(h[0], 1)
  assert.equal(h[23], 4)
  assert.equal(h.reduce((a, b) => a + b, 0), 5)
})
