// Unit tests: Results view display rules (frontend/lib/results.ts). Run: npm run test:frontend
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { bundleBadge, cameraText, checkStatusInfo, confidenceCell, recordingTimeText, resultsQuery } from '../lib/results.ts'

test('status badges: the three CSV values, coloured; anything else is neutral', () => {
  assert.equal(checkStatusInfo('clean').tone, 'ok')
  assert.equal(checkStatusInfo('litter_detected').tone, 'crit')
  assert.equal(checkStatusInfo('view_blocked').tone, 'warn')
  assert.equal(checkStatusInfo('DETECTED').tone, 'neutral')
})

test('recording time is shown as recorded (+05:30), never converted; missing -> time unknown', () => {
  assert.deepEqual(recordingTimeText('2026-11-19T16:30:41+05:30'), { text: '2026-11-19 16:30:41 +05:30', unknown: false })
  assert.deepEqual(recordingTimeText(null), { text: 'time unknown', unknown: true })
  assert.deepEqual(recordingTimeText(''), { text: 'time unknown', unknown: true })
})

test('upload or empty camera ids are flagged as no camera assigned', () => {
  assert.deepEqual(cameraText('upload'), { text: 'no camera assigned', missing: true })
  assert.deepEqual(cameraText(''), { text: 'no camera assigned', missing: true })
  assert.deepEqual(cameraText(null), { text: 'no camera assigned', missing: true })
  assert.deepEqual(cameraText('TEST_CAM_01'), { text: 'TEST_CAM_01', missing: false })
})

test('confidence stays the CSV text (0-100, empty for clean / blocked)', () => {
  assert.equal(confidenceCell('78.27'), '78.27')
  assert.equal(confidenceCell(''), '')
})

test('bundle badge says whether the job has an import-ready CSV and why not', () => {
  assert.equal(bundleBadge(null).label, 'no bundle')
  assert.equal(bundleBadge({ ok: true, import_ready: true, folder: 'F', csv: 'x.csv' }).tone, 'ok')
  const b = bundleBadge({ ok: true, import_ready: false, reasons: ['no camera assigned', 'no store assigned'] })
  assert.equal(b.label, 'no CSV')
  assert.equal(b.title, 'no camera assigned\nno store assigned')
  assert.equal(bundleBadge({ ok: false, error: 'boom' }).label, 'bundle failed')
})

test('filter query drops empty values', () => {
  assert.equal(resultsQuery({ store: 'S1', camera: '', status: 'clean', min_confidence: ' ' }), 'store=S1&status=clean')
})
