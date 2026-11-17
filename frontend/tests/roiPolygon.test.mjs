// Unit tests: ROI calibration polygon rules and labels (frontend/lib/roiPolygon.ts, lib/autoAnalysis.ts).
// Run: npm run test:frontend
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { labelAnchor, normalisePoint, polygonArea, polygonProblem, selfIntersects, svgPoints } from '../lib/roiPolygon.ts'
import { canCalibrate, roiLabel } from '../lib/autoAnalysis.ts'

const GROUND = [[0.08, 0.55], [0.92, 0.55], [1, 1], [0, 1]]

test('a ground polygon with perspective is valid; invalid shapes are explained', () => {
  assert.equal(polygonProblem(GROUND, 640, 360), null)
  assert.match(polygonProblem(GROUND.slice(0, 2), 640, 360), /at least 3/)
  assert.match(polygonProblem([[0.1, 0.5], [0.9, 1], [0.9, 0.5], [0.1, 1]], 640, 360), /crosses itself/)
  assert.match(polygonProblem([[0.5, 0.5], [0.52, 0.5], [0.51, 0.52]], 640, 360), /too small/)
  assert.match(polygonProblem([[0, 0], [1, 0], [1, 1], [0, 1]], 640, 360), /whole screenshot/)
  assert.match(polygonProblem([[0.1, 0.6], [0.1, 0.6], [0.5, 0.9], [0.2, 0.95]], 640, 360), /on top of each other/)
  assert.match(polygonProblem([[0.1, 0.6], [1.2, 0.6], [0.5, 0.9]], 640, 360), /on the screenshot/)
})

test('self-intersection and area helpers', () => {
  assert.equal(selfIntersects([[0, 0], [10, 0], [10, 10], [0, 10]]), false)
  assert.equal(selfIntersects([[0, 0], [10, 10], [10, 0], [0, 10]]), true)
  assert.equal(polygonArea([[0, 0], [10, 0], [10, 10], [0, 10]]), 100)
})

test('clicks are stored normalised, so the polygon stays aligned at any display or picture size', () => {
  // the same physical spot clicked on a 640x360 and a 1920x1080 rendering of one view
  assert.deepEqual(normalisePoint(320, 270, 640, 360), [0.5, 0.75])
  assert.deepEqual(normalisePoint(960, 810, 1920, 1080), [0.5, 0.75])
  assert.deepEqual(normalisePoint(-3, 362, 640, 360), [0, 1])           // just off the edge -> clamped in
  assert.equal(normalisePoint(-50, 100, 640, 360), null)                  // far outside -> ignored
  // drawn in the picture's own pixel space (SVG viewBox = picture size): scales exactly with the picture
  const nums = (s) => s.split(/[ ,]/).map(Number)
  const close = (a, b) => a.length === b.length && a.every((v, i) => Math.abs(v - b[i]) < 1e-6)
  assert.ok(close(nums(svgPoints(GROUND, 640, 360)), [51.2, 198, 588.8, 198, 640, 360, 0, 360]))
  assert.ok(close(nums(svgPoints(GROUND, 1280, 720)), [102.4, 396, 1177.6, 396, 1280, 720, 0, 720]))
  assert.ok(close(labelAnchor(GROUND, 640, 360), [51.2, 198]))
})

test('ROI labels and when the operator may calibrate', () => {
  assert.equal(roiLabel({ status: 'NOT_CONFIGURED', source: 'FULL_FRAME_FALLBACK' }).text.startsWith('ROI_NOT_CONFIGURED'), true)
  assert.equal(roiLabel({ status: 'APPROVED', source: 'OPERATOR_APPROVED_FOR_JOB', approved: true }).tone, 'ok')
  assert.match(roiLabel({ status: 'REVALIDATION_REQUIRED', source: 'SAVED_CAMERA_ROI' }).text, /revalidation/)
  const job = (policy, extra = {}) => ({ mode: 'auto', lifecycle: 'COMPLETED', roi_policy: policy, ...extra })
  assert.equal(canCalibrate(job({ status: 'NOT_CONFIGURED', source: 'FULL_FRAME_FALLBACK', approved: false })), true)
  assert.equal(canCalibrate(job({ status: 'REVALIDATION_REQUIRED', source: 'SAVED_CAMERA_ROI', approved: false })), true)
  assert.equal(canCalibrate(job({ status: 'APPROVED', source: 'OPERATOR_APPROVED_FOR_JOB', approved: true })), true)
  // a camera ROI in force is never re-asked; running / manual jobs cannot be calibrated here
  assert.equal(canCalibrate(job({ status: 'APPROVED', source: 'REGISTERED_CAMERA', approved: true })), false)
  assert.equal(canCalibrate(job({ status: 'NOT_CONFIGURED', source: 'FULL_FRAME_FALLBACK' }, { lifecycle: 'RUNNING_DETECTION' })), false)
  assert.equal(canCalibrate(job({ status: 'NOT_CONFIGURED', source: 'FULL_FRAME_FALLBACK' }, { mode: 'manual' })), false)
})
