// Unit tests: auto ROI badge + polygon editing rules (frontend/lib/autoRoi.ts). Run: npm run test:frontend
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { insertVertex, moveVertex, polygonArea, polygonValid, removeVertex, roiStatus } from '../lib/autoRoi.ts'

const SQ = [[0, 0.5], [1, 0.5], [1, 1], [0, 1]]

test('status badge: none, needs review, approved, manual, global fallback', () => {
  assert.equal(roiStatus(undefined).tone, 'neutral')
  assert.equal(roiStatus({ roi: null, resolved_from: 'none' }).label, 'No ROI · full frame')
  assert.deepEqual(roiStatus({ roi: { polygon: SQ, source: 'auto', needs_review: true }, resolved_from: 'camera' }), { label: 'Auto · needs review', tone: 'warn' })
  assert.equal(roiStatus({ roi: { polygon: SQ, source: 'auto', needs_review: false }, resolved_from: 'camera' }).label, 'Approved')
  assert.equal(roiStatus({ roi: { polygon: SQ, source: 'manual', needs_review: false }, resolved_from: 'camera' }).label, 'Manual · approved')
  assert.equal(roiStatus({ roi: { polygon: SQ, source: 'manual', needs_review: false }, resolved_from: 'global' }).tone, 'info')
})

test('moving a vertex clamps to the image and rounds to 4 decimals', () => {
  assert.deepEqual(moveVertex(SQ, 0, -0.2, 0.123456)[0], [0, 0.1235])
  assert.deepEqual(moveVertex(SQ, 2, 1.5, 2)[2], [1, 1])
  assert.deepEqual(moveVertex(SQ, 0, 0.1, 0.1).slice(1), SQ.slice(1))
})

test('insert / remove vertices; never below 3 points', () => {
  const p = insertVertex(SQ, 0)
  assert.equal(p.length, 5)
  assert.deepEqual(p[1], [0.5, 0.5])
  assert.equal(removeVertex(p, 1).length, 4)
  const tri = [[0, 0], [1, 0], [1, 1]]
  assert.equal(removeVertex(tri, 0).length, 3)
})

test('validity matches the API: area > 0 and inside 0..1', () => {
  assert.equal(polygonArea(SQ), 0.5)
  assert.equal(polygonValid(SQ), true)
  assert.equal(polygonValid([[0, 0], [0.5, 0.5], [1, 1]]), false)
  assert.equal(polygonValid([[0, 0], [2, 0], [1, 1]]), false)
})
