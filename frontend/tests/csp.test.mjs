// Unit tests: web UI Content-Security-Policy (frontend/lib/csp.ts). Run: npm run test:frontend
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { buildCsp } from '../lib/csp.ts'

const parse = (csp) => Object.fromEntries(csp.split('; ').map((d) => { const [k, ...v] = d.split(' '); return [k, v] }))

test('production policy: nonce + strict-dynamic, no eval, no inline scripts, no wildcards', () => {
  const p = parse(buildCsp({ nonce: 'abc123', dev: false }))
  assert.deepEqual(p['script-src'], ["'self'", "'nonce-abc123'", "'strict-dynamic'"])
  assert.deepEqual(p['style-src'], ["'self'", "'nonce-abc123'"])          // <style> elements need the nonce
  assert.deepEqual(p['style-src-attr'], ["'unsafe-inline'"])               // React style={} attributes only
  for (const [name, values] of Object.entries(p)) {
    assert.ok(!values.includes('*'), `${name} has a wildcard`)
    assert.ok(!values.includes("'unsafe-eval'"), `${name} allows eval`)
    if (name !== 'style-src-attr') assert.ok(!values.includes("'unsafe-inline'"), `${name} allows inline`)
  }
  assert.deepEqual(p['object-src'], ["'none'"])
  assert.deepEqual(p['frame-ancestors'], ["'none'"])
  assert.deepEqual(p['base-uri'], ["'self'"])
  assert.deepEqual(p['form-action'], ["'self'"])
  assert.equal(p['upgrade-insecure-requests'], undefined)                  // plain HTTP until HTTPS is set up
})

test('the same-origin API needs no extra source', () => {
  const p = parse(buildCsp({ nonce: 'n', dev: false }))
  for (const d of ['connect-src', 'img-src', 'media-src']) {   // fetch + event stream, snapshots/MJPEG, videos
    assert.ok(p[d].includes("'self'"), d)
    assert.ok(!p[d].some((s) => s.startsWith('http')), `${d} lists a separate origin`)
  }
  assert.ok(p['img-src'].includes('data:') && p['img-src'].includes('blob:'))
})

test('development adds only what Next.js dev needs', () => {
  const p = parse(buildCsp({ nonce: 'n', dev: true }))
  assert.ok(p['script-src'].includes("'unsafe-eval'"))
  assert.ok(p['style-src'].includes("'unsafe-inline'"))
  assert.ok(p['connect-src'].includes('ws://localhost:*'))
  const prod = parse(buildCsp({ nonce: 'n', dev: false }))
  assert.ok(!prod['connect-src'].some((s) => s.startsWith('ws:')))
})

test('a fresh nonce changes the policy', () => {
  assert.notEqual(buildCsp({ nonce: 'a', dev: false }), buildCsp({ nonce: 'b', dev: false }))
})
