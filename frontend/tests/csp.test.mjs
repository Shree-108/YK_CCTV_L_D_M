// Unit tests: web UI Content-Security-Policy (frontend/lib/csp.ts). Run: npm run test:frontend
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { apiOriginOf, buildCsp } from '../lib/csp.ts'

const API = 'http://localhost:8000'
const parse = (csp) => Object.fromEntries(csp.split('; ').map((d) => { const [k, ...v] = d.split(' '); return [k, v] }))

test('production policy: nonce + strict-dynamic, no eval, no inline scripts, no wildcards', () => {
  const p = parse(buildCsp({ nonce: 'abc123', apiOrigin: API, dev: false }))
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
  assert.equal(p['upgrade-insecure-requests'], undefined)                  // plain HTTP on this PC
})

test('the API origin is allowed exactly where the UI uses it', () => {
  const p = parse(buildCsp({ nonce: 'n', apiOrigin: API, dev: false }))
  assert.ok(p['connect-src'].includes(API))      // fetch + live event stream
  assert.ok(p['img-src'].includes(API))          // snapshots, camera frames, MJPEG
  assert.ok(p['media-src'].includes(API))        // processed videos
  assert.ok(p['img-src'].includes('data:') && p['img-src'].includes('blob:'))
  assert.ok(!p['script-src'].includes(API) && !p['default-src'].includes(API))
})

test('development adds only what Next.js dev needs', () => {
  const p = parse(buildCsp({ nonce: 'n', apiOrigin: API, dev: true }))
  assert.ok(p['script-src'].includes("'unsafe-eval'"))
  assert.ok(p['style-src'].includes("'unsafe-inline'"))
  assert.ok(p['connect-src'].includes('ws://localhost:*'))
  const prod = parse(buildCsp({ nonce: 'n', apiOrigin: API, dev: false }))
  assert.ok(!prod['connect-src'].some((s) => s.startsWith('ws:')))
})

test('a fresh nonce changes the policy', () => {
  assert.notEqual(buildCsp({ nonce: 'a', apiOrigin: API, dev: false }), buildCsp({ nonce: 'b', apiOrigin: API, dev: false }))
})

test('API origin is derived from NEXT_PUBLIC_API_URL', () => {
  assert.equal(apiOriginOf('http://localhost:8767/'), 'http://localhost:8767')
  assert.equal(apiOriginOf('http://127.0.0.1:8000/some/path'), 'http://127.0.0.1:8000')
  assert.equal(apiOriginOf(undefined), API)
  assert.equal(apiOriginOf('not a url'), API)
})
