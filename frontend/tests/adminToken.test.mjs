// Unit tests: legacy admin token policy (frontend/api/adminToken.ts). Run: npm run test:frontend
// Node's built-in test runner; Node strips the TypeScript types when importing the .ts module (no dependencies).
import assert from 'node:assert/strict'
import { beforeEach, test } from 'node:test'
import {
  ADMIN_TOKEN_KEY, adminTokenAllowed, applyAuthMode, getAdminToken, setAdminToken, _resetForTests,
} from '../api/adminToken.ts'

const SYNTHETIC = 'synthetic-test-token'          // not a real token
let events = 0

function fakeWindow() {
  const data = new Map()
  globalThis.window = {
    sessionStorage: {
      getItem: (k) => (data.has(k) ? data.get(k) : null),
      setItem: (k, v) => { data.set(k, String(v)) },
      removeItem: (k) => { data.delete(k) },
    },
    dispatchEvent: () => { events += 1; return true },
  }
  globalThis.Event = class { constructor(type) { this.type = type } }
  return data
}

let store
beforeEach(() => {
  store = fakeWindow()
  events = 0
  _resetForTests()
})

test('before the server mode is known the token is never used (default deny)', () => {
  store.set(ADMIN_TOKEN_KEY, SYNTHETIC)
  assert.equal(adminTokenAllowed(), false)
  assert.equal(getAdminToken(), null)
  setAdminToken('another')                          // cannot be stored either
  assert.equal(store.has(ADMIN_TOKEN_KEY), false)
})

test('login OFF keeps the existing workflow: store, read, clear', () => {
  applyAuthMode('off')
  assert.equal(adminTokenAllowed(), true)
  setAdminToken(SYNTHETIC)
  assert.equal(getAdminToken(), SYNTHETIC)
  assert.equal(store.get(ADMIN_TOKEN_KEY), SYNTHETIC)
  setAdminToken(null)
  assert.equal(getAdminToken(), null)
  assert.ok(events >= 2)
})

test('login OFF keeps a token stored earlier in this tab', () => {
  store.set(ADMIN_TOKEN_KEY, SYNTHETIC)
  applyAuthMode('off')
  assert.equal(getAdminToken(), SYNTHETIC)
})

test('login ON deletes a stored token, never returns one and refuses to store one', () => {
  store.set(ADMIN_TOKEN_KEY, SYNTHETIC)
  applyAuthMode('on')
  assert.equal(store.has(ADMIN_TOKEN_KEY), false)  // stale token removed from browser storage
  assert.equal(events, 1)                           // the app is told (admin badge updates)
  assert.equal(adminTokenAllowed(), false)
  assert.equal(getAdminToken(), null)
  setAdminToken(SYNTHETIC)
  assert.equal(store.has(ADMIN_TOKEN_KEY), false)
  assert.equal(getAdminToken(), null)
})

test('switching from ON back to OFF does not bring a cleared token back', () => {
  store.set(ADMIN_TOKEN_KEY, SYNTHETIC)
  applyAuthMode('on')
  applyAuthMode('off')
  assert.equal(getAdminToken(), null)
})

test('no window / unavailable storage never throws', () => {
  delete globalThis.window
  applyAuthMode('off')
  assert.equal(getAdminToken(), null)
  setAdminToken(SYNTHETIC)
  applyAuthMode('on')
})
