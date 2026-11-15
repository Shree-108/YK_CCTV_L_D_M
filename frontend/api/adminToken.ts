// Legacy ADMIN_API_TOKEN in the browser (API security Step 4, approved decision).
//
// - Login OFF (the default): unchanged - an admin may type the token in Settings -> Security; it is kept in this tab's
//   sessionStorage and sent as a Bearer header on admin calls.
// - Login ON: the token is not used at all. Any stored token is deleted and nothing can be stored; admins sign in
//   with an account. (The server still accepts the header from scripts - that compatibility is unchanged.)
// - Until the server has said which mode it is in, the token is not used either (default deny), so a stale token can
//   never reach a server that has login switched on.
//
// Plain TypeScript with no imports, so it can be unit-tested with Node's built-in test runner.

export const ADMIN_TOKEN_KEY = 'lm.adminToken'
export const ADMIN_TOKEN_EVENT = 'lm-admin-token'

type Mode = 'unknown' | 'login_off' | 'login_on'
let mode: Mode = 'unknown'

function storage(): Storage | null {
  try {
    return typeof window === 'undefined' ? null : window.sessionStorage
  } catch {
    return null
  }
}

function notify(): void {
  try {
    window.dispatchEvent(new Event(ADMIN_TOKEN_EVENT))
  } catch {
    /* no window (server render / tests) */
  }
}

/** Called by the login gate with the server's auth mode (from GET /api/auth/me). */
export function applyAuthMode(authMode: 'on' | 'off'): void {
  mode = authMode === 'on' ? 'login_on' : 'login_off'
  if (mode === 'login_on') {
    const s = storage()
    try {
      if (s && s.getItem(ADMIN_TOKEN_KEY) !== null) {
        s.removeItem(ADMIN_TOKEN_KEY)
        notify()
      }
    } catch {
      /* storage unavailable */
    }
  }
}

/** May the legacy token be used right now? Only once the server has confirmed that login is off. */
export function adminTokenAllowed(): boolean {
  return mode === 'login_off'
}

export function getAdminToken(): string | null {
  if (!adminTokenAllowed()) return null
  try {
    return storage()?.getItem(ADMIN_TOKEN_KEY) ?? null
  } catch {
    return null
  }
}

export function setAdminToken(token: string | null): void {
  const s = storage()
  try {
    if (token && adminTokenAllowed()) s?.setItem(ADMIN_TOKEN_KEY, token)
    else s?.removeItem(ADMIN_TOKEN_KEY)
  } catch {
    /* storage unavailable: nothing is kept */
  }
  notify()
}

/** Tests only: back to the initial (unknown) state. */
export function _resetForTests(): void {
  mode = 'unknown'
}
