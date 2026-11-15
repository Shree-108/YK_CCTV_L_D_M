// Login / session calls (API security Step 3). The session lives ONLY in the HttpOnly cookie set by the server:
// nothing here stores, returns or logs a password or session ID, and server error texts are never shown raw.
import { API_BASE } from './client'

export type AuthRole = 'viewer' | 'operator' | 'admin'

export interface AuthMe {
  auth_mode: 'off' | 'on'
  authenticated: boolean
  username?: string
  role?: AuthRole
  via?: 'session' | 'admin_token'
  idle_expires_at?: string | null
}

export type LoginResult =
  | { ok: true; me: AuthMe }
  | { ok: false; reason: 'invalid' | 'locked' | 'disabled' | 'unreachable' | 'rejected' | 'error'; message: string }

const opts: RequestInit = { credentials: 'include', cache: 'no-store' }

/** Current mode and session. Throws only when the server cannot be reached or answers unexpectedly. */
export async function fetchMe(signal?: AbortSignal): Promise<AuthMe> {
  const res = await fetch(`${API_BASE}/api/auth/me`, { ...opts, signal })
  if (!res.ok) throw new Error(`auth status unavailable (${res.status})`)
  return (await res.json()) as AuthMe
}

function minutes(retryAfter: string | null): string {
  const s = Number(retryAfter)
  if (!Number.isFinite(s) || s <= 0) return 'a few minutes'
  const m = Math.max(1, Math.ceil(s / 60))
  return m === 1 ? 'about a minute' : `about ${m} minutes`
}

export async function login(username: string, password: string): Promise<LoginResult> {
  let res: Response
  try {
    res = await fetch(`${API_BASE}/api/auth/login`, {
      ...opts, method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    })
  } catch {
    return { ok: false, reason: 'unreachable', message: 'Cannot reach the server. Check that the backend is running and try again.' }
  }
  if (res.ok) {
    try {
      return { ok: true, me: await fetchMe() }
    } catch {
      return { ok: false, reason: 'error', message: 'Signed in, but the session could not be confirmed. Please try again.' }
    }
  }
  switch (res.status) {
    case 401: return { ok: false, reason: 'invalid', message: 'Invalid username or password.' }
    case 429: return { ok: false, reason: 'locked', message: `Too many failed sign-in attempts. Try again in ${minutes(res.headers.get('Retry-After'))}.` }
    case 404: return { ok: false, reason: 'disabled', message: 'Sign-in is not enabled on this server.' }
    case 400: return { ok: false, reason: 'rejected', message: 'The sign-in request was not accepted. Please try again.' }
    case 403: return { ok: false, reason: 'rejected', message: 'Sign-in is only available on the computer running the application.' }
    default: return { ok: false, reason: 'error', message: 'Sign-in failed. Please try again.' }
  }
}

/**
 * Tells the server the person at this browser interacted with the app, which extends the idle timeout (never past the
 * 12-hour limit). The ONLY request that does: polling, streams and other API calls never extend a session.
 * Returns false when the session is no longer valid.
 */
export async function recordActivity(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/api/auth/activity`, { ...opts, method: 'POST' })
    return res.status !== 401
  } catch {
    return true                                    // offline: nothing to conclude; the regular re-check handles it
  }
}

/** Ends the server session; the server also clears the cookie. Network failures are ignored (the UI signs out). */
export async function logout(): Promise<void> {
  try {
    await fetch(`${API_BASE}/api/auth/logout`, { ...opts, method: 'POST' })
  } catch {
    /* offline: the cookie expires on its own */
  }
}
