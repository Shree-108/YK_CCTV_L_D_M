'use client'

// Login gate (API security Step 3).
// - LM_AUTH_MODE=off (the default): renders the application exactly as before - no login screen.
// - LM_AUTH_MODE=on: shows the sign-in screen until /api/auth/me reports a session, then the application.
//   The session is re-checked every 60 s and whenever the window regains focus; when it has ended the user sees
//   "Your session has expired". Idle timeout (approved policy, every role): only real user input - trusted
//   pointer, key, wheel and touch events while the tab is visible - extends the session, via POST /api/auth/activity
//   at most once a minute. Polling, streams and page timers never do. This gate is UI-only: whether API routes require a login is decided by the server
//   (backend/api/authz.py, Step 4), so nothing here should be read as protecting data.
// - Server unreachable: the application renders as before (its existing "Backend offline" status applies).
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { fetchMe, logout as apiLogout, recordActivity, type AuthMe } from '../../api/auth'
import { AUTH_REQUIRED_EVENT } from '../../api/client'
import { applyAuthMode } from '../../api/adminToken'
import { invalidate } from '../../hooks/useQuery'
import { LoginScreen, type LoginNotice } from './LoginScreen'

type Status = 'checking' | 'off' | 'unreachable' | 'signed_out' | 'signed_in'

interface AuthCtx {
  status: Status
  user: { username: string; role: string } | null
  signOut: () => Promise<void>
}

const Ctx = createContext<AuthCtx>({ status: 'off', user: null, signOut: async () => undefined })
export const useAuth = () => useContext(Ctx)

const RECHECK_MS = 60_000
const UNREACHABLE_RETRY_MS = 15_000
const ACTIVITY_PING_MS = 60_000              // report user activity at most once a minute
const ACTIVITY_EVENTS = ['pointerdown', 'keydown', 'wheel', 'touchstart'] as const

export function AuthGate({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<Status>('checking')
  const [user, setUser] = useState<AuthCtx['user']>(null)
  const [notice, setNotice] = useState<LoginNotice>(null)
  const statusRef = useRef<Status>('checking')
  statusRef.current = status

  const apply = useCallback((me: AuthMe) => {
    applyAuthMode(me.auth_mode === 'on' ? 'on' : 'off')   // login on: the legacy admin token is cleared, never sent
    if (me.auth_mode !== 'on') {
      setUser(null)
      setStatus('off')
      return
    }
    if (me.authenticated && me.via === 'session' && me.username && me.role) {
      setUser({ username: me.username, role: me.role })
      setStatus('signed_in')
      return
    }
    if (statusRef.current === 'signed_in') setNotice('expired')      // was signed in, session ended on the server
    setUser(null)
    setStatus('signed_out')
  }, [])

  const check = useCallback(async () => {
    try {
      apply(await fetchMe())
    } catch {
      // keep a known signed-in/out state on a transient error; only the very first check falls back to the app
      if (statusRef.current === 'checking' || statusRef.current === 'unreachable') setStatus('unreachable')
    }
  }, [apply])

  useEffect(() => {
    void check()
  }, [check])

  // periodic + on-focus re-check while signed in; retry while the server is unreachable
  useEffect(() => {
    if (status !== 'signed_in' && status !== 'unreachable') return
    const t = setInterval(() => void check(), status === 'signed_in' ? RECHECK_MS : UNREACHABLE_RETRY_MS)
    const onFocus = () => { if (document.visibilityState === 'visible') void check() }
    const onAuthRequired = () => void check()            // an API call answered 401: the session may have ended
    window.addEventListener('focus', onFocus)
    window.addEventListener(AUTH_REQUIRED_EVENT, onAuthRequired)
    document.addEventListener('visibilitychange', onFocus)
    return () => {
      clearInterval(t)
      window.removeEventListener('focus', onFocus)
      window.removeEventListener(AUTH_REQUIRED_EVENT, onAuthRequired)
      document.removeEventListener('visibilitychange', onFocus)
    }
  }, [status, check])

  // user activity -> idle timeout extended (signed in only). Synthetic (script-dispatched) events are ignored.
  useEffect(() => {
    if (status !== 'signed_in') return
    let lastSent = 0                            // first real input reports at once (e.g. right after a reload)
    let pending = false
    const send = () => {
      pending = false
      lastSent = Date.now()
      void recordActivity().then((ok) => { if (!ok) void check() })
    }
    const onInput = (e: Event) => {
      if (!e.isTrusted || document.visibilityState !== 'visible') return
      if (Date.now() - lastSent >= ACTIVITY_PING_MS) send()
      else pending = true
    }
    const t = setInterval(() => { if (pending && Date.now() - lastSent >= ACTIVITY_PING_MS) send() }, 10_000)
    ACTIVITY_EVENTS.forEach((ev) => window.addEventListener(ev, onInput, { capture: true, passive: true }))
    return () => {
      clearInterval(t)
      ACTIVITY_EVENTS.forEach((ev) => window.removeEventListener(ev, onInput, { capture: true }))
    }
  }, [status, check])

  const signOut = useCallback(async () => {
    await apiLogout()
    setUser(null)
    setNotice('logged_out')
    setStatus('signed_out')
  }, [])

  const onSignedIn = useCallback((me: AuthMe) => {
    setNotice(null)
    invalidate('')          // no page is mounted yet: drops every cached response of the previous session
    apply(me)
  }, [apply])

  const value = useMemo(() => ({ status, user, signOut }), [status, user, signOut])

  if (status === 'checking') {
    return (
      <div className="flex h-dvh items-center justify-center bg-bg text-[13px] text-fg-3" role="status" aria-live="polite">
        Checking sign-in…
      </div>
    )
  }
  if (status === 'signed_out') {
    return <LoginScreen notice={notice} onSignedIn={onSignedIn} onModeOff={() => void check()} />
  }
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}
