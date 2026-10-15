'use client'

// Login gate (API security Step 3).
// - LM_AUTH_MODE=off (the default): renders the application exactly as before - no login screen.
// - LM_AUTH_MODE=on: shows the sign-in screen until /api/auth/me reports a session, then the application.
//   The session is re-checked every 60 s and whenever the window regains focus; when it has ended the user sees
//   "Your session has expired". This gate is UI-only: whether API routes require a login is decided by the server
//   (route-level protection is a later step), so nothing here should be read as protecting data.
// - Server unreachable: the application renders as before (its existing "Backend offline" status applies).
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { fetchMe, logout as apiLogout, type AuthMe } from '../../api/auth'
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

export function AuthGate({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<Status>('checking')
  const [user, setUser] = useState<AuthCtx['user']>(null)
  const [notice, setNotice] = useState<LoginNotice>(null)
  const statusRef = useRef<Status>('checking')
  statusRef.current = status

  const apply = useCallback((me: AuthMe) => {
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
    window.addEventListener('focus', onFocus)
    document.addEventListener('visibilitychange', onFocus)
    return () => {
      clearInterval(t)
      window.removeEventListener('focus', onFocus)
      document.removeEventListener('visibilitychange', onFocus)
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
