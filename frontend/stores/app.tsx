'use client'

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { API_BASE, getAdminToken, setAdminToken } from '../api/client'
import { Api } from '../api/endpoints'
import type { Alert, LiveSummary, SystemEvent } from '../api/types'
import { invalidate } from '../hooks/useQuery'
import { useAuth } from '../features/auth/AuthGate'

// ---------------------------------------------------------------- toasts
export type Toast = { id: number; tone: 'ok' | 'warn' | 'crit' | 'info'; title: string; body?: string }

// ---------------------------------------------------------------- notifications (from real SSE events)
export type Notice = {
  id: string
  kind: 'litter' | 'camera' | 'model' | 'processing' | 'evaluation' | 'system'
  tone: Toast['tone']
  title: string
  body: string
  ts: string
  href?: string
}

type Theme = 'dark' | 'light' | 'system'
type StreamState = 'connecting' | 'open' | 'closed'

interface AppCtx {
  theme: Theme
  setTheme: (t: Theme) => void
  sidebarCollapsed: boolean
  setSidebarCollapsed: (v: boolean) => void
  mobileNavOpen: boolean
  setMobileNavOpen: (v: boolean) => void
  paletteOpen: boolean
  setPaletteOpen: (v: boolean) => void
  toasts: Toast[]
  toast: (t: Omit<Toast, 'id'>) => void
  dismissToast: (id: number) => void
  notices: Notice[]
  unread: number
  markAllRead: () => void
  stream: StreamState
  liveCameras: Record<string, { state: string; fps: number; inference_ms: number; summary: LiveSummary }>
  lastSummaryAt: string | null
  /** Effective role for showing/hiding controls (display only - the server enforces permissions). */
  role: UiRole
  /** Login is on and this browser is signed in with an account (the server enforces roles). */
  loginOn: boolean
  /** Admin via a verified ADMIN_API_TOKEN in this tab (as opposed to an admin account). */
  tokenAdmin: boolean
  isAdmin: boolean
  /** viewer = read-only; operator and admin may run detections and review alerts. */
  canOperate: boolean
  adminLogin: (token: string) => Promise<boolean>
  adminLogout: () => void
}

export type UiRole = 'viewer' | 'operator' | 'admin'

/** Text for a control the current role cannot use. */
export function lockedFor(need: 'operator' | 'admin', role: UiRole, loginOn: boolean): string | undefined {
  if (need === 'operator' && role !== 'viewer') return undefined
  if (need === 'admin' && role === 'admin') return undefined
  if (need === 'operator') return 'Your role (viewer) is read-only'
  return loginOn ? 'Only administrators can do this' : 'Admin sign-in required (Settings → Security)'
}

const Ctx = createContext<AppCtx | null>(null)

function readPref<T>(key: string, fallback: T): T {
  try {
    const v = window.localStorage.getItem(key)
    return v === null ? fallback : (JSON.parse(v) as T)
  } catch {
    return fallback
  }
}
function writePref(key: string, v: unknown) {
  try {
    window.localStorage.setItem(key, JSON.stringify(v))
  } catch {
    /* ignore */
  }
}

function applyTheme(t: Theme) {
  const dark = t === 'dark' || (t === 'system' && !window.matchMedia('(prefers-color-scheme: light)').matches)
  document.documentElement.dataset.theme = dark ? 'dark' : 'light'
}

function noticeFromEvent(e: SystemEvent): Notice | null {
  const tone = e.level === 'error' ? 'crit' : e.level === 'warning' ? 'warn' : 'info'
  const base = { id: `e${e.id}`, ts: e.ts, body: e.message, tone } as const
  if (e.source === 'camera' && /disconnect|error/.test(e.message))
    return { ...base, kind: 'camera', title: `Camera ${e.camera_id ?? ''} offline`, href: '/cameras' }
  if (e.source === 'camera' && /reconnected/.test(e.message))
    return { ...base, kind: 'camera', tone: 'ok', title: `Camera ${e.camera_id ?? ''} reconnected`, href: '/cameras' }
  if (e.source === 'models') return { ...base, kind: 'model', title: 'Model change', href: '/model-lab' }
  if (e.source === 'evaluation') return { ...base, kind: 'evaluation', title: e.level === 'error' ? 'Evaluation failed' : 'Evaluation completed', href: '/evaluation' }
  if (e.source === 'notifier') return { ...base, kind: 'system', title: 'Notification delivery failed', href: '/system' }
  if (e.level !== 'info') return { ...base, kind: 'system', title: `${e.source} ${e.level}`, href: '/system' }
  return null
}

export function AppProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<Theme>('dark')
  const [sidebarCollapsed, setCollapsed] = useState(false)
  const [mobileNavOpen, setMobileNavOpen] = useState(false)
  const [paletteOpen, setPaletteOpen] = useState(false)
  const [toasts, setToasts] = useState<Toast[]>([])
  const [notices, setNotices] = useState<Notice[]>([])
  const [readAt, setReadAt] = useState<string>('')
  const [stream, setStream] = useState<StreamState>('connecting')
  const [liveCameras, setLiveCameras] = useState<AppCtx['liveCameras']>({})
  const [lastSummaryAt, setLastSummaryAt] = useState<string | null>(null)
  const [tokenAdmin, setIsAdmin] = useState(false)
  // Role for the UI (display only: the server decides). Login on: the account's role. Login off: operator, as
  // before (anyone on this PC may view, detect and review). A verified admin token makes either an admin.
  const auth = useAuth()
  const loginOn = auth.status === 'signed_in'
  const accountRole: UiRole | null = loginOn && auth.user
    ? (['viewer', 'operator', 'admin'].includes(auth.user.role) ? auth.user.role as UiRole : 'viewer') : null
  // with login on only the account counts: a legacy admin token is never used (adminToken.ts clears it)
  const role: UiRole = loginOn ? (accountRole ?? 'viewer') : tokenAdmin ? 'admin' : 'operator'
  const isAdmin = role === 'admin'
  const canOperate = role !== 'viewer'
  const toastId = useRef(0)

  useEffect(() => {
    setThemeState(readPref<Theme>('lm.theme', 'dark'))
    setCollapsed(readPref('lm.sidebarCollapsed', false))
    setReadAt(readPref('lm.noticesReadAt', ''))
    if (getAdminToken()) Api.authCheck().then(() => setIsAdmin(true)).catch(() => setAdminToken(null))
  }, [])

  useEffect(() => {
    applyTheme(theme)
    if (theme !== 'system') return
    const mq = window.matchMedia('(prefers-color-scheme: light)')
    const h = () => applyTheme('system')
    mq.addEventListener('change', h)
    return () => mq.removeEventListener('change', h)
  }, [theme])

  const toast = useCallback((t: Omit<Toast, 'id'>) => {
    const id = ++toastId.current
    setToasts((ts) => [...ts.slice(-3), { ...t, id }])
    setTimeout(() => setToasts((ts) => ts.filter((x) => x.id !== id)), t.tone === 'crit' ? 8000 : 4500)
  }, [])

  const pushNotice = useCallback((n: Notice) => {
    setNotices((ns) => (ns.some((x) => x.id === n.id) ? ns : [n, ...ns].slice(0, 100)))
  }, [])

  // Live event stream (SSE). Reconnects with backoff; browser EventSource also retries on its own.
  useEffect(() => {
    let es: EventSource | null = null
    let retry: ReturnType<typeof setTimeout> | null = null
    let delay = 2000
    let closed = false
    const connect = () => {
      setStream('connecting')
      es = new EventSource(`${API_BASE}/api/stream`, { withCredentials: true })   // sends the session cookie
      es.onopen = () => {
        setStream('open')
        delay = 2000
      }
      es.addEventListener('summary', (ev) => {
        const d = JSON.parse((ev as MessageEvent).data)
        setLiveCameras(d.cameras)
        setLastSummaryAt(d.ts)
      })
      es.addEventListener('alert', (ev) => {
        const a = JSON.parse((ev as MessageEvent).data) as Pick<Alert, 'id' | 'ts' | 'camera_id' | 'class' | 'confidence' | 'severity'>
        pushNotice({
          id: `a${a.id}`, kind: 'litter', tone: a.severity === 'HIGH' ? 'crit' : a.severity === 'MEDIUM' ? 'warn' : 'info',
          title: `Litter confirmed: ${a.class.replace(/_/g, ' ')}`, body: `${a.camera_id} · ${(a.confidence * 100).toFixed(0)}% · ${a.severity}`,
          ts: a.ts, href: `/alerts?id=${a.id}`,
        })
        invalidate('alerts')
        invalidate('summary')
      })
      es.addEventListener('event', (ev) => {
        const n = noticeFromEvent(JSON.parse((ev as MessageEvent).data))
        if (n) pushNotice(n)
        invalidate('cameras')
      })
      es.onerror = () => {
        es?.close()
        setStream('closed')
        if (!closed) retry = setTimeout(connect, (delay = Math.min(delay * 2, 30000)))
      }
    }
    connect()
    return () => {
      closed = true
      es?.close()
      if (retry) clearTimeout(retry)
    }
  }, [pushNotice])

  const value = useMemo<AppCtx>(() => ({
    theme,
    setTheme: (t) => {
      setThemeState(t)
      writePref('lm.theme', t)
    },
    sidebarCollapsed,
    setSidebarCollapsed: (v) => {
      setCollapsed(v)
      writePref('lm.sidebarCollapsed', v)
    },
    mobileNavOpen, setMobileNavOpen, paletteOpen, setPaletteOpen,
    toasts, toast, dismissToast: (id) => setToasts((ts) => ts.filter((t) => t.id !== id)),
    notices,
    unread: notices.filter((n) => n.ts > readAt).length,
    markAllRead: () => {
      const now = new Date().toISOString()
      setReadAt(now)
      writePref('lm.noticesReadAt', now)
    },
    stream, liveCameras, lastSummaryAt,
    role, loginOn, tokenAdmin: tokenAdmin && !loginOn, isAdmin, canOperate,
    adminLogin: async (token) => {
      setAdminToken(token)
      try {
        await Api.authCheck()
        setIsAdmin(true)
        return true
      } catch {
        setAdminToken(null)
        setIsAdmin(false)
        return false
      }
    },
    adminLogout: () => {
      setAdminToken(null)
      setIsAdmin(false)
    },
  }), [theme, sidebarCollapsed, mobileNavOpen, paletteOpen, toasts, toast, notices, readAt, stream, liveCameras, lastSummaryAt, role, loginOn, tokenAdmin, isAdmin, canOperate])

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useApp(): AppCtx {
  const c = useContext(Ctx)
  if (!c) throw new Error('useApp must be used inside AppProvider')
  return c
}
