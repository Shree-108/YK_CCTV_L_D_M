'use client'

import Link from 'next/link'
import { usePathname, useRouter } from 'next/navigation'
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import {
  AlertTriangle, ArrowRight, Bell, ChevronDown, ChevronsLeft, ChevronsRight, CornerDownLeft, Cpu, LogOut, Menu, Search, Settings as SettingsIcon, X,
} from 'lucide-react'
import { API_BASE } from '../../api/client'
import { Api } from '../../api/endpoints'
import type { SearchResult } from '../../api/types'
import { useAuth } from '../../features/auth/AuthGate'
import { useQuery } from '../../hooks/useQuery'
import { ago, cx, className as cls, pct } from '../../lib/format'
import { BRAND, MODULE_NAME } from '../../lib/module'
import { useApp, type UiRole } from '../../stores/app'
import { Dot, IconButton, Kbd } from '../ui/primitives'
import { Drawer, Toaster } from '../ui/overlay'
import { EmptyState } from '../ui/states'
import { ALL_NAV, NAV, navFor, type BadgeKey } from './nav'

// ---------------------------------------------------------------- badge data (real API values)
function useBadges(): Partial<Record<BadgeKey, number>> {
  const { liveCameras } = useApp()
  const alerts = useQuery('alerts:newcount', () => Api.alerts({ status: 'new', limit: 1 }), { poll: 15000 })
  const cams = useQuery('cameras:list', Api.cameras, { poll: 20000 })
  const hard = useQuery('hard:pending', () => Api.hardExamples({ status: 'pending', limit: 1 }), { poll: 60000 })
  return {
    online: Object.values(liveCameras).filter((c) => c.state === 'online').length || undefined,
    newAlerts: alerts.data?.total || undefined,
    cameras: cams.data?.cameras.length || undefined,
    hardPending: hard.data?.total || undefined,
  }
}

/** Yatrikart wordmark + the active AI module (lib/module.ts). Logo is white; inverted in light theme. */
function Brand({ collapsed }: { collapsed?: boolean }) {
  return (
    <div className={cx('flex h-14 shrink-0 items-center border-b border-line', collapsed ? 'justify-center px-0' : 'px-5')}>
      {collapsed ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img src="/brand/yatrikart-mark.png" alt={`${BRAND} · ${MODULE_NAME}`} title={`${BRAND} · ${MODULE_NAME}`}
          className="brand-logo h-6 w-auto select-none" draggable={false} />
      ) : (
        <div className="flex min-w-0 flex-col gap-1">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/brand/yatrikart-wordmark.png" alt={BRAND} className="brand-logo h-[18px] w-auto self-start select-none" draggable={false} />
          <div className="truncate text-[9.5px] font-semibold uppercase tracking-[0.26em] text-fg-3">{MODULE_NAME}</div>
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------- backend connection status
type ConnTone = 'ok' | 'warn' | 'crit'
/** "Connected" only after /api/health actually answered; the live event stream is reported separately. */
function useBackendStatus(): { tone: ConnTone; label: string; short: string; hint?: string; detail: string } {
  const { stream } = useApp()
  const health = useQuery('health', Api.health, { poll: 15000 })
  if (health.error) {
    return { tone: 'crit', label: 'Backend offline', short: 'Offline', hint: API_BASE,
      detail: `${health.error.message} Retrying every 15 s. If the API is running, check that this page's address is listed in CORS_ORIGINS.` }
  }
  if (!health.data) return { tone: 'warn', label: 'Connecting to backend…', short: 'Connecting…', hint: API_BASE, detail: `Waiting for ${API_BASE}/api/health` }
  if (!health.data.model) {
    return { tone: 'warn', label: 'Backend online · no model', short: 'No model', hint: 'detection unavailable',
      detail: health.data.model_error ?? 'No model loaded' }
  }
  if (stream !== 'open') {
    return { tone: 'ok', label: 'Connected to backend', short: 'Connected', hint: 'live updates reconnecting…', detail: `${API_BASE} · event stream ${stream}` }
  }
  return { tone: 'ok', label: 'Connected to backend', short: 'Connected', detail: `${API_BASE} · model ${health.data.model.version}` }
}

const CONN_PILL: Record<ConnTone, string> = {
  ok: 'border-ok/30 bg-ok-bg text-ok', warn: 'border-warn/30 bg-warn-bg text-warn', crit: 'border-crit/30 bg-crit-bg text-crit',
}
const CONN_DOT: Record<ConnTone, string> = { ok: 'live-dot bg-ok', warn: 'bg-warn', crit: 'bg-crit' }

// ---------------------------------------------------------------- sidebar
function Sidebar({ collapsed, onNavigate, onToggle }: { collapsed: boolean; onNavigate?: () => void; onToggle?: () => void }) {
  const pathname = usePathname()
  const badges = useBadges()
  const active = navFor(pathname)
  const conn = useBackendStatus()
  return (
    <nav aria-label="Primary" className="flex h-full flex-col">
      <Brand collapsed={collapsed} />
      <div className="min-h-0 flex-1 overflow-y-auto py-3">
        {NAV.filter((g) => !g.hidden).map((g, gi) => (
          <div key={g.label} className="mb-3">
            {collapsed
              ? gi > 0 && <div className="mx-3 mb-3 border-t border-line" aria-hidden />
              : <div className="px-5 pb-1.5 text-[10px] font-semibold uppercase tracking-[0.16em] text-fg-3">{g.label}</div>}
            <ul className="flex flex-col gap-0.5">
              {g.items.map((it) => {
                const on = active?.href === it.href
                const b = it.badge ? badges[it.badge] : undefined
                return (
                  <li key={it.href} className="px-2.5">
                    <Link href={it.href} onClick={onNavigate} title={collapsed ? it.label : undefined}
                      aria-current={on ? 'page' : undefined}
                      className={cx('group relative flex h-9 items-center gap-3 rounded-md text-[13.5px] transition-colors',
                        collapsed ? 'justify-center' : 'px-3',
                        on ? 'bg-info-bg font-medium text-fg' : 'text-fg-2 hover:bg-hover hover:text-fg')}>
                      {on && <span className="absolute -left-2.5 bottom-1.5 top-1.5 w-[3px] rounded-r-full bg-info" aria-hidden />}
                      <it.icon className={cx('size-4 shrink-0', on ? 'text-fg' : 'text-fg-3 group-hover:text-fg-2')} aria-hidden />
                      {!collapsed && <span className="flex-1 truncate">{it.label}</span>}
                      {b !== undefined && (collapsed
                        ? <span className={cx('absolute right-1.5 top-1.5 size-1.5 rounded-full', it.badge === 'newAlerts' ? 'bg-crit' : it.badge === 'online' ? 'bg-ok' : 'bg-fg-3')} aria-label={`${b}`} />
                        : <span className={cx('tabular min-w-5 rounded-full px-1.5 text-center text-[11px] font-semibold leading-[18px]',
                            it.badge === 'newAlerts' ? 'bg-crit text-white' : it.badge === 'online' ? 'bg-ok-bg text-ok' : 'bg-panel-2 text-fg-3')}>
                            {b > 99 ? '99+' : b}
                          </span>)}
                    </Link>
                  </li>
                )
              })}
            </ul>
          </div>
        ))}
      </div>
      <div className={cx('flex shrink-0 items-center gap-2 border-t border-line py-2.5', collapsed ? 'flex-col px-2' : 'px-3')}>
        <Link href="/system" onClick={onNavigate} title={`${conn.label}${conn.hint ? ` · ${conn.hint}` : ''}\n${conn.detail}`}
          className={cx('flex min-w-0 items-center gap-2 rounded-md text-[12px] text-fg-2 hover:bg-hover hover:text-fg', collapsed ? 'size-8 justify-center' : 'h-8 flex-1 px-2')}>
          <span className={cx('size-2 shrink-0 rounded-full', CONN_DOT[conn.tone])} aria-hidden />
          {collapsed ? <span className="sr-only">{conn.label}</span> : <span className="truncate">{conn.label}</span>}
        </Link>
        {onToggle && (
          <button onClick={onToggle} aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'} title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            className="inline-flex size-8 shrink-0 items-center justify-center rounded-md text-fg-3 hover:bg-hover hover:text-fg">
            {collapsed ? <ChevronsRight className="size-4" /> : <ChevronsLeft className="size-4" />}
          </button>
        )}
      </div>
    </nav>
  )
}

// ---------------------------------------------------------------- command palette
type Cmd = { id: string; label: string; hint?: string; group: string; run: () => void }

function CommandPalette() {
  const { paletteOpen, setPaletteOpen } = useApp()
  const router = useRouter()
  const [q, setQ] = useState('')
  const [idx, setIdx] = useState(0)
  const [results, setResults] = useState<SearchResult | null>(null)
  const [recent, setRecent] = useState<string[]>([])
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (!paletteOpen) return
    setQ('')
    setIdx(0)
    setResults(null)
    try { setRecent(JSON.parse(localStorage.getItem('lm.recentSearches') ?? '[]')) } catch { setRecent([]) }
    setTimeout(() => inputRef.current?.focus(), 10)
  }, [paletteOpen])

  // debounced global search against the backend
  useEffect(() => {
    if (!paletteOpen || q.trim().length < 2) { setResults(null); return }
    const t = setTimeout(() => { Api.search(q.trim()).then(setResults).catch(() => setResults(null)) }, 180)
    return () => clearTimeout(t)
  }, [q, paletteOpen])

  const go = (href: string, remember?: string) => {
    if (remember) {
      const next = [remember, ...recent.filter((r) => r !== remember)].slice(0, 6)
      try { localStorage.setItem('lm.recentSearches', JSON.stringify(next)) } catch { /* ignore */ }
    }
    setPaletteOpen(false)
    router.push(href)
  }

  const cmds: Cmd[] = useMemo(() => {
    const ql = q.toLowerCase()
    const actions: Cmd[] = [
      { id: 'a-cam', label: 'Add camera', group: 'Actions', hint: 'Cameras', run: () => go('/cameras?new=1') },
      { id: 'a-img', label: 'Run image detection', group: 'Actions', run: () => go('/detect/image') },
      { id: 'a-vid', label: 'Analyse a video', group: 'Actions', run: () => go('/detect/video') },
      { id: 'a-eval', label: 'Run model evaluation', group: 'Actions', run: () => go('/evaluation') },
      { id: 'a-alerts', label: 'View new alerts', group: 'Actions', run: () => go('/alerts') },
    ]
    const nav: Cmd[] = ALL_NAV.map((n) => ({ id: `n-${n.href}`, label: `Go to ${n.label}`, hint: n.description, group: 'Navigation', run: () => go(n.href) }))
    const found: Cmd[] = []
    if (results) {
      results.cameras.forEach((c) => found.push({ id: `c-${c.id}`, label: `${c.name}`, hint: `${c.id} · ${c.location || 'no location'}`, group: 'Cameras', run: () => go(`/cameras?id=${encodeURIComponent(c.id)}`, q) }))
      results.alerts.forEach((a) => found.push({ id: `al-${a.id}`, label: `Alert #${a.id} · ${cls(a.class)}`, hint: `${a.camera_id} · ${pct(a.confidence, 0)} · ${a.severity} · ${ago(a.ts)}`, group: 'Alerts', run: () => go(`/alerts?id=${a.id}`, q) }))
      results.models.forEach((m) => found.push({ id: `m-${m.version}`, label: m.version, hint: m.active ? 'active model' : m.name, group: 'Models', run: () => go(`/model-lab?v=${encodeURIComponent(m.version)}`, q) }))
      results.experiments.forEach((e) => found.push({ id: `e-${e.id}`, label: e.name, hint: e.id, group: 'Experiments', run: () => go(`/experiments?id=${encodeURIComponent(e.id)}`, q) }))
      results.datasets.forEach((d) => found.push({ id: `d-${d.name}`, label: d.name, group: 'Datasets', run: () => go(`/datasets?name=${encodeURIComponent(d.name)}`, q) }))
    }
    const match = (c: Cmd) => !ql || c.label.toLowerCase().includes(ql) || (c.hint ?? '').toLowerCase().includes(ql)
    const rec: Cmd[] = !ql ? recent.map((r) => ({ id: `r-${r}`, label: r, group: 'Recent searches', run: () => setQ(r) })) : []
    return [...rec, ...found, ...actions.filter(match), ...nav.filter(match)]
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [q, results, recent])

  useEffect(() => setIdx(0), [cmds.length])
  if (!paletteOpen) return null
  let lastGroup = ''
  return (
    <div className="fixed inset-0 z-[55] flex items-start justify-center p-4 pt-[12vh]">
      <div className="absolute inset-0 bg-black/55" onClick={() => setPaletteOpen(false)} aria-hidden />
      <div role="dialog" aria-modal="true" aria-label="Command palette" className="relative w-[620px] max-w-full overflow-hidden rounded-lg border border-line bg-panel shadow-[var(--shadow)] animate-in fade-in zoom-in-95 duration-100">
        <div className="flex items-center gap-2 border-b border-line px-3">
          <Search className="size-4 text-fg-3" aria-hidden />
          <input ref={inputRef} value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search cameras, alerts, models, experiments, datasets… or type a command"
            aria-label="Search" role="combobox" aria-expanded aria-controls="palette-list" aria-activedescendant={cmds[idx]?.id}
            className="h-11 flex-1 bg-transparent text-[14px] text-fg placeholder:text-fg-3 focus:outline-none"
            onKeyDown={(e) => {
              if (e.key === 'ArrowDown') { e.preventDefault(); setIdx((i) => Math.min(i + 1, cmds.length - 1)) }
              if (e.key === 'ArrowUp') { e.preventDefault(); setIdx((i) => Math.max(i - 1, 0)) }
              if (e.key === 'Enter') { e.preventDefault(); cmds[idx]?.run() }
              if (e.key === 'Escape') setPaletteOpen(false)
            }} />
          <Kbd>Esc</Kbd>
        </div>
        <ul id="palette-list" role="listbox" className="max-h-[52vh] overflow-y-auto p-1.5">
          {cmds.length === 0 && <li className="px-3 py-6 text-center text-[12px] text-fg-3">No results for “{q}”</li>}
          {cmds.map((c, i) => {
            const header = c.group !== lastGroup ? (lastGroup = c.group) : null
            return (
              <li key={c.id} role="none">
                {header && <div className="px-2 pb-1 pt-2 text-[10px] font-semibold uppercase tracking-[0.1em] text-fg-3">{header}</div>}
                <button id={c.id} role="option" aria-selected={i === idx} onMouseEnter={() => setIdx(i)} onClick={c.run}
                  className={cx('flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left', i === idx ? 'bg-hover' : '')}>
                  <ArrowRight className="size-3.5 shrink-0 text-fg-3" aria-hidden />
                  <span className="truncate text-[13px] text-fg">{c.label}</span>
                  {c.hint && <span className="ml-auto truncate pl-3 text-[11px] text-fg-3">{c.hint}</span>}
                  {i === idx && <CornerDownLeft className="size-3 shrink-0 text-fg-3" aria-hidden />}
                </button>
              </li>
            )
          })}
        </ul>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------- notification center
function NotificationCenter({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { notices, markAllRead, stream } = useApp()
  const router = useRouter()
  useEffect(() => { if (open) markAllRead() }, [open]) // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <Drawer open={open} onClose={onClose} title="Notifications" width="w-[400px]"
      subtitle={stream === 'open' ? 'Live — receiving events from the backend' : 'Event stream disconnected; reconnecting…'}>
      {notices.length === 0 ? (
        <EmptyState icon={Bell} title="No notifications yet" body="New confirmed litter, camera outages, model changes and finished evaluations appear here as they happen while this page is open." />
      ) : (
        <ul className="flex flex-col gap-1.5">
          {notices.map((n) => (
            <li key={n.id}>
              <button onClick={() => { if (n.href) { onClose(); router.push(n.href) } }}
                className="flex w-full gap-2.5 rounded-md border border-line bg-panel-2 p-2.5 text-left hover:border-line-strong">
                <Dot tone={n.tone} />
                <div className="min-w-0 flex-1">
                  <div className="flex items-center justify-between gap-2">
                    <span className="truncate text-[13px] font-medium text-fg">{n.title}</span>
                    <span className="shrink-0 text-[11px] text-fg-3">{ago(n.ts)}</span>
                  </div>
                  <p className="truncate text-[12px] text-fg-3">{n.body}</p>
                </div>
              </button>
            </li>
          ))}
        </ul>
      )}
    </Drawer>
  )
}

// ---------------------------------------------------------------- top bar
const ROLE_LABEL: Record<UiRole, string> = { viewer: 'Viewer', operator: 'Operator', admin: 'Admin' }
const ROLE_INITIALS: Record<UiRole, string> = { viewer: 'VW', operator: 'OP', admin: 'AD' }
const ROLE_HINT: Record<UiRole, string> = {
  viewer: 'Viewer: read-only access.',
  operator: 'Operator: view, run detections and review alerts.',
  admin: 'Admin: full access, including configuration and exports.',
}

/** Signed-in user (login on) or the current UI role (login off), with role details and sign-out. */
function ProfileMenu() {
  const { role, loginOn, tokenAdmin } = useApp()
  const auth = useAuth()
  const [open, setOpen] = useState(false)
  const [signingOut, setSigningOut] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!open) return
    const close = (e: MouseEvent) => { if (!ref.current?.contains(e.target as Node)) setOpen(false) }
    const esc = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false) }
    document.addEventListener('mousedown', close)
    document.addEventListener('keydown', esc)
    return () => { document.removeEventListener('mousedown', close); document.removeEventListener('keydown', esc) }
  }, [open])
  const user = auth.status === 'signed_in' ? auth.user : null
  const name = user?.username ?? ROLE_LABEL[role]
  const initials = user ? user.username.slice(0, 2).toUpperCase() : ROLE_INITIALS[role]
  return (
    <div ref={ref} className="relative">
      <button onClick={() => setOpen((v) => !v)} aria-haspopup="menu" aria-expanded={open} aria-label={`Account: ${name}, role ${ROLE_LABEL[role]}`}
        className="flex h-9 items-center gap-2 rounded-md pl-1 pr-1.5 text-left hover:bg-hover">
        <span className={cx('flex size-7 items-center justify-center rounded-full text-[11px] font-semibold', role === 'admin' ? 'bg-exp-bg text-exp' : 'bg-info-bg text-fg')}>{initials}</span>
        <span className="hidden min-w-0 flex-col leading-tight xl:flex">
          <span className="max-w-[120px] truncate text-[12.5px] font-medium text-fg">{name}</span>
          <span className="text-[10.5px] text-fg-3">{user ? user.role : ROLE_LABEL[role]}</span>
        </span>
        <ChevronDown className="hidden size-3.5 text-fg-3 xl:block" aria-hidden />
      </button>
      {open && (
        <div role="menu" className="absolute right-0 z-40 mt-1.5 w-64 rounded-lg border border-line bg-elev p-1.5 shadow-[var(--shadow)]">
          <div className="border-b border-line px-2.5 pb-2.5 pt-1.5">
            <div className="truncate text-[13px] font-medium text-fg">{user ? `Signed in as ${user.username}` : `Role: ${ROLE_LABEL[role]}`}</div>
            <div className="mt-0.5 text-[11.5px] text-fg-3">
              {ROLE_HINT[role]}{tokenAdmin ? ' (admin token)' : loginOn ? '' : ' Sign in as admin in Settings → Security.'}
            </div>
          </div>
          <Link role="menuitem" href="/settings?tab=security" onClick={() => setOpen(false)}
            className="mt-1 flex items-center gap-2 rounded-md px-2.5 py-1.5 text-[13px] text-fg-2 hover:bg-hover hover:text-fg">
            <SettingsIcon className="size-3.5" aria-hidden /> Role &amp; security
          </Link>
          {user && (
            <button role="menuitem" disabled={signingOut} onClick={async () => { setSigningOut(true); await auth.signOut() }}
              className="flex w-full items-center gap-2 rounded-md px-2.5 py-1.5 text-left text-[13px] text-fg-2 hover:bg-hover hover:text-fg disabled:opacity-50">
              <LogOut className="size-3.5" aria-hidden /> Sign out
            </button>
          )}
        </div>
      )}
    </div>
  )
}

function Topbar() {
  const pathname = usePathname()
  const item = navFor(pathname)
  const { setPaletteOpen, setMobileNavOpen, unread } = useApp()
  const health = useQuery('health', Api.health, { poll: 15000 })
  const alerts = useQuery('alerts:newcount', () => Api.alerts({ status: 'new', limit: 1 }), { poll: 15000 })
  const [notifOpen, setNotifOpen] = useState(false)
  const conn = useBackendStatus()
  const model = health.data?.model
  const newAlerts = alerts.data?.total ?? 0

  return (
    <header className="sticky top-0 z-30 flex h-14 shrink-0 items-center gap-3 border-b border-line bg-panel/95 px-3 backdrop-blur md:px-5">
      <IconButton icon={Menu} label="Open navigation" className="lg:hidden" onClick={() => setMobileNavOpen(true)} />
      <div className="min-w-0 shrink">
        <div className="truncate text-[15px] font-semibold leading-tight text-fg">{item?.label ?? MODULE_NAME}</div>
        <div className="hidden max-w-[280px] truncate text-[11px] text-fg-3 sm:block">{item?.description ?? `${BRAND} · ${MODULE_NAME}`}</div>
      </div>
      <button onClick={() => setPaletteOpen(true)} aria-label="Open search (Ctrl+K)"
        className="mx-auto hidden h-9 w-full max-w-md items-center gap-2 rounded-md border border-line bg-panel-2 px-3 text-[12.5px] text-fg-3 transition-colors hover:border-line-strong md:flex">
        <Search className="size-4" aria-hidden /> Search cameras, alerts, models…
        <span className="ml-auto flex gap-0.5"><Kbd>Ctrl</Kbd><Kbd>K</Kbd></span>
      </button>
      <div className="ml-auto flex shrink-0 items-center gap-1.5 md:ml-0">
        <IconButton icon={Search} label="Search" className="md:hidden" onClick={() => setPaletteOpen(true)} />
        <Link href="/system" title={`${conn.label}${conn.hint ? ` · ${conn.hint}` : ''}\n${conn.detail}`} role="status"
          className={cx('hidden h-8 items-center gap-1.5 rounded-full border px-2.5 text-[12px] font-medium lg:flex', CONN_PILL[conn.tone])}>
          <span className={cx('size-1.5 rounded-full', CONN_DOT[conn.tone])} aria-hidden />{conn.short}
        </Link>
        {model && (
          <Link href="/model-lab" title={`Active model ${model.version} on ${model.device.toUpperCase()} · ${model.classes.length} classes`}
            className="hidden h-8 items-center gap-1.5 rounded-full border border-line bg-panel-2 px-2.5 text-[12px] text-fg-2 hover:border-line-strong hover:text-fg 2xl:flex">
            <Cpu className="size-3.5 text-exp" aria-hidden />
            <span className="max-w-[140px] truncate font-mono text-[11.5px]">{model.version}</span>
            <span className="text-[10.5px] uppercase text-fg-3">{model.device}</span>
          </Link>
        )}
        <Link href="/alerts" aria-label={`${newAlerts} new alerts`} title="New alerts"
          className={cx('flex h-8 items-center gap-1.5 rounded-full border px-2.5 text-[12px] font-medium',
            newAlerts > 0 ? 'border-crit/30 bg-crit-bg text-crit' : 'border-line bg-panel-2 text-fg-2 hover:text-fg')}>
          <AlertTriangle className="size-3.5" aria-hidden />
          <span className="tabular">{alerts.data ? newAlerts : '—'}</span>
          <span className="hidden sm:inline">new</span>
        </Link>
        <div className="relative">
          <IconButton icon={Bell} label={`Notifications (${unread} unread)`} onClick={() => setNotifOpen(true)} />
          {unread > 0 && <span className="pointer-events-none absolute right-0.5 top-0.5 flex min-w-4 items-center justify-center rounded-full bg-crit px-1 text-[9.5px] font-bold leading-4 text-white">{unread > 9 ? '9+' : unread}</span>}
        </div>
        <Link href="/settings" aria-label="Settings" title="Settings" className="hidden size-8 items-center justify-center rounded-md text-fg-2 hover:bg-hover hover:text-fg sm:inline-flex">
          <SettingsIcon className="size-4" aria-hidden />
        </Link>
        <span className="mx-1 hidden h-6 w-px bg-line sm:block" aria-hidden />
        <ProfileMenu />
      </div>
      <NotificationCenter open={notifOpen} onClose={() => setNotifOpen(false)} />
    </header>
  )
}

// ---------------------------------------------------------------- shell
export function AppShell({ children }: { children: ReactNode }) {
  const { sidebarCollapsed, setSidebarCollapsed, mobileNavOpen, setMobileNavOpen, setPaletteOpen, paletteOpen } = useApp()
  const pathname = usePathname()
  useEffect(() => setMobileNavOpen(false), [pathname]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        setPaletteOpen(!paletteOpen)
      }
    }
    window.addEventListener('keydown', h)
    return () => window.removeEventListener('keydown', h)
  }, [paletteOpen, setPaletteOpen])

  return (
    <div className="flex h-dvh overflow-hidden bg-bg">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-[70] focus:rounded focus:bg-info focus:px-2 focus:py-1 focus:text-on-accent">Skip to content</a>
      <aside className={cx('relative hidden shrink-0 border-r border-line bg-panel transition-[width] duration-150 lg:block', sidebarCollapsed ? 'w-[60px]' : 'w-[240px]')}>
        <Sidebar collapsed={sidebarCollapsed} onToggle={() => setSidebarCollapsed(!sidebarCollapsed)} />
      </aside>
      {mobileNavOpen && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <div className="absolute inset-0 bg-black/55" onClick={() => setMobileNavOpen(false)} aria-hidden />
          <aside className="relative h-full w-[264px] border-r border-line bg-panel animate-in slide-in-from-left-8 duration-150">
            <IconButton icon={X} label="Close navigation" className="absolute right-2 top-3 z-10" onClick={() => setMobileNavOpen(false)} />
            <Sidebar collapsed={false} onNavigate={() => setMobileNavOpen(false)} />
          </aside>
        </div>
      )}
      <div className="flex min-w-0 flex-1 flex-col">
        <Topbar />
        <main id="main" className="min-h-0 flex-1 overflow-y-auto">
          <div className="mx-auto w-full max-w-[1800px] p-3 md:p-6">{children}</div>
        </main>
      </div>
      <CommandPalette />
      <Toaster />
    </div>
  )
}
