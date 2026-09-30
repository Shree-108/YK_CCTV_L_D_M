'use client'

import Link from 'next/link'
import { usePathname, useRouter } from 'next/navigation'
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import {
  ArrowRight, Bell, ChevronsLeft, ChevronsRight, CornerDownLeft, Menu, Search, Settings as SettingsIcon, X, Zap,
} from 'lucide-react'
import { Api } from '../../api/endpoints'
import type { SearchResult } from '../../api/types'
import { useQuery } from '../../hooks/useQuery'
import { ago, cx, className as cls, pct } from '../../lib/format'
import { useApp } from '../../stores/app'
import { Badge, Dot, IconButton, Kbd } from '../ui/primitives'
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

// ---------------------------------------------------------------- sidebar
function Sidebar({ collapsed, onNavigate }: { collapsed: boolean; onNavigate?: () => void }) {
  const pathname = usePathname()
  const badges = useBadges()
  const active = navFor(pathname)
  const { stream } = useApp()
  return (
    <nav aria-label="Primary" className="flex h-full flex-col">
      <div className={cx('flex h-16 shrink-0 items-center gap-3', collapsed ? 'justify-center px-0' : 'px-5')}>
        <div className="flex size-9 items-center justify-center rounded-xl bg-info text-white shadow-sm"><Zap className="size-5" aria-hidden /></div>
        {!collapsed && (
          <div className="min-w-0 leading-tight">
            <div className="truncate text-[14px] font-bold tracking-[0.06em] text-fg">SIGHTLINE</div>
            <div className="truncate text-[10px] font-medium uppercase tracking-[0.18em] text-info">Litter detection</div>
          </div>
        )}
      </div>
      {!collapsed && (
        <div className={cx('mx-4 mb-2 flex items-center gap-2 rounded-lg px-3 py-2 text-[12px]', stream === 'open' ? 'bg-ok-bg text-ok' : 'bg-warn-bg text-warn')}>
          <span className={cx('size-1.5 rounded-full', stream === 'open' ? 'live-dot bg-ok' : 'bg-warn')} aria-hidden />
          {stream === 'open' ? 'Connected to backend' : 'Connecting to backend…'}
        </div>
      )}
      <div className="min-h-0 flex-1 overflow-y-auto py-2">
        {NAV.filter((g) => !g.hidden).map((g) => {
          const open = true
          return (
          <div key={g.label} className="mb-2">
            {!collapsed && <div className="px-5 pb-1 pt-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-fg-3">{g.label}</div>}
            {collapsed && <div className="mx-3 my-2 border-t border-line" aria-hidden />}
            {(open || collapsed) && <ul>
              {g.items.map((it) => {
                const on = active?.href === it.href
                const b = it.badge ? badges[it.badge] : undefined
                return (
                  <li key={it.href} className="px-2">
                    <Link href={it.href} onClick={onNavigate} title={collapsed ? it.label : undefined}
                      aria-current={on ? 'page' : undefined}
                      className={cx('group relative flex items-center gap-3 rounded-lg transition-colors',
                        'h-10 text-[14px]',
                        collapsed ? 'justify-center' : 'px-3',
                        on ? 'bg-info-bg font-medium text-info' : 'text-fg-2 hover:bg-hover hover:text-fg')}>
                      <it.icon className={cx('size-4 shrink-0', on ? 'text-info' : 'text-fg-3 group-hover:text-fg-2')} aria-hidden />
                      {!collapsed && <span className="flex-1 truncate">{it.label}</span>}
                      {b !== undefined && (collapsed
                        ? <span className="absolute right-1 top-1 size-1.5 rounded-full bg-info" aria-label={`${b}`} />
                        : <span className={cx('tabular rounded-sm px-1.5 text-[11px] font-medium',
                            it.badge === 'newAlerts' ? 'bg-crit-bg text-crit' : it.badge === 'online' ? 'bg-ok-bg text-ok' : 'bg-panel-2 text-fg-3')}>
                            {it.badge === 'online' && <span className="live-dot mr-1 inline-block size-1.5 rounded-full bg-ok align-middle" aria-hidden />}{b}
                          </span>)}
                    </Link>
                  </li>
                )
              })}
            </ul>}
          </div>
          )
        })}
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
function Topbar() {
  const pathname = usePathname()
  const item = navFor(pathname)
  const { setPaletteOpen, setMobileNavOpen, unread, stream, liveCameras, isAdmin } = useApp()
  const health = useQuery('health', Api.health, { poll: 15000 })
  const alerts = useQuery('alerts:newcount', () => Api.alerts({ status: 'new', limit: 1 }), { poll: 15000 })
  const [notifOpen, setNotifOpen] = useState(false)
  const cams = Object.values(liveCameras)
  const online = cams.filter((c) => c.state === 'online').length
  const warn = cams.filter((c) => c.state !== 'online').length
  const sysTone = health.error ? 'crit' : health.data?.status === 'ok' ? 'ok' : health.data ? 'warn' : 'neutral'
  const sysText = health.error ? 'Backend offline' : health.data?.status === 'ok' ? 'Operational' : health.data ? 'Degraded' : 'Checking…'

  return (
    <header className="sticky top-0 z-30 flex h-12 shrink-0 items-center gap-3 border-b border-line bg-bg/90 px-3 backdrop-blur md:px-4">
      <IconButton icon={Menu} label="Open navigation" className="lg:hidden" onClick={() => setMobileNavOpen(true)} />
      <div className="min-w-0">
        <div className="truncate text-[10px] font-semibold uppercase tracking-[0.12em] text-fg-3">{(ALL_NAV.find((n) => n.href === item?.href)?.group ?? 'AI Operations')}</div>
        <div className="truncate text-[13px] font-semibold text-fg">{item?.label ?? 'AI Litter Operations'}</div>
      </div>
      <button onClick={() => setPaletteOpen(true)} aria-label="Open search (Ctrl+K)"
        className="mx-auto hidden h-8 w-full max-w-md items-center gap-2 rounded-md border border-line bg-panel px-2.5 text-[12px] text-fg-3 hover:border-line-strong md:flex">
        <Search className="size-3.5" aria-hidden /> Search cameras, alerts, models…
        <span className="ml-auto flex gap-0.5"><Kbd>Ctrl</Kbd><Kbd>K</Kbd></span>
      </button>
      <div className="ml-auto flex items-center gap-1.5 md:ml-0">
        <IconButton icon={Search} label="Search" className="md:hidden" onClick={() => setPaletteOpen(true)} />
        <Link href="/system" className="hidden items-center gap-1.5 rounded-md px-2 py-1 text-[12px] text-fg-2 hover:bg-hover lg:flex" title={health.data?.model_error ?? undefined}>
          <Dot tone={sysTone} pulse={sysTone === 'ok'} /> {sysText}
        </Link>
        <Link href="/alerts" className="flex items-center gap-1.5 rounded-md px-2 py-1 text-[12px] text-fg-2 hover:bg-hover" aria-label={`${alerts.data?.total ?? 0} new alerts`}>
          <span className="hidden sm:inline">Alerts</span>
          <span className={cx('tabular rounded-sm px-1.5 font-mono text-[12px]', (alerts.data?.total ?? 0) > 0 ? 'bg-crit-bg text-crit' : 'bg-panel-2 text-fg-3')}>
            {alerts.data ? String(alerts.data.total).padStart(2, '0') : '—'}
          </span>
        </Link>
        <div className="relative">
          <IconButton icon={Bell} label={`Notifications (${unread} unread)`} onClick={() => setNotifOpen(true)} />
          {unread > 0 && <span className="pointer-events-none absolute right-1 top-1 flex size-3.5 items-center justify-center rounded-full bg-crit text-[9px] font-bold text-white">{unread > 9 ? '9+' : unread}</span>}
        </div>
        <Link href="/settings" aria-label="Settings" title="Settings" className="inline-flex size-8 items-center justify-center rounded-md text-fg-2 hover:bg-hover hover:text-fg">
          <SettingsIcon className="size-4" aria-hidden />
        </Link>
        <Link href="/settings?tab=security" title={isAdmin ? 'Admin session active' : 'Operator (read + review). Sign in as admin in Settings → Security'}
          className="flex h-8 items-center gap-1.5 rounded-md border border-line px-2 text-[12px] text-fg-2 hover:border-line-strong">
          <span className={cx('flex size-5 items-center justify-center rounded-full text-[10px] font-semibold', isAdmin ? 'bg-exp-bg text-exp' : 'bg-panel-2 text-fg-3')}>{isAdmin ? 'AD' : 'OP'}</span>
          <span className="hidden sm:inline">{isAdmin ? 'Admin' : 'Operator'}</span>
        </Link>
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
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-[70] focus:rounded focus:bg-info focus:px-2 focus:py-1 focus:text-white">Skip to content</a>
      <aside className={cx('relative hidden shrink-0 border-r border-line bg-panel transition-[width] duration-150 lg:block', sidebarCollapsed ? 'w-[56px]' : 'w-[228px]')}>
        <Sidebar collapsed={sidebarCollapsed} />
        <button onClick={() => setSidebarCollapsed(!sidebarCollapsed)} aria-label={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          className="absolute bottom-3 right-2 inline-flex size-7 items-center justify-center rounded-md text-fg-3 hover:bg-hover hover:text-fg">
          {sidebarCollapsed ? <ChevronsRight className="size-4" /> : <ChevronsLeft className="size-4" />}
        </button>
      </aside>
      {mobileNavOpen && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <div className="absolute inset-0 bg-black/55" onClick={() => setMobileNavOpen(false)} aria-hidden />
          <aside className="relative h-full w-[260px] border-r border-line bg-panel animate-in slide-in-from-left-8 duration-150">
            <IconButton icon={X} label="Close navigation" className="absolute right-2 top-2 z-10" onClick={() => setMobileNavOpen(false)} />
            <Sidebar collapsed={false} onNavigate={() => setMobileNavOpen(false)} />
          </aside>
        </div>
      )}
      <div className="flex min-w-0 flex-1 flex-col">
        <Topbar />
        <main id="main" className="min-h-0 flex-1 overflow-y-auto">
          <div className="mx-auto w-full max-w-[1800px] p-3 md:p-5">{children}</div>
        </main>
      </div>
      <CommandPalette />
      <Toaster />
    </div>
  )
}
