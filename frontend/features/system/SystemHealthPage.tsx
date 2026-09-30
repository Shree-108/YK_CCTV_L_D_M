'use client'

import { Activity, Cpu, Database, HardDrive, MemoryStick, Radio, Server, Sparkles, Bell, Microchip } from 'lucide-react'
import { Api } from '../../api/endpoints'
import { useQuery } from '../../hooks/useQuery'
import { dateTime, duration, num } from '../../lib/format'
import { useApp } from '../../stores/app'
import { Badge, PageHeader, Panel, StatusBadge } from '../../components/ui/primitives'
import { DataTable } from '../../components/ui/table'
import { EmptyState, ErrorState, LoadingState } from '../../components/ui/states'

const ICONS: Record<string, typeof Server> = {
  backend: Server, database: Database, inference: Sparkles, camera_services: Radio, cpu: Cpu, memory: MemoryStick,
  storage: HardDrive, gpu: Microchip, notifications: Bell,
}

export default function SystemHealthPage() {
  const sys = useQuery('system:health', Api.systemHealth, { poll: 10000 })
  const health = useQuery('health', Api.health, { poll: 10000 })
  const events = useQuery('events:system', () => Api.events(100), { poll: 15000 })
  const { stream, lastSummaryAt } = useApp()
  const h = health.data

  return (
    <div className="flex flex-col gap-3">
      <PageHeader title="System health" description="Live checks of every service. Each row shows when it was checked; nothing here is estimated."
        badge={sys.data && <StatusBadge status={sys.data.overall} />} />
      {sys.error && !sys.data ? <Panel><ErrorState error={sys.error} onRetry={sys.refetch} title="Backend offline" /></Panel> : !sys.data ? <Panel><LoadingState rows={6} /></Panel> : (
        <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
          {sys.data.components.map((c) => {
            const Icon = ICONS[c.component] ?? Activity
            const pctv = typeof c.percent === 'number' ? (c.percent as number) : null
            return (
              <section key={c.component} className="flex flex-col gap-2 rounded-lg border border-line bg-panel p-3.5">
                <div className="flex items-center justify-between gap-2">
                  <span className="flex items-center gap-2 text-[13px] font-medium capitalize text-fg"><Icon className="size-4 text-fg-3" aria-hidden />{c.component.replace('_', ' ')}</span>
                  <StatusBadge status={c.status} />
                </div>
                <p className="text-[12px] text-fg-2">{c.detail}</p>
                {pctv !== null && (
                  <div className="h-1.5 overflow-hidden rounded-full bg-panel-2" role="progressbar" aria-valuenow={pctv} aria-valuemin={0} aria-valuemax={100} aria-label={`${c.component} usage`}>
                    <div className="h-full rounded-full" style={{ width: `${pctv}%`, background: pctv > 90 ? 'var(--crit)' : pctv > 75 ? 'var(--warn)' : 'var(--ok)' }} />
                  </div>
                )}
                {Array.isArray(c.cameras) && (c.cameras as { id: string; state: string; last_error: string | null }[]).length > 0 && (
                  <ul className="flex flex-col gap-1 text-[12px]">
                    {(c.cameras as { id: string; state: string; last_error: string | null }[]).map((cam) => (
                      <li key={cam.id} className="flex items-center justify-between gap-2"><span className="font-mono">{cam.id}</span><span className="truncate text-fg-3" title={cam.last_error ?? ''}>{cam.last_error ?? ''}</span><StatusBadge status={cam.state} /></li>
                    ))}
                  </ul>
                )}
                <span className="mt-auto text-[11px] text-fg-3">checked {dateTime(c.checked_at)}{typeof c.latency_ms === 'number' ? ` · ${c.latency_ms} ms` : ''}{typeof c.size_mb === 'number' ? ` · ${c.size_mb} MB` : ''}</span>
              </section>
            )
          })}
        </div>
      )}
      <div className="grid gap-3 lg:grid-cols-3">
        <Panel title="Runtime">
          {!h ? <LoadingState /> : (
            <dl className="grid grid-cols-2 gap-y-2 text-[12.5px]">
              <dt className="text-fg-3">Uptime</dt><dd className="text-fg">{duration(h.uptime_sec)}</dd>
              <dt className="text-fg-3">Server time</dt><dd className="text-fg">{dateTime(h.time)}</dd>
              <dt className="text-fg-3">Model</dt><dd className="font-mono text-fg">{h.model?.version ?? '—'}</dd>
              <dt className="text-fg-3">Device</dt><dd className="font-mono text-fg">{h.model?.device ?? '—'}</dd>
              <dt className="text-fg-3">Cameras</dt><dd className="text-fg">{h.cameras.online}/{h.cameras.running} running online · {h.cameras.configured} configured</dd>
              <dt className="text-fg-3">Event stream</dt><dd className="text-fg">{stream}{lastSummaryAt ? ` · last ${dateTime(lastSummaryAt)}` : ''}</dd>
              <dt className="text-fg-3">Admin API</dt><dd className="text-fg">{h.admin_enabled ? 'enabled (token set)' : 'disabled (no ADMIN_API_TOKEN)'}</dd>
            </dl>
          )}
        </Panel>
        <Panel title="Hardware & recommendation">
          {!h ? <LoadingState /> : (
            <div className="flex flex-col gap-2 text-[12.5px]">
              <div className="flex justify-between"><span className="text-fg-3">CUDA GPU</span><span className="text-fg">{h.hardware.cuda ? `${h.hardware.gpu} (${h.hardware.vram_gb} GB)` : 'not available'}</span></div>
              <div className="flex justify-between"><span className="text-fg-3">CPU threads</span><span className="text-fg">{h.hardware.cpu_count}</span></div>
              <div className="flex justify-between"><span className="text-fg-3">CPU / RAM now</span><span className="text-fg">{num(h.system.cpu_percent)}% / {num(h.system.memory_percent)}%</span></div>
              <div className="mt-1 rounded-md border border-line bg-panel-2 p-2 text-[12px]">
                <div className="mb-1 text-[11px] uppercase tracking-wide text-fg-3">Practical model for this hardware</div>
                {Object.entries(h.hardware.recommended).map(([k, v]) => <div key={k} className="flex justify-between"><span className="text-fg-3">{k.replace('_', ' ')}</span><span className="font-mono text-fg">{String(v)}</span></div>)}
              </div>
            </div>
          )}
        </Panel>
        <Panel title="Notification providers">
          {!h ? <LoadingState /> : (
            <ul className="flex flex-col gap-2 text-[12.5px]">
              {h.notifications.map((n) => <li key={n.provider} className="flex items-center justify-between"><span className="capitalize text-fg">{n.provider}</span>{n.configured ? <Badge tone="ok" dot>configured</Badge> : <Badge>not configured</Badge>}</li>)}
              <li className="text-[11px] text-fg-3">Configure via environment variables on the server (see .env.example).</li>
            </ul>
          )}
        </Panel>
      </div>
      <Panel title="System events" noPad>
        {events.error && !events.data ? <ErrorState error={events.error} compact /> : (
          <DataTable rows={events.data?.events ?? []} rowKey={(e) => e.id} dense pageSize={20}
            empty={<EmptyState compact title="No events" />}
            columns={[
              { key: 't', header: 'Time', cell: (e) => dateTime(e.ts), sortValue: (e) => e.ts },
              { key: 'l', header: 'Level', cell: (e) => <Badge tone={e.level === 'error' ? 'crit' : e.level === 'warning' ? 'warn' : 'neutral'}>{e.level}</Badge> },
              { key: 's', header: 'Source', cell: (e) => e.source },
              { key: 'c', header: 'Camera', cell: (e) => <span className="font-mono">{e.camera_id ?? '—'}</span> },
              { key: 'm', header: 'Message', cell: (e) => <span className="text-fg">{e.message}</span> },
            ]} />
        )}
      </Panel>
    </div>
  )
}
