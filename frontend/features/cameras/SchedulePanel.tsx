'use client'

import { useEffect, useState } from 'react'
import { Camera as CameraIcon, Clock, Pause, Play, Save, Square, StepForward } from 'lucide-react'
import { url } from '../../api/client'
import { Api } from '../../api/endpoints'
import type { Capture, ScheduleConfig } from '../../api/types'
import { invalidate, useQuery } from '../../hooks/useQuery'
import { cx, dateTime, num, pct } from '../../lib/format'
import { useApp } from '../../stores/app'
import { Badge, Button, Field, Input, Panel, Select, type Tone } from '../../components/ui/primitives'
import { EmptyState } from '../../components/ui/states'
import { litterName } from '../detection/common'

const STATE: Record<string, { label: string; tone: Tone }> = {
  running: { label: 'Running', tone: 'ok' }, paused: { label: 'Paused', tone: 'warn' }, stopped: { label: 'Stopped', tone: 'neutral' },
}
const CAP_TONE: Record<string, Tone> = { done: 'ok', skipped: 'warn', failed: 'crit', interrupted: 'crit', running: 'info', invalid_frame: 'warn' }

function countdown(sec: number) {
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60
  return h ? `${h}h ${String(m).padStart(2, '0')}m` : `${m}m ${String(s).padStart(2, '0')}s`
}

/** Random-interval screenshot capture for one camera: controls, next capture, history. */
export function SchedulePanel({ cameraId, cameraState }: { cameraId: string; cameraState: string }) {
  const { isAdmin, toast } = useApp()
  const sched = useQuery(`schedule:${cameraId}`, () => Api.schedule(cameraId), { poll: 5000 })
  const caps = useQuery(`captures:${cameraId}`, () => Api.captures(cameraId, 12), { poll: 10000 })
  const s = sched.data
  const [cfg, setCfg] = useState<Pick<ScheduleConfig, 'min_minutes' | 'max_minutes' | 'task'>>({ min_minutes: 120, max_minutes: 180, task: 'litter' })
  const [busy, setBusy] = useState<string | null>(null)
  const [left, setLeft] = useState<number | null>(null)
  const [open, setOpen] = useState<Capture | null>(null)

  useEffect(() => { if (s) setCfg({ min_minutes: s.config.min_minutes, max_minutes: s.config.max_minutes, task: s.config.task }) },
    [s?.config.min_minutes, s?.config.max_minutes, s?.config.task]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {   // live countdown between polls
    const tick = () => setLeft(s?.state === 'running' && s.next_run_at ? Math.max(0, Math.round((Date.parse(s.next_run_at) - Date.now()) / 1000)) : null)
    tick()
    const t = setInterval(tick, 1000)
    return () => clearInterval(t)
  }, [s?.state, s?.next_run_at])

  const act = async (action: 'start' | 'pause' | 'resume' | 'stop' | 'capture-now') => {
    setBusy(action)
    try {
      const r = await Api.scheduleAction(cameraId, action)
      if (r.capture) {
        const c = r.capture
        toast({ tone: c.status === 'done' ? 'ok' : c.status === 'skipped' ? 'warn' : 'crit', title: `Capture ${c.status}`,
          body: c.status === 'done' ? `${c.counts.confirmed ?? 0} litter object(s) detected` : c.error ?? undefined })
      }
      invalidate(`schedule:${cameraId}`); invalidate(`captures:${cameraId}`); invalidate('alerts')
    } catch (e) {
      toast({ tone: 'crit', title: `${action} failed`, body: (e as Error).message })
    } finally { setBusy(null) }
  }
  const save = async () => {
    setBusy('save')
    try { await Api.saveSchedule(cameraId, cfg); invalidate(`schedule:${cameraId}`); toast({ tone: 'ok', title: 'Schedule saved' }) }
    catch (e) { toast({ tone: 'crit', title: 'Could not save schedule', body: (e as Error).message }) } finally { setBusy(null) }
  }

  const st = STATE[s?.state ?? 'stopped']
  return (
    <Panel title="Scheduled screenshots" icon={Clock}
      subtitle="One screenshot at a random time between the minimum and maximum wait, analysed by this camera's model"
      actions={<Badge tone={st.tone} dot>{st.label}</Badge>}>
      <div className="grid grid-cols-1 gap-3 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <div className="flex flex-col gap-2 text-[12.5px]">
          <div className="grid grid-cols-2 gap-x-3 gap-y-1.5">
            <span className="text-fg-3">Camera status</span><span className="text-fg">{cameraState}</span>
            <span className="text-fg-3">Next capture</span>
            <span className="text-fg">{s?.state === 'running' && s.next_run_at
              ? <>{dateTime(s.next_run_at)} <span className="font-mono text-info">· in {left != null ? countdown(left) : '—'}</span></>
              : s?.state === 'paused' ? 'Paused' : 'Not scheduled'}</span>
            <span className="text-fg-3">Last successful</span>
            <span className="text-fg">{s?.last_success_at ? dateTime(s.last_success_at) : '—'}</span>
            <span className="text-fg-3">Last attempt</span>
            <span className="text-fg">{s?.last_run_at ? <>{dateTime(s.last_run_at)} <Badge tone={CAP_TONE[s.last_status ?? ''] ?? 'neutral'}>{s.last_status}</Badge></> : '—'}</span>
            {s?.busy && <><span className="text-fg-3">Now</span><span className="text-info">Capturing…</span></>}
            {!!s?.consecutive_failures && <><span className="text-fg-3">Failures in a row</span><span className="text-crit">{s.consecutive_failures} — retrying every {s.config.retry_minutes} min</span></>}
          </div>
          {s?.last_error && <p className="rounded-md border border-crit/30 bg-crit-bg px-2 py-1 text-[11.5px] text-crit">{s.last_error}</p>}
          {s && !s.model_available && <p className="rounded-md border border-warn/30 bg-warn-bg px-2 py-1 text-[11.5px] text-warn">{s.model_note}</p>}
          <div className="mt-1 flex flex-wrap gap-1.5">
            {s?.state === 'stopped' && <Button size="sm" variant="primary" icon={Play} disabled={!isAdmin} loading={busy === 'start'} onClick={() => act('start')}>Start</Button>}
            {s?.state === 'running' && <Button size="sm" icon={Pause} disabled={!isAdmin} loading={busy === 'pause'} onClick={() => act('pause')}>Pause</Button>}
            {s?.state === 'paused' && <Button size="sm" variant="primary" icon={Play} disabled={!isAdmin} loading={busy === 'resume'} onClick={() => act('resume')}>Resume</Button>}
            {s && s.state !== 'stopped' && <Button size="sm" variant="ghost" icon={Square} disabled={!isAdmin} loading={busy === 'stop'} onClick={() => act('stop')}>Stop</Button>}
            <Button size="sm" icon={CameraIcon} disabled={!isAdmin || !!s?.busy} loading={busy === 'capture-now'} onClick={() => act('capture-now')}>Capture Now</Button>
          </div>
        </div>
        <div className="grid min-w-0 grid-cols-3 gap-2">
          <Field label="Min wait (min)"><Input inputMode="numeric" value={cfg.min_minutes} onChange={(e) => setCfg((c) => ({ ...c, min_minutes: Number(e.target.value) || 0 }))} /></Field>
          <Field label="Max wait (min)"><Input inputMode="numeric" value={cfg.max_minutes} onChange={(e) => setCfg((c) => ({ ...c, max_minutes: Number(e.target.value) || 0 }))} /></Field>
          <Field label="Model">
            <Select value={cfg.task} onChange={(e) => setCfg((c) => ({ ...c, task: e.target.value as ScheduleConfig['task'] }))}>
              <option value="litter">Outdoor · litter</option><option value="uniform">Indoor · uniform</option>
            </Select>
          </Field>
          <div className="col-span-3 flex items-center justify-between gap-2">
            <p className="text-[11px] text-fg-3">Random time between min and max (default 120–180 min). Same value in both = fixed interval (e.g. 45 or 120). Failed captures retry after {s?.config.retry_minutes ?? 10} min.</p>
            <Button size="sm" icon={Save} disabled={!isAdmin} loading={busy === 'save'} onClick={save}>Save</Button>
          </div>
        </div>
      </div>

      <h3 className="mb-2 mt-4 text-[11px] font-semibold uppercase tracking-wide text-fg-3">Recent screenshots</h3>
      {(caps.data?.captures ?? []).length === 0 ? <EmptyState compact title="No screenshots yet" body="Click “Capture Now” or start the schedule." /> : (
        <div className="grid grid-cols-2 gap-2 md:grid-cols-3 xl:grid-cols-4">
          {caps.data!.captures.map((c) => (
            <button key={c.id} onClick={() => setOpen(c)} className="flex flex-col overflow-hidden rounded-md border border-line bg-panel-2 text-left hover:border-info">
              {c.annotated_url || c.original_url
                // eslint-disable-next-line @next/next/no-img-element
                ? <img src={url((c.annotated_url ?? c.original_url)!)} alt={`Screenshot ${c.ts}`} loading="lazy" className="aspect-video w-full bg-video object-cover" />
                : <div className="flex aspect-video items-center justify-center bg-video text-[11px] text-crit">no image</div>}
              <div className="flex flex-col gap-0.5 px-2 py-1.5 text-[11px]">
                <span className="flex items-center justify-between gap-1">
                  <span className="text-fg">{dateTime(c.ts)}</span><Badge tone={CAP_TONE[c.status] ?? 'neutral'}>{c.status}</Badge>
                </span>
                <span className="truncate text-fg-3">{c.trigger} · {c.status === 'done'
                  ? `${num(c.counts.confirmed ?? 0)} litter${c.counts.highest_confidence ? ` · max ${pct(c.counts.highest_confidence, 0)}` : ''}`
                  : c.error ?? ''}</span>
              </div>
            </button>
          ))}
        </div>
      )}
      {open && <CaptureDetail c={open} onClose={() => setOpen(null)} />}
    </Panel>
  )
}

function CaptureDetail({ c, onClose }: { c: Capture; onClose: () => void }) {
  const [kind, setKind] = useState<'annotated' | 'original'>(c.annotated_url ? 'annotated' : 'original')
  const src = kind === 'annotated' ? c.annotated_url : c.original_url
  const litter = c.detections.filter((d) => d.status === 'confirmed')
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4" role="dialog" aria-modal onClick={onClose}>
      <div className="flex max-h-full w-full max-w-4xl flex-col gap-2 overflow-auto rounded-lg border border-line bg-panel p-3" onClick={(e) => e.stopPropagation()}>
        <div className="flex flex-wrap items-center gap-2 text-[12.5px]">
          <span className="font-medium text-fg">{dateTime(c.ts)}</span>
          <Badge tone={CAP_TONE[c.status] ?? 'neutral'}>{c.status}</Badge>
          <span className="text-fg-3">{c.trigger} · {c.task} · {c.model_version ?? 'no model'}</span>
          <div className="ml-auto flex gap-1">
            {(['annotated', 'original'] as const).map((k) => (
              <button key={k} disabled={!(k === 'annotated' ? c.annotated_url : c.original_url)} onClick={() => setKind(k)}
                className={cx('h-7 rounded-md border px-2 text-[12px] disabled:opacity-40', kind === k ? 'border-info text-info' : 'border-line text-fg-2')}>{k}</button>
            ))}
            <button onClick={onClose} className="h-7 rounded-md border border-line px-2 text-[12px] text-fg-2">Close</button>
          </div>
        </div>
        {src
          // eslint-disable-next-line @next/next/no-img-element
          ? <img src={url(src)} alt={`${kind} screenshot`} className="w-full rounded-md bg-video object-contain" />
          : <p className="text-[12px] text-crit">{c.error}</p>}
        {c.error && c.status !== 'failed' && <p className="text-[12px] text-warn">{c.error}</p>}
        {litter.length > 0 && (
          <ul className="flex flex-wrap gap-1.5 text-[12px]">
            {litter.map((d, i) => <li key={i} className="rounded border border-line bg-panel-2 px-2 py-0.5 text-fg">{litterName(d.class)} {pct(d.confidence, 0)}</li>)}
          </ul>
        )}
        {c.alert_ids.length > 0 && <p className="text-[12px] text-fg-3">New alerts: {c.alert_ids.map((id) => `#${id}`).join(', ')}</p>}
      </div>
    </div>
  )
}
