'use client'

import Link from 'next/link'
import { useEffect, useMemo, useState } from 'react'
import { Cctv, CheckCircle2, Plug, PlugZap, Save, ShieldCheck, Square, XCircle } from 'lucide-react'
import { url } from '../../api/client'
import { Api } from '../../api/endpoints'
import type { Camera, CctvTestResult, RtspConfig } from '../../api/types'
import { invalidate, useQuery } from '../../hooks/useQuery'
import { cx, num, pct, time } from '../../lib/format'
import { useApp } from '../../stores/app'
import { Badge, Button, Field, Input, PageHeader, Panel, Select, type Tone } from '../../components/ui/primitives'
import { MetricCard } from '../../components/ui/kpi'
import { DataTable } from '../../components/ui/table'
import { EmptyState } from '../../components/ui/states'
import { RoiSetup, litterName } from '../detection/common'
import { SchedulePanel } from './SchedulePanel'

const NEW = '__new__'

interface Form {
  id: string; name: string; host: string; port: string; channel: string; stream_type: 'main' | 'sub'
  username: string; password: string; path_template: string
  processing_fps: string; frame_skip: string; confidence: string; image_size: string
}

const EMPTY: Form = { id: 'cctv_cam1', name: '', host: '', port: '554', channel: '1', stream_type: 'sub', username: '', password: '',
  path_template: '', processing_fps: '5', frame_skip: '0', confidence: '', image_size: '' }

/** Connection state shown to operators (the backend worker drives it). */
function liveState(c: Camera | undefined): { label: string; tone: Tone; pulse?: boolean } {
  if (!c) return { label: 'CCTV configuration required', tone: 'neutral' }
  if (!c.enabled) return { label: 'Stopped', tone: 'neutral' }
  switch (c.status.state) {
    case 'online': return { label: 'CONNECTED', tone: 'ok', pulse: true }
    case 'connecting': return { label: 'CONNECTING…', tone: 'info', pulse: true }
    case 'reconnecting': return { label: (c.status.reconnects ?? 0) > 1 ? 'CONNECTION LOST — RECONNECTING…' : 'CONNECTING…', tone: 'warn', pulse: true }
    default: return { label: c.status.state.toUpperCase(), tone: 'crit' }
  }
}

function formFrom(c: Camera): Form {
  const s = c.settings as { video?: { live_process_fps?: number; live_frame_skip?: number }; confidence?: { confirmed_threshold?: number }; inference?: { image_size?: number } }
  return {
    id: c.id, name: c.name, host: c.rtsp?.host ?? '', port: String(c.rtsp?.port ?? 554), channel: String(c.rtsp?.channel ?? 1),
    stream_type: c.rtsp?.stream_type ?? 'sub', username: c.rtsp?.username ?? '', password: '', path_template: c.rtsp?.path_template ?? '',
    processing_fps: String(s.video?.live_process_fps ?? ''), frame_skip: String(s.video?.live_frame_skip ?? ''),
    confidence: String(s.confidence?.confirmed_threshold ?? ''), image_size: String(s.inference?.image_size ?? ''),
  }
}

export default function LiveCctvPage() {
  const { isAdmin, toast } = useApp()
  const cams = useQuery('cameras:list', Api.cameras, { poll: 2000 })
  const defaults = useQuery('cctv:defaults', Api.cctvDefaults)
  const cctvCams = useMemo(() => (cams.data?.cameras ?? []).filter((c) => c.source.startsWith('cctv:')), [cams.data])
  // sample-video / file cameras: scheduler testing before the authorised RTSP details are available
  const testCams = useMemo(() => (cams.data?.cameras ?? []).filter((c) => !c.source.startsWith('cctv:')), [cams.data])
  const [sel, setSel] = useState<string>(NEW)
  const testCam = testCams.find((c) => c.id === sel)
  const cam = cctvCams.find((c) => c.id === sel) ?? testCam
  const [form, setForm] = useState<Form>(EMPTY)
  const [test, setTest] = useState<CctvTestResult | null>(null)
  const [busy, setBusy] = useState<string | null>(null)

  // pick the first configured camera; prefill a new form from non-secret CCTV_* env defaults
  useEffect(() => { if (sel === NEW && cctvCams.length) setSel(cctvCams[0].id) }, [cctvCams.length]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (cam && !testCam) setForm(formFrom(cam))
    else if (sel === NEW) {
      const d = defaults.data?.defaults ?? {}
      setForm({ ...EMPTY, host: d.host ?? '', port: String(d.port ?? 554), channel: String(d.channel ?? 1),
        stream_type: (d.stream_type as Form['stream_type']) ?? 'sub', username: d.username ?? '', path_template: d.path_template ?? '' })
    }
    setTest(null)
  }, [sel, cam?.updated_at, defaults.data]) // eslint-disable-line react-hooks/exhaustive-deps

  const set = (k: keyof Form) => (e: { target: { value: string } }) => setForm((f) => ({ ...f, [k]: e.target.value }))
  const rtsp = (): RtspConfig => ({ host: form.host.trim(), port: Number(form.port) || 554, channel: Number(form.channel) || 1,
    stream_type: form.stream_type, username: form.username.trim(), path_template: form.path_template.trim() || (defaults.data?.path_template ?? '') })

  const run = async (what: string, fn: () => Promise<unknown>) => {
    setBusy(what)
    try { return await fn() } catch (e) { toast({ tone: 'crit', title: `${what} failed`, body: (e as Error).message }) } finally { setBusy(null) }
  }
  const save = () => run('Save', async () => {
    const c = await Api.cctvSave(form.id.trim(), {
      name: form.name.trim() || form.id, rtsp: rtsp(), password: form.password || undefined,
      processing_fps: form.processing_fps ? Number(form.processing_fps) : undefined, frame_skip: form.frame_skip ? Number(form.frame_skip) : undefined,
      confidence: form.confidence ? Number(form.confidence) : undefined, image_size: form.image_size ? Number(form.image_size) : undefined,
    })
    setForm((f) => ({ ...f, password: '' }))       // never keep the password in the page after saving
    invalidate('cameras:')
    setSel(c.id)
    toast({ tone: 'ok', title: 'CCTV configuration saved', body: c.has_password ? 'Password stored in the server .env (not the database).' : 'No password stored yet.' })
    return c
  })
  const testConn = () => run('Test connection', async () => {
    setTest(null)
    setTest(await Api.cctvTest({ camera_id: form.id.trim(), rtsp: rtsp(), password: form.password || undefined }))
  })
  const connect = () => run('Connect', async () => { if (form.password || !cam) await save(); await Api.cctvConnect(form.id.trim()); invalidate('cameras:') })
  const stop = () => run('Stop', async () => { await Api.cctvStop(form.id.trim()); invalidate('cameras:') })

  const st = liveState(cam)
  const s = cam?.status
  const live = !!cam?.enabled && (s?.state === 'online' || s?.state === 'reconnecting')
  const configured = !!cam?.rtsp?.host

  return (
    <div className="flex flex-col gap-3">
      <PageHeader title="Live CCTV"
        description="Authenticated RTSP from an authorized NVR/camera on the local network or VPN. Live frames go through the same YOLO model, ROI filter, tracking and alert rules as Image and Video Analysis." />
      {!isAdmin && (
        <div className="flex items-center gap-2 rounded-md border border-warn/40 bg-warn-bg px-3 py-2 text-[12px] text-warn">
          <ShieldCheck className="size-4" aria-hidden /> Configuring and starting a CCTV stream needs the admin token.
          <Link href="/settings?tab=security" className="underline">Sign in as admin</Link>
        </div>
      )}
      <div className="grid grid-cols-1 gap-3 xl:grid-cols-[minmax(0,1fr)_360px]">
        <div className="flex min-w-0 flex-col gap-3">
          <Panel title={cam ? cam.name : 'Live CCTV'} icon={Cctv}
            subtitle={testCam ? 'Test source — recorded video, not the NVR' : cam?.rtsp_url ?? 'CCTV configuration required'}
            actions={<Badge tone={st.tone} dot className={cx(st.pulse && 'animate-pulse')}>{st.label}</Badge>}>
            {live ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img key={`${cam!.id}-${cam!.status.reconnects}`} src={url(`/api/cameras/${encodeURIComponent(cam!.id)}/stream`)}
                alt={`Live CCTV ${cam!.name}`} className="aspect-video w-full rounded-md bg-video object-contain" />
            ) : (
              <div className="flex aspect-video flex-col items-center justify-center gap-2 rounded-md border border-line bg-video text-center">
                <Cctv className="size-8 text-fg-3" aria-hidden />
                <p className="text-[13px] text-fg">{!cam ? 'CCTV configuration required' : !cam.enabled ? 'Stream stopped' : 'Connecting to the NVR…'}</p>
                <p className="max-w-md text-[12px] text-fg-3">
                  {!cam ? 'Enter the NVR details provided by your CCTV administrator, then Test Connection. Image and Video Analysis keep working without it.'
                    : !cam.enabled ? (testCam ? 'Live view is off for this test source; scheduled screenshots below still work.' : 'Click “Connect Live CCTV” to start live litter detection.') : (s?.last_error ?? 'Waiting for the first frame.')}
                </p>
              </div>
            )}
            {cam && (
              <div className="mt-3 grid grid-cols-2 gap-2 md:grid-cols-6">
                <MetricCard label="Litter detected" value={num(s?.summary?.confirmed_litter ?? 0)} tone={(s?.summary?.confirmed_litter ?? 0) > 0 ? 'crit' : undefined} hint="confirmed, in view" />
                <MetricCard label="Capture FPS" value={num(s?.capture_fps, 1)} hint="from the NVR" />
                <MetricCard label="Processing FPS" value={num(s?.fps, 1)} hint="analysed by YOLO" />
                <MetricCard label="Inference" value={s?.inference_ms ? `${num(s.inference_ms, 0)} ms` : '—'} />
                <MetricCard label="Dropped frames" value={num(s?.frames_dropped)} hint="stale frames skipped" />
                <MetricCard label="Reconnects" value={num(s?.reconnects)} tone={(s?.reconnects ?? 0) > 1 ? 'warn' : undefined} />
              </div>
            )}
            {cam && !testCam && (
              <div className="mt-3 flex flex-wrap justify-end gap-2">
                {cam.enabled
                  ? <Button variant="danger" icon={Square} disabled={!isAdmin} loading={busy === 'Stop'} onClick={stop}>Disconnect</Button>
                  : <Button variant="primary" icon={PlugZap} disabled={!isAdmin || !configured} loading={busy === 'Connect'} onClick={connect}>Connect Live CCTV</Button>}
              </div>
            )}
          </Panel>
          {cam && <SchedulePanel cameraId={cam.id} cameraState={testCam ? `${st.label} (test source)` : st.label} />}
          {cam && s?.state === 'online' && <LiveRoi cam={cam} />}
          {cam && <LiveEvents cameraId={cam.id} />}
        </div>

        <Panel title="CCTV connection" icon={Plug}>
          <div className="flex flex-col gap-3">
            <Field label="Camera">
              <Select value={sel} onChange={(e) => setSel(e.target.value)}>
                {cctvCams.map((c) => <option key={c.id} value={c.id}>{c.name} ({c.id})</option>)}
                <option value={NEW}>+ New CCTV camera</option>
                {testCams.length > 0 && <optgroup label="Test sources (recorded video — not CCTV)">
                  {testCams.map((c) => <option key={c.id} value={c.id}>{c.name} ({c.id})</option>)}
                </optgroup>}
              </Select>
            </Field>
            {testCam ? (
              <p className="rounded-md border border-warn/40 bg-warn-bg px-2 py-1.5 text-[12px] text-warn">
                TEST SOURCE — this camera reads a recorded video file, not the NVR. Use it only to try the scheduler and models.
              </p>
            ) : <>
            <div className="grid grid-cols-2 gap-2">
              <Field label="Camera ID"><Input value={form.id} onChange={set('id')} disabled={!!cam} placeholder="cctv_cam1" /></Field>
              <Field label="Camera Name"><Input value={form.name} onChange={set('name')} placeholder="IFFCO CHOWK - CAM 1" /></Field>
            </div>
            <Field label="NVR/Camera IP or Host" hint="The LAN/VPN IP from your CCTV administrator — not the device serial/PSN.">
              <Input value={form.host} onChange={set('host')} placeholder="e.g. 192.168.x.x" autoComplete="off" />
            </Field>
            <div className="grid grid-cols-3 gap-2">
              <Field label="RTSP Port"><Input value={form.port} onChange={set('port')} inputMode="numeric" /></Field>
              <Field label="Channel"><Input value={form.channel} onChange={set('channel')} inputMode="numeric" /></Field>
              <Field label="Stream Type">
                <Select value={form.stream_type} onChange={(e) => setForm((f) => ({ ...f, stream_type: e.target.value as Form['stream_type'] }))}>
                  <option value="main">Main Stream</option><option value="sub">Sub Stream</option>
                </Select>
              </Field>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <Field label="Username"><Input value={form.username} onChange={set('username')} autoComplete="off" /></Field>
              <Field label="Password" hint={cam?.has_password ? 'Stored on the server — leave empty to keep it.' : undefined}>
                <Input type="password" value={form.password} onChange={set('password')} autoComplete="new-password"
                  placeholder={cam?.has_password ? '••••••••' : ''} />
              </Field>
            </div>
            <details className="rounded-md border border-line px-2 py-1.5">
              <summary className="cursor-pointer select-none text-[12px] text-fg-2">RTSP path & processing</summary>
              <div className="mt-2 flex flex-col gap-2">
                <Field label="RTSP path" hint={defaults.data?.template_note}>
                  <Input value={form.path_template} onChange={set('path_template')} placeholder={defaults.data?.path_template} className="font-mono text-[11px]" />
                </Field>
                <div className="grid grid-cols-2 gap-2">
                  <Field label="Processing FPS" hint="frames/s sent to YOLO"><Input value={form.processing_fps} onChange={set('processing_fps')} inputMode="decimal" /></Field>
                  <Field label="Frame skip" hint="analyse every (n+1)-th frame"><Input value={form.frame_skip} onChange={set('frame_skip')} inputMode="numeric" /></Field>
                  <Field label="YOLO confidence" hint="empty = system setting"><Input value={form.confidence} onChange={set('confidence')} inputMode="decimal" placeholder="0.60" /></Field>
                  <Field label="Image size" hint="empty = system setting"><Input value={form.image_size} onChange={set('image_size')} inputMode="numeric" placeholder="640" /></Field>
                </div>
              </div>
            </details>
            <div className="flex flex-wrap gap-2">
              <Button icon={Plug} disabled={!isAdmin || !form.host} loading={busy === 'Test connection'} onClick={testConn}>Test Connection</Button>
              <Button icon={Save} disabled={!isAdmin || !form.host} loading={busy === 'Save'} onClick={save}>Save</Button>
              {!cam?.enabled && <Button variant="primary" icon={PlugZap} disabled={!isAdmin || !form.host} loading={busy === 'Connect'} onClick={connect}>Connect Live CCTV</Button>}
            </div>
            {test && <TestResult r={test} />}
            </>}
            <p className="text-[11px] text-fg-3">Credentials are never shown, logged or stored in the database. Use the NVR only over the local network or a VPN — do not expose it to the internet.</p>
          </div>
        </Panel>
      </div>
    </div>
  )
}

function TestResult({ r }: { r: CctvTestResult }) {
  return (
    <div className={cx('rounded-md border px-3 py-2 text-[12px]', r.ok ? 'border-ok/40 bg-ok-bg' : 'border-crit/40 bg-crit-bg')} role="status">
      <p className={cx('flex items-center gap-1.5 font-semibold', r.ok ? 'text-ok' : 'text-crit')}>
        {r.ok ? <CheckCircle2 className="size-4" aria-hidden /> : <XCircle className="size-4" aria-hidden />}
        {r.ok ? 'CONNECTED' : r.code.replace(/_/g, ' ').toUpperCase()}
      </p>
      <p className="mt-0.5 text-fg">{r.message}</p>
      {r.ok && r.resolution && <p className="text-fg-2">{r.resolution.join('×')} · {num(r.fps, 1)} fps received</p>}
      <ul className="mt-1.5 flex flex-col gap-0.5 text-fg-2">
        {r.stages.map((s) => <li key={s.stage}>{s.ok ? '✓' : '✗'} {s.stage}: {s.detail}</li>)}
      </ul>
      {r.url && <p className="mt-1 break-all font-mono text-[10.5px] text-fg-3">{r.url}</p>}
    </div>
  )
}

/** Per-camera outdoor ROI, drawn on a real frame from the stream (same editor as uploads). */
function LiveRoi({ cam }: { cam: Camera }) {
  const [frame, setFrame] = useState<{ src: string; w: number; h: number } | null>(null)
  useEffect(() => {
    const u = url(`/api/cameras/${encodeURIComponent(cam.id)}/frame.jpg?raw=1&t=${Date.now()}`)
    const img = new Image()
    img.onload = () => setFrame({ src: u, w: img.naturalWidth, h: img.naturalHeight })
    img.src = u
  }, [cam.id])
  return (
    <Panel title="Detection ROI" subtitle="Only litter inside the outdoor ground area generates detections and events">
      {frame ? <RoiSetup cameraId={cam.id} src={frame.src} width={frame.w} height={frame.h} /> : <p className="text-[12px] text-fg-3">Loading a frame…</p>}
    </Panel>
  )
}

function LiveEvents({ cameraId }: { cameraId: string }) {
  const alerts = useQuery(`alerts:live:${cameraId}`, () => Api.alerts({ camera_id: cameraId, limit: 10 }), { poll: 5000 })
  return (
    <Panel title="Recent live events" noPad>
      <DataTable rows={alerts.data?.alerts ?? []} rowKey={(a) => a.id} dense
        empty={<EmptyState compact title="No litter events yet" body="An event is created once per confirmed litter object (temporal confirmation + de-duplication)." />}
        columns={[
          { key: 't', header: 'Time', cell: (a) => <span className="font-mono">{time(a.ts)}</span> },
          { key: 'o', header: 'Object', cell: (a) => <span className="text-fg">{litterName(a.class)}</span> },
          { key: 'c', header: 'Confidence', align: 'right', cell: (a) => pct(a.confidence, 0) },
          { key: 's', header: 'Status', cell: (a) => <Badge tone={a.status === 'new' ? 'warn' : 'neutral'}>{a.status.replace('_', ' ')}</Badge> },
          { key: 'e', header: 'Evidence', cell: (a) => a.snapshot_url ? <a className="text-info hover:underline" href={url(a.snapshot_url)} target="_blank" rel="noreferrer">frame</a> : '—' },
          { key: 'd', header: '', cell: (a) => <Link className="text-info hover:underline" href={`/alerts?id=${a.id}`}>#{a.id}</Link> },
        ]} />
    </Panel>
  )
}
