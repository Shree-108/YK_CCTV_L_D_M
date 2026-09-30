'use client'

import { useRouter, useSearchParams } from 'next/navigation'
import { useEffect, useState } from 'react'
import { Camera as CameraIcon, LayoutGrid, List, Pencil, Plus, Power, Trash2 } from 'lucide-react'
import { url } from '../../api/client'
import { Api } from '../../api/endpoints'
import type { Camera, RoiZone } from '../../api/types'
import { invalidate, useQuery } from '../../hooks/useQuery'
import { ago, cx, num } from '../../lib/format'
import { useApp } from '../../stores/app'
import { Badge, Button, Field, Input, PageHeader, Panel, Segmented, Select, Slider, StatusBadge, Tabs, Toggle } from '../../components/ui/primitives'
import { ConfirmDialog, Drawer } from '../../components/ui/overlay'
import { DataTable } from '../../components/ui/table'
import { EmptyState, ErrorState, LoadingState } from '../../components/ui/states'
import { RoiEditor } from '../../components/vision/Overlay'
import { CameraDrawer, CameraTile, DEFAULT_OVERLAY } from './live'

export default function CamerasPage() {
  const params = useSearchParams()
  const router = useRouter()
  const { isAdmin, toast } = useApp()
  const cams = useQuery('cameras:list', Api.cameras, { poll: 10000 })
  const [view, setView] = useState<'cards' | 'list'>('cards')
  const [detail, setDetail] = useState<string | null>(null)
  const [del, setDel] = useState<Camera | null>(null)
  const editId = params.get('id')
  const isNew = params.get('new') === '1'
  const list = cams.data?.cameras ?? []
  const editing = isNew ? null : list.find((c) => c.id === editId) ?? null
  const closeEditor = () => router.replace('/cameras')

  const toggle = async (c: Camera) => {
    try {
      await Api.updateCamera(c.id, { enabled: !c.enabled })
      toast({ tone: 'ok', title: `${c.id} ${c.enabled ? 'disabled' : 'enabled'}` })
      invalidate('cameras')
    } catch (e) { toast({ tone: 'crit', title: 'Update failed', body: (e as Error).message }) }
  }
  const lock = !isAdmin ? 'Admin sign-in required (Settings → Security)' : undefined

  return (
    <div className="flex flex-col gap-3">
      <PageHeader title="Cameras" description="Camera sources, regions of interest and per-camera detection overrides. Each camera runs in its own worker and reconnects automatically."
        actions={<>
          <Segmented label="View" value={view} onChange={setView} options={[{ value: 'cards', label: 'Cards', icon: LayoutGrid }, { value: 'list', label: 'List', icon: List }]} />
          <Button variant="primary" icon={Plus} disabled={!isAdmin} title={lock} onClick={() => router.replace('/cameras?new=1')}>Add camera</Button>
        </>} />
      {cams.error && !cams.data ? <ErrorState error={cams.error} onRetry={cams.refetch} /> : !cams.data ? <LoadingState rows={5} /> : list.length === 0 ? (
        <Panel><EmptyState icon={CameraIcon} title="No cameras configured" body="Connect your first CCTV camera to begin monitoring. RTSP, HTTP streams, video files (looped) and local webcams are supported."
          action={<Button variant="primary" icon={Plus} disabled={!isAdmin} title={lock} onClick={() => router.replace('/cameras?new=1')}>Add camera</Button>} /></Panel>
      ) : view === 'cards' ? (
        <div className="grid gap-3 md:grid-cols-2 2xl:grid-cols-3">
          {list.map((c) => (
            <section key={c.id} className="flex flex-col overflow-hidden rounded-lg border border-line bg-panel">
              <CameraTile cam={c} overlay={DEFAULT_OVERLAY} compact onOpen={() => setDetail(c.id)} heightClass="aspect-video rounded-none border-0 border-b" />
              <div className="flex flex-col gap-1 p-3 text-[12px]">
                <div className="flex items-center justify-between gap-2"><span className="truncate text-[13px] font-medium text-fg">{c.name}</span><StatusBadge status={c.enabled ? c.status.state : 'disabled'} /></div>
                <div className="flex justify-between text-fg-3"><span className="font-mono">{c.id}</span><span>{c.location || 'Location unavailable'}</span></div>
                <div className="flex justify-between text-fg-3">
                  <span>{c.status.state === 'online' ? `${num(c.status.fps, 1)} fps · ${num(c.status.inference_ms, 0)} ms` : c.status.last_error ?? '—'}</span>
                  <span>{c.roi.length} ROI zone{c.roi.length === 1 ? '' : 's'}</span>
                </div>
              </div>
              <div className="mt-auto flex gap-1 border-t border-line p-2">
                <Button size="sm" onClick={() => setDetail(c.id)}>Open</Button>
                <Button size="sm" variant="ghost" icon={Pencil} disabled={!isAdmin} title={lock} onClick={() => router.replace(`/cameras?id=${encodeURIComponent(c.id)}`)}>Configure</Button>
                <Button size="sm" variant="ghost" icon={Power} disabled={!isAdmin} title={lock} onClick={() => toggle(c)}>{c.enabled ? 'Disable' : 'Enable'}</Button>
                <Button size="sm" variant="ghost" icon={Trash2} disabled={!isAdmin} title={lock} className="ml-auto text-crit" onClick={() => setDel(c)} aria-label={`Delete ${c.id}`} />
              </div>
            </section>
          ))}
        </div>
      ) : (
        <Panel noPad>
          <DataTable rows={list} rowKey={(c) => c.id} onRowClick={(c) => setDetail(c.id)} columns={[
            { key: 'name', header: 'Name', cell: (c) => <span className="text-fg">{c.name}</span>, sortValue: (c) => c.name },
            { key: 'id', header: 'ID', cell: (c) => <span className="font-mono">{c.id}</span>, sortValue: (c) => c.id },
            { key: 'loc', header: 'Location', cell: (c) => c.location || <span className="text-fg-3">unavailable</span> },
            { key: 'st', header: 'Status', cell: (c) => <StatusBadge status={c.enabled ? c.status.state : 'disabled'} />, sortValue: (c) => c.status.state },
            { key: 'fps', header: 'FPS', align: 'right', cell: (c) => num(c.status.fps, 1) },
            { key: 'lat', header: 'Latency', align: 'right', cell: (c) => c.status.inference_ms ? `${num(c.status.inference_ms, 0)} ms` : '—' },
            { key: 'model', header: 'Model', cell: () => <span className="text-fg-3">active (global)</span> },
            { key: 'seen', header: 'Last frame', cell: (c) => c.status.last_frame_age_sec != null ? `${num(c.status.last_frame_age_sec, 1)} s ago` : '—' },
            { key: 'upd', header: 'Updated', cell: (c) => ago(c.updated_at), hideable: true },
            { key: 'act', header: '', cell: (c) => (
              <div className="flex gap-1" onClick={(e) => e.stopPropagation()}>
                <Button size="sm" variant="ghost" icon={Pencil} disabled={!isAdmin} title={lock} onClick={() => router.replace(`/cameras?id=${encodeURIComponent(c.id)}`)}>Edit</Button>
                <Button size="sm" variant="ghost" icon={Power} disabled={!isAdmin} title={lock} onClick={() => toggle(c)}>{c.enabled ? 'Disable' : 'Enable'}</Button>
              </div>) },
          ]} />
        </Panel>
      )}
      <CameraDrawer cam={list.find((c) => c.id === detail) ?? null} onClose={() => setDetail(null)} />
      <CameraEditor open={isNew || !!editing} cam={editing} onClose={closeEditor} existingIds={list.map((c) => c.id)} />
      <ConfirmDialog open={!!del} onClose={() => setDel(null)} danger confirmLabel="Delete camera" title={`Delete ${del?.id}?`}
        body={<>The camera configuration and ROI are removed and its stream is stopped. Alerts, detections and snapshots already recorded are kept.</>}
        onConfirm={async () => {
          try { await Api.deleteCamera(del!.id); toast({ tone: 'ok', title: `${del!.id} deleted` }); invalidate('cameras') }
          catch (e) { toast({ tone: 'crit', title: 'Delete failed', body: (e as Error).message }) }
          setDel(null)
        }} />
    </div>
  )
}

type Section = 'general' | 'stream' | 'model' | 'detection' | 'roi' | 'tracking' | 'alerts'

interface Form {
  id: string; name: string; source: string; location: string; latitude: string; longitude: string; enabled: boolean; roi: RoiZone[]
  conf: number | null; imgsz: string; tiling: '' | 'auto' | 'always' | 'off'; liveFps: string
  window: string; minConf: string; maxMissed: string
  alertsEnabled: boolean | null; cooldown: string; notifyEnabled: boolean; minSeverity: string; providers: string[]
}

function toForm(c: Camera | null): Form {
  const s = (c?.settings ?? {}) as Record<string, Record<string, unknown>>
  const n = (c?.notify ?? {}) as Record<string, unknown>
  const str = (v: unknown) => (v === undefined || v === null ? '' : String(v))
  return {
    id: c?.id ?? '', name: c?.name ?? '', source: c?.source ?? '', location: c?.location ?? '',
    latitude: str(c?.latitude), longitude: str(c?.longitude), enabled: c?.enabled ?? true, roi: c?.roi ?? [],
    conf: (s.confidence?.confirmed_threshold as number) ?? null, imgsz: str(s.inference?.image_size),
    tiling: s.tiling ? (s.tiling.enabled === false ? 'off' : (s.tiling.mode as 'auto' | 'always') ?? 'auto') : '',
    liveFps: str(s.video?.live_process_fps), window: str(s.tracking?.temporal_window), minConf: str(s.tracking?.min_confirmations),
    maxMissed: str(s.tracking?.max_missed_frames), alertsEnabled: (s.alerts?.enabled as boolean) ?? null, cooldown: str(s.alerts?.dedup_window_sec),
    notifyEnabled: (n.enabled as boolean) ?? true, minSeverity: str(n.min_severity) || '', providers: (n.providers as string[]) ?? [],
  }
}

function buildSettings(f: Form, base: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, Record<string, unknown>> = JSON.parse(JSON.stringify(base ?? {}))
  const set = (sec: string, k: string, v: unknown) => {
    out[sec] = out[sec] ?? {}
    if (v === '' || v === null || v === undefined) delete out[sec][k]
    else out[sec][k] = v
    if (!Object.keys(out[sec]).length) delete out[sec]
  }
  set('confidence', 'confirmed_threshold', f.conf)
  set('inference', 'image_size', f.imgsz ? Number(f.imgsz) : '')
  if (f.tiling === '') delete out.tiling
  else out.tiling = f.tiling === 'off' ? { enabled: false } : { enabled: true, mode: f.tiling }
  set('video', 'live_process_fps', f.liveFps ? Number(f.liveFps) : '')
  set('tracking', 'temporal_window', f.window ? Number(f.window) : '')
  set('tracking', 'min_confirmations', f.minConf ? Number(f.minConf) : '')
  set('tracking', 'max_missed_frames', f.maxMissed ? Number(f.maxMissed) : '')
  set('alerts', 'enabled', f.alertsEnabled)
  set('alerts', 'dedup_window_sec', f.cooldown ? Number(f.cooldown) : '')
  return out
}

function CameraEditor({ open, cam, onClose, existingIds }: { open: boolean; cam: Camera | null; onClose: () => void; existingIds: string[] }) {
  const { toast } = useApp()
  const settings = useQuery('settings', Api.settings)
  const health = useQuery('health', Api.health)
  const [f, setF] = useState<Form>(toForm(cam))
  const [sec, setSec] = useState<Section>('general')
  const [saving, setSaving] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [frame, setFrame] = useState<{ src: string; w: number; h: number } | null>(null)
  const [frameErr, setFrameErr] = useState<string | null>(null)
  useEffect(() => { if (open) { setF(toForm(cam)); setSec('general'); setErr(null); setFrame(null); setFrameErr(null) } }, [open, cam?.id]) // eslint-disable-line react-hooks/exhaustive-deps
  const g = settings.data?.settings
  const isNew = !cam
  const idOk = /^[A-Za-z0-9_-]{1,40}$/.test(f.id) && (!isNew || !existingIds.includes(f.id))

  const loadFrame = async () => {
    if (!cam) return
    setFrameErr(null)
    try {
      const res = await fetch(url(`/api/cameras/${encodeURIComponent(cam.id)}/frame.jpg?t=${Date.now()}`))
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail ?? `HTTP ${res.status}`)
      const blob = await res.blob()
      const src = URL.createObjectURL(blob)
      const img = new Image()
      img.onload = () => setFrame({ src, w: img.naturalWidth, h: img.naturalHeight })
      img.src = src
    } catch (e) { setFrameErr((e as Error).message) }
  }
  useEffect(() => { if (open && sec === 'roi' && !frame) void loadFrame() }, [sec, open]) // eslint-disable-line react-hooks/exhaustive-deps

  const save = async () => {
    setSaving(true)
    setErr(null)
    const body = {
      name: f.name, location: f.location, enabled: f.enabled, roi: f.roi,
      // the API returns credentials masked; only send the source when it was actually edited
      ...(isNew || f.source !== cam!.source ? { source: f.source } : {}),
      latitude: f.latitude === '' ? null : Number(f.latitude), longitude: f.longitude === '' ? null : Number(f.longitude),
      settings: buildSettings(f, isNew ? {} : cam!.settings),
      notify: { enabled: f.notifyEnabled, ...(f.minSeverity ? { min_severity: f.minSeverity } : {}), ...(f.providers.length ? { providers: f.providers } : {}) },
    }
    try {
      if (isNew) await Api.createCamera({ id: f.id, ...body } as Partial<Camera>)
      else await Api.updateCamera(cam!.id, body as Partial<Camera>)
      toast({ tone: 'ok', title: isNew ? `Camera ${f.id} added` : `Camera ${cam!.id} saved`, body: f.enabled ? 'The stream (re)starts automatically.' : undefined })
      invalidate('cameras')
      onClose()
    } catch (e) { setErr((e as Error).message) } finally { setSaving(false) }
  }

  const SECTIONS: { value: Section; label: string }[] = [
    { value: 'general', label: 'General' }, { value: 'stream', label: 'Stream' }, { value: 'model', label: 'AI model' }, { value: 'detection', label: 'Detection' },
    { value: 'roi', label: 'ROI' }, { value: 'tracking', label: 'Tracking' }, { value: 'alerts', label: 'Alerts' },
  ]
  const idx = SECTIONS.findIndex((s) => s.value === sec)
  const inherit = (v: unknown) => `inherit (${v ?? '—'})`

  return (
    <Drawer open={open} onClose={onClose} width="w-[720px]" title={isNew ? 'Add camera' : `Configure ${cam?.id}`}
      subtitle={isNew ? `Step ${idx + 1} of ${SECTIONS.length}` : 'Empty fields inherit the global settings'}
      footer={<>
        {err && <span className="mr-auto max-w-[60%] text-[12px] text-crit">{err}</span>}
        {isNew && idx > 0 && <Button variant="ghost" onClick={() => setSec(SECTIONS[idx - 1].value)}>Back</Button>}
        {isNew && idx < SECTIONS.length - 1 ? <Button variant="primary" disabled={idx === 0 && (!idOk || !f.name)} onClick={() => setSec(SECTIONS[idx + 1].value)}>Next</Button>
          : <Button variant="primary" loading={saving} disabled={!f.name || !f.source || (isNew && !idOk)} onClick={save}>{isNew ? 'Create camera' : 'Save changes'}</Button>}
      </>}>
      <div className="flex flex-col gap-4">
        <Tabs value={sec} onChange={setSec} tabs={SECTIONS} />
        {sec === 'general' && (
          <div className="grid gap-3 md:grid-cols-2">
            <Field label="Camera ID" hint={isNew ? 'letters, digits, - and _; cannot be changed later' : 'cannot be changed'}>
              <Input value={f.id} disabled={!isNew} onChange={(e) => setF({ ...f, id: e.target.value.toUpperCase() })} placeholder="CAM01" aria-invalid={isNew && !!f.id && !idOk} />
            </Field>
            <Field label="Name"><Input value={f.name} onChange={(e) => setF({ ...f, name: e.target.value })} placeholder="Main Entrance" /></Field>
            <Field label="Location"><Input value={f.location} onChange={(e) => setF({ ...f, location: e.target.value })} placeholder="Front sidewalk" /></Field>
            <div className="grid grid-cols-2 gap-2">
              <Field label="Latitude"><Input inputMode="decimal" value={f.latitude} onChange={(e) => setF({ ...f, latitude: e.target.value })} placeholder="optional" /></Field>
              <Field label="Longitude"><Input inputMode="decimal" value={f.longitude} onChange={(e) => setF({ ...f, longitude: e.target.value })} placeholder="optional" /></Field>
            </div>
            <div className="md:col-span-2"><Toggle label="Enabled" checked={f.enabled} onChange={(v) => setF({ ...f, enabled: v })} description="Disabled cameras keep their configuration but no stream is opened." /></div>
          </div>
        )}
        {sec === 'stream' && (
          <div className="flex flex-col gap-3">
            <Field label="Source" hint={<>RTSP (<code>rtsp://…</code>), HTTP/MJPEG, a video file path (looped — for demos), a webcam index (<code>0</code>), or <code>env:VAR_NAME</code> to keep credentials in the server’s .env. Credentials are masked in API responses.</>}>
              <Input value={f.source} onChange={(e) => setF({ ...f, source: e.target.value })} placeholder="env:CAM01_RTSP_URL" className="font-mono" />
            </Field>
            {!isNew && cam!.source.includes('***') && <p className="text-[12px] text-fg-3">The password is masked. Leave the field unchanged to keep the stored URL, or enter a complete new URL.</p>}
            <Field label="Live processing FPS" hint={`Frames analysed per second. Global: ${g?.video.live_process_fps ?? '—'}. On CPU each frame takes ~${num(health.data?.model ? 250 : null)} ms.`}>
              <Select value={f.liveFps} onChange={(e) => setF({ ...f, liveFps: e.target.value })}><option value="">{inherit(g?.video.live_process_fps)}</option>{[0.5, 1, 2, 3, 5, 10].map((v) => <option key={v} value={v}>{v}</option>)}</Select>
            </Field>
          </div>
        )}
        {sec === 'model' && (
          <div className="rounded-md border border-line bg-panel-2 p-3 text-[13px]">
            <div className="flex items-center justify-between"><span>Active model</span><span className="font-mono text-fg">{health.data?.model?.version ?? '—'}</span></div>
            <p className="mt-2 text-[12px] text-fg-3">All cameras share the globally active model (one model in memory, serialized inference). Per-camera model selection is not supported; change the active model in the Model Lab.</p>
          </div>
        )}
        {sec === 'detection' && (
          <div className="flex flex-col gap-3">
            <div className="flex items-center gap-3">
              <div className="flex-1"><Slider label="Confirmed threshold" value={f.conf ?? g?.confidence.confirmed_threshold ?? 0.6} min={0.2} max={0.95} step={0.05} onChange={(v) => setF({ ...f, conf: v })} /></div>
              <Button size="sm" variant="ghost" disabled={f.conf === null} onClick={() => setF({ ...f, conf: null })}>Inherit</Button>
            </div>
            <p className="-mt-2 text-[11px] text-fg-3">{f.conf === null ? `Inheriting global ${g?.confidence.confirmed_threshold}` : 'Camera override'} · weak {g?.confidence.weak_threshold} / candidate {g?.confidence.candidate_threshold} (global)</p>
            <div className="grid grid-cols-2 gap-3">
              <Field label="Image size"><Select value={f.imgsz} onChange={(e) => setF({ ...f, imgsz: e.target.value })}><option value="">{inherit(g?.inference.image_size)}</option>{[480, 640, 800, 960, 1280].map((v) => <option key={v} value={v}>{v}px</option>)}</Select></Field>
              <Field label="Tiled inference"><Select value={f.tiling} onChange={(e) => setF({ ...f, tiling: e.target.value as Form['tiling'] })}>
                <option value="">{inherit(g ? (g.tiling.enabled ? g.tiling.mode : 'off') : undefined)}</option><option value="auto">auto</option><option value="always">always</option><option value="off">off</option></Select></Field>
            </div>
          </div>
        )}
        {sec === 'roi' && (
          isNew ? <p className="text-[12px] text-fg-3">Save the camera first; the ROI editor needs a frame from its stream. Without zones the whole frame is monitored.</p> :
          frame ? <RoiEditor src={frame.src} width={frame.w} height={frame.h} zones={f.roi} onChange={(roi) => setF({ ...f, roi })} /> :
          frameErr ? <ErrorState compact error={new Error(frameErr)} title="Could not grab a frame" onRetry={loadFrame} /> : <LoadingState rows={4} label="Grabbing a frame" />
        )}
        {sec === 'tracking' && (
          <div className="grid grid-cols-3 gap-3">
            <Field label="Tracker"><Select disabled><option>ByteTrack-style IoU</option></Select></Field>
            <Field label="Temporal window" hint="frames"><Select value={f.window} onChange={(e) => setF({ ...f, window: e.target.value })}><option value="">{inherit(g?.tracking.temporal_window)}</option>{[3, 5, 7, 10, 15].map((v) => <option key={v}>{v}</option>)}</Select></Field>
            <Field label="Min confirmations"><Select value={f.minConf} onChange={(e) => setF({ ...f, minConf: e.target.value })}><option value="">{inherit(g?.tracking.min_confirmations)}</option>{[1, 2, 3, 4, 5, 7].map((v) => <option key={v}>{v}</option>)}</Select></Field>
            <Field label="Max age" hint="missed frames before a track ends"><Select value={f.maxMissed} onChange={(e) => setF({ ...f, maxMissed: e.target.value })}><option value="">{inherit(g?.tracking.max_missed_frames)}</option>{[5, 10, 15, 30, 60].map((v) => <option key={v}>{v}</option>)}</Select></Field>
          </div>
        )}
        {sec === 'alerts' && (
          <div className="flex flex-col gap-3">
            <Toggle label="Alerts enabled" checked={f.alertsEnabled ?? g?.alerts.enabled ?? true} onChange={(v) => setF({ ...f, alertsEnabled: v })} description={f.alertsEnabled === null ? 'inheriting global' : 'camera override'} />
            <div className="grid grid-cols-2 gap-3">
              <Field label="Cooldown / dedup window" hint="seconds a re-detected object at the same place is merged"><Select value={f.cooldown} onChange={(e) => setF({ ...f, cooldown: e.target.value })}>
                <option value="">{inherit(g?.alerts.dedup_window_sec)}</option>{[60, 300, 900, 1800, 3600].map((v) => <option key={v} value={v}>{v}s</option>)}</Select></Field>
              <Field label="Minimum confidence" hint="= confirmed threshold (Detection tab)"><Input disabled value={String(f.conf ?? g?.confidence.confirmed_threshold ?? '')} /></Field>
            </div>
            <Toggle label="Send notifications" checked={f.notifyEnabled} onChange={(v) => setF({ ...f, notifyEnabled: v })} />
            <div className="grid grid-cols-2 gap-3">
              <Field label="Notify from severity"><Select value={f.minSeverity} onChange={(e) => setF({ ...f, minSeverity: e.target.value })}><option value="">server default</option><option>LOW</option><option>MEDIUM</option><option>HIGH</option></Select></Field>
              <Field label="Channels">
                <div className="flex flex-wrap gap-2 pt-1">
                  {(health.data?.notifications ?? []).map((n) => (
                    <label key={n.provider} className={cx('flex items-center gap-1.5 text-[12px]', !n.configured && 'text-fg-3')}>
                      <input type="checkbox" checked={f.providers.includes(n.provider)} onChange={(e) => setF({ ...f, providers: e.target.checked ? [...f.providers, n.provider] : f.providers.filter((p) => p !== n.provider) })} />
                      {n.provider}{!n.configured && <Badge>not configured</Badge>}
                    </label>
                  ))}
                </div>
              </Field>
            </div>
            <p className="text-[11px] text-fg-3">No channel selected = all configured channels. Provider credentials come only from the server environment (.env).</p>
          </div>
        )}
      </div>
    </Drawer>
  )
}
