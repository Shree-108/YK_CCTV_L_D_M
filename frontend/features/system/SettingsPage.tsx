'use client'

import Link from 'next/link'
import { useRouter, useSearchParams } from 'next/navigation'
import { useEffect, useState, type ReactNode } from 'react'
import { KeyRound, LogOut, Monitor, Moon, RotateCcw, Save, Sun } from 'lucide-react'
import { API_BASE } from '../../api/client'
import { Api } from '../../api/endpoints'
import type { Settings } from '../../api/types'
import { invalidate, useQuery } from '../../hooks/useQuery'
import { cx } from '../../lib/format'
import { useApp } from '../../stores/app'
import { Badge, Button, Field, Input, PageHeader, Panel, Segmented, Select, Slider, Toggle, Unavailable } from '../../components/ui/primitives'
import { ErrorState, LoadingState } from '../../components/ui/states'

type Tab = 'general' | 'appearance' | 'detection' | 'tracking' | 'alerts' | 'cameras' | 'models' | 'storage' | 'notifications' | 'security'
const TABS: { value: Tab; label: string }[] = [
  { value: 'general', label: 'General' }, { value: 'appearance', label: 'Appearance' }, { value: 'detection', label: 'Detection' },
  { value: 'tracking', label: 'Tracking' }, { value: 'alerts', label: 'Alerts' }, { value: 'cameras', label: 'Cameras' }, { value: 'models', label: 'Models' },
  { value: 'storage', label: 'Storage' }, { value: 'notifications', label: 'Notifications' }, { value: 'security', label: 'Security' },
]

type Patch = { [K in keyof Settings]?: Partial<Settings[K]> }

export default function SettingsPage() {
  const params = useSearchParams()
  const router = useRouter()
  const tab = (params.get('tab') as Tab) ?? 'general'
  const { isAdmin, toast } = useApp()
  const q = useQuery('settings', Api.settings)
  const [draft, setDraft] = useState<Settings | null>(null)
  const [saving, setSaving] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  useEffect(() => { if (q.data) setDraft(q.data.settings) }, [q.data])
  const dirty = !!draft && !!q.data && JSON.stringify(draft) !== JSON.stringify(q.data.settings)

  const set = <K extends keyof Settings>(sec: K, patch: Partial<Settings[K]>) => setDraft((d) => (d ? { ...d, [sec]: { ...d[sec], ...patch } } : d))

  const save = async () => {
    if (!draft || !q.data) return
    // send only changed sections/keys
    const patch: Patch = {}
    for (const k of Object.keys(draft) as (keyof Settings)[]) {
      const a = draft[k] as Record<string, unknown>, b = q.data.settings[k] as Record<string, unknown>
      const changed = Object.fromEntries(Object.entries(a).filter(([kk, v]) => JSON.stringify(v) !== JSON.stringify(b[kk])))
      if (Object.keys(changed).length) (patch as Record<string, unknown>)[k] = changed
    }
    setSaving(true)
    setErr(null)
    try {
      await Api.saveSettings(patch)
      toast({ tone: 'ok', title: 'Settings saved', body: 'Running cameras pick them up on the next frame.' })
      invalidate('settings')
    } catch (e) { setErr((e as Error).message) } finally { setSaving(false) }
  }

  const detectionTabs: Tab[] = ['detection', 'tracking', 'alerts', 'storage']
  return (
    <div className="flex flex-col gap-3">
      <PageHeader title="Settings" description="Global detection settings are stored in the database and apply to every camera without its own override. Secrets live only in the server’s environment."
        actions={detectionTabs.includes(tab) && <>
          <Button variant="ghost" icon={RotateCcw} disabled={!dirty} onClick={() => q.data && setDraft(q.data.settings)}>Discard</Button>
          <Button variant="primary" icon={Save} disabled={!dirty || !isAdmin} loading={saving} title={!isAdmin ? 'Admin sign-in required' : undefined} onClick={save}>Save changes</Button>
        </>} />
      <div className="grid gap-3 lg:grid-cols-[200px_minmax(0,1fr)]">
        <nav aria-label="Settings sections" className="flex gap-1 overflow-x-auto lg:flex-col">
          {TABS.map((t) => (
            <button key={t.value} onClick={() => router.replace(`/settings?tab=${t.value}`)} aria-current={tab === t.value ? 'page' : undefined}
              className={cx('shrink-0 rounded-md px-3 py-1.5 text-left text-[13px]', tab === t.value ? 'bg-hover text-fg' : 'text-fg-2 hover:bg-hover/60')}>{t.label}</button>
          ))}
        </nav>
        <div className="min-w-0">
          {err && <p className="mb-2 rounded-md border border-crit/40 bg-crit-bg p-2 text-[12px] text-crit">{err}</p>}
          {!isAdmin && detectionTabs.includes(tab) && <p className="mb-2 rounded-md border border-line bg-panel-2 p-2 text-[12px] text-fg-3">Read-only: <Link href="/settings?tab=security" className="text-info hover:underline">sign in as admin</Link> to change settings.</p>}
          {tab === 'general' && <General />}
          {tab === 'appearance' && <Appearance />}
          {tab === 'security' && <Security />}
          {tab === 'notifications' && <Notifications />}
          {tab === 'cameras' && <Section title="Cameras"><p className="text-[13px] text-fg-2">Camera sources, ROI zones and per-camera overrides are managed on the <Link href="/cameras" className="text-info hover:underline">Cameras</Link> page.</p></Section>}
          {tab === 'models' && <Section title="Models"><p className="text-[13px] text-fg-2">Deployment, rollback and registration are in the <Link href="/model-lab" className="text-info hover:underline">Model Lab</Link>; the operating threshold can be applied from the <Link href="/evaluation" className="text-info hover:underline">Evaluation → Threshold lab</Link>.</p></Section>}
          {detectionTabs.includes(tab) && (q.error && !q.data ? <ErrorState error={q.error} onRetry={q.refetch} /> : !draft ? <LoadingState rows={6} /> : (
            <fieldset disabled={!isAdmin} className="flex flex-col gap-3">
              {tab === 'detection' && <>
                <Section title="Confidence tiers" hint="weak ≤ candidate ≤ confirmed. Weak detections are logged only; candidates count as temporal hits; confirmation needs the mean hit confidence ≥ confirmed.">
                  <div className="grid gap-4 md:grid-cols-3">
                    <Slider label="Weak" value={draft.confidence.weak_threshold} min={0.05} max={0.9} step={0.01} onChange={(v) => set('confidence', { weak_threshold: v })} />
                    <Slider label="Candidate" value={draft.confidence.candidate_threshold} min={0.05} max={0.95} step={0.01} onChange={(v) => set('confidence', { candidate_threshold: v })} />
                    <Slider label="Confirmed" value={draft.confidence.confirmed_threshold} min={0.05} max={0.95} step={0.01} onChange={(v) => set('confidence', { confirmed_threshold: v })} />
                  </div>
                  {!(draft.confidence.weak_threshold <= draft.confidence.candidate_threshold && draft.confidence.candidate_threshold <= draft.confidence.confirmed_threshold) &&
                    <p className="mt-2 text-[12px] text-crit">Order must be weak ≤ candidate ≤ confirmed — the server will reject this.</p>}
                </Section>
                <Section title="Inference">
                  <div className="grid gap-3 md:grid-cols-4">
                    <Field label="Image size"><Select value={draft.inference.image_size} onChange={(e) => set('inference', { image_size: Number(e.target.value) })}>{[480, 640, 800, 960, 1280, 1600].map((v) => <option key={v} value={v}>{v}px</option>)}</Select></Field>
                    <Field label="NMS IoU"><Input type="number" step={0.05} min={0.1} max={0.9} value={draft.inference.iou_threshold} onChange={(e) => set('inference', { iou_threshold: Number(e.target.value) })} /></Field>
                    <Field label="Max detections"><Input type="number" min={1} max={3000} value={draft.inference.max_detections} onChange={(e) => set('inference', { max_detections: Number(e.target.value) })} /></Field>
                    <Field label="Device"><Select value={draft.inference.device} onChange={(e) => set('inference', { device: e.target.value })}>{['auto', 'cpu', 'cuda', 'cuda:0', 'mps'].map((d) => <option key={d}>{d}</option>)}</Select></Field>
                  </div>
                  <div className="mt-3"><Toggle label="Class-agnostic NMS" checked={draft.inference.agnostic_nms} onChange={(v) => set('inference', { agnostic_nms: v })} description="Off: a packet lying on cardboard can keep both boxes." /></div>
                </Section>
                <Section title="Tiled inference (small objects)">
                  <div className="grid gap-3 md:grid-cols-4">
                    <div className="md:col-span-4"><Toggle label="Enabled" checked={draft.tiling.enabled} onChange={(v) => set('tiling', { enabled: v })} /></div>
                    <Field label="Mode"><Select value={draft.tiling.mode} onChange={(e) => set('tiling', { mode: e.target.value as 'auto' | 'always' })}><option value="auto">auto (large frames)</option><option value="always">always</option></Select></Field>
                    <Field label="Tile size"><Select value={draft.tiling.tile_size} onChange={(e) => set('tiling', { tile_size: Number(e.target.value) })}>{[512, 640, 768, 1024, 1280].map((v) => <option key={v} value={v}>{v}px</option>)}</Select></Field>
                    <Field label="Overlap"><Input type="number" step={0.05} min={0} max={0.5} value={draft.tiling.overlap} onChange={(e) => set('tiling', { overlap: Number(e.target.value) })} /></Field>
                    <Field label="Auto if longer side >" hint="× tile size"><Input type="number" step={0.05} min={1} value={draft.tiling.auto_min_ratio} onChange={(e) => set('tiling', { auto_min_ratio: Number(e.target.value) })} /></Field>
                  </div>
                  <div className="mt-3"><Toggle label="Also run on the full frame" checked={draft.tiling.include_full_frame} onChange={(v) => set('tiling', { include_full_frame: v })} description="Catches objects larger than a tile." /></div>
                </Section>
                <Section title="Filters">
                  <div className="grid gap-3 md:grid-cols-4">
                    <Field label="Min box area" hint="px²"><Input type="number" min={0} value={draft.filters.min_box_area_px} onChange={(e) => set('filters', { min_box_area_px: Number(e.target.value) })} /></Field>
                    <Field label="Max box area" hint="fraction of frame"><Input type="number" step={0.01} min={0.01} max={1} value={draft.filters.max_box_area_frac} onChange={(e) => set('filters', { max_box_area_frac: Number(e.target.value) })} /></Field>
                    <Field label="Max aspect ratio"><Input type="number" min={1} value={draft.filters.max_aspect_ratio} onChange={(e) => set('filters', { max_aspect_ratio: Number(e.target.value) })} /></Field>
                    <Field label="ROI anchor"><Select value={draft.filters.roi_anchor} onChange={(e) => set('filters', { roi_anchor: e.target.value as 'footpoint' | 'center' })}><option value="footpoint">foot point</option><option value="center">center</option></Select></Field>
                    <Field label="ROI margin" hint="tolerance, fraction"><Input type="number" step={0.01} min={0} max={0.2} value={draft.filters.roi_margin_frac} onChange={(e) => set('filters', { roi_margin_frac: Number(e.target.value) })} /></Field>
                    <Field label="Blocked classes" hint="comma separated"><Input value={draft.filters.class_blocklist.join(', ')} onChange={(e) => set('filters', { class_blocklist: e.target.value.split(',').map((x) => x.trim()).filter(Boolean) })} /></Field>
                  </div>
                  <div className="mt-3"><Toggle label="Apply ROI zones" checked={draft.filters.roi_enabled} onChange={(v) => set('filters', { roi_enabled: v })} /></div>
                </Section>
              </>}
              {tab === 'tracking' && <>
                <Section title="Temporal confirmation" hint="An object is confirmed after ≥ min confirmations candidate hits within the last window of processed frames.">
                  <div className="grid gap-3 md:grid-cols-3">
                    <Field label="Temporal window"><Input type="number" min={1} max={100} value={draft.tracking.temporal_window} onChange={(e) => set('tracking', { temporal_window: Number(e.target.value) })} /></Field>
                    <Field label="Min confirmations"><Input type="number" min={1} max={100} value={draft.tracking.min_confirmations} onChange={(e) => set('tracking', { min_confirmations: Number(e.target.value) })} /></Field>
                    <Field label="Max missed frames" hint="before a track ends"><Input type="number" min={1} value={draft.tracking.max_missed_frames} onChange={(e) => set('tracking', { max_missed_frames: Number(e.target.value) })} /></Field>
                    <Field label="Match IoU"><Input type="number" step={0.05} min={0.05} max={0.95} value={draft.tracking.iou_match} onChange={(e) => set('tracking', { iou_match: Number(e.target.value) })} /></Field>
                    <Field label="Low-score match IoU"><Input type="number" step={0.05} min={0.05} max={0.95} value={draft.tracking.low_iou_match} onChange={(e) => set('tracking', { low_iou_match: Number(e.target.value) })} /></Field>
                  </div>
                </Section>
                <Section title="Stationarity">
                  <Toggle label="Reject moving objects" checked={draft.tracking.stationarity_enabled} onChange={(v) => set('tracking', { stationarity_enabled: v })} description="Items being carried never confirm." />
                  <div className="mt-3 max-w-xs"><Field label="Max centre shift" hint="fraction of frame diagonal"><Input type="number" step={0.01} min={0.01} max={1} value={draft.tracking.max_center_shift_frac} onChange={(e) => set('tracking', { max_center_shift_frac: Number(e.target.value) })} /></Field></div>
                </Section>
                <Section title="Video & live processing">
                  <div className="grid gap-3 md:grid-cols-4">
                    <Field label="Video sample FPS"><Input type="number" step={0.5} min={0.5} max={60} value={draft.video.sample_fps} onChange={(e) => set('video', { sample_fps: Number(e.target.value) })} /></Field>
                    <Field label="Live FPS per camera"><Input type="number" step={0.5} min={0.5} max={60} value={draft.video.live_process_fps} onChange={(e) => set('video', { live_process_fps: Number(e.target.value) })} /></Field>
                    <Field label="Live log interval" hint="s between logged rows"><Input type="number" step={0.5} min={0} value={draft.video.live_log_interval_sec} onChange={(e) => set('video', { live_log_interval_sec: Number(e.target.value) })} /></Field>
                    <Field label="Max video length" hint="seconds"><Input type="number" min={1} value={draft.video.max_video_seconds} onChange={(e) => set('video', { max_video_seconds: Number(e.target.value) })} /></Field>
                  </div>
                </Section>
              </>}
              {tab === 'alerts' && <>
                <Section title="Alert generation">
                  <div className="flex flex-col gap-3">
                    <Toggle label="Alerts enabled" checked={draft.alerts.enabled} onChange={(v) => set('alerts', { enabled: v })} />
                    <Toggle label="One alert per physical object (track)" checked={draft.alerts.one_alert_per_track} onChange={(v) => set('alerts', { one_alert_per_track: v })} />
                    <Toggle label="Create alerts from image uploads" checked={draft.alerts.alert_on_image_uploads} onChange={(v) => set('alerts', { alert_on_image_uploads: v })} />
                  </div>
                </Section>
                <Section title="Deduplication" hint="A new confirmation is merged into an existing alert of the same camera when it is within the time window, overlaps (IoU) or is close (centre distance), and has a similar class.">
                  <div className="grid gap-3 md:grid-cols-4">
                    <Field label="Time window" hint="seconds"><Input type="number" min={0} value={draft.alerts.dedup_window_sec} onChange={(e) => set('alerts', { dedup_window_sec: Number(e.target.value) })} /></Field>
                    <Field label="IoU"><Input type="number" step={0.05} min={0} max={1} value={draft.alerts.dedup_iou} onChange={(e) => set('alerts', { dedup_iou: Number(e.target.value) })} /></Field>
                    <Field label="Centre distance" hint="fraction of diagonal"><Input type="number" step={0.01} min={0} max={1} value={draft.alerts.dedup_center_frac} onChange={(e) => set('alerts', { dedup_center_frac: Number(e.target.value) })} /></Field>
                    <Field label="Class similarity"><Select value={draft.alerts.class_match} onChange={(e) => set('alerts', { class_match: e.target.value as 'exact' | 'group' | 'any' })}><option value="exact">exact class</option><option value="group">class group</option><option value="any">any class</option></Select></Field>
                  </div>
                </Section>
                <Section title="Severity rules" hint="HIGH if any HIGH rule matches, else MEDIUM if any MEDIUM rule matches, else LOW. Priorities come from class metadata.">
                  {(['high', 'medium'] as const).map((lvl) => (
                    <div key={lvl} className="mb-3">
                      <div className="mb-1 text-[12px] font-medium uppercase text-fg-2">{lvl}</div>
                      {draft.severity[lvl].map((r, i) => (
                        <div key={i} className="mb-1.5 grid grid-cols-2 gap-2 md:grid-cols-4">
                          <Field label="Priorities"><Input value={r.priorities.join(', ')} onChange={(e) => set('severity', { [lvl]: draft.severity[lvl].map((x, j) => j === i ? { ...x, priorities: e.target.value.split(',').map((s) => s.trim()).filter(Boolean) } : x) })} /></Field>
                          <Field label="Classes"><Input value={r.classes.join(', ')} onChange={(e) => set('severity', { [lvl]: draft.severity[lvl].map((x, j) => j === i ? { ...x, classes: e.target.value.split(',').map((s) => s.trim()).filter(Boolean) } : x) })} /></Field>
                          <Field label="Min confidence"><Input type="number" step={0.05} min={0} max={1} value={r.min_confidence} onChange={(e) => set('severity', { [lvl]: draft.severity[lvl].map((x, j) => j === i ? { ...x, min_confidence: Number(e.target.value) } : x) })} /></Field>
                          <Field label="Min active objects"><Input type="number" min={0} value={r.min_active_objects} onChange={(e) => set('severity', { [lvl]: draft.severity[lvl].map((x, j) => j === i ? { ...x, min_active_objects: Number(e.target.value) } : x) })} /></Field>
                        </div>
                      ))}
                    </div>
                  ))}
                </Section>
              </>}
              {tab === 'storage' && (
                <Section title="Snapshots">
                  <div className="grid gap-3 md:grid-cols-3">
                    <Field label="JPEG quality"><Input type="number" min={30} max={100} value={draft.alerts.snapshot_jpeg_quality} onChange={(e) => set('alerts', { snapshot_jpeg_quality: Number(e.target.value) })} /></Field>
                    <Field label="Crop padding" hint="fraction of box size"><Input type="number" step={0.05} min={0} max={2} value={draft.alerts.crop_padding_frac} onChange={(e) => set('alerts', { crop_padding_frac: Number(e.target.value) })} /></Field>
                    <Field label="Annotate every video frame"><Select value={String(draft.video.annotate_all_frames)} onChange={(e) => set('video', { annotate_all_frames: e.target.value === 'true' })}><option value="true">yes (original fps)</option><option value="false">sampled frames only</option></Select></Field>
                  </div>
                  <p className="mt-3 text-[12px] text-fg-3">Snapshots are written to <span className="font-mono">storage/alerts/&lt;date&gt;/&lt;camera&gt;/</span>; video outputs to <span className="font-mono">storage/videos/&lt;job&gt;/</span>. Only paths are stored in the database. <Unavailable>automatic retention cleanup not implemented</Unavailable></p>
                </Section>
              )}
            </fieldset>
          ))}
        </div>
      </div>
    </div>
  )
}

function Section({ title, hint, children }: { title: string; hint?: string; children: ReactNode }) {
  return (
    <Panel title={title}>
      {hint && <p className="-mt-1 mb-3 text-[12px] text-fg-3">{hint}</p>}
      {children}
    </Panel>
  )
}

function General() {
  const h = useQuery('health', Api.health)
  return (
    <Section title="General">
      <dl className="grid gap-y-2 text-[13px] md:grid-cols-[200px_1fr]">
        <dt className="text-fg-3">API endpoint</dt><dd className="font-mono text-fg">{API_BASE}</dd>
        <dt className="text-fg-3">Backend status</dt><dd className="text-fg">{h.error ? 'not reachable' : h.data?.status ?? '…'}</dd>
        <dt className="text-fg-3">Active model</dt><dd className="font-mono text-fg">{h.data?.model?.version ?? '—'}</dd>
        <dt className="text-fg-3">Classes in model</dt><dd className="text-fg">{h.data?.model?.classes.length ?? '—'}</dd>
      </dl>
      <p className="mt-3 text-[12px] text-fg-3">The frontend reads <span className="font-mono">NEXT_PUBLIC_API_URL</span> at build/start time.</p>
    </Section>
  )
}

function Appearance() {
  const { theme, setTheme, sidebarCollapsed, setSidebarCollapsed } = useApp()
  return (
    <Section title="Appearance">
      <div className="flex flex-col gap-4">
        <Field label="Theme">
          <Segmented label="Theme" value={theme} onChange={setTheme} options={[{ value: 'dark', label: 'Dark', icon: Moon }, { value: 'light', label: 'Light', icon: Sun }, { value: 'system', label: 'System', icon: Monitor }]} />
        </Field>
        <Toggle label="Collapsed sidebar" checked={sidebarCollapsed} onChange={setSidebarCollapsed} />
        <p className="text-[12px] text-fg-3">Stored in this browser only.</p>
      </div>
    </Section>
  )
}

function Notifications() {
  const h = useQuery('health', Api.health)
  return (
    <Section title="Notification providers" hint="Providers are enabled only when their credentials exist in the server environment. Per-camera channel and minimum severity are set on the camera's Alerts tab.">
      <ul className="flex flex-col gap-2 text-[13px]">
        {(h.data?.notifications ?? []).map((n) => (
          <li key={n.provider} className="flex items-center justify-between rounded-md border border-line bg-panel-2 px-3 py-2">
            <span className="capitalize text-fg">{n.provider}</span>
            {n.configured ? <Badge tone="ok" dot>configured</Badge> : <Badge>not configured</Badge>}
          </li>
        ))}
      </ul>
      <pre className="mt-3 overflow-x-auto rounded-md border border-line bg-panel-2 p-2 font-mono text-[11.5px] text-fg-2">{`# server .env
NOTIFY_MIN_SEVERITY=HIGH
SMTP_HOST=  SMTP_PORT=587  SMTP_USER=  SMTP_PASSWORD=  ALERT_EMAIL_TO=
ALERT_WEBHOOK_URL=
TELEGRAM_BOT_TOKEN=  TELEGRAM_CHAT_ID=`}</pre>
    </Section>
  )
}

function Security() {
  const { isAdmin, adminLogin, adminLogout, toast } = useApp()
  const h = useQuery('health', Api.health)
  const [token, setToken] = useState('')
  const [busy, setBusy] = useState(false)
  return (
    <Section title="Admin access" hint="Configuration changes (cameras, settings, models, labels) require the admin token configured on the server as ADMIN_API_TOKEN. The token is kept only in this browser tab's session storage.">
      {h.data && !h.data.admin_enabled && <p className="mb-3 rounded-md border border-warn/40 bg-warn-bg p-2 text-[12px] text-warn">The server has no ADMIN_API_TOKEN configured, so admin functions are disabled.</p>}
      {isAdmin ? (
        <div className="flex items-center justify-between rounded-md border border-line bg-panel-2 p-3">
          <span className="flex items-center gap-2 text-[13px] text-fg"><Badge tone="exp" dot>admin</Badge> Signed in for this session</span>
          <Button icon={LogOut} onClick={() => { adminLogout(); toast({ tone: 'info', title: 'Admin session ended' }) }}>Sign out</Button>
        </div>
      ) : (
        <form className="flex max-w-md flex-col gap-2" onSubmit={async (e) => {
          e.preventDefault()
          setBusy(true)
          const ok = await adminLogin(token)
          setBusy(false)
          setToken('')
          toast(ok ? { tone: 'ok', title: 'Admin session started' } : { tone: 'crit', title: 'Invalid admin token' })
        }}>
          <Field label="Admin token"><Input type="password" autoComplete="current-password" value={token} onChange={(e) => setToken(e.target.value)} /></Field>
          <Button type="submit" variant="primary" icon={KeyRound} loading={busy} disabled={!token}>Sign in</Button>
        </form>
      )}
      <ul className="mt-4 list-disc pl-5 text-[12px] text-fg-3">
        <li>Uploads are validated by type and size on the server and stored under random names.</li>
        <li>RTSP credentials can be referenced as <span className="font-mono">env:VAR</span> and are masked in every API response.</li>
        <li>Operators without the token can view, review alerts and run detections.</li>
      </ul>
    </Section>
  )
}

