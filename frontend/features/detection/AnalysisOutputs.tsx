'use client'

// Building blocks shared by Image and Video Analysis results: processing status, identity (store / camera) with its
// warnings, detection status badges, confidence and authenticated downloads. Nothing here recomputes a status: the
// server's values are shown as they are.
import { useState } from 'react'
import { AlertTriangle, Download } from 'lucide-react'
import { downloadAuthed } from '../../api/client'
import type { Identity } from '../../api/types'
import { cx } from '../../lib/format'
import { confidenceText, detectionStatusInfo, fileNameFromPath, processingStatusInfo } from '../../lib/litter'
import { useApp } from '../../stores/app'
import { Badge, Button } from '../../components/ui/primitives'

export function DetectionStatusBadge({ status, reason }: { status: string; reason?: string | null }) {
  const i = detectionStatusInfo(status, reason)
  return <Badge tone={i.tone} dot title={i.hint}>{i.label}</Badge>
}

export function ProcessingStatusBadge({ status }: { status: string | null | undefined }) {
  const i = processingStatusInfo(status)
  return <Badge tone={i.tone} dot title={i.hint}>{i.label}</Badge>
}

/** Confidence as "0.87 · 87%" with a small bar. */
export function ConfidenceCell({ value }: { value: number }) {
  return (
    <span className="flex items-center gap-2.5">
      <span className="h-1.5 w-20 overflow-hidden rounded-full bg-panel-2" aria-hidden>
        <span className={cx('block h-full rounded-full', value >= 0.75 ? 'bg-ok' : value >= 0.5 ? 'bg-warn' : 'bg-fg-3')} style={{ width: `${value * 100}%` }} />
      </span>
      <span className="tabular font-mono text-[12px] text-fg">{confidenceText(value)}</span>
    </span>
  )
}

/** The analysis outcome in words. Zero detections reads "No litter detected by the model", never "clean". */
export function ProcessingBanner({ status, detail }: { status: string | null | undefined; detail?: string | null }) {
  const i = processingStatusInfo(status)
  return (
    <div role="status" className={cx('flex flex-wrap items-center gap-x-3 gap-y-1 rounded-lg border px-3 py-2.5',
      i.tone === 'crit' ? 'border-crit/40 bg-crit-bg' : i.tone === 'info' ? 'border-info/40 bg-info-bg' : 'border-line bg-panel-2')}>
      <span className={cx('text-[15px] font-semibold', i.tone === 'crit' ? 'text-crit' : 'text-fg')}>{i.label}</span>
      <span className="text-[12.5px] text-fg-2">{detail ?? i.hint}</span>
    </div>
  )
}

export function Warnings({ title, items }: { title: string; items: string[] | null | undefined }) {
  if (!items?.length) return null
  return (
    <div role="alert" className="rounded-lg border border-warn/40 bg-warn-bg px-3 py-2 text-[12.5px] text-warn">
      <div className="flex items-center gap-1.5 font-medium"><AlertTriangle className="size-3.5 shrink-0" aria-hidden />{title}</div>
      <ul className="mt-1 flex list-disc flex-col gap-0.5 pl-5">{items.map((w, i) => <li key={i}>{w}</li>)}</ul>
    </div>
  )
}

/** Store / camera / recording the result is recorded under, with the server's identity warnings shown visibly. */
export function IdentityBlock({ identity }: { identity: Identity | null | undefined }) {
  if (!identity) return <p className="text-[12px] text-fg-3">No identity block in this result (older analysis).</p>
  const unknown = (v: string) => !v || v === 'UNKNOWN'
  const item = (k: string, id: string, name?: string) => (
    <div className="min-w-0 rounded-lg border border-line bg-panel-2 px-3 py-2">
      <dt className="text-[11px] font-medium text-fg-3">{k}</dt>
      <dd className={cx('truncate text-[13px] font-medium', unknown(id) ? 'text-warn' : 'text-fg')} title={id}>
        <span className="font-mono">{id || 'UNKNOWN'}</span>{name && !unknown(name) && <span className="font-normal text-fg-3"> · {name}</span>}
      </dd>
    </div>
  )
  return (
    <div className="flex flex-col gap-2">
      <dl className="grid grid-cols-1 gap-2 sm:grid-cols-3">
        {item('Store', identity.store_id, identity.store_name)}
        {item('Camera', identity.camera_id, identity.camera_name)}
        {item('Recording / analysis ID', identity.recording_id)}
      </dl>
      <Warnings title="Identity warnings" items={identity.warnings} />
    </div>
  )
}

export interface DownloadItem { label: string; path: string | null | undefined; filename: string }

/** Authenticated downloads (plain links cannot send the bearer token). Missing files are not offered. */
export function Downloads({ items }: { items: DownloadItem[] }) {
  const { toast } = useApp()
  const [busy, setBusy] = useState<string | null>(null)
  const present = items.filter((i): i is DownloadItem & { path: string } => !!i.path)
  if (!present.length) return <p className="text-[12px] text-fg-3">No files to download.</p>
  const get = async (i: DownloadItem & { path: string }) => {
    setBusy(i.label)
    try { await downloadAuthed(i.path, fileNameFromPath(i.path, i.filename)) } catch (e) {
      toast({ tone: 'crit', title: 'Download failed', body: (e as Error).message })
    } finally { setBusy(null) }
  }
  return (
    <div className="flex flex-wrap gap-2">
      {present.map((i) => <Button key={i.label} size="sm" icon={Download} loading={busy === i.label} onClick={() => get(i)}>{i.label}</Button>)}
    </div>
  )
}
