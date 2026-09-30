'use client'

import Link from 'next/link'
import { useEffect, useState } from 'react'
import { Check, EyeOff, ScanSearch, X } from 'lucide-react'
import { url } from '../../api/client'
import { Api } from '../../api/endpoints'
import type { HardExample } from '../../api/types'
import { invalidate, useQuery } from '../../hooks/useQuery'
import { className as cls, dateTime, pct } from '../../lib/format'
import { useApp } from '../../stores/app'
import { Badge, Button, PageHeader, Panel, Segmented, StatusBadge, Tabs } from '../../components/ui/primitives'
import { EmptyState, ErrorState, LoadingState } from '../../components/ui/states'

const KINDS = [
  { value: 'false_positive', label: 'False positives' }, { value: 'false_negative', label: 'False negatives' },
  { value: 'tiny', label: 'Tiny objects' }, { value: 'blurred', label: 'Blurred' }, { value: 'occluded', label: 'Occluded' },
  { value: 'low_confidence', label: 'Low confidence' },
] as const
type Kind = (typeof KINDS)[number]['value']

export default function HardExamplesPage() {
  const [kind, setKind] = useState<Kind>('false_negative')
  const [status, setStatus] = useState<'pending' | 'approved' | 'rejected' | 'ignored' | 'all'>('pending')
  const [offset, setOffset] = useState(0)
  const LIMIT = 36
  const q = useQuery(`hard:${kind}:${status}:${offset}`, () => Api.hardExamples({ kind, status: status === 'all' ? undefined : status, offset, limit: LIMIT }))
  const counts = useQuery('hard:counts', () => Api.hardExamples({ limit: 1 }))
  useEffect(() => setOffset(0), [kind, status])
  const countFor = (k: string) => (counts.data?.counts ?? []).filter((c) => c.kind === k && (status === 'all' || c.review_status === status)).reduce((a, c) => a + c.n, 0)

  return (
    <div className="flex flex-col gap-3">
      <PageHeader title="Hard examples" description="Failure cases mined automatically by evaluations (misses, false alarms, tiny/blurred/low-confidence objects) and flagged manually in the Test Lab. Approved items are the queue for the next labelling and retraining round." />
      <Panel noPad>
        <Tabs value={kind} onChange={setKind} tabs={KINDS.map((k) => ({ value: k.value, label: k.label, count: counts.data ? countFor(k.value) : undefined }))} />
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-line p-3">
          <Segmented label="Review status" value={status} onChange={setStatus} size="sm"
            options={[{ value: 'pending', label: 'Pending' }, { value: 'approved', label: 'Approved' }, { value: 'rejected', label: 'Rejected' }, { value: 'ignored', label: 'Ignored' }, { value: 'all', label: 'All' }]} />
          <span className="text-[12px] text-fg-3">{q.data ? `${q.data.total} items` : ''}</span>
        </div>
        <div className="p-3">
          {kind === 'occluded' && <p className="mb-3 text-[12px] text-fg-3">Occlusion cannot be measured automatically; items appear here only when tagged manually.</p>}
          {q.error ? <ErrorState error={q.error} compact onRetry={q.refetch} /> : !q.data ? <LoadingState rows={4} /> : q.data.total === 0 ? (
            <EmptyState icon={ScanSearch} title="Nothing here" body={<>Run an <Link href="/evaluation" className="text-info hover:underline">evaluation</Link> to mine hard examples, or mark predictions in the <Link href="/test-lab" className="text-info hover:underline">Test Lab</Link>.</>} />
          ) : (
            <>
              <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3 2xl:grid-cols-4">
                {q.data.items.map((h) => <HardCard key={h.id} h={h} />)}
              </div>
              {q.data.total > LIMIT && (
                <div className="mt-3 flex justify-end gap-1">
                  <Button size="sm" disabled={offset === 0} onClick={() => setOffset(offset - LIMIT)}>Previous</Button>
                  <Button size="sm" disabled={offset + LIMIT >= q.data.total} onClick={() => setOffset(offset + LIMIT)}>Next</Button>
                </div>
              )}
            </>
          )}
        </div>
      </Panel>
    </div>
  )
}

/** Crop preview rendered from the dataset image via CSS (no extra server work). */
function Crop({ h }: { h: HardExample }) {
  const [dims, setDims] = useState<[number, number] | null>(null)
  const src = h.dataset && h.image ? url(`/api/datasets/${encodeURIComponent(h.dataset)}/image?id=${encodeURIComponent(h.image)}`) : null
  useEffect(() => {
    if (!src) return
    const img = new Image()
    img.onload = () => setDims([img.naturalWidth, img.naturalHeight])
    img.src = src
  }, [src])
  if (!src) return <div className="aspect-[4/3] bg-panel-2" />
  const bw = Math.max(8, h.x2 - h.x1), bh = Math.max(8, h.y2 - h.y1)
  const pad = Math.max(bw, bh) * 0.9
  const vx = h.x1 - pad, vy = h.y1 - pad, vw = bw + 2 * pad, vh = bh + 2 * pad
  return (
    <div className="relative aspect-[4/3] overflow-hidden bg-video">
      {dims && (
        <svg viewBox={`${vx} ${vy} ${vw} ${vh}`} preserveAspectRatio="xMidYMid meet" className="absolute inset-0 size-full" role="img" aria-label={`${h.class} crop`}>
          <image href={src} x={0} y={0} width={dims[0]} height={dims[1]} />
          <rect x={h.x1} y={h.y1} width={bw} height={bh} fill="none" stroke={h.kind === 'false_positive' ? 'var(--crit)' : 'var(--warn)'} strokeWidth={Math.max(1, vw / 150)} />
        </svg>
      )}
    </div>
  )
}

function HardCard({ h }: { h: HardExample }) {
  const { toast } = useApp()
  const [busy, setBusy] = useState(false)
  const set = async (s: string) => {
    setBusy(true)
    try {
      await Api.updateHardExample(h.id, s)
      invalidate('hard')
    } catch (e) {
      toast({ tone: 'crit', title: 'Update failed', body: (e as Error).message })
    } finally {
      setBusy(false)
    }
  }
  return (
    <article className="flex flex-col overflow-hidden rounded-md border border-line bg-panel-2">
      <Crop h={h} />
      <div className="flex flex-col gap-1 p-2.5 text-[12px]">
        <div className="flex items-center justify-between gap-2">
          <span className="truncate font-medium uppercase text-fg">{h.class ? cls(h.class) : '—'}</span>
          <StatusBadge status={h.review_status} />
        </div>
        <div className="flex flex-wrap gap-x-3 text-fg-3">
          <span>conf {h.confidence != null ? pct(h.confidence, 0) : 'none'}</span>
          <span className="font-mono">{h.model_version}</span>
        </div>
        {h.image && <Link href={`/ground-truth?dataset=${encodeURIComponent(h.dataset ?? '')}&id=${encodeURIComponent(h.image)}`} className="truncate font-mono text-[11px] text-info hover:underline">{h.image}</Link>}
        <div className="flex items-center justify-between text-[11px] text-fg-3"><span>{h.source.split(':')[0]}</span><span>{dateTime(h.created_at)}</span></div>
      </div>
      <div className="mt-auto flex gap-1 border-t border-line p-1.5">
        <Button size="sm" variant="ghost" icon={Check} disabled={busy} onClick={() => set('approved')} title="Keep for the next training round">Approve</Button>
        <Button size="sm" variant="ghost" icon={X} disabled={busy} onClick={() => set('rejected')} title="Not a real failure (e.g. label error)">Reject</Button>
        <Button size="sm" variant="ghost" icon={EyeOff} disabled={busy} onClick={() => set('ignored')}>Ignore</Button>
        <Badge className="ml-auto self-center" title="Copying into a dataset version is done via the Dataset Lab + Ground Truth editor">add via GT</Badge>
      </div>
    </article>
  )
}
