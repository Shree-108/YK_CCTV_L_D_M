'use client'

import Link from 'next/link'
import { useRouter, useSearchParams } from 'next/navigation'
import { useEffect, useMemo, useRef, useState } from 'react'
import { Cctv, Expand, LayoutGrid, Map as MapIcon, MapPinOff, Pause, Play, Plus } from 'lucide-react'
import { Api } from '../../api/endpoints'
import type { Camera } from '../../api/types'
import { useQuery } from '../../hooks/useQuery'
import { cx } from '../../lib/format'
import { Button, Dot, PageHeader, Segmented, Select, Toggle } from '../../components/ui/primitives'
import { EmptyState, ErrorState, Skeleton } from '../../components/ui/states'
import { CameraDrawer, CameraTile, DEFAULT_OVERLAY, camTone, type OverlayOpts } from './live'

const LAYOUTS = [1, 2, 4, 6, 9] as const
type Layout = (typeof LAYOUTS)[number]
const GRID: Record<Layout, string> = {
  1: 'grid-cols-1', 2: 'grid-cols-1 md:grid-cols-2', 4: 'grid-cols-1 sm:grid-cols-2', 6: 'grid-cols-1 sm:grid-cols-2 xl:grid-cols-3', 9: 'grid-cols-2 xl:grid-cols-3',
}

function readOverlay(): OverlayOpts {
  try { return { ...DEFAULT_OVERLAY, ...JSON.parse(localStorage.getItem('lm.liveOverlay') ?? '{}') } } catch { return DEFAULT_OVERLAY }
}

export default function LivePage() {
  const params = useSearchParams()
  const router = useRouter()
  const focus = params.get('focus')
  const cams = useQuery('cameras:list', Api.cameras, { poll: 10000 })
  const [layout, setLayout] = useState<Layout>(4)
  const [view, setView] = useState<'grid' | 'map'>('grid')
  const [filter, setFilter] = useState<'all' | 'online' | 'issues'>('all')
  const [overlay, setOverlay] = useState<OverlayOpts>(DEFAULT_OVERLAY)
  const [paused, setPaused] = useState(false)
  const [page, setPage] = useState(0)
  const [drawer, setDrawer] = useState<string | null>(null)
  const wallRef = useRef<HTMLDivElement>(null)

  useEffect(() => setOverlay(readOverlay()), [])
  useEffect(() => { try { localStorage.setItem('lm.liveOverlay', JSON.stringify(overlay)) } catch { /* ignore */ } }, [overlay])
  useEffect(() => { if (focus) { setLayout(1); setView('grid') } }, [focus])

  const all = cams.data?.cameras ?? []
  const list = useMemo(() => {
    let l = all
    if (focus) l = all.filter((c) => c.id === focus)
    else if (filter === 'online') l = all.filter((c) => c.status.state === 'online')
    else if (filter === 'issues') l = all.filter((c) => c.enabled && c.status.state !== 'online')
    return l
  }, [all, filter, focus])
  const pages = Math.max(1, Math.ceil(list.length / layout))
  const shown = list.slice(page * layout, page * layout + layout)
  useEffect(() => { if (page >= pages) setPage(0) }, [pages, page])
  const drawerCam = all.find((c) => c.id === drawer) ?? null

  return (
    <div>
      <PageHeader title="Live CCTV"
        description="Every box is a real detection from the active model on the latest processed frame. Solid green = confirmed (temporal), dashed amber = candidate."
        actions={<>
          {focus && <Button variant="ghost" onClick={() => router.push('/live')}>Show all cameras</Button>}
          <Segmented label="View" value={view} onChange={setView} options={[{ value: 'grid', label: 'Grid', icon: LayoutGrid }, { value: 'map', label: 'Map', icon: MapIcon }]} />
          {view === 'grid' && !focus && (
            <Segmented label="Layout" value={String(layout) as `${Layout}`} onChange={(v) => { setLayout(Number(v) as Layout); setPage(0) }}
              options={LAYOUTS.map((n) => ({ value: String(n) as `${Layout}`, label: String(n), title: `${n} camera${n > 1 ? 's' : ''}` }))} />
          )}
          <Button icon={paused ? Play : Pause} onClick={() => setPaused((p) => !p)}>{paused ? 'Resume all' : 'Pause all'}</Button>
          <Button icon={Expand} onClick={() => wallRef.current?.requestFullscreen?.()} disabled={view !== 'grid'}>Fullscreen</Button>
        </>} />

      <div className="mb-3 flex flex-wrap items-center gap-x-5 gap-y-2 rounded-lg border border-line bg-panel px-3 py-2">
        {!focus && (
          <Select aria-label="Camera filter" value={filter} onChange={(e) => { setFilter(e.target.value as typeof filter); setPage(0) }} className="w-44">
            <option value="all">All cameras ({all.length})</option>
            <option value="online">Online ({all.filter((c) => c.status.state === 'online').length})</option>
            <option value="issues">With issues ({all.filter((c) => c.enabled && c.status.state !== 'online').length})</option>
          </Select>
        )}
        {([['detections', 'Detections'], ['ids', 'Track IDs'], ['confidence', 'Confidence'], ['roi', 'ROI'], ['weak', 'Weak boxes']] as const).map(([k, l]) => (
          <div key={k} className="w-[128px]"><Toggle label={l} checked={overlay[k]} onChange={(v) => setOverlay((o) => ({ ...o, [k]: v }))} /></div>
        ))}
        {pages > 1 && view === 'grid' && (
          <div className="ml-auto flex items-center gap-1 text-[12px] text-fg-3">
            {Array.from({ length: pages }, (_, i) => (
              <button key={i} onClick={() => setPage(i)} aria-label={`Page ${i + 1}`} aria-current={i === page}
                className={cx('size-6 rounded-md', i === page ? 'bg-hover text-fg' : 'hover:bg-hover')}>{i + 1}</button>
            ))}
          </div>
        )}
      </div>

      {cams.error && !cams.data ? <ErrorState error={cams.error} onRetry={cams.refetch} /> :
        !cams.data ? <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">{[0, 1, 2, 3].map((i) => <Skeleton key={i} className="aspect-video" />)}</div> :
        all.length === 0 ? (
          <EmptyState icon={Cctv} title="No cameras configured" body="Connect your first CCTV camera to begin monitoring."
            action={<Link href="/cameras?new=1"><Button variant="primary" icon={Plus}>Add camera</Button></Link>} />
        ) : list.length === 0 ? (
          <EmptyState title="No cameras match this filter" action={<Button onClick={() => setFilter('all')}>Show all</Button>} />
        ) : view === 'map' ? (
          <CameraMap cameras={list} onOpen={(c) => setDrawer(c.id)} />
        ) : (
          <div ref={wallRef} className={cx('grid gap-2 bg-bg', GRID[layout])}>
            {shown.map((c) => (
              <CameraTile key={c.id} cam={c} overlay={overlay} paused={paused} onOpen={() => setDrawer(c.id)}
                compact={layout >= 6} heightClass={layout === 1 ? 'aspect-video max-h-[78vh]' : 'aspect-video'} />
            ))}
          </div>
        )}
      <CameraDrawer cam={drawerCam} onClose={() => setDrawer(null)} />
    </div>
  )
}

/** Schematic map: positions cameras by their configured coordinates. No basemap tiles are loaded. */
function CameraMap({ cameras, onOpen }: { cameras: Camera[]; onOpen: (c: Camera) => void }) {
  const located = cameras.filter((c) => c.latitude != null && c.longitude != null)
  const missing = cameras.filter((c) => c.latitude == null || c.longitude == null)
  const lats = located.map((c) => c.latitude!), lons = located.map((c) => c.longitude!)
  const [minLa, maxLa, minLo, maxLo] = [Math.min(...lats), Math.max(...lats), Math.min(...lons), Math.max(...lons)]
  const pos = (c: Camera) => ({
    left: `${located.length > 1 && maxLo > minLo ? 8 + ((c.longitude! - minLo) / (maxLo - minLo)) * 84 : 50}%`,
    top: `${located.length > 1 && maxLa > minLa ? 8 + (1 - (c.latitude! - minLa) / (maxLa - minLa)) * 84 : 50}%`,
  })
  return (
    <div className="grid gap-3 lg:grid-cols-[1fr_280px]">
      <div className="relative h-[60vh] min-h-[360px] overflow-hidden rounded-lg border border-line bg-panel"
        style={{ backgroundImage: 'linear-gradient(var(--line) 1px, transparent 1px), linear-gradient(90deg, var(--line) 1px, transparent 1px)', backgroundSize: '48px 48px' }}>
        <span className="absolute left-3 top-2 text-[11px] text-fg-3">Schematic layout from configured coordinates (no basemap)</span>
        {located.length === 0 && <div className="absolute inset-0 flex items-center justify-center"><EmptyState icon={MapPinOff} title="No camera has coordinates" body="Add latitude/longitude in the camera configuration to place it here." /></div>}
        {located.map((c) => (
          <button key={c.id} onClick={() => onOpen(c)} style={pos(c)} aria-label={`${c.name}: ${c.status.state}`}
            className="group absolute -translate-x-1/2 -translate-y-1/2 flex flex-col items-center gap-1">
            <span className={cx('flex size-7 items-center justify-center rounded-full border-2 bg-panel',
              camTone(c.status.state) === 'ok' ? 'border-ok' : camTone(c.status.state) === 'crit' ? 'border-crit' : 'border-warn')}>
              <Cctv className="size-3.5 text-fg" aria-hidden />
            </span>
            <span className="rounded-sm bg-elev px-1.5 text-[11px] text-fg shadow-[var(--shadow)]">{c.id}</span>
          </button>
        ))}
      </div>
      <div className="rounded-lg border border-line bg-panel p-3">
        <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-fg-3">Location unavailable ({missing.length})</h3>
        {missing.length === 0 ? <p className="text-[12px] text-fg-3">All cameras have coordinates.</p> : (
          <ul className="flex flex-col gap-1">
            {missing.map((c) => (
              <li key={c.id}><button onClick={() => onOpen(c)} className="flex w-full items-center gap-2 rounded-md px-1.5 py-1 text-left text-[12px] hover:bg-hover">
                <Dot tone={camTone(c.status.state)} /> <span className="text-fg">{c.name}</span><span className="ml-auto font-mono text-fg-3">{c.id}</span>
              </button></li>
            ))}
          </ul>
        )}
        <div className="mt-3 flex flex-col gap-1 border-t border-line pt-2 text-[11px] text-fg-3">
          <span className="flex items-center gap-1.5"><Dot tone="ok" /> online</span>
          <span className="flex items-center gap-1.5"><Dot tone="warn" /> connecting / degraded</span>
          <span className="flex items-center gap-1.5"><Dot tone="crit" /> offline / reconnecting</span>
        </div>
      </div>
    </div>
  )
}
