'use client'

import { useEffect, useMemo, useState } from 'react'
import { Columns2, Eye, ImageIcon, Layers, Play, RotateCcw, ScanSearch } from 'lucide-react'
import { Api } from '../../api/endpoints'
import type { Detection, ImageDetectResult } from '../../api/types'
import { invalidate, useQuery } from '../../hooks/useQuery'
import { className as cls, cx, num, pct, time } from '../../lib/format'
import { useApp } from '../../stores/app'
import { Badge, Button, PageHeader, Panel, Segmented, StatusBadge, Toggle } from '../../components/ui/primitives'
import { MetricCard } from '../../components/ui/kpi'
import { DataTable } from '../../components/ui/table'
import { EmptyState, ErrorState } from '../../components/ui/states'
import { AspectBox, ImageOverlay, STATUS_COLOR, type OverlayBox } from '../../components/vision/Overlay'
import { DetectionSettingsForm, DropZone, RoiSetup, litterName, toForm, type DetectConfig } from './common'

type View = 'annotated' | 'original' | 'compare'

const LITTER_COLOR = '#ff5a1f'

export default function ImageDetectionPage() {
  const { toast } = useApp()
  const settings = useQuery('settings', Api.settings)
  const classes = useQuery('classes', Api.classes)
  const classInfo = useMemo(() => Object.fromEntries((classes.data?.classes ?? []).map((c) => [c.name, c])), [classes.data])
  const [file, setFile] = useState<File | null>(null)
  const [src, setSrc] = useState<string | null>(null)
  const [dims, setDims] = useState<[number, number]>([0, 0])
  const [cfg, setCfg] = useState<DetectConfig>({ model_version: '', confidence: 0.6, image_size: '', tiling: 'auto', camera_id: 'upload', use_roi: true })
  const [createAlerts, setCreateAlerts] = useState(false)
  const [result, setResult] = useState<ImageDetectResult | null>(null)
  const [error, setError] = useState<Error | null>(null)
  const [pending, setPending] = useState(false)
  const [view, setView] = useState<View>('annotated')
  const [showRejected, setShowRejected] = useState(false)
  const [selected, setSelected] = useState<number | null>(null)
  const [detectedAt, setDetectedAt] = useState<string | null>(null)

  useEffect(() => {
    const t = settings.data?.settings.confidence.confirmed_threshold
    if (t) setCfg((c) => ({ ...c, confidence: t }))
  }, [settings.data])

  const onFile = (f: File) => {
    if (src) URL.revokeObjectURL(src)
    const u = URL.createObjectURL(f)
    setFile(f)
    setSrc(u)
    setResult(null)
    setError(null)
    const img = new Image()
    img.onload = () => setDims([img.naturalWidth, img.naturalHeight])
    img.src = u
  }

  const run = async () => {
    if (!file) return
    setPending(true)
    setError(null)
    try {
      const r = await Api.detectImage(toForm(file, cfg, { create_alerts: String(createAlerts), return_image: 'false' }))
      setResult(r)
      setDetectedAt(new Date().toISOString())
      setSelected(null)
      if (r.alerts.length) { toast({ tone: 'warn', title: `${r.alerts.length} alert(s) created`, body: `Camera ${r.camera_id}` }); invalidate('alerts') }
    } catch (e) {
      setError(e as Error)
    } finally {
      setPending(false)
    }
  }

  const dets = useMemo(() => (result?.detections ?? []).map((d, i) => ({ ...d, _i: i })), [result])
  const boxes: OverlayBox[] = useMemo(() => dets
    .filter((d) => showRejected || d.status === 'confirmed')
    .map((d) => ({
      key: d._i, bbox: d.bbox, color: d.status === 'confirmed' ? LITTER_COLOR : STATUS_COLOR[d.status], dashed: d.status === 'candidate', thin: d.status !== 'confirmed',
      label: d.status === 'weak' || d.status === 'rejected' ? undefined : litterName(d.class), sub: pct(d.confidence, 0),
      selected: selected === d._i, onClick: () => setSelected(d._i),
    })), [dets, showRejected, selected])

  const [w, h] = result?.image_size ?? dims

  return (
    <div className="flex flex-col gap-3">
      <PageHeader title="Image detection" description="Upload Image → Define ROI (the outdoor ground area) → Analyze Image. Litter is detected only inside the ROI; a still image has no temporal evidence, so an object counts as litter when it reaches the confirmed threshold." />
      <div className="grid gap-3 xl:grid-cols-[minmax(0,1fr)_320px]">
        <Panel title={file ? file.name : 'Image'} icon={ImageIcon} noPad
          actions={file && <>
            <Segmented size="sm" label="Image view" value={view} onChange={setView} options={[
              { value: 'annotated', label: 'Annotated', icon: ScanSearch }, { value: 'original', label: 'Original', icon: Eye }, { value: 'compare', label: 'Compare', icon: Columns2 }]} />
            <Button size="sm" variant="ghost" icon={RotateCcw} onClick={() => { setFile(null); setSrc(null); setResult(null) }}>Clear</Button>
          </>}>
          <div className="p-2">
            {!file ? <DropZone accept="image/jpeg,image/png,image/webp,image/bmp" onFile={onFile} label="Drop an image or click to browse" hint="JPEG, PNG, WebP or BMP · the server enforces the size limit" /> : (
              <div className={cx('grid gap-2', view === 'compare' && 'md:grid-cols-2')}>
                {(view === 'compare' || view === 'original') && (
                  <AspectBox width={w} height={h} className="overflow-hidden rounded-md border border-line">
                    <ImageOverlay src={src} width={w} height={h} className="absolute inset-0" alt="Original image" />
                  </AspectBox>
                )}
                {view === 'compare' && (
                  <AspectBox width={w} height={h} className="overflow-hidden rounded-md border border-line">
                    <ImageOverlay src={src} width={w} height={h} boxes={result ? boxes : []} className="absolute inset-0" alt="Image with detections" />
                  </AspectBox>
                )}
                {view === 'annotated' && <RoiSetup cameraId={cfg.camera_id} src={src} width={w} height={h} boxes={result ? boxes : []} disabled={pending} />}
              </div>
            )}
          </div>
        </Panel>
        <Panel title="Detection settings" icon={Layers}>
          <DetectionSettingsForm cfg={cfg} setCfg={setCfg}>
            <Toggle label="Create alerts" checked={createAlerts} onChange={setCreateAlerts} description="Off = test only (detections are still logged)." />
            <Button variant="primary" icon={Play} disabled={!file} loading={pending} onClick={run} className="mt-1 w-full">Analyze Image</Button>
          </DetectionSettingsForm>
        </Panel>
      </div>

      {error && <Panel><ErrorState error={error} onRetry={run} title="Detection failed" compact /></Panel>}
      {result && (
        <Panel title="Detection results" subtitle={`model ${result.model_version} · ${result.tiled ? `${result.tiles} tiles` : 'full frame'} · ${result.image_size.join('×')}px`}
          actions={<Toggle label="Show low-confidence & ignored" checked={showRejected} onChange={setShowRejected} />} noPad>
          <div className="grid grid-cols-2 gap-2 p-3 md:grid-cols-6">
            <MetricCard label="Litter detected" value={result.counts.confirmed} tone={result.counts.confirmed ? 'crit' : 'ok'} />
            <MetricCard label="Litter types" value={Object.keys(result.by_class).length} hint={Object.entries(result.by_class).map(([k, v]) => `${v}× ${litterName(k)}`).join(', ') || '—'} />
            <MetricCard label="Highest confidence" value={result.highest_confidence != null ? pct(result.highest_confidence, 0) : '—'} />
            <MetricCard label="Average confidence" value={result.average_confidence != null ? pct(result.average_confidence, 0) : '—'} />
            <MetricCard label="Inside ROI" value={result.roi_active ? result.counts.confirmed : '—'} hint={result.roi_active ? `${num(result.outside_roi_ignored ?? 0)} outside-ROI ignored` : 'ROI not set'} />
            <MetricCard label="Inference time" value={`${num(result.inference_ms, 0)} ms`} />
          </div>
          <DataTable<Detection & { _i: number }> rows={dets.filter((d) => showRejected || d.status === 'confirmed')} rowKey={(d) => d._i} selectedKey={selected}
            onRowClick={(d) => setSelected(d._i)} dense
            empty={<EmptyState compact title="No litter detected" body="Nothing inside the ROI reached the confirmed threshold. Turn on “Show low-confidence & ignored” to see weaker or outside-ROI detections." />}
            columns={[
              { key: 'n', header: '#', cell: (d) => <span className="text-fg-3">{d._i + 1}</span>, width: '40px' },
              { key: 'class', header: 'Detected object', cell: (d) => <span className="rounded-md border border-line bg-panel-2 px-2 py-0.5 text-[12.5px] text-fg">{litterName(d.class)}</span>, sortValue: (d) => d.class },
              { key: 'conf', header: 'Confidence', cell: (d) => (
                <span className="flex items-center gap-2.5">
                  <span className="h-1.5 w-20 overflow-hidden rounded-full bg-panel-2"><span className="block h-full rounded-full bg-info" style={{ width: `${d.confidence * 100}%` }} /></span>
                  <span className="tabular text-fg">{pct(d.confidence, 1)}</span>
                </span>), sortValue: (d) => d.confidence },
              { key: 'prio', header: 'Priority', cell: (d) => { const p = classInfo[d.class]?.priority ?? 'medium'; return <Badge tone={p === 'critical' ? 'crit' : p === 'high' ? 'warn' : p === 'medium' ? 'info' : 'ok'} dot>{p[0].toUpperCase() + p.slice(1)}</Badge> }, sortValue: (d) => ({ critical: 4, high: 3, medium: 2, low: 1 } as Record<string, number>)[classInfo[d.class]?.priority ?? 'medium'] },
              { key: 'loc', header: 'Location / Area', cell: (d) => <span><span className="text-fg">{region(d.bbox, w, h)}</span> <span className="text-fg-3">· {Math.round((d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1])).toLocaleString()} px²</span></span>, sortValue: (d) => (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]) },
              { key: 'time', header: 'Detection time', cell: () => <span className="text-fg-2">{detectedAt ? time(detectedAt) : '—'}</span> },
              { key: 'status', header: 'Status', cell: (d) => <StatusBadge status={d.status} />, sortValue: (d) => d.status, hideable: true },
              { key: 'reason', header: 'Reason', cell: (d) => d.reason ? <Badge>{d.reason.replace(/_/g, ' ')}</Badge> : '—', hideable: true, defaultHidden: true },
            ]} />
        </Panel>
      )}
    </div>
  )
}

/** Human-readable position of a box in the frame, e.g. "bottom-left". */
function region(b: number[], w: number, h: number): string {
  if (!w || !h) return '—'
  const cx = (b[0] + b[2]) / 2 / w, cy = (b[1] + b[3]) / 2 / h
  const v = cy < 1 / 3 ? 'top' : cy < 2 / 3 ? 'middle' : 'bottom'
  const hz = cx < 1 / 3 ? 'left' : cx < 2 / 3 ? 'center' : 'right'
  return v === 'middle' && hz === 'center' ? 'center' : `${v}-${hz}`
}
