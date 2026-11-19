'use client'

import { useEffect, useMemo, useState } from 'react'
import { Columns2, Eye, FileDown, ImageIcon, Layers, ListChecks, Play, RotateCcw, ScanSearch } from 'lucide-react'
import { Api } from '../../api/endpoints'
import type { Detection, ImageDetectResult } from '../../api/types'
import { invalidate, useQuery } from '../../hooks/useQuery'
import { cx, num } from '../../lib/format'
import { confidenceText, detectionStatusOf } from '../../lib/litter'
import { lockedFor, useApp } from '../../stores/app'
import { Button, PageHeader, Panel, Segmented, Toggle } from '../../components/ui/primitives'
import { MetricCard } from '../../components/ui/kpi'
import { DataTable } from '../../components/ui/table'
import { EmptyState, ErrorState } from '../../components/ui/states'
import { AspectBox, ImageOverlay, type OverlayBox } from '../../components/vision/Overlay'
import { StoreCameraSelect, type StoreCamera } from '../stores/StoreCameraSelect'
import { ConfidenceCell, DetectionStatusBadge, Downloads, IdentityBlock, ProcessingBanner } from './AnalysisOutputs'
import { DetectionSettingsForm, DropZone, FileInfo, Steps, fileExt, fileSize, litterName, toForm, type DetectConfig } from './common'

type View = 'annotated' | 'original' | 'compare'
type Row = Detection & { _i: number; _status: string }

/** Box colour per detection status: DETECTED solid orange, UNDER_REVIEW dashed amber, REJECTED dashed grey. */
const BOX: Record<string, { color: string; dashed: boolean; thin: boolean }> = {
  DETECTED: { color: '#ff5a1f', dashed: false, thin: false },
  UNDER_REVIEW: { color: 'var(--warn)', dashed: true, thin: true },
  REJECTED: { color: 'var(--fg-3)', dashed: true, thin: true },
}

export default function ImageDetectionPage() {
  const { toast, canOperate, role, loginOn } = useApp()
  const opLock = lockedFor('operator', role, loginOn)
  const settings = useQuery('settings', Api.settings)
  const [file, setFile] = useState<File | null>(null)
  const [src, setSrc] = useState<string | null>(null)
  const [dims, setDims] = useState<[number, number]>([0, 0])
  const [cfg, setCfg] = useState<DetectConfig>({ model_version: '', confidence: 0.6, image_size: '', tiling: 'auto' })
  const [who, setWho] = useState<StoreCamera>({ store_id: '', camera_id: '' })
  const [createAlerts, setCreateAlerts] = useState(false)
  const [result, setResult] = useState<ImageDetectResult | null>(null)
  const [error, setError] = useState<Error | null>(null)
  const [pending, setPending] = useState(false)
  const [view, setView] = useState<View>('annotated')
  const [selected, setSelected] = useState<number | null>(null)

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
      const r = await Api.detectImage(toForm(file, cfg, who, { create_alerts: String(createAlerts), return_image: 'false', save_outputs: 'true' }))
      setResult(r)
      setSelected(null)
      invalidate('litter')
      if (r.alerts.length) { toast({ tone: 'warn', title: `${r.alerts.length} alert(s) created` }); invalidate('alerts') }
    } catch (e) {
      setError(e as Error)
    } finally {
      setPending(false)
    }
  }

  const rows: Row[] = useMemo(() => (result?.detections ?? []).map((d, i) => ({ ...d, _i: i, _status: detectionStatusOf(d) })), [result])
  const boxes: OverlayBox[] = useMemo(() => rows.map((d) => {
    const b = BOX[d._status] ?? BOX.UNDER_REVIEW
    return {
      key: d._i, bbox: d.bbox, color: b.color, dashed: b.dashed, thin: b.thin,
      label: d._status === 'REJECTED' ? undefined : `${d._i + 1} ${litterName(d.class)}`, sub: d.confidence.toFixed(2),
      selected: selected === d._i, onClick: () => setSelected(d._i),
    }
  }), [rows, selected])
  const count = (s: string) => rows.filter((r) => r._status === s).length

  const [w, h] = result?.image_size ?? dims
  const a = result?.analysis
  const stamp = a?.analysis_id ?? 'image'
  return (
    <div className="flex flex-col gap-4">
      <PageHeader title="Image Analysis" description="The whole image is analysed. Every detection the model returns is listed with its confidence and status: only DETECTED objects count as litter events; UNDER REVIEW and REJECTED are shown for transparency." />
      <div className="rounded-xl border border-line bg-panel px-4 py-3">
        <Steps steps={['Upload image', 'Store & camera', 'Analyze', 'Review results']} current={!file ? 0 : result ? 3 : pending ? 2 : 1} />
      </div>
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_340px]">
        <Panel title={file ? file.name : 'Image'} icon={ImageIcon} noPad
          actions={file && <>
            <Segmented size="sm" label="Image view" value={view} onChange={setView} options={[
              { value: 'annotated', label: 'Annotated', icon: ScanSearch }, { value: 'original', label: 'Original', icon: Eye }, { value: 'compare', label: 'Compare', icon: Columns2 }]} />
            <Button size="sm" variant="ghost" icon={RotateCcw} onClick={() => { setFile(null); setSrc(null); setResult(null) }}>Clear</Button>
          </>}>
          <div className="flex flex-col gap-3 p-3">
            {!file ? <DropZone accept="image/jpeg,image/png,image/webp,image/bmp" onFile={onFile} label="Drag and drop a CCTV image here" hint="JPEG, PNG, WebP or BMP · the server enforces the size limit" /> : (<>
              <FileInfo items={[['File name', file.name], ['Size', fileSize(file.size)], ['Format', fileExt(file)], ['Resolution', w && h ? `${w} × ${h}px` : '—']]} />
              <div className={cx('grid gap-2', view === 'compare' && 'md:grid-cols-2')}>
                {view !== 'annotated' && (
                  <AspectBox width={w} height={h} className="overflow-hidden rounded-md border border-line">
                    <ImageOverlay src={src} width={w} height={h} className="absolute inset-0" alt="Original image" />
                  </AspectBox>
                )}
                {view !== 'original' && (
                  <AspectBox width={w} height={h} className="overflow-hidden rounded-md border border-line">
                    <ImageOverlay src={src} width={w} height={h} boxes={result ? boxes : []} className="absolute inset-0" alt="Image with every detection" />
                  </AspectBox>
                )}
              </div>
              {result && <BoxLegend />}
            </>)}
          </div>
        </Panel>
        <Panel title="Analysis settings" icon={Layers}>
          <div className="flex flex-col gap-3.5">
            <StoreCameraSelect value={who} onChange={setWho} disabled={pending} />
            <DetectionSettingsForm cfg={cfg} setCfg={setCfg}>
              <Toggle label="Create alerts" checked={createAlerts} onChange={setCreateAlerts} description="Off = test only (detections are still recorded)." />
              <Button variant="primary" icon={Play} disabled={!file || !canOperate} title={opLock} loading={pending} onClick={run} className="mt-1 w-full">Analyze Image</Button>
            </DetectionSettingsForm>
          </div>
        </Panel>
      </div>

      {error && <Panel><ErrorState error={error} onRetry={run} title="Analysis failed" compact /></Panel>}
      {result && (
        <Panel title="Analysis result" icon={FileDown} subtitle={`${a?.analysis_id ?? 'not saved'} · model ${result.model_version} · ${result.tiled ? `${result.tiles} tiles` : 'full frame'} · ${result.image_size.join('×')}px`}>
          <div className="flex flex-col gap-3">
            <ProcessingBanner status={a?.processing_status ?? (count('DETECTED') ? 'COMPLETED_LITTER_DETECTED' : 'COMPLETED_NO_LITTER_DETECTED')} />
            <IdentityBlock identity={result.identity} />
            <div className="grid grid-cols-2 gap-2.5 md:grid-cols-3 xl:grid-cols-6">
              <MetricCard label="Litter events (DETECTED)" value={a?.unique_events ?? count('DETECTED')} tone={count('DETECTED') ? 'crit' : undefined} />
              <MetricCard label="Under review" value={count('UNDER_REVIEW')} tone={count('UNDER_REVIEW') ? 'warn' : undefined} hint="not counted as events" />
              <MetricCard label="Rejected" value={count('REJECTED')} hint="removed by a filter" />
              <MetricCard label="All detections" value={rows.length} />
              <MetricCard label="Highest confidence" value={result.highest_confidence != null ? confidenceText(result.highest_confidence) : '—'} />
              <MetricCard label="Inference time" value={`${num(result.inference_ms, 0)} ms`} />
            </div>
            <Downloads items={[
              { label: 'Detections CSV', path: a?.urls.detections_csv, filename: `${stamp}_detections.csv` },
              { label: 'Summary CSV', path: a?.urls.summary_csv, filename: `${stamp}_summary.csv` },
              { label: 'Annotated image', path: a?.urls.annotated, filename: `${stamp}_annotated.jpg` },
              { label: 'Metadata', path: a?.urls.metadata, filename: `${stamp}_metadata.json` },
            ]} />
          </div>
        </Panel>
      )}
      {result && (
        <Panel title="Every detection" icon={ListChecks} subtitle="Numbers match the labels on the image; click a row to highlight its box" noPad>
          <DataTable<Row> rows={rows} rowKey={(d) => d._i} selectedKey={selected} onRowClick={(d) => setSelected(d._i)} dense pageSize={50}
            empty={<EmptyState compact title="No litter detected by the model" body="The model returned no detection above the weak threshold. This is not a guarantee that the area is litter-free." />}
            columns={[
              { key: 'n', header: '#', cell: (d) => <span className="text-fg-3">{d._i + 1}</span>, width: '44px', sortValue: (d) => d._i },
              { key: 'class', header: 'Class', cell: (d) => <span className="font-medium text-fg">{litterName(d.class)}</span>, sortValue: (d) => d.class },
              { key: 'conf', header: 'Confidence', cell: (d) => <ConfidenceCell value={d.confidence} />, sortValue: (d) => d.confidence },
              { key: 'status', header: 'Status', cell: (d) => <DetectionStatusBadge status={d._status} reason={d.reason} />, sortValue: (d) => d._status },
              { key: 'loc', header: 'Position', cell: (d) => <span><span className="text-fg">{region(d.bbox, w, h)}</span> <span className="text-fg-3">· {Math.round((d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1])).toLocaleString()} px²</span></span>, sortValue: (d) => (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]) },
              { key: 'event', header: 'Event', cell: (d) => d.event_id ? <span className="font-mono text-[11.5px]">{d.event_id}</span> : <span className="text-fg-3">—</span>, hideable: true },
            ]} />
        </Panel>
      )}
    </div>
  )
}

function BoxLegend() {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11.5px] text-fg-3">
      {Object.entries(BOX).map(([k, b]) => (
        <span key={k} className="inline-flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-4 rounded-[2px] border-2" style={{ borderColor: b.color, borderStyle: b.dashed ? 'dashed' : 'solid' }} aria-hidden />
          {k.replace('_', ' ')}
        </span>
      ))}
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
