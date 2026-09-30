'use client'

import Link from 'next/link'
import { useEffect, useMemo, useState } from 'react'
import { CheckCircle2, CircleHelp, Microscope, Minus, Play, Plus, ShieldX, Upload, XCircle } from 'lucide-react'
import { url } from '../../api/client'
import { Api } from '../../api/endpoints'
import type { GtBox, TestLabPred, TestLabResponse, TestLabResult } from '../../api/types'
import { useQuery } from '../../hooks/useQuery'
import { className as cls, cx, num, pct } from '../../lib/format'
import { useApp } from '../../stores/app'
import { Badge, Button, Field, IconButton, PageHeader, Panel, Segmented, Select, Slider, Toggle } from '../../components/ui/primitives'
import { Columns } from '../../components/ui/charts'
import { MetricCard } from '../../components/ui/kpi'
import { EmptyState, ErrorState, LoadingState } from '../../components/ui/states'
import { AspectBox, ImageOverlay, STATUS_COLOR, type OverlayBox } from '../../components/vision/Overlay'
import { DropZone } from '../detection/common'

type Source = 'dataset' | 'upload'
type Verdict = 'true_positive' | 'false_positive' | 'false_negative' | 'uncertain'

export default function TestLabPage() {
  const { toast } = useApp()
  const models = useQuery('models', Api.models)
  const datasets = useQuery('datasets', Api.datasets)
  const [source, setSource] = useState<Source>('dataset')
  const [ds, setDs] = useState('')
  const [split, setSplit] = useState('test')
  const [image, setImage] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [fileSrc, setFileSrc] = useState<string | null>(null)
  const [versions, setVersions] = useState<string[]>([])
  const [conf, setConf] = useState(0.25)
  const [imgsz, setImgsz] = useState<number | ''>('')
  const [tiling, setTiling] = useState<'off' | 'on' | 'auto'>('off')
  const [res, setRes] = useState<TestLabResponse | null>(null)
  const [err, setErr] = useState<Error | null>(null)
  const [pending, setPending] = useState(false)
  const [review, setReview] = useState(false)
  const [showGt, setShowGt] = useState(true)
  const [marks, setMarks] = useState<Record<string, Verdict>>({})

  const images = useQuery(ds ? `ds:imgs:${ds}:${split}` : null, () => Api.datasetImages(ds, { split, limit: 500 }))
  useEffect(() => { if (!ds && datasets.data?.datasets[0]) setDs(datasets.data.datasets[0].name) }, [datasets.data, ds])
  useEffect(() => { if (images.data?.items[0] && !images.data.items.some((i) => i.id === image)) setImage(images.data.items[0].id) }, [images.data]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { if (!versions.length && models.data?.active) setVersions([models.data.active]) }, [models.data]) // eslint-disable-line react-hooks/exhaustive-deps

  const run = async () => {
    setPending(true)
    setErr(null)
    setMarks({})
    try {
      if (source === 'dataset') {
        setRes(await Api.testlabRun({ dataset: ds, image, versions, confidence: conf, image_size: imgsz || undefined, tiling }))
      } else if (file) {
        const f = new FormData()
        f.append('file', file)
        f.append('versions', versions.join(','))
        f.append('confidence', String(conf))
        f.append('tiling', tiling)
        if (imgsz) f.append('image_size', String(imgsz))
        setRes(await Api.testlabUpload(f))
      }
    } catch (e) {
      setErr(e as Error)
    } finally {
      setPending(false)
    }
  }

  const mark = async (r: TestLabResult, key: string, verdict: Verdict, box: { bbox: number[]; class?: string; confidence?: number }) => {
    if (source !== 'dataset') return
    try {
      await Api.addReview({ dataset: ds, image, model_version: r.model_version, verdict, cls: box.class, confidence: box.confidence, bbox: box.bbox })
      setMarks((m) => ({ ...m, [key]: verdict }))
      if (verdict === 'false_positive' || verdict === 'false_negative') toast({ tone: 'info', title: 'Saved to Hard Examples', body: verdict.replace('_', ' ') })
    } catch (e) {
      toast({ tone: 'crit', title: 'Could not save review', body: (e as Error).message })
    }
  }

  const imgSrc = source === 'dataset' && image ? url(`/api/datasets/${encodeURIComponent(ds)}/image?id=${encodeURIComponent(image)}`) : fileSrc
  const available = models.data?.models.filter((m) => m.file_exists) ?? []
  const canRun = versions.length > 0 && (source === 'dataset' ? !!image : !!file)

  return (
    <div className="flex flex-col gap-3">
      <PageHeader title="Model Test Lab" badge={<Badge tone="exp">research</Badge>}
        description="Run several registered model versions on exactly the same input and compare their raw detections, latency and — for labelled dataset images — true/false positives and misses against ground truth." />
      <Panel title="Test configuration" icon={Microscope}>
        <div className="grid gap-4 lg:grid-cols-[1.2fr_1fr_1fr]">
          <div className="flex flex-col gap-3">
            <Segmented label="Input source" value={source} onChange={(v) => { setSource(v); setRes(null) }}
              options={[{ value: 'dataset', label: 'Dataset image (with ground truth)' }, { value: 'upload', label: 'Upload image', icon: Upload }]} />
            {source === 'dataset' ? (
              <div className="grid grid-cols-[1fr_90px] gap-2">
                <Field label="Dataset"><Select value={ds} onChange={(e) => setDs(e.target.value)}>
                  {datasets.data?.datasets.map((d) => <option key={d.name}>{d.name}</option>)}</Select></Field>
                <Field label="Split"><Select value={split} onChange={(e) => setSplit(e.target.value)}>{['test', 'val', 'train'].map((s) => <option key={s}>{s}</option>)}</Select></Field>
                <div className="col-span-2"><Field label={`Image (${images.data?.total ?? 0})`}><Select value={image} onChange={(e) => setImage(e.target.value)}>
                  {images.data?.items.map((i) => <option key={i.id} value={i.id}>{i.file} · {i.status === 'negative' ? 'negative' : `${i.objects} obj`}</option>)}</Select></Field></div>
                {datasets.data?.datasets.length === 0 && <p className="col-span-2 text-[12px] text-fg-3">No datasets found in the dataset folder.</p>}
              </div>
            ) : (
              <DropZone compact accept="image/*" label={file ? file.name : 'Choose an image'} hint="No ground truth → TP/FP/FN cannot be computed"
                onFile={(f) => { if (fileSrc) URL.revokeObjectURL(fileSrc); setFile(f); setFileSrc(URL.createObjectURL(f)); setRes(null) }} />
            )}
          </div>
          <div className="flex flex-col gap-2">
            <span className="text-[11px] font-medium uppercase tracking-wide text-fg-3">Models ({versions.length}/4)</span>
            {versions.map((v, i) => (
              <div key={i} className="flex gap-1.5">
                <Select aria-label={`Model ${i + 1}`} value={v} onChange={(e) => setVersions((vs) => vs.map((x, j) => (j === i ? e.target.value : x)))}>
                  {available.map((m) => <option key={m.version} value={m.version}>{m.version}{m.active ? ' (active)' : ''}</option>)}
                </Select>
                <IconButton icon={Minus} label="Remove model" onClick={() => setVersions((vs) => vs.filter((_, j) => j !== i))} />
              </div>
            ))}
            <Button size="sm" icon={Plus} disabled={versions.length >= 4 || available.length === 0} onClick={() => setVersions((vs) => [...vs, available.find((m) => !vs.includes(m.version))?.version ?? available[0].version])}>Add model</Button>
            {available.length < 2 && <p className="text-[11px] text-fg-3">Only {available.length} model version is registered. <Link href="/model-lab" className="text-info hover:underline">Register an experiment</Link> to compare architectures or training runs.</p>}
          </div>
          <div className="flex flex-col gap-3">
            <Slider label="Confidence threshold" value={conf} min={0.05} max={0.95} step={0.05} onChange={setConf} />
            <div className="grid grid-cols-2 gap-2">
              <Field label="Image size"><Select value={String(imgsz)} onChange={(e) => setImgsz(e.target.value ? Number(e.target.value) : '')}>
                <option value="">Default</option>{[480, 640, 960, 1280].map((v) => <option key={v} value={v}>{v}</option>)}</Select></Field>
              <Field label="Inference mode"><Select value={tiling} onChange={(e) => setTiling(e.target.value as typeof tiling)}>
                <option value="off">Full frame</option><option value="on">Tiled</option><option value="auto">Auto</option></Select></Field>
            </div>
            <Button variant="primary" icon={Play} disabled={!canRun} loading={pending} onClick={run}>Run all</Button>
          </div>
        </div>
      </Panel>

      {err && <Panel><ErrorState error={err} onRetry={run} compact /></Panel>}
      {pending && !res && <Panel><LoadingState rows={4} label="Running models" /></Panel>}
      {res && (
        <>
          <div className="flex flex-wrap items-center gap-4 rounded-lg border border-line bg-panel px-3 py-2">
            {res.ground_truth && <>
              <Toggle label="Show ground truth" checked={showGt} onChange={setShowGt} />
              <Toggle label="Review mode" checked={review} onChange={setReview} description="Mark predictions TP/FP/uncertain and misses FN" />
            </>}
            <div className="ml-auto flex flex-wrap gap-3 text-[11px] text-fg-3">
              {res.ground_truth ? <>
                <span className="flex items-center gap-1"><span className="size-2 rounded-sm bg-ok" />true positive</span>
                <span className="flex items-center gap-1"><span className="size-2 rounded-sm bg-crit" />false positive</span>
                <span className="flex items-center gap-1"><span className="size-2 rounded-sm bg-warn" />missed (FN)</span>
                <span className="flex items-center gap-1"><span className="size-2 rounded-sm bg-info" />ground truth</span>
              </> : <span>Uploaded image: predictions only (no ground truth)</span>}
            </div>
          </div>
          <div className={cx('grid gap-3', res.results.length === 1 ? 'grid-cols-1 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]' : res.results.length === 2 ? 'lg:grid-cols-2' : 'lg:grid-cols-2 2xl:grid-cols-3')}>
            {res.results.map((r) => (
              <ModelColumn key={r.model_version} r={r} res={res} src={imgSrc} showGt={showGt} review={review} marks={marks} onMark={mark} />
            ))}
            {res.results.length === 1 && res.ground_truth && (
              <Panel title="Ground truth" subtitle={`${res.ground_truth.length} labelled objects`}>
                <AspectBox width={res.image.width} height={res.image.height} className="overflow-hidden rounded-md border border-line">
                  <ImageOverlay src={imgSrc} width={res.image.width} height={res.image.height} className="absolute inset-0" alt="Ground truth"
                    boxes={res.ground_truth.map((g) => ({ key: g.index, bbox: g.bbox, color: STATUS_COLOR.gt, label: cls(g.class).toUpperCase(), sub: g.size }))} />
                </AspectBox>
              </Panel>
            )}
          </div>
        </>
      )}
      {!res && !pending && !err && <Panel><EmptyState icon={Microscope} title="Configure a test and press “Run all”" body="Results appear side by side for each selected model." /></Panel>}
    </div>
  )
}

function ModelColumn({ r, res, src, showGt, review, marks, onMark }: {
  r: TestLabResult; res: TestLabResponse; src: string | null; showGt: boolean; review: boolean; marks: Record<string, Verdict>
  onMark: (r: TestLabResult, key: string, v: Verdict, box: { bbox: number[]; class?: string; confidence?: number }) => void
}) {
  const [sel, setSel] = useState<string | null>(null)
  const gt = res.ground_truth
  const boxes = useMemo<OverlayBox[]>(() => {
    if (r.error) return []
    const out: OverlayBox[] = []
    if (gt && showGt) gt.forEach((g) => out.push({ key: `g${g.index}`, bbox: g.bbox, color: STATUS_COLOR.gt, thin: true, dashed: true }))
    if (gt) (r.missed_gt ?? []).forEach((i) => {
      const g = gt.find((x) => x.index === i)!
      out.push({ key: `m${i}`, bbox: g.bbox, color: STATUS_COLOR.fn, label: `MISSED ${cls(g.class).toUpperCase()}`, onClick: () => setSel(`m${i}`), selected: sel === `m${i}` })
    })
    ;(r.detections ?? []).forEach((d, i) => out.push({
      key: `p${i}`, bbox: d.bbox, color: d.match ? STATUS_COLOR[d.match] : 'var(--info)', label: d.class.toUpperCase(), sub: pct(d.confidence, 0),
      onClick: () => setSel(`p${i}`), selected: sel === `p${i}`,
    }))
    return out
  }, [r, gt, showGt, sel])

  if (r.error) return <Panel title={r.model_version}><ErrorState error={new Error(r.error)} compact /></Panel>
  const c = r.counts
  const selPred: TestLabPred | undefined = sel?.startsWith('p') ? r.detections?.[Number(sel.slice(1))] : undefined
  const selGt: GtBox | undefined = sel?.startsWith('m') && gt ? gt.find((g) => g.index === Number(sel.slice(1))) : undefined

  return (
    <Panel title={r.model_version} subtitle={`${r.device} · ${r.tiled ? `${r.tiles} tiles` : 'full frame'}`} noPad>
      <AspectBox width={res.image.width} height={res.image.height} className="border-b border-line">
        <ImageOverlay src={src} width={res.image.width} height={res.image.height} boxes={boxes} className="absolute inset-0" alt={`Predictions of ${r.model_version}`} />
      </AspectBox>
      <div className="flex flex-col gap-3 p-3">
        <div className="grid grid-cols-3 gap-2">
          <MetricCard label="Objects" value={r.detections?.length ?? 0} />
          <MetricCard label="Latency" value={`${num(r.latency_ms, 0)} ms`} />
          <MetricCard label="FPS" value={num(r.fps, 1)} />
        </div>
        {c && (
          <div className="grid grid-cols-5 gap-2">
            <MetricCard label="Ground truth" value={c.ground_truth} />
            <MetricCard label="Model" value={c.predicted} />
            <MetricCard label="Correct" value={c.true_positive} tone="ok" />
            <MetricCard label="Missed" value={c.missed} tone={c.missed ? 'warn' : undefined} />
            <MetricCard label="Extra" value={c.false_positive} tone={c.false_positive ? 'crit' : undefined} />
          </div>
        )}
        <div>
          <span className="text-[11px] uppercase tracking-wide text-fg-3">Confidence distribution</span>
          <Columns ariaLabel={`Confidence distribution for ${r.model_version}`} height={60} color="var(--exp)"
            data={(r.confidence_histogram ?? []).map((v, i) => ({ label: (i / 10).toFixed(1), value: v }))} />
        </div>
        {review && (selPred || selGt) && (
          <div className="rounded-md border border-line bg-panel-2 p-2.5">
            <div className="mb-2 text-[12px] text-fg">
              {selPred ? <>Prediction: <b>{cls(selPred.class)}</b> {pct(selPred.confidence, 0)} · auto-match <Badge tone={selPred.match === 'tp' ? 'ok' : 'crit'}>{selPred.match}</Badge></>
                : <>Ground truth missed: <b>{cls(selGt!.class)}</b> ({selGt!.size})</>}
              {marks[`${r.model_version}:${sel}`] && <Badge tone="info" className="ml-2">marked {marks[`${r.model_version}:${sel}`].replace('_', ' ')}</Badge>}
            </div>
            <div className="flex flex-wrap gap-1.5">
              {selPred ? <>
                <Button size="sm" icon={CheckCircle2} onClick={() => onMark(r, `${r.model_version}:${sel}`, 'true_positive', selPred)}>True positive</Button>
                <Button size="sm" icon={XCircle} variant="danger" onClick={() => onMark(r, `${r.model_version}:${sel}`, 'false_positive', selPred)}>False positive</Button>
              </> : (
                <Button size="sm" icon={ShieldX} onClick={() => onMark(r, `${r.model_version}:${sel}`, 'false_negative', { bbox: selGt!.bbox, class: selGt!.class })}>Confirm false negative</Button>
              )}
              <Button size="sm" variant="ghost" icon={CircleHelp} onClick={() => onMark(r, `${r.model_version}:${sel}`, 'uncertain', selPred ?? { bbox: selGt!.bbox, class: selGt!.class })}>Uncertain</Button>
            </div>
          </div>
        )}
        {review && !sel && <p className="text-[11px] text-fg-3">Click a box on the image to review it.</p>}
      </div>
    </Panel>
  )
}
