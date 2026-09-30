'use client'

import { useRouter, useSearchParams } from 'next/navigation'
import { useEffect, useMemo, useRef, useState } from 'react'
import { ClipboardCheck, Plus, RotateCcw, Save, Sparkles, Trash2 } from 'lucide-react'
import { url } from '../../api/client'
import { Api } from '../../api/endpoints'
import type { Bbox, TestLabPred } from '../../api/types'
import { invalidate, useQuery } from '../../hooks/useQuery'
import { className as cls, cx, pct } from '../../lib/format'
import { useApp } from '../../stores/app'
import { Badge, Button, Field, Input, PageHeader, Panel, Select, Toggle } from '../../components/ui/primitives'
import { EmptyState, ErrorState, LoadingState } from '../../components/ui/states'
import { AspectBox, ImageOverlay, STATUS_COLOR, type OverlayBox } from '../../components/vision/Overlay'

type EditBox = { key: string; cls: number; bbox: Bbox }

export default function GroundTruthPage() {
  const params = useSearchParams()
  const router = useRouter()
  const { isAdmin, toast } = useApp()
  const datasets = useQuery('datasets', Api.datasets)
  const models = useQuery('models', Api.models)
  const ds = params.get('dataset') ?? datasets.data?.datasets[0]?.name ?? ''
  const [split, setSplit] = useState('')
  const [search, setSearch] = useState('')
  const imgs = useQuery(ds ? `gt:imgs:${ds}:${split}` : null, () => Api.datasetImages(ds, { split: split || undefined, limit: 500 }))
  const id = params.get('id') ?? imgs.data?.items[0]?.id ?? ''
  const ann = useQuery(ds && id ? `ann:${ds}:${id}` : null, () => Api.annotation(ds, id))
  const names = datasets.data?.datasets.find((d) => d.name === ds)?.classes ?? []
  const [boxes, setBoxes] = useState<EditBox[]>([])
  const [sel, setSel] = useState<string | null>(null)
  const [drawCls, setDrawCls] = useState(0)
  const [drawing, setDrawing] = useState(false)
  const [corner, setCorner] = useState<[number, number] | null>(null)
  const [preds, setPreds] = useState<TestLabPred[] | null>(null)
  const [showPreds, setShowPreds] = useState(true)
  const [saving, setSaving] = useState(false)
  const a = ann.data
  const dirty = useRef(false)

  useEffect(() => {
    if (!a) return
    setBoxes(a.ground_truth.map((g) => ({ key: `g${g.index}`, cls: g.cls_id, bbox: g.bbox })))
    setSel(null)
    setPreds(null)
    setCorner(null)
    dirty.current = false
  }, [a])

  const go = (imageId: string) => {
    if (dirty.current && !confirm('Discard unsaved annotation changes?')) return
    router.replace(`/ground-truth?dataset=${encodeURIComponent(ds)}&id=${encodeURIComponent(imageId)}`)
  }
  const edit = (fn: (b: EditBox[]) => EditBox[]) => { dirty.current = true; setBoxes(fn) }

  const runModel = async () => {
    const v = models.data?.active
    if (!v || !a) return
    try {
      const r = await Api.testlabRun({ dataset: ds, image: id, versions: [v], confidence: 0.25, tiling: 'off' })
      setPreds(r.results[0].detections ?? [])
    } catch (e) {
      toast({ tone: 'crit', title: 'Prediction failed', body: (e as Error).message })
    }
  }

  const save = async () => {
    if (!a) return
    setSaving(true)
    try {
      const out = boxes.map((b) => ({
        cls: b.cls, cx: (b.bbox[0] + b.bbox[2]) / 2 / a.width, cy: (b.bbox[1] + b.bbox[3]) / 2 / a.height,
        w: (b.bbox[2] - b.bbox[0]) / a.width, h: (b.bbox[3] - b.bbox[1]) / a.height,
      }))
      const r = await Api.saveAnnotation(ds, id, out)
      toast({ tone: 'ok', title: 'Labels saved', body: `${r.objects} objects · previous file kept as .txt.bak` })
      dirty.current = false
      invalidate(`ann:${ds}:${id}`)
      invalidate(`dataset:${ds}`)
    } catch (e) {
      toast({ tone: 'crit', title: 'Could not save labels', body: (e as Error).message })
    } finally {
      setSaving(false)
    }
  }

  const overlay = useMemo<OverlayBox[]>(() => [
    ...(showPreds && preds ? preds.map((p, i) => ({ key: `p${i}`, bbox: p.bbox, color: p.match === 'fp' ? STATUS_COLOR.fp : 'var(--exp)', dashed: true, thin: true, label: undefined })) : []),
    ...boxes.map((b) => ({ key: b.key, bbox: b.bbox, color: STATUS_COLOR.gt, label: cls(names[b.cls] ?? `#${b.cls}`).toUpperCase(), selected: sel === b.key, onClick: () => setSel(b.key) })),
  ], [boxes, preds, showPreds, sel, names])

  const selected = boxes.find((b) => b.key === sel)
  const list = (imgs.data?.items ?? []).filter((i) => !search || i.file.toLowerCase().includes(search.toLowerCase()))

  return (
    <div className="flex flex-col gap-3">
      <PageHeader title="Ground truth" description="Review and correct YOLO labels. Saving rewrites the image's label file (the previous version is kept as .txt.bak) and invalidates dataset statistics."
        actions={<Select aria-label="Dataset" value={ds} onChange={(e) => router.replace(`/ground-truth?dataset=${encodeURIComponent(e.target.value)}`)} className="w-60">
          {datasets.data?.datasets.map((d) => <option key={d.name}>{d.name}</option>)}</Select>} />
      {datasets.data?.datasets.length === 0 ? <EmptyState icon={ClipboardCheck} title="No datasets" /> : (
        <div className="grid gap-3 xl:grid-cols-[240px_minmax(0,1fr)_300px]">
          <Panel title="Images" noPad bodyClass="flex flex-col">
            <div className="flex gap-1.5 border-b border-line p-2">
              <Input aria-label="Filter images" placeholder="Filter…" value={search} onChange={(e) => setSearch(e.target.value)} />
              <Select aria-label="Split" value={split} onChange={(e) => setSplit(e.target.value)} className="w-20"><option value="">all</option>{['train', 'val', 'test'].map((s) => <option key={s}>{s}</option>)}</Select>
            </div>
            <ul className="max-h-[70vh] overflow-y-auto">
              {!imgs.data ? <li className="p-2"><LoadingState /></li> : list.map((i) => (
                <li key={i.id}>
                  <button onClick={() => go(i.id)} className={cx('flex w-full items-center justify-between gap-2 px-3 py-1.5 text-left text-[12px] hover:bg-hover', i.id === id && 'bg-info-bg text-fg')}>
                    <span className="truncate">{i.file}</span>
                    <span className={cx('shrink-0 font-mono text-[11px]', i.status === 'issues' ? 'text-crit' : 'text-fg-3')}>{i.status === 'negative' ? 'neg' : i.objects}</span>
                  </button>
                </li>
              ))}
            </ul>
          </Panel>
          <Panel title={id || 'Image'} noPad actions={a && <>
            <Button size="sm" icon={Sparkles} onClick={runModel} disabled={!models.data?.active}>Show predictions</Button>
            <Button size="sm" variant={drawing ? 'primary' : 'secondary'} icon={Plus} disabled={!a.editable} onClick={() => { setDrawing((d) => !d); setCorner(null) }}>{drawing ? 'Click 2 corners…' : 'Add box'}</Button>
          </>}>
            {ann.error ? <ErrorState error={ann.error} compact onRetry={ann.refetch} /> : !a ? <div className="p-3"><LoadingState rows={6} /></div> : (
              <AspectBox width={a.width} height={a.height}>
                <ImageOverlay src={url(`/api/datasets/${encodeURIComponent(ds)}/image?id=${encodeURIComponent(id)}`)} width={a.width} height={a.height} boxes={overlay}
                  className="absolute inset-0" alt={`Dataset image ${id}`}
                  onImageClick={drawing ? (nx, ny) => {
                    const p: [number, number] = [nx * a.width, ny * a.height]
                    if (!corner) { setCorner(p); return }
                    const bb: Bbox = [Math.min(corner[0], p[0]), Math.min(corner[1], p[1]), Math.max(corner[0], p[0]), Math.max(corner[1], p[1])]
                    if (bb[2] - bb[0] >= 2 && bb[3] - bb[1] >= 2) {
                      const key = `n${Date.now()}`
                      edit((bs) => [...bs, { key, cls: drawCls, bbox: bb }])
                      setSel(key)
                    }
                    setCorner(null)
                    setDrawing(false)
                  } : undefined}>
                  {corner && <svg viewBox={`0 0 ${a.width} ${a.height}`} className="pointer-events-none absolute inset-0 size-full"><circle cx={corner[0]} cy={corner[1]} r={Math.max(3, a.width / 200)} fill="var(--warn)" /></svg>}
                </ImageOverlay>
              </AspectBox>
            )}
          </Panel>
          <Panel title="Annotations" bodyClass="flex flex-col gap-3">
            {!a ? <LoadingState /> : (
              <>
                <div className="flex flex-wrap gap-1.5 text-[11px]">
                  <Badge tone="info">GT {boxes.length}</Badge>
                  {preds && <Badge tone="exp">predictions {preds.length}</Badge>}
                  {preds && <Badge tone="crit">unmatched {preds.filter((p) => p.match === 'fp').length}</Badge>}
                  {a.negative && <Badge>negative image</Badge>}
                </div>
                {a.problems.length > 0 && <ul className="rounded-md border border-crit/40 bg-crit-bg p-2 text-[11px] text-crit">{a.problems.map((p) => <li key={p}>{p}</li>)}</ul>}
                {preds && <Toggle label="Show predictions" checked={showPreds} onChange={setShowPreds} />}
                <Field label="Class for new boxes"><Select value={drawCls} onChange={(e) => setDrawCls(Number(e.target.value))}>{names.map((n, i) => <option key={n} value={i}>{cls(n)}</option>)}</Select></Field>
                <ul className="flex max-h-[38vh] flex-col gap-1 overflow-y-auto">
                  {boxes.map((b, i) => (
                    <li key={b.key}>
                      <button onClick={() => setSel(b.key)} className={cx('flex w-full items-center gap-2 rounded-md border px-2 py-1 text-left text-[12px]', sel === b.key ? 'border-info bg-info-bg' : 'border-line hover:bg-hover')}>
                        <span className="font-mono text-fg-3">{i + 1}</span><span className="flex-1 truncate text-fg">{cls(names[b.cls] ?? `#${b.cls}`)}</span>
                        <span className="font-mono text-[11px] text-fg-3">{Math.round(b.bbox[2] - b.bbox[0])}×{Math.round(b.bbox[3] - b.bbox[1])}</span>
                      </button>
                    </li>
                  ))}
                  {boxes.length === 0 && <li className="text-[12px] text-fg-3">No objects labelled.</li>}
                </ul>
                {selected && (
                  <div className="flex flex-col gap-2 rounded-md border border-line bg-panel-2 p-2">
                    <Field label="Change class"><Select value={selected.cls} disabled={!a.editable} onChange={(e) => edit((bs) => bs.map((b) => (b.key === selected.key ? { ...b, cls: Number(e.target.value) } : b)))}>
                      {names.map((n, i) => <option key={n} value={i}>{cls(n)}</option>)}</Select></Field>
                    <Button size="sm" variant="danger" icon={Trash2} disabled={!a.editable} onClick={() => { edit((bs) => bs.filter((b) => b.key !== selected.key)); setSel(null) }}>Delete box</Button>
                  </div>
                )}
                {preds && preds.some((p) => p.match === 'fp') && (
                  <div className="text-[11px] text-fg-3">
                    Predictions without a matching label (possible missing annotations):
                    <ul className="mt-1 flex flex-col gap-1">
                      {preds.filter((p) => p.match === 'fp').map((p, i) => (
                        <li key={i} className="flex items-center justify-between gap-2">
                          <span>{cls(p.class)} {pct(p.confidence, 0)}</span>
                          <Button size="sm" variant="ghost" disabled={!a.editable} onClick={() => edit((bs) => [...bs, { key: `p${Date.now()}${i}`, cls: p.cls_id, bbox: p.bbox }])}>Add as label</Button>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                <div className="mt-auto flex gap-2 border-t border-line pt-3">
                  <Button variant="ghost" icon={RotateCcw} onClick={() => { if (a) { setBoxes(a.ground_truth.map((g) => ({ key: `g${g.index}`, cls: g.cls_id, bbox: g.bbox }))); dirty.current = false } }}>Reset</Button>
                  <Button variant="primary" icon={Save} className="flex-1" loading={saving} disabled={!isAdmin || !a.editable} title={!isAdmin ? 'Admin sign-in required to write label files' : undefined} onClick={save}>Save labels</Button>
                </div>
                {!a.editable && <p className="text-[11px] text-fg-3">Negative images have no label file; move the image into a labelled split to annotate it.</p>}
              </>
            )}
          </Panel>
        </div>
      )}
    </div>
  )
}
