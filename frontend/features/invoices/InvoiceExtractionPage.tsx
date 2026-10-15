'use client'

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { AlertTriangle, Download, FileSpreadsheet, FileWarning, ImageIcon, Play, ReceiptText, RotateCcw, UploadCloud, X } from 'lucide-react'
import { Api } from '../../api/endpoints'
import type { InvoiceExtractResponse, InvoiceExtraction, InvoiceProductRow } from '../../api/types'
import { useQuery } from '../../hooks/useQuery'
import { cx } from '../../lib/format'
import { useApp } from '../../stores/app'
import { Badge, Button, KeyValue, PageHeader, Panel, Segmented, type Tone } from '../../components/ui/primitives'
import { MetricCard } from '../../components/ui/kpi'
import { DataTable } from '../../components/ui/table'
import { Drawer } from '../../components/ui/overlay'
import { EmptyState, ErrorState } from '../../components/ui/states'

const ACCEPT = '.jpg,.jpeg,.png,image/jpeg,image/png'
const MAX_FILES = 25
const STATUS: Record<InvoiceExtraction['status'], { tone: Tone; label: string }> = {
  ok: { tone: 'ok', label: 'Extracted' },
  needs_review: { tone: 'warn', label: 'Needs review' },
  failed: { tone: 'crit', label: 'Failed' },
  duplicate: { tone: 'neutral', label: 'Duplicate' },
}

type Picked = { file: File; url: string; key: string }
type Row = InvoiceProductRow & { _k: string }

function downloadCsv(text: string, name: string) {
  // BOM so Excel opens UTF-8 (₹, non-English product names) correctly
  const blob = new Blob(['﻿' + text], { type: 'text/csv;charset=utf-8' })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = name
  a.click()
  setTimeout(() => URL.revokeObjectURL(a.href), 1000)
}

/** Local-time file stamp, e.g. 20241014_1336. */
const stamp = () => {
  const d = new Date()
  const p = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}`
}

export default function InvoiceExtractionPage() {
  const { toast } = useApp()
  const engine = useQuery('invoice:engine', Api.invoiceEngine)
  const [files, setFiles] = useState<Picked[]>([])
  const [over, setOver] = useState(false)
  const [pending, setPending] = useState(false)
  const [error, setError] = useState<Error | null>(null)
  const [result, setResult] = useState<InvoiceExtractResponse | null>(null)
  const [filter, setFilter] = useState<'all' | 'review'>('all')
  const [selected, setSelected] = useState<Row | null>(null)
  const input = useRef<HTMLInputElement>(null)
  const abort = useRef<AbortController | null>(null)
  const filesRef = useRef(files)
  filesRef.current = files

  useEffect(() => () => {
    filesRef.current.forEach((f) => URL.revokeObjectURL(f.url))
    abort.current?.abort()
  }, [])

  const add = useCallback((list: FileList | null) => {
    if (!list) return
    const ok = Array.from(list).filter((f) => /\.(jpe?g|png)$/i.test(f.name))
    if (ok.length < list.length) toast({ tone: 'warn', title: 'Some files were skipped', body: 'Only JPG, JPEG and PNG images are supported.' })
    setFiles((cur) => {
      const keys = new Set(cur.map((c) => c.key))
      const next = [...cur]
      for (const f of ok) {
        const key = `${f.name}:${f.size}:${f.lastModified}`
        if (!keys.has(key) && next.length < MAX_FILES) { next.push({ file: f, url: URL.createObjectURL(f), key }); keys.add(key) }
      }
      if (cur.length + ok.length > MAX_FILES) toast({ tone: 'warn', title: `Maximum ${MAX_FILES} images per run` })
      return next
    })
  }, [toast])

  const remove = (key: string) => setFiles((cur) => {
    const f = cur.find((c) => c.key === key)
    if (f) URL.revokeObjectURL(f.url)
    return cur.filter((c) => c.key !== key)
  })

  const clear = () => {
    files.forEach((f) => URL.revokeObjectURL(f.url))
    setFiles([]); setResult(null); setError(null); setSelected(null)
  }

  const run = async () => {
    if (!files.length) return
    const form = new FormData()
    files.forEach((f) => form.append('files', f.file, f.file.name))
    abort.current?.abort()
    abort.current = new AbortController()
    setPending(true); setError(null); setSelected(null)
    try {
      const r = await Api.extractInvoices(form, abort.current.signal)
      setResult(r)
      setFilter('all')
      toast({ tone: r.totals.rows_needing_review || r.totals.failed ? 'warn' : 'ok',
        title: `${r.totals.rows} product row(s) from ${r.totals.invoices} invoice(s)`,
        body: r.totals.rows_needing_review ? `${r.totals.rows_needing_review} row(s) need manual review` : undefined })
    } catch (e) {
      if ((e as Error).name !== 'AbortError') setError(e as Error)
    } finally {
      setPending(false)
    }
  }

  const rows: Row[] = useMemo(() => (result?.products ?? []).map((p) => ({ ...p, _k: `${p.source_file}#${p.line_no}` })), [result])
  const shown = filter === 'review' ? rows.filter((r) => r.needs_review === 'YES') : rows
  const invoiceOf = (file: string) => result?.invoices.find((i) => i.source_file === file)
  const imageOf = (file: string) => files.find((f) => f.file.name === file)?.url

  return (
    <div className="flex flex-col gap-3">
      <PageHeader title="Invoice extraction"
        description="Upload invoice images (JPG, JPEG, PNG) → Extract products → Download the CSV. Text is read by a local OCR engine on this machine; nothing is sent to a cloud service."
        badge={engine.data && (engine.data.available
          ? <Badge tone="ok" dot>OCR: {engine.data.engine}</Badge>
          : <Badge tone="crit" dot title={engine.data.error}>OCR engine not installed</Badge>)} />

      <div role="note" className="flex items-start gap-2 rounded-lg border border-line bg-warn-bg px-3.5 py-2.5 text-[12.5px] text-warn">
        <AlertTriangle className="mt-0.5 size-4 shrink-0" aria-hidden />
        <span>OCR extraction is <b>not 100% accurate</b>. Missing or uncertain values are left as read (never guessed) and the row is marked
          <b> Needs review</b>. Check those rows against the invoice image before using the data.</span>
      </div>

      <Panel title="Invoice images" icon={ImageIcon} subtitle={files.length ? `${files.length} selected` : undefined}
        actions={files.length > 0 && <>
          <Button size="sm" variant="ghost" icon={RotateCcw} onClick={clear} disabled={pending}>Clear</Button>
          <Button size="sm" variant="primary" icon={Play} onClick={run} loading={pending} disabled={engine.data?.available === false}>
            {pending ? 'Extracting…' : 'Extract products'}
          </Button>
        </>}>
        <div
          role="button" tabIndex={0} aria-label="Add invoice images"
          onClick={() => input.current?.click()}
          onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && input.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setOver(true) }}
          onDragLeave={() => setOver(false)}
          onDrop={(e) => { e.preventDefault(); setOver(false); add(e.dataTransfer.files) }}
          className={cx('flex cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border border-dashed text-center transition-colors',
            files.length ? 'p-5' : 'min-h-[220px] p-10', over ? 'border-info bg-info-bg' : 'border-line-strong bg-panel hover:border-info/60')}>
          <UploadCloud className="size-6 text-fg-3" aria-hidden />
          <p className="text-[13px] font-medium text-fg">{files.length ? 'Add more invoice images' : 'Drop invoice images or click to browse'}</p>
          <p className="text-[12px] text-fg-3">JPG, JPEG or PNG · up to {MAX_FILES} images per run · several invoices can be processed together</p>
          <input ref={input} type="file" accept={ACCEPT} multiple className="hidden" onChange={(e) => { add(e.target.files); e.target.value = '' }} />
        </div>
        {files.length > 0 && (
          <ul className="mt-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
            {files.map((f) => {
              const inv = invoiceOf(f.file.name)
              return (
                <li key={f.key} className="flex min-w-0 items-center gap-2.5 rounded-md border border-line bg-panel-2 p-2">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={f.url} alt="" className="size-12 shrink-0 rounded object-cover" />
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-[12.5px] text-fg" title={f.file.name}>{f.file.name}</div>
                    <div className="text-[11px] text-fg-3">{(f.file.size / 1024).toFixed(0)} KB</div>
                    {inv && <Badge tone={STATUS[inv.status].tone} className="mt-0.5">{STATUS[inv.status].label}</Badge>}
                  </div>
                  <button type="button" aria-label={`Remove ${f.file.name}`} disabled={pending} onClick={() => remove(f.key)}
                    className="rounded p-1 text-fg-3 hover:bg-hover hover:text-fg disabled:opacity-40"><X className="size-3.5" /></button>
                </li>
              )
            })}
          </ul>
        )}
        {engine.data?.available === false && (
          <p className="mt-3 text-[12px] text-crit">{engine.data.error} — run <code>pip install -r invoice_extractor/requirements.txt</code> and restart the backend.</p>
        )}
      </Panel>

      {error && <Panel><ErrorState error={error} onRetry={run} title="Extraction failed" compact /></Panel>}

      {result && (
        <>
          <Panel title="Extracted products" icon={FileSpreadsheet}
            subtitle={`OCR engine ${result.engine}${result.skipped_identical.length ? ` · ${result.skipped_identical.length} identical upload(s) skipped` : ''}`}
            actions={<>
              <Button size="sm" variant="primary" icon={Download} disabled={!rows.length}
                onClick={() => downloadCsv(result.csv.products, `products_${stamp()}.csv`)}>products.csv</Button>
              <Button size="sm" icon={FileWarning} onClick={() => downloadCsv(result.csv.review, `extraction_errors_${stamp()}.csv`)}>Review report</Button>
              <Button size="sm" variant="subtle" icon={Download} onClick={() => downloadCsv(result.csv.summary, `invoices_summary_${stamp()}.csv`)}>Summary</Button>
            </>} noPad>
            <div className="grid grid-cols-2 gap-2 p-3 md:grid-cols-5">
              <MetricCard label="Invoices" value={result.totals.invoices} />
              <MetricCard label="Product rows" value={result.totals.rows} />
              <MetricCard label="Rows needing review" value={result.totals.rows_needing_review} tone={result.totals.rows_needing_review ? 'warn' : 'ok'} />
              <MetricCard label="Failed images" value={result.totals.failed} tone={result.totals.failed ? 'crit' : undefined} />
              <MetricCard label="Duplicates" value={result.totals.duplicates} hint={result.totals.duplicates ? 'left out of products.csv' : undefined} />
            </div>
            <div className="flex items-center justify-between gap-2 border-t border-line px-3 py-2">
              <Segmented size="sm" label="Row filter" value={filter} onChange={setFilter} options={[
                { value: 'all', label: `All rows (${rows.length})` },
                { value: 'review', label: `Needs review (${rows.filter((r) => r.needs_review === 'YES').length})` }]} />
              <span className="text-[11.5px] text-fg-3">Click a row to compare it with the image</span>
            </div>
            <DataTable<Row> rows={shown} rowKey={(r) => r._k} dense onRowClick={setSelected} selectedKey={selected?._k}
              empty={<EmptyState compact title={filter === 'review' ? 'No rows need review' : 'No product rows extracted'}
                body={filter === 'review' ? undefined : 'See the invoices below and the review report for the reason.'} />}
              columns={[
                { key: 'file', header: 'Image', cell: (r) => <span className="block max-w-[160px] truncate text-fg-2" title={r.source_file}>{r.source_file}</span>, sortValue: (r) => r.source_file },
                { key: 'inv', header: 'Invoice no.', cell: (r) => r.invoice_number || <span className="text-fg-3">—</span>, sortValue: (r) => r.invoice_number },
                { key: 'line', header: '#', cell: (r) => <span className="text-fg-3">{r.line_no}</span>, width: '36px', sortValue: (r) => Number(r.line_no) },
                { key: 'name', header: 'Product', cell: (r) => r.product_name || <span className="text-crit">missing</span>, sortValue: (r) => r.product_name },
                { key: 'sku', header: 'SKU / code', cell: (r) => r.sku || '—', hideable: true },
                { key: 'hsn', header: 'HSN', cell: (r) => r.hsn_code || '—', hideable: true },
                { key: 'qty', header: 'Qty', align: 'right', cell: (r) => <span className="tabular">{r.quantity || '—'}</span> },
                { key: 'unit', header: 'Unit', cell: (r) => r.unit || '—', hideable: true },
                { key: 'price', header: 'Unit price', align: 'right', cell: (r) => <span className="tabular">{r.unit_price || '—'}</span> },
                { key: 'disc', header: 'Disc.', align: 'right', cell: (r) => <span className="tabular">{r.discount || '—'}</span>, hideable: true },
                { key: 'taxable', header: 'Taxable', align: 'right', cell: (r) => <span className="tabular">{r.taxable_value || '—'}</span>, hideable: true, defaultHidden: true },
                { key: 'rate', header: 'Tax %', align: 'right', cell: (r) => <span className="tabular">{r.tax_rate || '—'}</span>, hideable: true },
                { key: 'tax', header: 'Tax amt', align: 'right', cell: (r) => <span className="tabular">{r.tax_amount || '—'}</span>, hideable: true },
                { key: 'total', header: 'Line total', align: 'right', cell: (r) => <span className="tabular font-medium text-fg">{r.line_total || '—'}</span>, sortValue: (r) => Number(r.line_total) || 0 },
                { key: 'check', header: 'Qty × price', cell: (r) => <Badge tone={r.arithmetic_check === 'ok' ? 'ok' : r.arithmetic_check === 'mismatch' ? 'crit' : 'neutral'}>{r.arithmetic_check.replace('_', ' ')}</Badge>, hideable: true },
                { key: 'review', header: 'Status', cell: (r) => r.needs_review === 'YES'
                  ? <span title={r.review_reasons}><Badge tone="warn" dot>Needs review</Badge></span>
                  : <Badge tone="ok" dot>OK</Badge>, sortValue: (r) => r.needs_review },
              ]} />
          </Panel>

          <Panel title="Invoices" icon={ReceiptText} noPad>
            <ul className="divide-y divide-line">
              {result.invoices.map((inv) => {
                const issues = inv.issues.filter((i) => i.severity !== 'info' && i.line_no == null)
                return (
                  <li key={inv.sha256 || inv.source_file} className="flex flex-col gap-2 p-3 md:flex-row md:items-start">
                    <div className="min-w-0 md:w-64">
                      <div className="truncate text-[13px] font-medium text-fg" title={inv.source_file}>{inv.source_file}</div>
                      <div className="mt-1 flex flex-wrap gap-1.5">
                        <Badge tone={STATUS[inv.status].tone} dot>{STATUS[inv.status].label}</Badge>
                        <Badge>{inv.items.length} item(s)</Badge>
                      </div>
                    </div>
                    <div className="min-w-0 flex-1">
                      <KeyValue cols={3} items={[
                        ['Invoice no.', inv.invoice_number || '—'], ['Date', inv.invoice_date || '—'], ['Vendor', inv.vendor_name || '—'],
                        ['GSTIN', inv.vendor_gstin || '—'], ['Printed total', inv.invoice_total || '—'], ['Sum of lines', inv.quality.sum_line_totals ?? '—'],
                      ]} />
                      {issues.length > 0 && (
                        <ul className="mt-2 space-y-0.5 text-[12px]">
                          {issues.map((i, k) => (
                            <li key={k} className={i.severity === 'error' ? 'text-crit' : i.severity === 'review' ? 'text-warn' : 'text-fg-3'}>• {i.detail}</li>
                          ))}
                        </ul>
                      )}
                    </div>
                  </li>
                )
              })}
            </ul>
          </Panel>
        </>
      )}

      <Drawer open={!!selected} onClose={() => setSelected(null)} width="w-[640px]"
        title={selected ? selected.product_name || `Line ${selected.line_no}` : ''}
        subtitle={selected ? `${selected.source_file} · invoice ${selected.invoice_number || '—'} · line ${selected.line_no}` : undefined}>
        {selected && (
          <div className="flex flex-col gap-3">
            {selected.needs_review === 'YES' && (
              <div className="rounded-md bg-warn-bg px-3 py-2 text-[12.5px] text-warn">
                <div className="mb-1 font-semibold">Why this row needs review</div>
                <ul className="space-y-0.5">{selected.review_reasons.split('; ').map((r, i) => <li key={i}>• {r}</li>)}</ul>
              </div>
            )}
            <KeyValue cols={2} items={[
              ['Product', selected.product_name || '—'], ['SKU / code', selected.sku || '—'], ['HSN', selected.hsn_code || '—'],
              ['Quantity', `${selected.quantity || '—'} ${selected.unit}`], ['Unit price', selected.unit_price || '—'], ['Discount', selected.discount || '—'],
              ['Taxable value', selected.taxable_value || '—'], ['Tax', `${selected.tax_rate ? selected.tax_rate + '% · ' : ''}${selected.tax_amount || '—'}`],
              ['Line total', selected.line_total || '—'], ['OCR confidence', selected.ocr_confidence || '—'],
            ]} />
            <div>
              <div className="mb-1 text-[11px] uppercase tracking-wide text-fg-3">Text as read by OCR</div>
              <pre className="whitespace-pre-wrap rounded-md border border-line bg-panel-2 p-2 text-[12px] text-fg-2">{selected.raw_text}</pre>
            </div>
            {imageOf(selected.source_file) && (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={imageOf(selected.source_file)} alt={`Invoice ${selected.source_file}`} className="w-full rounded-md border border-line" />
            )}
          </div>
        )}
      </Drawer>
    </div>
  )
}
