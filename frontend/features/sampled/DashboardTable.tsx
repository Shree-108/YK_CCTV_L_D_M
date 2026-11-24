'use client'

// Video Analysis results in the dashboard import schema (backend/services/dashboard_rows.py via
// GET /api/detect/video/{job_id}/report). The table and both CSV downloads come from the same server rows.
// Default: one row per screenshot; the toggle switches to one row per detected object. All rows are paged.
import { useState } from 'react'
import { Download, FileText } from 'lucide-react'
import { downloadAuthed, url } from '../../api/client'
import { Api, videoDashboardReportPath } from '../../api/endpoints'
import type { DashboardRow, DashboardStatus } from '../../api/sampledTypes'
import { useQuery } from '../../hooks/useQuery'
import { useApp } from '../../stores/app'
import { Badge, Button, Panel, Toggle, type Tone } from '../../components/ui/primitives'
import { DataTable } from '../../components/ui/table'
import { EmptyState, ErrorState } from '../../components/ui/states'

const STATUS: Record<DashboardStatus, { tone: Tone; hint: string }> = {
  litter_detected: { tone: 'crit', hint: 'At least one confirmed litter detection in the screenshot' },
  clean: { tone: 'ok', hint: 'No confirmed litter (possible litter is listed in the remark)' },
  view_blocked: { tone: 'warn', hint: 'The screenshot shows no usable view; the reason is in the remark' },
}

export function DashboardTable({ jobId, detailedCsv }: { jobId: string; detailedCsv?: string | null }) {
  const { toast } = useApp()
  const [perObject, setPerObject] = useState(false)
  const [busy, setBusy] = useState<string | null>(null)
  const q = useQuery(`sampled:dashboard:${jobId}:${perObject ? 'obj' : 'shot'}`, () => Api.videoDashboardReport(jobId, perObject))
  const download = async (key: string, path: string, name: string) => {
    setBusy(key)
    try { await downloadAuthed(path, name) } catch (e) { toast({ tone: 'crit', title: 'Download failed', body: (e as Error).message }) } finally { setBusy(null) }
  }
  const suffix = perObject ? '_objects' : ''
  const s = q.data?.summary
  return (
    <Panel title="Results" icon={FileText}
      subtitle={perObject ? 'One row per detected object (confirmed and possible litter)' : 'One row per screenshot, in the dashboard import format'}
      actions={<>
        <Button size="sm" variant="primary" icon={Download} loading={busy === 'csv'}
          onClick={() => download('csv', videoDashboardReportPath(jobId, 'csv', perObject), `${jobId}_dashboard_import${suffix}.csv`)}>Download CSV (dashboard import)</Button>
        <Button size="sm" icon={Download} loading={busy === 'full'}
          onClick={() => download('full', videoDashboardReportPath(jobId, 'csv_full', perObject), `${jobId}_dashboard_import_full${suffix}.csv`)}>Download CSV (full)</Button>
        {detailedCsv && <Button size="sm" variant="ghost" icon={Download} loading={busy === 'detailed'}
          onClick={() => download('detailed', detailedCsv, `${jobId}_detection_report.csv`)}>Detailed CSV</Button>}
      </>}>
      <div className="flex flex-col gap-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <Toggle label="One row per detected object" checked={perObject} onChange={setPerObject} />
          {s && (
            <div className="flex flex-wrap gap-1.5 text-[12px]">
              {(Object.keys(STATUS) as DashboardStatus[]).map((k) => <Badge key={k} tone={STATUS[k].tone}>{k} {s[k]}</Badge>)}
            </div>
          )}
        </div>
        {q.error && !q.data ? <ErrorState error={q.error} onRetry={q.refetch} compact /> : (
          <DataTable<DashboardRow> rows={q.data?.rows ?? []} pageSize={25} dense
            rowKey={(r) => `${r.screenshot}:${r.row_kind}:${r.remark}`}
            toolbar={<span className="text-[12px] text-fg-3">{q.data ? `${q.data.total} row${q.data.total === 1 ? '' : 's'}` : 'Loading…'}</span>}
            empty={q.data ? <EmptyState compact title="No rows" /> : <div className="p-3"><div className="skeleton h-24 rounded-md" /></div>}
            columns={[
              { key: 'timestamp', header: 'Captured At', sortValue: (r) => r.timestamp, cell: (r) => <span className="whitespace-nowrap font-mono text-[12px]">{r.timestamp}</span> },
              { key: 'camera_id', header: 'Camera ID', cell: (r) => <span className="whitespace-nowrap font-mono text-[12px]">{r.camera_id}</span> },
              { key: 'image_name', header: 'Image Name', cell: (r) => r.image_url
                ? <a href={url(r.image_url)} className="font-mono text-[11.5px] text-info underline-offset-2 hover:underline" title="Download the annotated JPG">{r.image_name}</a>
                : <span className="font-mono text-[11.5px]">{r.image_name}</span> },
              { key: 'litter_class', header: 'Litter Class', cell: (r) => <span className="font-mono text-[12px]">{r.litter_class}</span> },
              { key: 'confidence_score', header: 'Confidence Score', align: 'right', sortValue: (r) => (r.confidence_score ? Number(r.confidence_score) : -1),
                cell: (r) => <span className="tabular">{r.confidence_score}</span> },
              { key: 'detection_status', header: 'Detection Status', sortValue: (r) => r.detection_status,
                cell: (r) => <Badge tone={STATUS[r.detection_status]?.tone ?? 'neutral'} dot title={STATUS[r.detection_status]?.hint}>{r.detection_status}</Badge> },
              { key: 'remark', header: 'Remark', className: 'min-w-[260px]', cell: (r) => <span className="text-[12px] text-fg-2">{r.remark}</span> },
              { key: 'store_name', header: 'Store Name', hideable: true, cell: (r) => r.store_name },
              { key: 'source_file', header: 'Source File', hideable: true, cell: (r) => <span className="break-all font-mono text-[11.5px]">{r.source_file}</span> },
            ]} />
        )}
        <p className="text-[11.5px] text-fg-3">Empty cells mean no value. Store Name and Source File are not importer columns: they are only in “Download CSV (full)”.</p>
      </div>
    </Panel>
  )
}
