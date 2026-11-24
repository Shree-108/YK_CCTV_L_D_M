'use client'

// CSV export (alerts / detections). Downloads through fetch so the session cookie or admin token is sent and a refusal
// becomes a short message instead of the server's JSON. With login on, exports are admin-only (the server enforces
// it; the button is only disabled here so other roles are not offered an action that will be refused).
// Optional importer format (``importerQuery``): offers the ai_garbage_detection import CSV (9 columns, one row per
// camera check, built by backend/services/garbage_export.py) next to the old table export, which stays available.
// Pages that do not pass ``importerQuery`` render exactly the button they always had.
import { useState } from 'react'
import { Download, type LucideIcon } from 'lucide-react'
import { ApiError, downloadAuthed } from '../../api/client'
import { Button, Select } from '../../components/ui/primitives'
import { lockedFor, useApp } from '../../stores/app'

const stamp = () => new Date().toISOString().slice(0, 19).replace(/[-:]/g, '').replace('T', '_')

type Format = 'importer' | 'legacy'

export function ExportCsvButton({ table, query, label = 'Export CSV', icon = Download, importerQuery }: {
  table: 'alerts' | 'detections'
  query?: string
  label?: string
  icon?: LucideIcon
  importerQuery?: string
}) {
  const { role, loginOn, toast } = useApp()
  const [busy, setBusy] = useState(false)
  const [format, setFormat] = useState<Format>('importer')
  const locked = loginOn ? lockedFor('admin', role, loginOn) : undefined          // login off: open, as before
  const importer = importerQuery !== undefined && format === 'importer'
  const button = (
    <Button icon={icon} loading={busy} disabled={!!locked}
      title={locked ? 'Only administrators can export CSV files'
        : importer ? 'Download the import-ready checks in the importer CSV format (9 columns)' : `Download ${table} as CSV`}
      onClick={async () => {
        setBusy(true)
        try {
          if (importer) await downloadAuthed(`/api/results/checks.csv${importerQuery ? `?${importerQuery}` : ''}`, `ai_garbage_detection_${stamp()}.csv`)
          else await downloadAuthed(`/api/export/${table}.csv${query ? `?${query}` : ''}`, `${table}_${stamp()}.csv`)
        } catch (e) {
          const status = e instanceof ApiError ? e.status : -1
          toast({ tone: 'crit', title: status === 403 ? 'Permission denied' : 'Export failed',
                  body: e instanceof ApiError ? e.message : 'The download failed. Please try again.' })
        } finally {
          setBusy(false)
        }
      }}>{label}</Button>
  )
  if (importerQuery === undefined) return button
  return (
    <span className="inline-flex items-center gap-2">
      <Select aria-label="CSV format" className="w-auto" value={format} onChange={(e) => setFormat(e.target.value as Format)}>
        <option value="importer">Importer format (9 columns)</option>
        <option value="legacy">Old format ({table} CSV)</option>
      </Select>
      {button}
    </span>
  )
}
