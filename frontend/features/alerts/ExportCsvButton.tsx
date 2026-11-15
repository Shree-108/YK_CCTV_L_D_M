'use client'

// CSV export (alerts / detections). Downloads through fetch so the session cookie or admin token is sent and a refusal
// becomes a short message instead of the server's JSON. With login on, exports are admin-only (the server enforces
// it; the button is only disabled here so other roles are not offered an action that will be refused).
import { useState } from 'react'
import { Download, type LucideIcon } from 'lucide-react'
import { ApiError, downloadAuthed } from '../../api/client'
import { Button } from '../../components/ui/primitives'
import { lockedFor, useApp } from '../../stores/app'

const stamp = () => new Date().toISOString().slice(0, 19).replace(/[-:]/g, '').replace('T', '_')

export function ExportCsvButton({ table, query, label = 'Export CSV', icon = Download }: {
  table: 'alerts' | 'detections'
  query?: string
  label?: string
  icon?: LucideIcon
}) {
  const { role, loginOn, toast } = useApp()
  const [busy, setBusy] = useState(false)
  const locked = loginOn ? lockedFor('admin', role, loginOn) : undefined          // login off: open, as before
  return (
    <Button icon={icon} loading={busy} disabled={!!locked}
      title={locked ? 'Only administrators can export CSV files' : `Download ${table} as CSV`}
      onClick={async () => {
        setBusy(true)
        try {
          await downloadAuthed(`/api/export/${table}.csv${query ? `?${query}` : ''}`, `${table}_${stamp()}.csv`)
        } catch (e) {
          const status = e instanceof ApiError ? e.status : -1
          toast({ tone: 'crit', title: status === 403 ? 'Permission denied' : 'Export failed',
                  body: e instanceof ApiError ? e.message : 'The download failed. Please try again.' })
        } finally {
          setBusy(false)
        }
      }}>{label}</Button>
  )
}
