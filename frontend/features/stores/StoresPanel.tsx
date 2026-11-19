'use client'

import { useState } from 'react'
import { Check, Pencil, Plus, Store as StoreIcon, X } from 'lucide-react'
import { Api } from '../../api/endpoints'
import type { Store } from '../../api/types'
import { invalidate, useQuery } from '../../hooks/useQuery'
import { STORE_ID_RE } from '../../lib/litter'
import { lockedFor, useApp } from '../../stores/app'
import { Button, Input, Panel } from '../../components/ui/primitives'
import { DataTable } from '../../components/ui/table'
import { EmptyState, ErrorState, LoadingState } from '../../components/ui/states'

/** Stores list with admin add / rename (POST / PUT /api/stores). Cameras are attached on the camera's own form. */
export function StoresPanel() {
  const { isAdmin, role, loginOn, toast } = useApp()
  const lock = lockedFor('admin', role, loginOn)
  const q = useQuery('stores:list', Api.stores)
  const [adding, setAdding] = useState(false)
  const [draft, setDraft] = useState({ id: '', name: '' })
  const [renaming, setRenaming] = useState<{ id: string; name: string } | null>(null)
  const [busy, setBusy] = useState(false)
  const stores = q.data?.stores ?? []
  const idOk = STORE_ID_RE.test(draft.id.trim()) && !stores.some((s) => s.id === draft.id.trim())

  const run = async (fn: () => Promise<unknown>, ok: string, done: () => void) => {
    setBusy(true)
    try {
      await fn()
      toast({ tone: 'ok', title: ok })
      invalidate('stores')
      invalidate('cameras')
      done()
    } catch (e) {
      toast({ tone: 'crit', title: 'Store not saved', body: (e as Error).message })
    } finally {
      setBusy(false)
    }
  }
  const create = () => run(() => Api.createStore({ id: draft.id.trim(), name: draft.name.trim() }), `Store ${draft.id.trim()} added`,
    () => { setDraft({ id: '', name: '' }); setAdding(false) })
  const rename = () => renaming && run(() => Api.updateStore(renaming.id, { name: renaming.name.trim() }), `Store ${renaming.id} renamed`, () => setRenaming(null))

  return (
    <Panel title="Stores" icon={StoreIcon} subtitle="Every camera can belong to one store; analyses are reported per store" noPad
      actions={<Button size="sm" icon={Plus} disabled={!isAdmin || adding} title={lock} onClick={() => setAdding(true)}>Add store</Button>}>
      {adding && (
        <div className="flex flex-wrap items-end gap-2 border-b border-line px-4 py-3">
          <Input className="w-40" value={draft.id} maxLength={64} placeholder="STORE_001" aria-label="Store ID" aria-invalid={!!draft.id && !idOk}
            onChange={(e) => setDraft({ ...draft, id: e.target.value })} />
          <Input className="min-w-40 flex-1" value={draft.name} maxLength={120} placeholder="Main Street" aria-label="Store name"
            onChange={(e) => setDraft({ ...draft, name: e.target.value })} />
          <Button size="sm" variant="primary" icon={Check} loading={busy} disabled={!idOk || !draft.name.trim()} onClick={create}>Save</Button>
          <Button size="sm" variant="ghost" icon={X} onClick={() => { setAdding(false); setDraft({ id: '', name: '' }) }}>Cancel</Button>
          {draft.id && !idOk && <p className="w-full text-[11px] text-crit">Letters, digits, space, . _ - only (max 64), and not already used.</p>}
        </div>
      )}
      {q.error && !q.data ? <ErrorState error={q.error} onRetry={q.refetch} compact /> : !q.data ? <div className="p-4"><LoadingState rows={2} /></div> : (
        <DataTable<Store> rows={stores} rowKey={(s) => s.id} dense
          empty={<EmptyState compact icon={StoreIcon} title="No stores yet" body="Add a store, then pick it on each camera. Analyses without a store are recorded under UNKNOWN." />}
          columns={[
            { key: 'id', header: 'Store ID', cell: (s) => <span className="font-mono text-fg">{s.id}</span>, sortValue: (s) => s.id },
            { key: 'name', header: 'Name', sortValue: (s) => s.name, cell: (s) => renaming?.id === s.id ? (
              <span className="flex items-center gap-1.5">
                <Input value={renaming.name} maxLength={120} aria-label={`New name for ${s.id}`} autoFocus
                  onChange={(e) => setRenaming({ id: s.id, name: e.target.value })} onKeyDown={(e) => e.key === 'Enter' && renaming.name.trim() && void rename()} />
                <Button size="sm" variant="primary" icon={Check} loading={busy} disabled={!renaming.name.trim()} onClick={rename} aria-label="Save name" />
                <Button size="sm" variant="ghost" icon={X} onClick={() => setRenaming(null)} aria-label="Cancel rename" />
              </span>) : <span className="text-fg">{s.name}</span> },
            { key: 'cams', header: 'Cameras', cell: (s) => s.cameras.length ? <span className="font-mono text-[12px]">{s.cameras.map((c) => c.id).join(', ')}</span> : <span className="text-fg-3">none</span> },
            { key: 'act', header: '', align: 'right', cell: (s) => (
              <Button size="sm" variant="ghost" icon={Pencil} disabled={!isAdmin || renaming?.id === s.id} title={lock}
                onClick={() => setRenaming({ id: s.id, name: s.name })}>Rename</Button>) },
          ]} />
      )}
    </Panel>
  )
}
